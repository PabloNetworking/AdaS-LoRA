"""
AdaS-LayerAware utility functions: layer-aware component identity.

This module provides functions for layer-aware-specific operations:
- ComponentID = (layer_key, global_idx) internal representation
- Grouped payload metadata for wire format
- Layer-aware scoring, costing, and selection
- Residual buffers keyed by ComponentID

Key differences from AdaS:
- AdaS: flat integer component index shared across all layers
- layer-aware: exact layer-specific ComponentID = (layer_key, global_idx)
- layer-aware: scoring is per ComponentID, not collapsed across layers
"""
import torch
import logging
from typing import List, Dict, Tuple, Optional, Set, Any

logger = logging.getLogger(__name__)

# Type alias for clarity
ComponentID = Tuple[str, int]  # (layer_key, global_idx)


def _cfg_debug(cfg) -> bool:
    """Return True iff top-level cfg.debug is True."""
    if cfg is None:
        return False
    return bool(getattr(cfg, 'debug', False))


# =============================================================================
# Layer Key Helpers
# =============================================================================

def canonicalize_lora_layer_key(key: str) -> str:
    """
    Convert a LoRA parameter key to its canonical layer key.
    
    The canonical layer key is the exact path to the LoRA adapter layer,
    excluding the lora_A/lora_B and weight suffixes. This is used internally
    for layer-aware grouping and ComponentID construction.
    
    Note: This is DIFFERENT from the distributed payload key format.
    For distributed payloads, use get_canonical_lora_param_key() which
    preserves the full parameter path.
    
    Examples:
        'model.layers.0.self_attn.q_proj.lora_A.default.weight' -> 'model.layers.0.self_attn.q_proj'
        'base_model.model.encoder.layer.3.attention.self.query.lora_B.weight' -> 'base_model.model.encoder.layer.3.attention.self.query'
    
    Args:
        key: Full LoRA parameter key from state dict
        
    Returns:
        Canonical layer key (exact path without lora_A/B suffix)
    """
    # First strip any rank suffix (e.g., ".8" at the end)
    result = strip_rank_suffix(key)
    
    # Remove common LoRA suffixes to get base layer path
    suffixes_to_remove = [
        '.lora_A.default.weight',
        '.lora_B.default.weight',
        '.lora_A.weight',
        '.lora_B.weight',
        '.lora_A',
        '.lora_B',
    ]
    
    for suffix in suffixes_to_remove:
        if result.endswith(suffix):
            result = result[:-len(suffix)]
            break
    
    return result


def strip_rank_suffix(key: str) -> str:
    """
    Strip a trailing rank suffix (e.g., '.8') from a distributed payload key.
    
    Examples:
        'base_model.model.layers.0.q_proj.lora_A.default.weight.8' -> 'base_model.model.layers.0.q_proj.lora_A.default.weight'
        'base_model.model.layers.0.q_proj.lora_A.default.weight' -> 'base_model.model.layers.0.q_proj.lora_A.default.weight'
    
    Args:
        key: Parameter key (may or may not have rank suffix)
        
    Returns:
        Key without rank suffix
    """
    parts = key.split('.')
    if parts and parts[-1].isdigit():
        return '.'.join(parts[:-1])
    return key


def get_canonical_lora_param_key(key: str) -> str:
    """
    Get the canonical LoRA parameter key for use in distributed payloads.
    
    This function uses the same canonicalization as the heterolora loading path
    to ensure compatibility with load_weight_local().
    
    The canonical format:
    - Strips any trailing rank suffix (e.g., ".8")
    - Strips leading 'model.' prefix to align server/client keys
    
    Args:
        key: Full LoRA parameter key
        
    Returns:
        Canonical key suitable for distributed payload format
    """
    from federatedscope.contrib.common.heterolora_utils import _canonical_lora_key
    return _canonical_lora_key(key)


def infer_layer_keys_from_state_dict(state_dict: dict) -> List[str]:
    """
    Infer all unique layer keys from a state dict containing LoRA parameters.
    
    Args:
        state_dict: Model state dict with LoRA parameters
        
    Returns:
        Sorted list of unique canonical layer keys
    """
    layer_keys = set()
    
    for key in state_dict.keys():
        if 'lora_A' in key or 'lora_B' in key:
            layer_key = canonicalize_lora_layer_key(key)
            layer_keys.add(layer_key)
    
    return sorted(layer_keys)


def get_lora_keys_for_layer(state_dict: dict, layer_key: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Find the full lora_A and lora_B keys for a given layer key.
    
    This function handles both:
    - Original model state dict keys (full parameter names)
    - Distributed payload keys (with or without rank suffix)
    
    Args:
        state_dict: Model state dict or distributed payload dict
        layer_key: Canonical layer key (without lora_A/B suffix)
        
    Returns:
        Tuple of (a_key, b_key) or (None, None) if not found
    """
    a_key = None
    b_key = None
    
    for key in state_dict.keys():
        # Canonicalize to layer key for comparison
        canonical = canonicalize_lora_layer_key(key)
        if canonical == layer_key:
            if 'lora_A' in key and 'lora_B' not in key:
                a_key = key
            elif 'lora_B' in key:
                b_key = key
    
    return a_key, b_key


# =============================================================================
# ComponentID Helpers: Conversion between grouped and flattened formats
# =============================================================================

def flatten_grouped_indices_by_layer(
    grouped_dict: Dict[str, List[int]]
) -> List[ComponentID]:
    """
    Flatten a grouped indices dictionary to a list of ComponentIDs.
    
    Wire format (grouped dict):
        {"layer.0.q_proj": [0, 3, 5], "layer.0.v_proj": [1, 2]}
        
    Internal format (flattened ComponentIDs):
        [("layer.0.q_proj", 0), ("layer.0.q_proj", 3), ("layer.0.q_proj", 5),
         ("layer.0.v_proj", 1), ("layer.0.v_proj", 2)]
    
    Args:
        grouped_dict: Dict mapping layer_key -> list of global indices
        
    Returns:
        List of ComponentID tuples, sorted by (layer_key, global_idx)
    """
    component_ids = []
    
    for layer_key in sorted(grouped_dict.keys()):
        indices = grouped_dict[layer_key]
        for global_idx in sorted(indices):
            component_ids.append((layer_key, global_idx))
    
    return component_ids


def group_component_ids_by_layer(
    component_ids: List[ComponentID]
) -> Dict[str, List[int]]:
    """
    Group ComponentIDs back into a layer-keyed dictionary.
    
    Internal format (flattened ComponentIDs):
        [("layer.0.q_proj", 0), ("layer.0.q_proj", 3), ("layer.0.v_proj", 1)]
        
    Wire format (grouped dict):
        {"layer.0.q_proj": [0, 3], "layer.0.v_proj": [1]}
    
    Args:
        component_ids: List of ComponentID tuples
        
    Returns:
        Dict mapping layer_key -> sorted list of global indices
    """
    grouped = {}
    
    for layer_key, global_idx in component_ids:
        if layer_key not in grouped:
            grouped[layer_key] = []
        grouped[layer_key].append(global_idx)
    
    # Sort indices within each layer
    for layer_key in grouped:
        grouped[layer_key] = sorted(grouped[layer_key])
    
    return grouped


def normalize_indices_to_grouped(
    payload: Any,
    layer_keys: Optional[List[str]] = None,
    max_rank: Optional[int] = None
) -> Dict[str, List[int]]:
    """
    Normalize various payload formats to grouped indices dict.
    
    Layer-aware components (client, server, aggregator) call this function on
    incoming index metadata. After normalization, the layer-aware path uses:
    - grouped metadata by exact layer key (on the wire)
    - ComponentID = (layer_key, global_idx) internally
    
    A flat index list applies to every layer (AdaS semantics), so it is
    expanded to all layers.
    
    Handles:
    - Already grouped dict: {"layer_key": [0, 1, 2]} - returned as-is
    - Flat list: [0, 1, 2] - expanded to all layers if layer_keys provided
    - None: returns empty dict
    
    Args:
        payload: Input payload (grouped dict, flat list, or None)
        layer_keys: List of canonical layer keys (required for flat->grouped expansion)
        max_rank: Unused
        
    Returns:
        Normalized grouped indices dict with exact layer keys
    """
    if payload is None:
        return {}
    
    if isinstance(payload, dict):
        # Already grouped format - return as-is
        return dict(payload)
    
    if isinstance(payload, (list, tuple)):
        # Flat format: the same indices apply to every layer (AdaS semantics)
        flat_indices = list(payload)
        
        if layer_keys is None or not layer_keys:
            logger.warning(
                "normalize_indices_to_grouped: flat index payload received but no layer_keys "
                "provided for expansion. Returning empty dict."
            )
            return {}
        
        # Expand flat indices to all layers
        grouped = {}
        for layer_key in layer_keys:
            grouped[layer_key] = list(flat_indices)
        
        return grouped
    
    return {}


# =============================================================================
# Layer-Aware Scoring Helpers
# =============================================================================
#
# Layer-aware scoring:
# - AdaS scores score[p], where p is a slot index shared across all layers
# - AdaS-LayerAware scores score[(layer_key, global_idx)] for each layer-specific
#   component, so (layer_X, idx) and (layer_Y, idx) are distinct components with
#   separate scores
# - All scoring functions in this module key scores by ComponentID
#

def compute_component_scores_grouped(
    model,
    current_survivors_by_layer: Dict[str, List[int]]
) -> Dict[ComponentID, float]:
    """
    Compute per-ComponentID importance scores using norm-product.
    
    layer-aware key difference: Each ComponentID = (layer_key, global_idx) gets its own score.
    We do NOT collapse same-slot components from different layers into one shared score.
    
    For each ComponentID (layer_key, p):
        score = ||A_row_p||_2 * ||B_col_p||_2 for that specific layer
    
    Args:
        model: Model with LoRA adapters
        current_survivors_by_layer: Dict mapping layer_key -> list of survivor global indices
        
    Returns:
        Dict mapping ComponentID -> importance score
    """
    scores = {}
    state_dict = model.state_dict()
    
    for layer_key, survivor_indices in current_survivors_by_layer.items():
        a_key, b_key = get_lora_keys_for_layer(state_dict, layer_key)
        
        if a_key is None or b_key is None:
            continue
        
        A = state_dict[a_key]
        B = state_dict[b_key]
        
        # For each survivor component in this layer
        for local_pos, global_idx in enumerate(survivor_indices):
            if local_pos >= A.shape[0] or local_pos >= B.shape[1]:
                continue
            
            a_norm = torch.norm(A[local_pos, :], p=2).item()
            b_norm = torch.norm(B[:, local_pos], p=2).item()
            
            component_id = (layer_key, global_idx)
            scores[component_id] = a_norm * b_norm
    
    return scores


def group_component_scores_by_layer(
    scores: Dict[ComponentID, float]
) -> Dict[str, List[Tuple[int, float]]]:
    """
    Group component scores by layer.
    
    Args:
        scores: Dict mapping ComponentID -> score
        
    Returns:
        Dict mapping layer_key -> [(global_idx, score), ...]
    """
    grouped = {}
    
    for (layer_key, global_idx), score in scores.items():
        if layer_key not in grouped:
            grouped[layer_key] = []
        grouped[layer_key].append((global_idx, score))
    
    return grouped


def compute_layer_median_scores(
    scores: Dict[ComponentID, float],
    eps: float = 1e-12
) -> Dict[str, float]:
    """
    Compute per-layer median score.
    
    Args:
        scores: Dict mapping ComponentID -> score
        eps: Small value to avoid division by zero
        
    Returns:
        Dict mapping layer_key -> median_score
    """
    grouped = group_component_scores_by_layer(scores)
    layer_medians = {}
    
    for layer_key, idx_score_pairs in grouped.items():
        if not idx_score_pairs:
            layer_medians[layer_key] = 1.0
            continue
        
        score_list = [s for _, s in idx_score_pairs]
        median_val = torch.tensor(score_list).median().item()
        
        if median_val <= 0:
            median_val = eps
        
        layer_medians[layer_key] = median_val
    
    return layer_medians


def normalize_component_scores_by_layer_median(
    scores: Dict[ComponentID, float],
    eps: float = 1e-12
) -> Tuple[Dict[ComponentID, float], Dict[str, float]]:
    """
    Normalize component scores by the median score of their respective layer.
    
    Args:
        scores: Dict mapping ComponentID -> score
        eps: Small value to avoid division by zero
        
    Returns:
        Tuple of:
            - normalized_scores: Dict mapping ComponentID -> normalized_score
            - layer_medians: Dict mapping layer_key -> median_score
    """
    layer_medians = compute_layer_median_scores(scores, eps)
    normalized_scores = {}
    
    for (layer_key, global_idx), raw_score in scores.items():
        median = layer_medians.get(layer_key, 1.0)
        normalized_scores[(layer_key, global_idx)] = raw_score / (median + eps)
    
    return normalized_scores, layer_medians


def apply_layer_importance_prior(
    normalized_scores: Dict[ComponentID, float],
    layer_importance: Optional[Dict[str, float]],
    default_importance: float = 1.0,
    importance_strength: float = 1.0
) -> Dict[ComponentID, float]:
    """
    Apply layer-importance prior to normalized scores.
    
    Lower importance layers get smaller adjusted scores, leading to
    more prune pressure in the global low-set selection.
    
    Args:
        normalized_scores: Dict mapping ComponentID -> normalized_score
        layer_importance: Dict mapping layer_key -> importance (0 to 1)
        default_importance: Default importance for layers not in the dict
        importance_strength: Exponent for importance weighting
        
    Returns:
        adjusted_scores: Dict mapping ComponentID -> adjusted_score
    """
    adjusted_scores = {}
    
    for (layer_key, global_idx), norm_score in normalized_scores.items():
        if layer_importance is not None:
            layer_weight = layer_importance.get(layer_key, default_importance)
        else:
            layer_weight = default_importance
        
        adjusted_scores[(layer_key, global_idx)] = norm_score * (layer_weight ** importance_strength)
    
    return adjusted_scores


def compute_stage1_adjusted_scores_grouped(
    raw_scores: Dict[ComponentID, float],
    layer_importance: Optional[Dict[str, float]] = None,
    eps: float = 1e-12,
    importance_strength: float = 1.0
) -> Tuple[Dict[ComponentID, float], Dict[ComponentID, float], Dict[str, float]]:
    """
    Compute Stage 1 adjusted scores with layer normalization and importance prior.
    
    This is the main entry point for Stage 1 score adjustment:
    1. Normalize raw scores by layer median
    2. Apply layer-importance prior
    
    The adjusted scores feed into the global low-set selection. Lower-importance
    layers get smaller adjusted scores, leading to more prune pressure.
    
    Args:
        raw_scores: Dict mapping ComponentID -> raw importance score
        layer_importance: Optional dict mapping layer_key -> importance (0 to 1)
        eps: Small value to avoid division by zero
        importance_strength: Exponent for importance weighting
        
    Returns:
        Tuple of:
            - adjusted_scores: Final scores for low-set selection
            - normalized_scores: Intermediate median-normalized scores
            - layer_medians: Per-layer median values used for normalization
    """
    normalized_scores, layer_medians = normalize_component_scores_by_layer_median(
        raw_scores, eps
    )
    
    adjusted_scores = apply_layer_importance_prior(
        normalized_scores,
        layer_importance,
        default_importance=1.0,
        importance_strength=importance_strength
    )
    
    return adjusted_scores, normalized_scores, layer_medians


def compute_lowset_grouped(
    scores: Dict[ComponentID, float],
    gamma: float,
    rank_min: int,
    global_competition: bool = False
) -> Tuple[List[ComponentID], float]:
    """
    Compute the low-set ComponentIDs and their total score.
    
    This function selects from whatever score dict it receives. The scores may
    already include layer normalization and layer-importance weighting (via
    compute_stage1_adjusted_scores_grouped), or they may be raw scores.
    
    rank_min is a single minimum on the total number of survivors across all
    layers, not a per-layer minimum:
    - It applies in both layer-wise and global-competition modes
    - Total survivors across all layers must remain >= rank_min
    
    Args:
        scores: Dict mapping ComponentID -> importance score (raw or adjusted)
        gamma: Decay factor (0 < gamma < 1)
        rank_min: SHARED/GLOBAL minimum total survivors (NOT per-layer)
        global_competition: If True, selection is from one global pool;
                           if False, selection is per-layer (but min floor is still global)
        
    Returns:
        Tuple of:
            - low_component_ids: List of ComponentIDs in the low-set
            - low_score: Sum of scores for low-set components
    """
    if not scores:
        return [], 0.0
    
    # Total components across all layers
    total_r = len(scores)
    
    # Shared/global minimum floor: total survivors must stay >= rank_min
    # This applies in BOTH layer-wise and global-competition modes
    k_target_global = max(rank_min, int(total_r * gamma))
    max_prunable_global = total_r - k_target_global
    
    if max_prunable_global <= 0:
        return [], 0.0
    
    if global_competition:
        # Global competition: one pool across all layers
        # Selection is global, minimum floor is global
        all_components = list(scores.keys())
        
        # Sort by score (ascending) - lowest scores are pruning candidates
        sorted_components = sorted(all_components, key=lambda cid: scores[cid])
        low_component_ids = sorted_components[:max_prunable_global]
        low_score = sum(scores[cid] for cid in low_component_ids)
        
        return low_component_ids, low_score
    
    else:
        # Per-layer selection with a shared global minimum floor
        # Selection is layer-wise (low-set computed per layer), but the total
        # number pruned is capped by the global rank_min constraint
        grouped = group_component_ids_by_layer(list(scores.keys()))
        
        # First pass: compute per-layer low-sets based on gamma only (no per-layer rank_min)
        layer_low_candidates = []  # List of (ComponentID, score) tuples
        
        for layer_key, indices in grouped.items():
            layer_scores = [(idx, scores[(layer_key, idx)]) for idx in indices]
            r = len(layer_scores)
            
            # Per-layer target based on gamma (NO per-layer rank_min constraint)
            k_target_layer = int(r * gamma)
            m_layer = r - k_target_layer
            
            if m_layer <= 0:
                continue
            
            # Sort by score (ascending) within this layer
            sorted_layer = sorted(layer_scores, key=lambda x: x[1])
            layer_low = sorted_layer[:m_layer]
            
            for idx, score in layer_low:
                layer_low_candidates.append(((layer_key, idx), score))
        
        # Second pass: enforce GLOBAL rank_min by taking only the lowest-scoring
        # candidates up to max_prunable_global
        layer_low_candidates.sort(key=lambda x: x[1])  # Sort by score ascending
        selected_low = layer_low_candidates[:max_prunable_global]
        
        all_low_ids = [cid for cid, _ in selected_low]
        total_low_score = sum(score for _, score in selected_low)
        
        return all_low_ids, total_low_score


# =============================================================================
# Stage 2 Scoring Helpers (Layer-Aware)
# =============================================================================

def compute_stage2_upload_scores_grouped(
    effective_update_dict: dict,
    survivors_by_layer: Dict[str, List[int]]
) -> Dict[ComponentID, float]:
    """
    Compute Stage 2 upload scores from residual-corrected effective updates.
    
    Score for each ComponentID = ||effective_A_row||_2 * ||effective_B_col||_2
    computed on the exact layer's tensors.
    
    Args:
        effective_update_dict: Effective updates (delta + residual)
        survivors_by_layer: Dict mapping layer_key -> list of survivor global indices
        
    Returns:
        Dict mapping ComponentID -> upload_score
    """
    scores = {}
    
    for layer_key, survivor_indices in survivors_by_layer.items():
        a_key, b_key = get_lora_keys_for_layer(effective_update_dict, layer_key)
        
        if a_key is None or b_key is None:
            continue
        
        A = effective_update_dict[a_key]
        B = effective_update_dict[b_key]
        
        for local_pos, global_idx in enumerate(survivor_indices):
            if local_pos >= A.shape[0] or local_pos >= B.shape[1]:
                continue
            
            a_norm = torch.norm(A[local_pos, :], p=2).item()
            b_norm = torch.norm(B[:, local_pos], p=2).item()
            
            component_id = (layer_key, global_idx)
            scores[component_id] = a_norm * b_norm
    
    return scores


def compute_stage2_downlink_scores_grouped(
    aggregated_global_updates: dict,
    survivors_by_layer: Dict[str, List[int]]
) -> Dict[ComponentID, float]:
    """
    Compute Stage 2 downlink scores from aggregated global updates.
    
    Args:
        aggregated_global_updates: Aggregated global updates from server
        survivors_by_layer: Dict mapping layer_key -> list of survivor global indices
        
    Returns:
        Dict mapping ComponentID -> downlink_score
    """
    scores = {}
    
    for layer_key, survivor_indices in survivors_by_layer.items():
        a_key, b_key = get_lora_keys_for_layer(aggregated_global_updates, layer_key)
        
        if a_key is None or b_key is None:
            continue
        
        A = aggregated_global_updates[a_key]
        B = aggregated_global_updates[b_key]
        
        # For global aggregated updates, indices are at global positions
        for global_idx in survivor_indices:
            if global_idx >= A.shape[0] or global_idx >= B.shape[1]:
                continue
            
            a_norm = torch.norm(A[global_idx, :], p=2).item()
            b_norm = torch.norm(B[:, global_idx], p=2).item()
            
            component_id = (layer_key, global_idx)
            scores[component_id] = a_norm * b_norm
    
    return scores


# =============================================================================
# Layer-Aware Cost Helpers
# =============================================================================

def compute_component_upload_cost_grouped(
    effective_update_dict: dict,
    survivors_by_layer: Dict[str, List[int]],
    q_bits: int = 8,
    cmeta_bits: int = 32
) -> Dict[ComponentID, float]:
    """
    Compute per-ComponentID upload cost in bits.
    
    Cost = (num_params_per_component * q_bits) + cmeta_bits
    
    Args:
        effective_update_dict: Effective updates to send
        survivors_by_layer: Dict mapping layer_key -> list of survivor global indices
        q_bits: Quantization bits per parameter value
        cmeta_bits: Bits for metadata per component (index storage)
        
    Returns:
        Dict mapping ComponentID -> cost_in_bits
    """
    costs = {}
    
    for layer_key, survivor_indices in survivors_by_layer.items():
        a_key, b_key = get_lora_keys_for_layer(effective_update_dict, layer_key)
        
        if a_key is None or b_key is None:
            continue
        
        A = effective_update_dict[a_key]
        B = effective_update_dict[b_key]
        
        # Parameters per component for this layer
        params_per_component = A.shape[1] + B.shape[0]  # in_features + out_features
        cost_per_component = (params_per_component * q_bits) + cmeta_bits
        
        for global_idx in survivor_indices:
            component_id = (layer_key, global_idx)
            costs[component_id] = cost_per_component
    
    return costs


def compute_component_downlink_cost_grouped(
    aggregated_global_updates: dict,
    survivors_by_layer: Dict[str, List[int]],
    q_bits: int = 8,
    cmeta_bits: int = 32
) -> Dict[ComponentID, float]:
    """
    Compute per-ComponentID downlink cost in bits.
    
    Args:
        aggregated_global_updates: Aggregated global updates
        survivors_by_layer: Dict mapping layer_key -> list of survivor global indices
        q_bits: Quantization bits per parameter value
        cmeta_bits: Bits for metadata per component
        
    Returns:
        Dict mapping ComponentID -> cost_in_bits
    """
    costs = {}
    
    for layer_key, survivor_indices in survivors_by_layer.items():
        a_key, b_key = get_lora_keys_for_layer(aggregated_global_updates, layer_key)
        
        if a_key is None or b_key is None:
            continue
        
        A = aggregated_global_updates[a_key]
        B = aggregated_global_updates[b_key]
        
        params_per_component = A.shape[1] + B.shape[0]
        cost_per_component = (params_per_component * q_bits) + cmeta_bits
        
        for global_idx in survivor_indices:
            component_id = (layer_key, global_idx)
            costs[component_id] = cost_per_component
    
    return costs


# =============================================================================
# Selection Helpers (Layer-Aware)
# =============================================================================

def greedy_select_by_score_cost_ratio_grouped(
    scores: Dict[ComponentID, float],
    costs: Dict[ComponentID, float],
    budget: float,
    global_competition: bool = False
) -> List[ComponentID]:
    """
    Greedy selection by score-to-cost ratio under budget constraint.
    
    Args:
        scores: Dict mapping ComponentID -> score
        costs: Dict mapping ComponentID -> cost
        budget: Total budget in bits
        global_competition: If True, select globally; if False, select per-layer proportionally
        
    Returns:
        List of selected ComponentIDs
    """
    if budget <= 0:
        return []
    
    if global_competition:
        # Global competition: one greedy selection across all layers
        ratios = []
        for cid in scores.keys():
            score = scores.get(cid, 0.0)
            cost = costs.get(cid, 1.0)
            ratio = score / cost if cost > 0 else (float('inf') if score > 0 else 0.0)
            ratios.append((cid, score, cost, ratio))
        
        ratios.sort(key=lambda x: x[3], reverse=True)
        
        selected = []
        remaining_budget = budget
        
        for cid, score, cost, ratio in ratios:
            if cost <= remaining_budget:
                selected.append(cid)
                remaining_budget -= cost
        
        return selected
    
    else:
        # Per-layer selection with proportional budget allocation
        grouped = group_component_ids_by_layer(list(scores.keys()))
        n_layers = len(grouped)
        
        if n_layers == 0:
            return []
        
        # Allocate budget proportionally to layer sizes
        total_components = sum(len(indices) for indices in grouped.values())
        
        all_selected = []
        
        for layer_key, indices in grouped.items():
            layer_budget = budget * (len(indices) / total_components) if total_components > 0 else 0
            
            layer_ratios = []
            for global_idx in indices:
                cid = (layer_key, global_idx)
                score = scores.get(cid, 0.0)
                cost = costs.get(cid, 1.0)
                ratio = score / cost if cost > 0 else (float('inf') if score > 0 else 0.0)
                layer_ratios.append((cid, cost, ratio))
            
            layer_ratios.sort(key=lambda x: x[2], reverse=True)
            
            remaining = layer_budget
            for cid, cost, ratio in layer_ratios:
                if cost <= remaining:
                    all_selected.append(cid)
                    remaining -= cost
        
        return all_selected


# =============================================================================
# Layer-Aware Slicing / Distribution Helpers
# =============================================================================

def distribute_weights_by_layer_indices(
    server_lora_dict: dict,
    survivors_by_layer: Dict[str, List[int]],
    download_components: List[ComponentID],
    max_rank: int,
    debug: bool = False
) -> dict:
    """
    Distribute server LoRA weights to a client based on layer-aware indices.
    
    Each layer is sliced with its own selected indices. The output keys use
    the same canonical format as distribute_weight_fast, so that
    load_weight_local can load them.
    
    Key format: '{canonical_base_key}.{rank}' where canonical_base_key is the
    full parameter path (e.g., 'base_model.model.layers.0.q_proj.lora_A.default.weight')
    with leading 'model.' stripped if present.
    
    Args:
        server_lora_dict: Server LoRA weights at max_rank
        survivors_by_layer: Client's survivor indices per layer
        download_components: ComponentIDs to download
        max_rank: Global maximum rank
        debug: Enable debug logging
        
    Returns:
        Dict of distributed LoRA weights with canonical keys compatible with load_weight_local
    """
    download_by_layer = group_component_ids_by_layer(download_components)
    result = {}
    
    for layer_key, survivor_indices in survivors_by_layer.items():
        a_key, b_key = get_lora_keys_for_layer(server_lora_dict, layer_key)
        
        if a_key is None or b_key is None:
            continue
        
        download_indices = download_by_layer.get(layer_key, [])
        
        if not download_indices:
            continue
        
        A = server_lora_dict[a_key]
        B = server_lora_dict[b_key]
        
        # Create index tensor for global positions
        indices_tensor = torch.tensor(download_indices, dtype=torch.long, device=A.device)
        
        # Select rows from A and columns from B at global positions
        sliced_A = A.index_select(0, indices_tensor)
        sliced_B = B.index_select(1, indices_tensor)
        
        # Use the canonical key format expected by load_weight_local()
        canonical_a = get_canonical_lora_param_key(a_key)
        canonical_b = get_canonical_lora_param_key(b_key)
        
        # For layer-aware, each layer may have different effective rank based on download selection
        # Use the downloaded component count for this specific layer
        layer_rank = len(download_indices)
        
        dist_a_key = f"{canonical_a}.{layer_rank}"
        dist_b_key = f"{canonical_b}.{layer_rank}"
        
        result[dist_a_key] = sliced_A
        result[dist_b_key] = sliced_B
    
    if debug:
        n_layers = len(download_by_layer)
        n_components = len(download_components)
        sample_keys = list(result.keys())[:4]
        logger.debug(
            f"layer-aware Distributed LoRA weights: {n_layers} layers, {n_components} components, "
            f"sample keys: {sample_keys}"
        )
    
    return result


def slice_model_update_by_component_ids(
    model_update_dict: dict,
    survivors_by_layer: Dict[str, List[int]],
    selected_components: List[ComponentID]
) -> dict:
    """
    Slice model updates to include only selected ComponentIDs.
    
    Args:
        model_update_dict: Full model updates for all survivors
        survivors_by_layer: Dict mapping layer_key -> list of all survivor global indices
        selected_components: List of selected ComponentIDs
        
    Returns:
        Dict containing model updates only for selected components
    """
    if not selected_components:
        return {}
    
    selected_by_layer = group_component_ids_by_layer(selected_components)
    result = {}
    
    for layer_key, survivor_indices in survivors_by_layer.items():
        selected_in_layer = selected_by_layer.get(layer_key, [])
        
        if not selected_in_layer:
            continue
        
        a_key, b_key = get_lora_keys_for_layer(model_update_dict, layer_key)
        
        if a_key is None or b_key is None:
            continue
        
        A = model_update_dict[a_key]
        B = model_update_dict[b_key]
        
        # Map global indices to local positions
        survivor_to_local = {gidx: i for i, gidx in enumerate(survivor_indices)}
        selected_local_positions = [
            survivor_to_local[gidx] for gidx in selected_in_layer
            if gidx in survivor_to_local
        ]
        
        if not selected_local_positions:
            continue
        
        positions_tensor = torch.tensor(selected_local_positions, dtype=torch.long, device=A.device)
        
        # Select rows from A and columns from B
        result[a_key] = A.index_select(0, positions_tensor).clone()
        result[b_key] = B.index_select(1, positions_tensor).clone()
    
    return result


# =============================================================================
# Residual Buffer Helpers (Keyed by ComponentID)
# =============================================================================

def update_residual_buffers_after_upload_grouped(
    residual_buffers: Dict[ComponentID, Dict[str, torch.Tensor]],
    effective_update_dict: dict,
    upload_components: List[ComponentID],
    survivors_by_layer: Dict[str, List[int]],
    cfg=None
) -> Dict[ComponentID, Dict[str, torch.Tensor]]:
    """
    Update residual buffers by subtracting what was actually sent.
    
    For uploaded components: residual = 0 (clear)
    For non-uploaded survivors: residual = effective_update (store for next round)
    
    Residual buffers are stored on CPU to reduce GPU memory pressure.
    
    Args:
        residual_buffers: Current residual buffers keyed by ComponentID
        effective_update_dict: Effective updates (delta + old residual)
        upload_components: ComponentIDs that were actually uploaded
        survivors_by_layer: All survivor indices per layer
        
    Returns:
        Updated residual buffers keyed by ComponentID
    """
    upload_set = set(upload_components)
    new_residuals = {}
    logged_cpu_storage = False
    
    for layer_key, survivor_indices in survivors_by_layer.items():
        a_key, b_key = get_lora_keys_for_layer(effective_update_dict, layer_key)
        
        if a_key is None or b_key is None:
            continue
        
        A = effective_update_dict[a_key]
        B = effective_update_dict[b_key]
        
        for local_pos, global_idx in enumerate(survivor_indices):
            component_id = (layer_key, global_idx)
            
            if component_id in upload_set:
                # Uploaded - clear residual
                continue
            
            # Not uploaded - store effective update as new residual (on CPU)
            component_residual = {}
            
            if local_pos < A.shape[0]:
                residual_slice = A[local_pos, :].clone().cpu()
                component_residual[a_key] = residual_slice
                if not logged_cpu_storage and A.device.type != 'cpu' and _cfg_debug(cfg):
                    logger.debug(
                        f"update_residual_buffers_after_upload_grouped: "
                        f"storing residuals on CPU (source was {A.device})"
                    )
                    logged_cpu_storage = True
            
            if local_pos < B.shape[1]:
                residual_slice = B[:, local_pos].clone().cpu()
                component_residual[b_key] = residual_slice
            
            if component_residual:
                new_residuals[component_id] = component_residual
    
    return new_residuals


def apply_residual_to_update_grouped(
    delta_dict: dict,
    residual_buffers: Dict[ComponentID, Dict[str, torch.Tensor]],
    survivors_by_layer: Dict[str, List[int]],
    cfg=None
) -> dict:
    """
    Construct residual-aware effective update by adding residual buffers.
    
    effective_update = delta + residual
    
    Handles device/dtype alignment for CPU-backed residuals.
    
    Args:
        delta_dict: Fresh local model updates (typically on GPU)
        residual_buffers: Dict mapping ComponentID to {key: residual_tensor} (CPU-backed)
        survivors_by_layer: Dict mapping layer_key -> list of survivor global indices
        
    Returns:
        Dict containing effective updates (delta + residual)
    """
    effective_dict = {}
    logged_alignment = False
    
    for layer_key, survivor_indices in survivors_by_layer.items():
        a_key, b_key = get_lora_keys_for_layer(delta_dict, layer_key)
        
        if a_key is None or b_key is None:
            continue
        
        A_delta = delta_dict[a_key]
        B_delta = delta_dict[b_key]
        
        effective_A = A_delta.clone()
        effective_B = B_delta.clone()
        
        for local_pos, global_idx in enumerate(survivor_indices):
            component_id = (layer_key, global_idx)
            
            if component_id not in residual_buffers:
                continue
            
            component_residual = residual_buffers[component_id]
            
            if a_key in component_residual and local_pos < effective_A.shape[0]:
                residual_tensor = component_residual[a_key]
                if residual_tensor.device != effective_A.device or residual_tensor.dtype != effective_A.dtype:
                    if not logged_alignment and _cfg_debug(cfg):
                        logger.debug(
                            f"apply_residual_to_update_grouped: aligning residual device/dtype"
                        )
                        logged_alignment = True
                    residual_aligned = residual_tensor.to(device=effective_A.device, dtype=effective_A.dtype)
                else:
                    residual_aligned = residual_tensor
                effective_A[local_pos, :] += residual_aligned
            
            if b_key in component_residual and local_pos < effective_B.shape[1]:
                residual_tensor = component_residual[b_key]
                if residual_tensor.device != effective_B.device or residual_tensor.dtype != effective_B.dtype:
                    residual_aligned = residual_tensor.to(device=effective_B.device, dtype=effective_B.dtype)
                else:
                    residual_aligned = residual_tensor
                effective_B[:, local_pos] += residual_aligned
        
        effective_dict[a_key] = effective_A
        effective_dict[b_key] = effective_B
    
    return effective_dict


def prune_residual_buffers_grouped(
    residual_buffers: Dict[ComponentID, Dict[str, torch.Tensor]],
    pruned_components: List[ComponentID]
) -> Dict[ComponentID, Dict[str, torch.Tensor]]:
    """
    Remove residual buffers for structurally pruned components.
    
    Args:
        residual_buffers: Current residual buffers keyed by ComponentID
        pruned_components: ComponentIDs that were pruned by Stage 1
        
    Returns:
        Updated residual buffers with pruned components removed
    """
    pruned_set = set(pruned_components)
    return {cid: buffers for cid, buffers in residual_buffers.items() if cid not in pruned_set}


# =============================================================================
# Model Update Computation Helpers
# =============================================================================

def compute_model_update_from_snapshot_grouped(
    current_state_dict: dict,
    snapshot_state_dict: dict,
    survivors_by_layer: Dict[str, List[int]],
    cfg=None
) -> dict:
    """
    Compute fresh local model update by comparing post-training tensors to snapshot.
    
    delta_dict[key] = current_state_dict[key] - snapshot_state_dict[key]
    
    Handles device/dtype alignment for CPU-backed snapshots.
    
    Args:
        current_state_dict: Current LoRA state dict after training
        snapshot_state_dict: Pre-round snapshot of LoRA state dict
        survivors_by_layer: Dict mapping layer_key -> list of survivor global indices
        
    Returns:
        Dict containing model updates (deltas) for survivor components
    """
    delta_dict = {}
    logged_alignment = False
    
    for layer_key in survivors_by_layer.keys():
        a_key, b_key = get_lora_keys_for_layer(current_state_dict, layer_key)
        snap_a_key, snap_b_key = get_lora_keys_for_layer(snapshot_state_dict, layer_key)
        
        if a_key is None or snap_a_key is None:
            continue
        
        current_A = current_state_dict[a_key]
        current_B = current_state_dict[b_key]
        snap_A = snapshot_state_dict[snap_a_key]
        snap_B = snapshot_state_dict[snap_b_key]
        
        # Align device and dtype
        if snap_A.device != current_A.device or snap_A.dtype != current_A.dtype:
            if not logged_alignment and _cfg_debug(cfg):
                logger.debug(
                    f"compute_model_update_from_snapshot_grouped: aligning snapshot device/dtype"
                )
                logged_alignment = True
            snap_A = snap_A.to(device=current_A.device, dtype=current_A.dtype)
            snap_B = snap_B.to(device=current_B.device, dtype=current_B.dtype)
        
        if current_A.shape == snap_A.shape:
            delta_dict[a_key] = current_A - snap_A
        else:
            logger.warning(f"Shape mismatch for {a_key}: {current_A.shape} vs {snap_A.shape}")
        
        if current_B.shape == snap_B.shape:
            delta_dict[b_key] = current_B - snap_B
        else:
            logger.warning(f"Shape mismatch for {b_key}: {current_B.shape} vs {snap_B.shape}")
    
    return delta_dict


# =============================================================================
# Validation Helpers
# =============================================================================

def compute_residual_norm_summary_grouped(
    residual_buffers: Dict[ComponentID, Dict[str, torch.Tensor]]
) -> Dict[str, float]:
    """
    Compute summary statistics of residual buffer norms.
    
    Args:
        residual_buffers: Current residual buffers keyed by ComponentID
        
    Returns:
        Dict with min, avg, max, total norms
    """
    if not residual_buffers:
        return {'min': 0.0, 'avg': 0.0, 'max': 0.0, 'total': 0.0, 'count': 0}
    
    norms = []
    for cid, buffers in residual_buffers.items():
        component_norm = 0.0
        for key, tensor in buffers.items():
            if isinstance(tensor, torch.Tensor):
                component_norm += torch.norm(tensor, p=2).item() ** 2
        norms.append(component_norm ** 0.5)
    
    if not norms:
        return {'min': 0.0, 'avg': 0.0, 'max': 0.0, 'total': 0.0, 'count': 0}
    
    return {
        'min': min(norms),
        'avg': sum(norms) / len(norms),
        'max': max(norms),
        'total': sum(norms),
        'count': len(norms)
    }


def validate_layeraware_state(
    survivors_by_layer: Dict[str, List[int]],
    survivor_components: List[ComponentID],
    upload_components: Optional[List[ComponentID]] = None,
    download_components: Optional[List[ComponentID]] = None,
    context: str = "layer-aware state",
    raise_on_error: bool = False
) -> bool:
    """
    Validate consistency of the layer-aware survivor, upload and download state.
    
    This helper verifies that:
    1. survivors_by_layer contains proper layer keys (not placeholders)
    2. survivor_components matches the flattened survivors_by_layer
    3. upload_components (if provided) are subset of survivors
    4. download_components (if provided) are subset of survivors
    
    Args:
        survivors_by_layer: Current per-layer survivor state
        survivor_components: Current flattened ComponentID list
        upload_components: Optional upload selection to validate
        download_components: Optional download selection to validate
        context: Context string for logging
        raise_on_error: If True, raise ValueError on error; else return False
        
    Returns:
        True if valid, False otherwise (or raises ValueError)
    """
    issues = []
    
    # Check 1: No placeholder keys in survivors_by_layer
    for key in survivors_by_layer.keys():
        if key.startswith("__pending"):
            issues.append(f"Placeholder key '{key}' still in survivors_by_layer")
    
    # Check 2: survivor_components matches flattened survivors_by_layer
    expected_components = flatten_grouped_indices_by_layer(survivors_by_layer)
    if set(survivor_components) != set(expected_components):
        missing = set(expected_components) - set(survivor_components)
        extra = set(survivor_components) - set(expected_components)
        if missing:
            issues.append(f"survivor_components missing {len(missing)} components")
        if extra:
            issues.append(f"survivor_components has {len(extra)} extra components")
    
    # Check 3: upload_components are subset of survivors
    if upload_components is not None:
        survivor_set = set(expected_components)
        invalid_uploads = [cid for cid in upload_components if cid not in survivor_set]
        if invalid_uploads:
            issues.append(f"{len(invalid_uploads)} upload components not in survivor set")
    
    # Check 4: download_components are subset of survivors
    if download_components is not None:
        survivor_set = set(expected_components)
        invalid_downloads = [cid for cid in download_components if cid not in survivor_set]
        if invalid_downloads:
            issues.append(f"{len(invalid_downloads)} download components not in survivor set")
    
    if issues:
        msg = f"{context}: Validation FAILED - " + "; ".join(issues)
        if raise_on_error:
            raise ValueError(msg)
        logger.warning(msg)
        return False
    
    return True
