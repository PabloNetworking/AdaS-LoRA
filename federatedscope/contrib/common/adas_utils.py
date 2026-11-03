"""
Utility functions shared by the AdaS variants.

This module provides functions for:
- Component score computation (per rank-1 component importance)
- Index slicing for server broadcast
- Update slicing for client upload
- Index validation
"""
import torch
import logging
from typing import List, Dict, Tuple, Optional

import federatedscope.contrib.common as fs_common

logger = logging.getLogger(__name__)


def _cfg_debug(cfg) -> bool:
    """Return True iff top-level cfg.debug is True via shared common helpers."""
    if cfg is None:
        return False
    safe_getattr = getattr(fs_common, "_safe_getattr", getattr)
    return bool(safe_getattr(cfg, "debug", False))


def compute_component_scores(model, current_rank: Optional[int] = None) -> torch.Tensor:
    """
    Compute per-component importance scores using norm-product (vectorized, differentiable).
    
    For each local component position p in [0, r-1]:
        score[p] = sum over all LoRA layers of (||A_row_p||_2 * ||B_col_p||_2)
    
    This implementation is vectorized to preserve autograd gradients:
    - Uses torch.norm with dim= to compute all row/column norms at once
    - Accumulates via in-place sliced addition to maintain gradient flow
    
    Args:
        model: Model with LoRA adapters
        current_rank: Logical active rank (auto-detected if None)
        
    Returns:
        Tensor of shape (r,) with importance scores per component (with gradient linkage)
    """
    from federatedscope.contrib.common.heterolora_utils import iter_lora_pairs
    
    if current_rank is None:
        current_rank = getattr(model, "adas_current_rank", None)
        if current_rank is None:
            current_rank = getattr(model, "hetlora_current_rank", None)
    
    # Determine rank from first LoRA pair if not specified
    detected_rank = None
    for A, B, _ in iter_lora_pairs(model):
        detected_rank = min(A.shape[0], B.shape[1])
        break
    
    if current_rank is None:
        current_rank = detected_rank
    
    if current_rank is None or current_rank <= 0:
        return torch.tensor([])
    
    # Effective rank is the minimum of specified rank and detected rank
    if detected_rank is not None:
        r_eff = min(int(current_rank), detected_rank)
    else:
        r_eff = int(current_rank)
    
    # Choose device robustly
    try:
        device = next(model.parameters()).device
    except Exception:
        device = torch.device("cpu")
    
    # Initialize scores as None; we'll accumulate
    scores = None
    
    for A, B, _ in iter_lora_pairs(model):
        # A: (r, in_features), B: (out_features, r)
        r_A = A.shape[0]
        r_B = B.shape[1]
        r_layer = min(r_A, r_B, r_eff)
        
        # Slice to effective rank
        A_eff = A[:r_layer, :]  # (r_layer, in_features)
        B_eff = B[:, :r_layer]  # (out_features, r_layer)
        
        # Vectorized norm computation (preserves gradients)
        # a_norms[p] = ||A_row_p||_2 for all p
        a_norms = torch.norm(A_eff, p=2, dim=1)  # shape (r_layer,)
        # b_norms[p] = ||B_col_p||_2 for all p
        b_norms = torch.norm(B_eff, p=2, dim=0)  # shape (r_layer,)
        
        # Norm-product for this layer
        layer_scores = a_norms * b_norms  # shape (r_layer,)
        
        # Accumulate into scores tensor
        if scores is None:
            # Initialize with zeros, then add (preserves gradient from layer_scores)
            scores = torch.zeros(r_eff, device=device, dtype=layer_scores.dtype)
        
        # Vectorized accumulation via slicing (preserves gradients)
        scores = scores.clone()  # Ensure we don't modify in place during backward
        scores[:r_layer] = scores[:r_layer] + layer_scores
    
    if scores is None:
        scores = torch.zeros(r_eff, device=device)
    
    return scores


def compute_lowset_and_score(
    scores: torch.Tensor,
    gamma: float,
    rank_min: int
) -> Tuple[List[int], float]:
    """
    Compute the low-set positions and their total score.
    
    Args:
        scores: Per-component importance scores (shape: (r,))
        gamma: Decay factor (0 < gamma < 1)
        rank_min: Minimum rank to maintain
        
    Returns:
        Tuple of:
            - low_positions: List of local position indices in the low-set
            - low_score: Sum of scores for low-set positions
    """
    r = len(scores)
    if r == 0:
        return [], 0.0
    
    # k_target = max(rank_min, floor(r * gamma))
    k_target = max(rank_min, int(r * gamma))
    
    # m = number of positions to consider for low-set
    m = r - k_target
    
    if m <= 0:
        return [], 0.0
    
    # Find the m positions with smallest scores
    # argsort returns indices that would sort the array (ascending)
    sorted_indices = torch.argsort(scores)
    low_positions = sorted_indices[:m].tolist()
    
    # Compute low-set score
    low_score = sum(scores[p].item() for p in low_positions)
    
    return low_positions, low_score


def slice_update_by_keep_positions(
    state_dict: dict,
    old_indices: List[int],
    keep_positions: List[int]
) -> Tuple[dict, List[int]]:
    """
    Slice and reorder LoRA weights in a state dict for upload after pruning.
    
    Args:
        state_dict: State dict containing LoRA parameters
        old_indices: Current global indices mapping
        keep_positions: Local positions to keep
        
    Returns:
        Tuple of (new_state_dict, new_indices):
            - new_state_dict: Dict with sliced/reordered LoRA weights
            - new_indices: New global indices list (sorted ascending)
    """
    # Compute new indices (sorted ascending)
    new_indices_unsorted = [old_indices[p] for p in keep_positions]
    sorted_pairs = sorted(enumerate(new_indices_unsorted), key=lambda x: x[1])
    reorder_map = [keep_positions[i] for i, _ in sorted_pairs]
    new_indices = [idx for _, idx in sorted_pairs]
    
    positions_tensor = None
    result = {}
    
    for key, value in state_dict.items():
        if not isinstance(value, torch.Tensor):
            result[key] = value
            continue
        
        if 'lora_A' in key and 'lora_B' not in key:
            # lora_A: (r, in_features) -> select rows
            if positions_tensor is None or positions_tensor.device != value.device:
                positions_tensor = torch.tensor(reorder_map, dtype=torch.long, device=value.device)
            result[key] = value.index_select(0, positions_tensor).clone()
        elif 'lora_B' in key:
            # lora_B: (out_features, r) -> select columns
            if positions_tensor is None or positions_tensor.device != value.device:
                positions_tensor = torch.tensor(reorder_map, dtype=torch.long, device=value.device)
            result[key] = value.index_select(1, positions_tensor).clone()
        else:
            result[key] = value
    
    return result, new_indices


def validate_indices(
    indices: List[int],
    max_rank: int,
    context: str = ""
) -> bool:
    """
    Validate that indices are valid global component indices.
    
    Checks:
        - All indices are unique
        - All indices are in range [0, max_rank - 1]
        
    Args:
        indices: List of global component indices
        max_rank: Maximum rank (global rank space size)
        context: Optional context string for error messages
        
    Returns:
        True if valid, raises ValueError otherwise
    """
    if not indices:
        return True
    
    # Check uniqueness
    if len(indices) != len(set(indices)):
        raise ValueError(
            f"{context} Indices contain duplicates: {indices}"
        )
    
    # Check range
    for idx in indices:
        if not (0 <= idx < max_rank):
            raise ValueError(
                f"{context} Index {idx} out of range [0, {max_rank - 1}]"
            )
    
    return True


def get_component_norm(A_row: torch.Tensor, B_col: torch.Tensor) -> float:
    """
    Compute the norm-product for a single rank-1 component.
    
    Args:
        A_row: Single row from LoRA A (shape: (in_features,))
        B_col: Single column from LoRA B (shape: (out_features,))
        
    Returns:
        ||A_row||_2 * ||B_col||_2
    """
    return (torch.norm(A_row, p=2) * torch.norm(B_col, p=2)).item()


def compute_per_component_norms_from_state_dict(
    state_dict: dict,
    indices: List[int],
    current_rank: Optional[int] = None
) -> Dict[int, float]:
    """
    Compute norm-product for each global index in the state dict.
    
    Used for sparsity-weighted aggregation.
    
    Args:
        state_dict: Dict of LoRA parameters
        indices: Global indices corresponding to local positions
        current_rank: Local rank (auto-detected if None)
        
    Returns:
        Dict mapping global_index -> total_norm_product
    """
    from federatedscope.contrib.common.heterolora_utils import _canonical_lora_key
    
    # Collect A/B pairs
    lora_pairs = {}
    
    for key, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        if 'lora_A' in key and 'lora_B' not in key:
            base = _canonical_lora_key(key)
            base = base.replace('.lora_A.default.weight', '').replace('.lora_A.weight', '')
            if base not in lora_pairs:
                lora_pairs[base] = {'A': None, 'B': None}
            lora_pairs[base]['A'] = tensor
        elif 'lora_B' in key:
            base = _canonical_lora_key(key)
            base = base.replace('.lora_B.default.weight', '').replace('.lora_B.weight', '')
            if base not in lora_pairs:
                lora_pairs[base] = {'A': None, 'B': None}
            lora_pairs[base]['B'] = tensor
    
    # Compute per-component norms
    component_norms = {idx: 0.0 for idx in indices}
    
    for base, params in lora_pairs.items():
        A, B = params['A'], params['B']
        if A is None or B is None:
            continue
        
        local_rank = min(A.shape[0], B.shape[1], len(indices))
        
        for local_pos in range(local_rank):
            global_idx = indices[local_pos]
            norm = get_component_norm(A[local_pos, :], B[:, local_pos])
            component_norms[global_idx] += norm
    
    return component_norms


def distribute_weights_by_indices(
    server_lora_dict: dict,
    client_indices: List[int],
    max_rank: int,
    debug: bool = False
) -> dict:
    """
    Distribute server LoRA weights to a client based on their global indices.
    
    Creates distributed-format LoRA keys (with .rank suffix), the key format
    read by heterolora_utils.load_weight_local.
    
    Args:
        server_lora_dict: Server LoRA weights at max_rank
        client_indices: Client's active global indices
        max_rank: Global maximum rank
        debug: Enable debug logging
        
    Returns:
        Dict of distributed LoRA weights with .rank suffix keys
    """
    from federatedscope.contrib.common.heterolora_utils import _canonical_lora_key
    
    client_rank = len(client_indices)
    if client_rank == 0:
        return {}
    
    # Create index tensor
    indices_tensor = None
    result = {}
    
    for key, tensor in server_lora_dict.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        
        base_key = _canonical_lora_key(key)
        
        if 'lora_A' in key and 'lora_B' not in key:
            # Verify max_rank
            if tensor.shape[0] != max_rank:
                logger.warning(
                    f"Server LoRA A tensor {key} has shape {tensor.shape}, "
                    f"expected first dim = max_rank={max_rank}"
                )
                continue
            
            # Create index tensor on correct device
            if indices_tensor is None or indices_tensor.device != tensor.device:
                indices_tensor = torch.tensor(client_indices, dtype=torch.long, device=tensor.device)
            
            # Select rows by indices
            sliced = tensor.index_select(0, indices_tensor)
            
            # Create distributed key
            dist_key = f"{base_key}.{client_rank}"
            result[dist_key] = sliced
            
        elif 'lora_B' in key:
            # Verify max_rank
            if tensor.shape[1] != max_rank:
                logger.warning(
                    f"Server LoRA B tensor {key} has shape {tensor.shape}, "
                    f"expected second dim = max_rank={max_rank}"
                )
                continue
            
            # Create index tensor on correct device
            if indices_tensor is None or indices_tensor.device != tensor.device:
                indices_tensor = torch.tensor(client_indices, dtype=torch.long, device=tensor.device)
            
            # Select columns by indices
            sliced = tensor.index_select(1, indices_tensor)
            
            # Create distributed key
            dist_key = f"{base_key}.{client_rank}"
            result[dist_key] = sliced
    
    if debug:
        logger.debug(
            f"Distributed {len(result)} LoRA weights for rank={len(client_indices)}, indices={client_indices[:5]}..."
        )
    
    return result


# =============================================================================
# Stage 2 Utilities for AdaS
# =============================================================================

def compute_model_update_from_snapshot(
    current_state_dict: dict,
    snapshot_state_dict: dict,
    survivor_indices: List[int],
    cfg=None
) -> dict:
    """
    Compute fresh local model update by comparing post-training tensors to snapshot.
    
    delta_dict[key][local_pos] = current_state_dict[key][local_pos] - snapshot_state_dict[key][local_pos]
    
    Note: The snapshot may be CPU-backed (to reduce GPU memory pressure) while
    the current model tensors are on GPU. This function handles device/dtype
    alignment by moving a temporary copy of the snapshot tensor to the current
    tensor's device and dtype before subtraction.
    
    Args:
        current_state_dict: Current LoRA state dict after training (typically on GPU)
        snapshot_state_dict: Pre-round snapshot of LoRA state dict (may be CPU-backed)
        survivor_indices: List of global indices for survivors
        
    Returns:
        Dict containing model updates (deltas) for survivor components
    """
    delta_dict = {}
    logged_alignment = False
    
    for key in current_state_dict:
        if key not in snapshot_state_dict:
            continue
        
        current_tensor = current_state_dict[key]
        snapshot_tensor = snapshot_state_dict[key]
        
        if not isinstance(current_tensor, torch.Tensor) or not isinstance(snapshot_tensor, torch.Tensor):
            continue
        
        if 'lora_A' in key or 'lora_B' in key:
            # Compute element-wise difference
            if current_tensor.shape == snapshot_tensor.shape:
                # Align device and dtype: move snapshot to current tensor's device/dtype
                # (the stored snapshot remains CPU-backed; we use a temporary aligned copy)
                if snapshot_tensor.device != current_tensor.device or snapshot_tensor.dtype != current_tensor.dtype:
                    if not logged_alignment and _cfg_debug(cfg):
                        logger.debug(
                            f"Compute_model_update_from_snapshot: "
                            f"aligning CPU-backed snapshot ({snapshot_tensor.device}, {snapshot_tensor.dtype}) "
                            f"to model device ({current_tensor.device}, {current_tensor.dtype})"
                        )
                        logged_alignment = True
                    snapshot_aligned = snapshot_tensor.to(device=current_tensor.device, dtype=current_tensor.dtype)
                else:
                    snapshot_aligned = snapshot_tensor
                
                delta_dict[key] = current_tensor - snapshot_aligned
            else:
                logger.warning(
                    f"Shape mismatch for {key}: "
                    f"current={current_tensor.shape}, snapshot={snapshot_tensor.shape}"
                )
    
    return delta_dict


def apply_residual_to_update(
    delta_dict: dict,
    residual_buffers: Dict[int, Dict[str, torch.Tensor]],
    survivor_indices: List[int],
    cfg=None
) -> dict:
    """
    Construct residual-aware effective update by adding residual buffers.
    
    effective_update = delta + residual
    
    Note: Residual buffers may be CPU-backed (to reduce GPU memory pressure) while
    delta tensors are on GPU. This function handles device/dtype alignment by
    moving a temporary copy of each residual tensor to the effective tensor's
    device and dtype before addition.
    
    Args:
        delta_dict: Fresh local model updates (typically on GPU)
        residual_buffers: Dict mapping global_idx to {key: residual_tensor} (may be CPU-backed)
        survivor_indices: List of global indices for survivors
        
    Returns:
        Dict containing effective updates (delta + residual) for survivor components
    """
    effective_dict = {}
    logged_alignment = False
    
    for key, delta_tensor in delta_dict.items():
        if not isinstance(delta_tensor, torch.Tensor):
            continue
        
        effective_tensor = delta_tensor.clone()
        
        # Add residuals for survivor components
        if 'lora_A' in key and 'lora_B' not in key:
            # lora_A: (r, in_features) - add to rows
            for local_pos, global_idx in enumerate(survivor_indices):
                if global_idx in residual_buffers and key in residual_buffers[global_idx]:
                    if local_pos < effective_tensor.shape[0]:
                        residual_tensor = residual_buffers[global_idx][key]
                        # Align device and dtype: move residual to effective tensor's device/dtype
                        # (the stored residual remains CPU-backed; we use a temporary aligned copy)
                        if residual_tensor.device != effective_tensor.device or residual_tensor.dtype != effective_tensor.dtype:
                            if not logged_alignment and _cfg_debug(cfg):
                                logger.debug(
                                    f"apply_residual_to_update: "
                                    f"aligning CPU-backed residual ({residual_tensor.device}, {residual_tensor.dtype}) "
                                    f"to delta device ({effective_tensor.device}, {effective_tensor.dtype})"
                                )
                                logged_alignment = True
                            residual_aligned = residual_tensor.to(device=effective_tensor.device, dtype=effective_tensor.dtype)
                        else:
                            residual_aligned = residual_tensor
                        effective_tensor[local_pos, :] += residual_aligned
        elif 'lora_B' in key:
            # lora_B: (out_features, r) - add to columns
            for local_pos, global_idx in enumerate(survivor_indices):
                if global_idx in residual_buffers and key in residual_buffers[global_idx]:
                    if local_pos < effective_tensor.shape[1]:
                        residual_tensor = residual_buffers[global_idx][key]
                        # Align device and dtype: move residual to effective tensor's device/dtype
                        # (the stored residual remains CPU-backed; we use a temporary aligned copy)
                        if residual_tensor.device != effective_tensor.device or residual_tensor.dtype != effective_tensor.dtype:
                            if not logged_alignment and _cfg_debug(cfg):
                                logger.debug(
                                    f"apply_residual_to_update: "
                                    f"aligning CPU-backed residual ({residual_tensor.device}, {residual_tensor.dtype}) "
                                    f"to delta device ({effective_tensor.device}, {effective_tensor.dtype})"
                                )
                                logged_alignment = True
                            residual_aligned = residual_tensor.to(device=effective_tensor.device, dtype=effective_tensor.dtype)
                        else:
                            residual_aligned = residual_tensor
                        effective_tensor[:, local_pos] += residual_aligned
        
        effective_dict[key] = effective_tensor
    
    return effective_dict


def compute_stage2_upload_scores(
    effective_update_dict: dict,
    survivor_indices: List[int]
) -> Dict[int, float]:
    """
    Compute Stage 2 upload scores from residual-corrected effective updates.
    
    Score for component j = sum over all LoRA layers of ||effective_A_row_j||_2 * ||effective_B_col_j||_2
    
    Args:
        effective_update_dict: Effective updates (delta + residual)
        survivor_indices: List of global indices for survivors
        
    Returns:
        Dict mapping global_index -> upload_score
    """
    from federatedscope.contrib.common.heterolora_utils import _canonical_lora_key
    
    # Collect A/B pairs
    lora_pairs = {}
    for key, tensor in effective_update_dict.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        if 'lora_A' in key and 'lora_B' not in key:
            base = _canonical_lora_key(key)
            base = base.replace('.lora_A.default.weight', '').replace('.lora_A.weight', '')
            if base not in lora_pairs:
                lora_pairs[base] = {'A': None, 'B': None}
            lora_pairs[base]['A'] = tensor
        elif 'lora_B' in key:
            base = _canonical_lora_key(key)
            base = base.replace('.lora_B.default.weight', '').replace('.lora_B.weight', '')
            if base not in lora_pairs:
                lora_pairs[base] = {'A': None, 'B': None}
            lora_pairs[base]['B'] = tensor
    
    # Compute scores
    scores = {gidx: 0.0 for gidx in survivor_indices}
    
    for base, params in lora_pairs.items():
        A, B = params['A'], params['B']
        if A is None or B is None:
            continue
        
        local_rank = min(A.shape[0], B.shape[1], len(survivor_indices))
        
        for local_pos in range(local_rank):
            if local_pos >= len(survivor_indices):
                break
            global_idx = survivor_indices[local_pos]
            a_norm = torch.norm(A[local_pos, :], p=2).item()
            b_norm = torch.norm(B[:, local_pos], p=2).item()
            scores[global_idx] += a_norm * b_norm
    
    return scores


def compute_stage2_downlink_scores(
    aggregated_global_updates: dict,
    survivor_indices: List[int]
) -> Dict[int, float]:
    """
    Compute Stage 2 downlink scores from aggregated global updates.
    
    Score for component j = sum over all LoRA layers of ||agg_A_row_j||_2 * ||agg_B_col_j||_2
    
    Args:
        aggregated_global_updates: Aggregated global updates from server
        survivor_indices: List of global indices for survivors
        
    Returns:
        Dict mapping global_index -> downlink_score
    """
    from federatedscope.contrib.common.heterolora_utils import _canonical_lora_key
    
    # Collect A/B pairs
    lora_pairs = {}
    for key, tensor in aggregated_global_updates.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        if 'lora_A' in key and 'lora_B' not in key:
            base = _canonical_lora_key(key)
            base = base.replace('.lora_A.default.weight', '').replace('.lora_A.weight', '')
            if base not in lora_pairs:
                lora_pairs[base] = {'A': None, 'B': None}
            lora_pairs[base]['A'] = tensor
        elif 'lora_B' in key:
            base = _canonical_lora_key(key)
            base = base.replace('.lora_B.default.weight', '').replace('.lora_B.weight', '')
            if base not in lora_pairs:
                lora_pairs[base] = {'A': None, 'B': None}
            lora_pairs[base]['B'] = tensor
    
    # Compute scores
    scores = {gidx: 0.0 for gidx in survivor_indices}
    
    for base, params in lora_pairs.items():
        A, B = params['A'], params['B']
        if A is None or B is None:
            continue
        
        # For global aggregated updates, indices are at global positions
        for global_idx in survivor_indices:
            if global_idx < A.shape[0] and global_idx < B.shape[1]:
                a_norm = torch.norm(A[global_idx, :], p=2).item()
                b_norm = torch.norm(B[:, global_idx], p=2).item()
                scores[global_idx] += a_norm * b_norm
    
    return scores


def compute_component_upload_cost(
    effective_update_dict: dict,
    survivor_indices: List[int],
    q_bits: int = 8,
    cmeta_bits: int = 32
) -> Dict[int, float]:
    """
    Compute per-component upload cost in bits.
    
    Cost = (num_params_per_component * q_bits) + cmeta_bits
    
    Args:
        effective_update_dict: Effective updates to send
        survivor_indices: List of global indices for survivors
        q_bits: Quantization bits per parameter value
        cmeta_bits: Bits for metadata per component (index storage)
        
    Returns:
        Dict mapping global_index -> cost_in_bits
    """
    # Count parameters per component
    params_per_component = 0
    
    for key, tensor in effective_update_dict.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        if 'lora_A' in key and 'lora_B' not in key:
            # lora_A: (r, in_features) - one row per component
            params_per_component += tensor.shape[1]
        elif 'lora_B' in key:
            # lora_B: (out_features, r) - one column per component
            params_per_component += tensor.shape[0]
    
    # Cost per component
    cost_per_component = (params_per_component * q_bits) + cmeta_bits
    
    costs = {gidx: cost_per_component for gidx in survivor_indices}
    return costs


def compute_component_downlink_cost(
    aggregated_global_updates: dict,
    survivor_indices: List[int],
    q_bits: int = 8,
    cmeta_bits: int = 32
) -> Dict[int, float]:
    """
    Compute per-component downlink cost in bits.
    
    Args:
        aggregated_global_updates: Aggregated global updates
        survivor_indices: List of global indices for survivors
        q_bits: Quantization bits per parameter value
        cmeta_bits: Bits for metadata per component
        
    Returns:
        Dict mapping global_index -> cost_in_bits
    """
    # Count parameters per component
    params_per_component = 0
    
    for key, tensor in aggregated_global_updates.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        if 'lora_A' in key and 'lora_B' not in key:
            params_per_component += tensor.shape[1]
        elif 'lora_B' in key:
            params_per_component += tensor.shape[0]
    
    cost_per_component = (params_per_component * q_bits) + cmeta_bits
    
    costs = {gidx: cost_per_component for gidx in survivor_indices}
    return costs


def greedy_select_by_score_cost_ratio(
    scores: Dict[int, float],
    costs: Dict[int, float],
    budget: float,
    survivor_indices: List[int]
) -> List[int]:
    """
    Greedy selection by score-to-cost ratio under budget constraint.
    
    Algorithm:
    1. Compute score/cost ratio for each component
    2. Sort by ratio (descending)
    3. Select greedily until budget exhausted
    
    Args:
        scores: Dict mapping global_index -> score
        costs: Dict mapping global_index -> cost
        budget: Total budget in bits
        survivor_indices: List of global indices for survivors
        
    Returns:
        List of selected global indices
    """
    if budget <= 0:
        return []
    
    # Compute ratios
    ratios = []
    for gidx in survivor_indices:
        score = scores.get(gidx, 0.0)
        cost = costs.get(gidx, 1.0)
        if cost > 0:
            ratio = score / cost
        else:
            ratio = float('inf') if score > 0 else 0.0
        ratios.append((gidx, score, cost, ratio))
    
    # Sort by ratio (descending)
    ratios.sort(key=lambda x: x[3], reverse=True)
    
    # Greedy selection
    selected = []
    remaining_budget = budget
    
    for gidx, score, cost, ratio in ratios:
        if cost <= remaining_budget:
            selected.append(gidx)
            remaining_budget -= cost
    
    return selected


def slice_model_update_by_indices(
    model_update_dict: dict,
    survivor_indices: List[int],
    selected_indices: List[int]
) -> dict:
    """
    Slice model updates to include only selected components.
    
    Args:
        model_update_dict: Full model updates for all survivors
        survivor_indices: List of all survivor global indices
        selected_indices: List of selected global indices (subset of survivors)
        
    Returns:
        Dict containing model updates only for selected components
    """
    if not selected_indices:
        return {}
    
    # Map global indices to local positions
    survivor_to_local = {gidx: i for i, gidx in enumerate(survivor_indices)}
    selected_local_positions = [survivor_to_local[gidx] for gidx in selected_indices if gidx in survivor_to_local]
    
    if not selected_local_positions:
        return {}
    
    positions_tensor = None
    result = {}
    
    for key, tensor in model_update_dict.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        
        if 'lora_A' in key and 'lora_B' not in key:
            if positions_tensor is None or positions_tensor.device != tensor.device:
                positions_tensor = torch.tensor(selected_local_positions, dtype=torch.long, device=tensor.device)
            # Select rows
            if max(selected_local_positions) < tensor.shape[0]:
                result[key] = tensor.index_select(0, positions_tensor).clone()
        elif 'lora_B' in key:
            if positions_tensor is None or positions_tensor.device != tensor.device:
                positions_tensor = torch.tensor(selected_local_positions, dtype=torch.long, device=tensor.device)
            # Select columns
            if max(selected_local_positions) < tensor.shape[1]:
                result[key] = tensor.index_select(1, positions_tensor).clone()
    
    return result


def apply_sparse_update_to_model(
    model_state_dict: dict,
    sparse_update_dict: dict,
    download_indices: List[int],
    survivor_indices: List[int]
) -> dict:
    """
    Apply sparse model updates to local model at downloaded positions.
    
    Args:
        model_state_dict: Current model state dict
        sparse_update_dict: Sparse updates received for download_indices
        download_indices: List of global indices that were downloaded
        survivor_indices: Full survivor indices list
        
    Returns:
        Updated model state dict
    """
    if not download_indices:
        return model_state_dict
    
    # Map download indices to positions in sparse_update_dict
    download_to_sparse_pos = {gidx: i for i, gidx in enumerate(download_indices)}
    
    # Map survivor indices to positions in model
    survivor_to_local = {gidx: i for i, gidx in enumerate(survivor_indices)}
    
    result = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in model_state_dict.items()}
    
    for key, sparse_tensor in sparse_update_dict.items():
        if key not in result or not isinstance(sparse_tensor, torch.Tensor):
            continue
        
        model_tensor = result[key]
        
        if 'lora_A' in key and 'lora_B' not in key:
            # Apply downloaded rows to model
            for gidx in download_indices:
                if gidx in survivor_to_local and gidx in download_to_sparse_pos:
                    model_pos = survivor_to_local[gidx]
                    sparse_pos = download_to_sparse_pos[gidx]
                    if model_pos < model_tensor.shape[0] and sparse_pos < sparse_tensor.shape[0]:
                        model_tensor[model_pos, :] = sparse_tensor[sparse_pos, :]
        elif 'lora_B' in key:
            # Apply downloaded columns to model
            for gidx in download_indices:
                if gidx in survivor_to_local and gidx in download_to_sparse_pos:
                    model_pos = survivor_to_local[gidx]
                    sparse_pos = download_to_sparse_pos[gidx]
                    if model_pos < model_tensor.shape[1] and sparse_pos < sparse_tensor.shape[1]:
                        model_tensor[:, model_pos] = sparse_tensor[:, sparse_pos]
        
        result[key] = model_tensor
    
    return result


def update_residual_buffers_after_upload(
    residual_buffers: Dict[int, Dict[str, torch.Tensor]],
    effective_update_dict: dict,
    upload_indices: List[int],
    survivor_indices: List[int],
    cfg=None
) -> Dict[int, Dict[str, torch.Tensor]]:
    """
    Update residual buffers by subtracting what was actually sent.
    
    For uploaded components: residual = effective_update - sent_update = 0
    For non-uploaded survivors: residual remains (or is initialized from effective_update)
    
    Note: Residual buffers are stored on CPU to reduce GPU memory pressure.
    The effective updates may be on GPU, so we clone and move to CPU when storing.
    
    Args:
        residual_buffers: Current residual buffers (CPU-backed)
        effective_update_dict: Effective updates (delta + old residual), may be on GPU
        upload_indices: Components that were actually uploaded
        survivor_indices: All survivor indices
        
    Returns:
        Updated residual buffers (CPU-backed)
    """
    upload_set = set(upload_indices)
    survivor_to_local = {gidx: i for i, gidx in enumerate(survivor_indices)}
    
    # Clear residuals for uploaded components, keep/update for others
    new_residuals = {}
    logged_cpu_storage = False
    
    for gidx in survivor_indices:
        if gidx in upload_set:
            # Uploaded - clear residual
            continue
        
        local_pos = survivor_to_local.get(gidx)
        if local_pos is None:
            continue
        
        # Not uploaded - store effective update as new residual (on CPU)
        component_residual = {}
        for key, tensor in effective_update_dict.items():
            if not isinstance(tensor, torch.Tensor):
                continue
            
            if 'lora_A' in key and 'lora_B' not in key:
                if local_pos < tensor.shape[0]:
                    # Clone and move to CPU for storage
                    residual_slice = tensor[local_pos, :].clone().cpu()
                    component_residual[key] = residual_slice
                    if not logged_cpu_storage and tensor.device.type != 'cpu' and _cfg_debug(cfg):
                        logger.debug(
                            f"update_residual_buffers_after_upload: "
                            f"storing residuals on CPU (source was {tensor.device})"
                        )
                        logged_cpu_storage = True
            elif 'lora_B' in key:
                if local_pos < tensor.shape[1]:
                    # Clone and move to CPU for storage
                    residual_slice = tensor[:, local_pos].clone().cpu()
                    component_residual[key] = residual_slice
                    if not logged_cpu_storage and tensor.device.type != 'cpu' and _cfg_debug(cfg):
                        logger.debug(
                            f"update_residual_buffers_after_upload: "
                            f"storing residuals on CPU (source was {tensor.device})"
                        )
                        logged_cpu_storage = True
        
        if component_residual:
            new_residuals[gidx] = component_residual
    
    return new_residuals


def prune_residual_buffers(
    residual_buffers: Dict[int, Dict[str, torch.Tensor]],
    pruned_indices: List[int]
) -> Dict[int, Dict[str, torch.Tensor]]:
    """
    Remove residual buffers for structurally pruned components.
    
    Args:
        residual_buffers: Current residual buffers
        pruned_indices: Components that were pruned by Stage 1
        
    Returns:
        Updated residual buffers with pruned components removed
    """
    pruned_set = set(pruned_indices)
    return {gidx: buffers for gidx, buffers in residual_buffers.items() if gidx not in pruned_set}


def validate_upload_subset(
    upload_indices: List[int],
    survivor_indices: List[int],
    context: str = ""
) -> bool:
    """
    Validate that upload indices are subset of survivors.
    
    Args:
        upload_indices: Proposed upload indices
        survivor_indices: Current survivor indices
        context: Context string for error messages
        
    Returns:
        True if valid, raises ValueError otherwise
    """
    survivor_set = set(survivor_indices)
    for idx in upload_indices:
        if idx not in survivor_set:
            raise ValueError(
                f"{context} Upload index {idx} not in survivor set."
            )
    return True


def compute_residual_norm_summary(
    residual_buffers: Dict[int, Dict[str, torch.Tensor]]
) -> Dict[str, float]:
    """
    Compute summary statistics of residual buffer norms.
    
    Args:
        residual_buffers: Current residual buffers
        
    Returns:
        Dict with min, avg, max, total norms
    """
    if not residual_buffers:
        return {'min': 0.0, 'avg': 0.0, 'max': 0.0, 'total': 0.0, 'count': 0}
    
    norms = []
    for gidx, buffers in residual_buffers.items():
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
