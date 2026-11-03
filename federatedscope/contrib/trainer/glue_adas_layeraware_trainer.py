"""AdaS-LayerAware Stage 1 regularizer, shared by the GLUE and LLM trainers.

This module provides the Stage 1 regularizer for AdaS-LayerAware that:
- Computes penalties over exact layer-aware ComponentIDs
- Does NOT collapse same-slot components from different layers into one shared score
- Supports both layer-wise and global competition modes

layer-aware SCORING DESIGN:
- AdaS: score[p] where p is a shared slot index (one score for all layers' slot p)
- layer-aware: score[(layer_key, global_idx)] (one score per exact layer+index pair)
This regularizer uses the layer-aware ComponentID-based scoring throughout.
"""

import logging
from typing import Dict, List

import torch
import federatedscope.contrib.common as fs_common
from federatedscope.contrib.common.adas_layeraware_utils import ComponentID, canonicalize_lora_layer_key

logger = logging.getLogger(__name__)


def compute_adas_layeraware_regularizer(ctx):
    """
    Compute the AdaS-LayerAware Stage 1 regularizer.

    layer-aware key difference: Regularizer is computed over exact layer-aware ComponentIDs,
    not a flat shared low-set vector.

    For each low-set ComponentID (layer_key, global_idx):
        penalty += score(layer_key, global_idx)

    Args:
        ctx: Trainer context with model and cfg

    Returns:
        Regularizer loss term (weighted sum of low-set component scores), or None
    """
    adas_layeraware_cfg = fs_common.get_adas_layeraware_cfg(ctx.cfg)
    if adas_layeraware_cfg is None:
        return None

    stage1_cfg = getattr(adas_layeraware_cfg, 'stage1', None)
    if stage1_cfg is None:
        return None

    reg_weight = getattr(stage1_cfg, 'regularizer_weight', 0.01)
    if reg_weight <= 0:
        return None

    # Get layer-aware low candidates from model (set by client)
    low_candidates = getattr(ctx.model, 'adas_layeraware_low_candidates', None)
    if low_candidates is None or len(low_candidates) == 0:
        return None

    # Get survivors by layer from model
    survivors_by_layer = getattr(ctx.model, 'adas_layeraware_survivors_by_layer', None)
    if survivors_by_layer is None or not survivors_by_layer:
        return None

    try:
        # Compute per-ComponentID scores directly from live trainable parameters.
        # Important: never fall back to state_dict() here, because detached tensors
        # would break autograd and make the regularizer ineffective.
        scores = _compute_component_scores_grouped_differentiable(
            ctx.model, survivors_by_layer
        )

        if not scores:
            return None

        # Compute penalty over low-set ComponentIDs
        valid_candidates = [cid for cid in low_candidates if cid in scores]
        if not valid_candidates:
            return None

        low_penalty = torch.stack([scores[cid] for cid in valid_candidates]).sum()

        if not hasattr(ctx, '_adas_layeraware_reg_logged') or not ctx._adas_layeraware_reg_logged:
            has_grad = bool(getattr(low_penalty, 'requires_grad', False))
            if has_grad:
                if bool(getattr(ctx.cfg, 'debug', False)):
                    logger.debug(
                        f"Stage1 layer-aware Regularizer: reg_weight={reg_weight}, "
                        f"low_set_size={len(valid_candidates)}, "
                        f"low_penalty_value={low_penalty.item():.6f}, "
                        f"requires_grad=True"
                    )
            else:
                logger.warning(
                    f"Stage1 layer-aware Regularizer WARNING: requires_grad=False! "
                    f"reg_weight={reg_weight}, low_set_size={len(valid_candidates)}, "
                    f"Regularizer may be ineffective."
                )
            ctx._adas_layeraware_reg_logged = True

        return reg_weight * low_penalty

    except Exception as e:
        logger.warning(f"Failed to compute Stage1 layer-aware low-set penalty: {e}")
        return None


def _compute_component_scores_grouped_differentiable(
    model,
    survivors_by_layer: Dict[str, List[int]]
) -> Dict[ComponentID, torch.Tensor]:
    """
    Compute per-ComponentID importance scores with gradient preservation.

    This is similar to the layer-aware utility scoring, but it is intentionally trainer-
    side and strictly autograd-safe: it only uses live tensors from
    model.named_parameters() and never uses state_dict() fallbacks.

    Args:
        model: Model with LoRA adapters
        survivors_by_layer: Dict mapping layer_key -> list of survivor global indices

    Returns:
        Dict mapping ComponentID -> importance score tensor (with gradients)
    """
    scores: Dict[ComponentID, torch.Tensor] = {}

    # Use only live trainable parameters so autograd stays intact.
    named_params = dict(model.named_parameters())

    # Build a canonical layer -> LoRA parameter-name map once.
    layer_to_keys: Dict[str, Dict[str, str]] = {}
    for name in named_params.keys():
        canonical = _canonicalize_param_name(name)
        if canonical == name:
            continue

        entry = layer_to_keys.setdefault(canonical, {})
        if 'lora_A' in name and 'lora_B' not in name:
            entry['A'] = name
        elif 'lora_B' in name:
            entry['B'] = name

    skipped_layers = 0

    for layer_key, survivor_indices in survivors_by_layer.items():
        entry = layer_to_keys.get(layer_key, None)
        if entry is None:
            skipped_layers += 1
            continue

        a_key = entry.get('A', None)
        b_key = entry.get('B', None)
        if a_key is None or b_key is None:
            skipped_layers += 1
            continue

        A = named_params.get(a_key, None)
        B = named_params.get(b_key, None)
        if A is None or B is None:
            skipped_layers += 1
            continue

        # Score each survivor component in this exact layer.
        for local_pos, global_idx in enumerate(survivor_indices):
            if local_pos >= A.shape[0] or local_pos >= B.shape[1]:
                continue

            a_row = A[local_pos, :]
            b_col = B[:, local_pos]

            # Norm-product score preserves gradient as long as A/B are live params.
            a_norm = torch.norm(a_row, p=2)
            b_norm = torch.norm(b_col, p=2)

            component_id = (layer_key, global_idx)
            scores[component_id] = a_norm * b_norm

    if skipped_layers > 0 and bool(getattr(model, 'debug', False)):
        logger.debug(
            "Stage1 layer-aware Regularizer: skipped %d layers due to missing live LoRA params",
            skipped_layers,
        )

    return scores


def _canonicalize_param_name(name: str) -> str:
    """Convert a live named-parameter key to the exact layer-aware canonical layer key.

    This follows the same base policy as the layer-aware utils canonicalization used by
    the client/server path, then removes the outer trainer/model wrapper prefix
    so trainer-side named_parameters() keys align with survivors_by_layer keys.

    Example:
        model.base_model.model.deberta.encoder.layer.0.attention.output.dense.lora_A.default.weight
        -> base_model.model.deberta.encoder.layer.0.attention.output.dense
    """
    # Reuse the exact layer-key canonicalization policy from the layer-aware utils.
    result = canonicalize_lora_layer_key(name)

    # Trainer-side named_parameters() on GLUEAdapterModel adds an outer "model."
    # prefix that is not present in the client/server survivors_by_layer keys.
    if result.startswith("model."):
        result = result[len("model."):]

    return result
