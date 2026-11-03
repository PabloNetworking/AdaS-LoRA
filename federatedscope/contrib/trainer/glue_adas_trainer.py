"""AdaS Stage 1 regularizer, shared by the GLUE and LLM trainers."""

import logging

import torch
import federatedscope.contrib.common as fs_common

logger = logging.getLogger(__name__)


def compute_adas_regularizer(ctx):
    adas_cfg = fs_common.get_adas_cfg(ctx.cfg)
    if adas_cfg is None:
        return None

    stage1_cfg = getattr(adas_cfg, 'stage1', None)
    if stage1_cfg is None:
        return None

    reg_weight = getattr(stage1_cfg, 'regularizer_weight', 0.01)
    if reg_weight <= 0:
        return None

    low_positions = getattr(ctx.model, 'adas_low_positions', None)
    if low_positions is None or len(low_positions) == 0:
        return None

    try:
        from federatedscope.contrib.common.adas_utils import compute_component_scores
        current_rank = getattr(ctx.model, 'adas_current_rank', None)
        scores = compute_component_scores(ctx.model, current_rank=current_rank)

        if len(scores) == 0:
            return None

        valid_positions = [p for p in low_positions if p < len(scores)]
        if not valid_positions:
            return None

        low_penalty = torch.stack([scores[p] for p in valid_positions]).sum()
        if not hasattr(ctx, '_adas_reg_logged') or not ctx._adas_reg_logged:
            has_grad = bool(getattr(low_penalty, 'requires_grad', False))
            if has_grad:
                if bool(getattr(ctx.cfg, 'debug', False)):
                    logger.debug(
                        f"Stage1 Regularizer: reg_weight={reg_weight}, "
                        f"low_set_size={len(valid_positions)}, "
                        f"low_penalty_value={low_penalty.item():.6f}, "
                        f"requires_grad=True"
                    )
            else:
                logger.warning(
                    f"Stage1 Regularizer WARNING: requires_grad=False! "
                    f"reg_weight={reg_weight}, low_set_size={len(valid_positions)}, "
                    f"low_penalty_value={low_penalty.item() if hasattr(low_penalty, 'item') else low_penalty:.6f}. "
                    f"Regularizer may be ineffective."
                )
            ctx._adas_reg_logged = True
        return reg_weight * low_penalty
    except Exception as e:
        logger.warning(f"Failed to compute Stage1 low-set penalty: {e}")
        return None