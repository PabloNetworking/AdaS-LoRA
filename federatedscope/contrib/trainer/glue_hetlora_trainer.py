"""HetLoRA rank self-pruning regularizer (decayed tail penalty), shared by the GLUE and LLM trainers."""

import logging

import federatedscope.contrib.common as fs_common

logger = logging.getLogger(__name__)


def compute_hetlora_regularizer(ctx):
    hetlora_cfg = fs_common.get_hetlora_cfg(ctx.cfg)
    if hetlora_cfg is None:
        return None

    pruning_cfg = getattr(hetlora_cfg, 'pruning', None)
    if not (pruning_cfg and getattr(pruning_cfg, 'enabled', True)):
        return None

    reg_weight = getattr(pruning_cfg, 'regularizer_weight', 0.01)
    decay = getattr(pruning_cfg, 'decay', 0.99)
    if reg_weight <= 0:
        return None

    try:
        from federatedscope.contrib.common.heterolora_utils import tail_penalty
        penalty = tail_penalty(ctx.model, decay)
        return reg_weight * penalty
    except Exception as e:
        logger.warning(f"[HetLoRA] Failed to compute tail penalty: {e}")
        return None