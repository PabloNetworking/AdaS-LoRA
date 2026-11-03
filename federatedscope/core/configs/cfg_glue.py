"""
Configuration for GLUE benchmark tasks with FAH-QLoRA support.

This module extends the configuration with GLUE-specific options for
sequence classification and regression tasks (SST-2, QNLI, MRPC, STS-B, RTE, etc.)
with DeBERTa and other transformer models.
"""
import logging
from federatedscope.core.configs.config import CN
from federatedscope.register import register_config

logger = logging.getLogger(__name__)


def extend_glue_cfg(cfg):
    """
    Extend configuration with GLUE-specific options.
    
    This includes:
    - Sequence length and tokenization settings
    - Task-specific configurations
    - Adapter settings mirroring llm.adapter structure
    - FAH-QLoRA configuration for GLUE tasks
    """
    # ---------------------------------------------------------------------- #
    # GLUE related options
    # ---------------------------------------------------------------------- #
    cfg.glue = CN()
    cfg.glue.max_length = 128  # Maximum sequence length for tokenization
    cfg.glue.retry_on_nan_loss = True  # Retry batch if loss is NaN
    cfg.glue.task_name = ''  # Populated automatically from data.type during dataset loading
    cfg.glue.mnli_eval_split = 'matched'  # 'matched' or 'mismatched'

    # Cache directory for models and tokenizers
    cfg.glue.cache_dir = ''
    
    # ---------------------------------------------------------------------- #
    # Adapters for GLUE (mirrors llm.adapter structure for consistency)
    # ---------------------------------------------------------------------- #
    cfg.glue.adapter = CN()
    cfg.glue.adapter.use = False
    cfg.glue.adapter.args = [{}]
    cfg.glue.adapter.mv_to_cpu = False
    
    # HeteroLoRA specific configurations
    cfg.glue.adapter.max_rank = 64  # Maximum rank for zero-padding/truncation
    
    # Client configuration distribution strategy
    # Options: 'homo', 'random', 'heavy_tail', 'heavy_tail_strong', 'normal', 'custom'
    cfg.glue.adapter.hetero_strategy = 'homo'
    
    # Client-specific configurations (optional, can be auto-generated)
    cfg.glue.adapter.hetero_ranks = CN(new_allowed=True)
    cfg.glue.adapter.hetero_alpha = CN(new_allowed=True)

    # ---------------------------------------------------------------------- #
    # Base Model Quantization (usable with or without FAH-QLoRA)
    # This mirrors cfg.*.adapter.fah.base_quant, but is NOT gated by fah.enabled.
    # ---------------------------------------------------------------------- #
    cfg.glue.adapter.base_quant = CN()
    cfg.glue.adapter.base_quant.enabled = False
    cfg.glue.adapter.base_quant.distribution = CN(new_allowed=True)

    # ---------------------------------------------------------------------- #
    # FAH-QLoRA: Federated Adaptive Heterogeneous QLoRA for GLUE
    # ---------------------------------------------------------------------- #
    cfg.glue.adapter.fah = CN()
    cfg.glue.adapter.fah.enabled = False
    cfg.glue.adapter.fah.init_rank = 8
    cfg.glue.adapter.fah.r_min = 2
    cfg.glue.adapter.fah.r_max = 64
    cfg.glue.adapter.fah.lambda_dec = 1
    cfg.glue.adapter.fah.lambda_inc = 1
    cfg.glue.adapter.fah.warmup_rounds = 1
    cfg.glue.adapter.fah.alpha_fraction = 0.3
    cfg.glue.adapter.fah.validation_fraction = 0.1
    cfg.glue.adapter.fah.validation_steps = 10
    
    # ---------------------------------------------------------------------- #
    # Heterogeneous base-model quantization, as described in the FAH-QLoRA paper: "Devices are divided into three groups: one-third utilize
    # 4-bit quantized base models, another third use 8-bit quantized base 
    # models, and the remaining third employ 16-bit quantized base models.
    # The default LoRA data type is FP32."
    # ---------------------------------------------------------------------- #
    cfg.glue.adapter.fah.base_quant = CN()
    cfg.glue.adapter.fah.base_quant.enabled = False  # Enable heterogeneous base quantization
    # Distribution of quantization bits across clients
    # Keys are bit-widths (4, 8, 16), values are fractions summing to 1.0
    # Default: 1/3 each
    cfg.glue.adapter.fah.base_quant.distribution = CN(new_allowed=True)
    # Note: The actual default values {4: 0.333, 8: 0.333, 16: 0.334} are set
    # programmatically in fed_runner.py since CN doesn't support numeric keys

    # ---------------------------------------------------------------------- #
    # HetLoRA: Rank self-pruning + Sparsity-weighted aggregation
    # Mirrors llm.adapter.hetlora structure for GLUE tasks
    # ---------------------------------------------------------------------- #
    cfg.glue.adapter.hetlora = CN()
    cfg.glue.adapter.hetlora.enabled = False
    
    # Rank bounds
    cfg.glue.adapter.hetlora.rank_min = 4
    cfg.glue.adapter.hetlora.rank_max = 64
    cfg.glue.adapter.hetlora.init_rank = 64
    
    # Pruning configuration
    cfg.glue.adapter.hetlora.pruning = CN()
    cfg.glue.adapter.hetlora.pruning.enabled = True
    cfg.glue.adapter.hetlora.pruning.decay = 0.99     # Decay factor for prune threshold
    cfg.glue.adapter.hetlora.pruning.regularizer_weight = 0.01
    
    # Aggregation configuration
    cfg.glue.adapter.hetlora.aggregation = CN()
    cfg.glue.adapter.hetlora.aggregation.mode = 'sparsity_weighted'
    cfg.glue.adapter.hetlora.aggregation.epsilon = 1e-8

    # ---------------------------------------------------------------------- #
    # AdaS-PruneOnly: Index-aware rank-1 component aggregation
    # Mirrors llm.adapter.adas_pruneonly structure for GLUE tasks
    # ---------------------------------------------------------------------- #
    cfg.glue.adapter.adas_pruneonly = CN()
    cfg.glue.adapter.adas_pruneonly.enabled = False
    
    # Rank bounds
    cfg.glue.adapter.adas_pruneonly.rank_min = 4
    cfg.glue.adapter.adas_pruneonly.rank_max = 64
    cfg.glue.adapter.adas_pruneonly.init_rank = 64
    
    # Pruning configuration (component-based, ratio-based)
    cfg.glue.adapter.adas_pruneonly.pruning = CN()
    cfg.glue.adapter.adas_pruneonly.pruning.enabled = True
    cfg.glue.adapter.adas_pruneonly.pruning.gamma = 0.9
    cfg.glue.adapter.adas_pruneonly.pruning.regularizer_weight = 0.01
    
    # Aggregation configuration (index-aware)
    cfg.glue.adapter.adas_pruneonly.aggregation = CN()
    cfg.glue.adapter.adas_pruneonly.aggregation.mode = 'sparsity_weighted' #or sample_size
    cfg.glue.adapter.adas_pruneonly.aggregation.epsilon = 1e-8
    
    # ---------------------------------------------------------------------- #
    # AdaS: Two-stage sparse federated LoRA
    # Stage 1: Structural sparsity over survivor set (same as prune-only)
    # Stage 2: Communication sparsity via residual-aware budgeted selection
    # Mirrors llm.adapter.adas structure for GLUE tasks
    # ---------------------------------------------------------------------- #
    cfg.glue.adapter.adas = CN()
    cfg.glue.adapter.adas.enabled = False  # Master switch
    
    # Rank bounds
    cfg.glue.adapter.adas.rank_min = 4
    cfg.glue.adapter.adas.rank_max = 64
    cfg.glue.adapter.adas.init_rank = 64
    
    # Stage 1 configuration (structural sparsity over survivor set)
    cfg.glue.adapter.adas.stage1 = CN()
    cfg.glue.adapter.adas.stage1.gamma = 0.9
    cfg.glue.adapter.adas.stage1.regularizer_weight = 0.01
    
    # Stage 2 configuration (communication sparsity over survivors)
    cfg.glue.adapter.adas.stage2 = CN()
    cfg.glue.adapter.adas.stage2.enabled = True
    cfg.glue.adapter.adas.stage2.q_up_bits = 8
    cfg.glue.adapter.adas.stage2.q_down_bits = 32
    cfg.glue.adapter.adas.stage2.cmeta_bits = 32
    cfg.glue.adapter.adas.stage2.uplink_budget_window_s = 1.0
    cfg.glue.adapter.adas.stage2.downlink_budget_window_s = 1.0
    cfg.glue.adapter.adas.stage2.residual_enabled = True   
 
    # Aggregation configuration
    cfg.glue.adapter.adas.aggregation = CN()
    cfg.glue.adapter.adas.aggregation.mode = 'sample_size'
    cfg.glue.adapter.adas.aggregation.epsilon = 1e-8
    
    # ---------------------------------------------------------------------- #
    # AdaS-LayerAware: layer-aware component identity
    # Key difference from AdaS: exact layer-aware ComponentID = (layer_key, global_idx)
    # Supports grouped payloads and cross-layer global competition
    # ---------------------------------------------------------------------- #
    cfg.glue.adapter.adas_layeraware = CN()
    cfg.glue.adapter.adas_layeraware.enabled = False  # Master switch
    
    # Rank bounds
    # rank_min is a single global floor shared by all layers (not per-layer):
    # the total number of survivors across all layers stays >= rank_min.
    cfg.glue.adapter.adas_layeraware.rank_min = 4
    cfg.glue.adapter.adas_layeraware.rank_max = 64
    cfg.glue.adapter.adas_layeraware.init_rank = 64
    
    # Stage 1 configuration (structural sparsity over survivor set)
    cfg.glue.adapter.adas_layeraware.stage1 = CN()
    cfg.glue.adapter.adas_layeraware.stage1.gamma = 0.9
    cfg.glue.adapter.adas_layeraware.stage1.regularizer_weight = 0.01
    
    # Stage 2 configuration (communication sparsity over survivors)
    cfg.glue.adapter.adas_layeraware.stage2 = CN()
    cfg.glue.adapter.adas_layeraware.stage2.enabled = True
    cfg.glue.adapter.adas_layeraware.stage2.q_up_bits = 8
    cfg.glue.adapter.adas_layeraware.stage2.q_down_bits = 8
    cfg.glue.adapter.adas_layeraware.stage2.cmeta_bits = 32
    cfg.glue.adapter.adas_layeraware.stage2.uplink_budget_window_s = 1.0
    cfg.glue.adapter.adas_layeraware.stage2.downlink_budget_window_s = 1.0
    cfg.glue.adapter.adas_layeraware.stage2.residual_enabled = True
      
    # Aggregation configuration
    cfg.glue.adapter.adas_layeraware.aggregation = CN()
    cfg.glue.adapter.adas_layeraware.aggregation.mode = 'sample_size'
    cfg.glue.adapter.adas_layeraware.aggregation.epsilon = 1e-8
    
    # layer-aware-specific: Layer-aware component identity configuration
    
    # Competition scope for stage-1 pruning and stage-2 selection:
    # False = per-layer selection, True = one cross-layer global pool
    cfg.glue.adapter.adas_layeraware.stage1_global_competition = True
    cfg.glue.adapter.adas_layeraware.stage2_global_competition = True


register_config('glue', extend_glue_cfg)

