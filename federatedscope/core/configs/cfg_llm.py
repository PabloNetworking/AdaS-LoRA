import json
import logging

from federatedscope.core.configs.config import CN
from federatedscope.register import register_config

logger = logging.getLogger(__name__)


def extend_llm_cfg(cfg):
    # ---------------------------------------------------------------------- #
    # LLM related options
    # ---------------------------------------------------------------------- #
    cfg.llm = CN()
    cfg.llm.tok_len = 128
    # When True and the tokenizer has a chat_template, LLMDataset formats
    # prompts with the model's native chat template; otherwise it uses the
    # Alpaca PROMPT_DICT (base-model prompts).
    cfg.llm.use_chat_template = False
    cfg.llm.retry_on_nan_loss = False

    # ---------------------------------------------------------------------- #
    # Cache for LLM
    # ---------------------------------------------------------------------- #
    cfg.llm.cache = CN()
    cfg.llm.cache.model = ''

    # ---------------------------------------------------------------------- #
    # Chat tools for LLM
    # ---------------------------------------------------------------------- #
    cfg.llm.chat = CN()
    cfg.llm.chat.max_history_len = 10
    cfg.llm.chat.max_len = 100

    # ---------------------------------------------------------------------- #
    # Deepspeed related options
    # ---------------------------------------------------------------------- #
    cfg.llm.deepspeed = CN()
    cfg.llm.deepspeed.use = False
    cfg.llm.deepspeed.ds_config = ''

    # ---------------------------------------------------------------------- #
    # Adapters for LLM
    # ---------------------------------------------------------------------- #
    cfg.llm.adapter = CN()
    cfg.llm.adapter.use = False
    cfg.llm.adapter.args = [{}]
    # Move adapter to `cpu` after training, which can save memory but cost
    # more time.
    cfg.llm.adapter.mv_to_cpu = False
    
    # HeteroLoRA specific configurations
    cfg.llm.adapter.max_rank = 64  # Maximum rank for zero-padding/truncation
    
    # Client configuration distribution strategy
    # Options: 'homo', 'random', 'heavy_tail', 'heavy_tail_strong', 'normal', 'custom'
    cfg.llm.adapter.hetero_strategy = 'homo'
    
    # Client-specific rank configurations (optional, can be auto-generated)
    # Format: {client_id: {module_name: rank, ...}, ...}
    # Example: {0: {'q_proj': 8, 'v_proj': 8}, 1: {'q_proj': 16, 'v_proj': 16}}
    # If not provided and hetero_strategy != 'homo', will be auto-generated
    cfg.llm.adapter.hetero_ranks = CN(new_allowed=True)
    
    # Client-specific alpha values (optional)
    # Format: {client_id: alpha_value, ...}
    # Example: {0: 16, 1: 32, 2: 16}
    cfg.llm.adapter.hetero_alpha = CN(new_allowed=True)
    
    # Client-specific target modules (optional)
    # Format: {client_id: [module_list], ...}
    # Example: {0: ['q_proj', 'v_proj'], 1: ['q_proj', 'v_proj', 'k_proj']}

    # ---------------------------------------------------------------------- #
    # Base Model Quantization (usable with or without FAH-QLoRA)
    # This mirrors cfg.*.adapter.fah.base_quant, but is NOT gated by fah.enabled.
    # ---------------------------------------------------------------------- #
    cfg.llm.adapter.base_quant = CN()
    cfg.llm.adapter.base_quant.enabled = False
    cfg.llm.adapter.base_quant.distribution = CN(new_allowed=True)

    # ---------------------------------------------------------------------- #
    # FAH-QLoRA: Federated Adaptive Heterogeneous QLoRA configuration
    # Implements two-stage rank selection:
    #   Stage 1: Adapt average LoRA rank across rounds by maximizing loss 
    #            decrease rate
    #   Stage 2: Per-device rank assignment to minimize round completion time
    # ---------------------------------------------------------------------- #
    cfg.llm.adapter.fah = CN()
    cfg.llm.adapter.fah.enabled = False  # Whether FAH-QLoRA dynamic rank is active
    cfg.llm.adapter.fah.init_rank = 8    # Initial homogeneous rank r_0 for warm-up
    cfg.llm.adapter.fah.r_min = 2        # Global minimum rank constraint
    cfg.llm.adapter.fah.r_max = 64       # Global maximum rank constraint
    cfg.llm.adapter.fah.lambda_dec = 1   # Step size λ1 for rank decrease (eq. 11)
    cfg.llm.adapter.fah.lambda_inc = 1   # Step size λ2 for rank increase (eq. 11)
    cfg.llm.adapter.fah.warmup_rounds = 1  # Number of homogeneous FedAvg rounds
    cfg.llm.adapter.fah.alpha_fraction = 0.3 # Heuristic calculation of rank-independent compute time (e.g. forward pass, data loading, everything not scaling with rank)
    cfg.llm.adapter.fah.validation_fraction = 0.1  # Fraction of local data for FAH eval
    cfg.llm.adapter.fah.validation_steps = 10  # Max validation steps per FAH eval
    
    # ---------------------------------------------------------------------- #
    # Heterogeneous base-model quantization, as described in the FAH-QLoRA paper: "Devices are divided into three groups: one-third utilize
    # 4-bit quantized base models, another third use 8-bit quantized base 
    # models, and the remaining third employ 16-bit quantized base models.
    # The default LoRA data type is FP32."
    # ---------------------------------------------------------------------- #
    cfg.llm.adapter.fah.base_quant = CN()
    cfg.llm.adapter.fah.base_quant.enabled = False  # Enable heterogeneous base quantization
    # Distribution of quantization bits across clients
    # Keys are bit-widths (4, 8, 16), values are fractions summing to 1.0
    # Default: 1/3 each
    cfg.llm.adapter.fah.base_quant.distribution = CN(new_allowed=True)
    # Note: The actual default values {4: 0.333, 8: 0.333, 16: 0.334} are set
    # programmatically in fed_runner.py since CN doesn't support numeric keys

    # ---------------------------------------------------------------------- #
    # HetLoRA: Rank self-pruning + Sparsity-weighted aggregation
    # Implements the HetLoRA baseline from the paper with:
    #   - Client-side rank self-pruning via tail-rank regularizer
    #   - Server-side sparsity-weighted aggregation
    # Gated by federate.method == "hetlora" (not "heterolora" or "fah_qlora")
    # ---------------------------------------------------------------------- #
    cfg.llm.adapter.hetlora = CN()
    cfg.llm.adapter.hetlora.enabled = False  # Master switch for HetLoRA
    
    # Rank bounds
    cfg.llm.adapter.hetlora.rank_min = 2      # Minimum rank after pruning
    cfg.llm.adapter.hetlora.rank_max = 64     # Maximum rank (should match max_rank)
    cfg.llm.adapter.hetlora.init_rank = 64    # Initial rank for clients
    
    # Pruning configuration (client-side rank self-pruning)
    cfg.llm.adapter.hetlora.pruning = CN()
    cfg.llm.adapter.hetlora.pruning.enabled = True  # Enable pruning (within hetlora)
    cfg.llm.adapter.hetlora.pruning.decay = 0.99     # Decay factor for prune threshold
    cfg.llm.adapter.hetlora.pruning.regularizer_weight = 0.01  # Lambda for tail regularizer
    
    # Aggregation configuration (server-side sparsity-weighted)
    cfg.llm.adapter.hetlora.aggregation = CN()
    # Mode: 'sample_size' (default FedAvg), 'sparsity_weighted' (HetLoRA paper)
    cfg.llm.adapter.hetlora.aggregation.mode = 'sparsity_weighted'
    cfg.llm.adapter.hetlora.aggregation.epsilon = 1e-8  # Numerical stability for norm

    # ---------------------------------------------------------------------- #
    # AdaS-PruneOnly: Index-aware rank-1 component aggregation
    # Implements adaptive sparse LoRA with:
    #   - Global component identity space (rank indices)
    #   - Per-client active subset of global indices
    #   - Index-aware aggregation (per rank-1 component)
    #   - Component-based importance pruning
    # Gated by federate.method == "adas_pruneonly"
    # ---------------------------------------------------------------------- #
    cfg.llm.adapter.adas_pruneonly = CN()
    cfg.llm.adapter.adas_pruneonly.enabled = False  # Master switch
    
    # Rank bounds
    cfg.llm.adapter.adas_pruneonly.rank_min = 2      # Minimum active components
    cfg.llm.adapter.adas_pruneonly.rank_max = 64     # Maximum rank (global max_rank)
    cfg.llm.adapter.adas_pruneonly.init_rank = 64    # Initial active set size
    
    # Pruning configuration (component-based, ratio-based)
    cfg.llm.adapter.adas_pruneonly.pruning = CN()
    cfg.llm.adapter.adas_pruneonly.pruning.enabled = True
    cfg.llm.adapter.adas_pruneonly.pruning.gamma = 0.9  # Decay factor (<1), target = gamma * current
    cfg.llm.adapter.adas_pruneonly.pruning.regularizer_weight = 0.01  # Lambda for low-set regularizer
    
    # Aggregation configuration (index-aware)
    cfg.llm.adapter.adas_pruneonly.aggregation = CN()
    # Mode: 'sample_size' (weighted by sample count), 'sparsity_weighted' (weighted by component norms)
    cfg.llm.adapter.adas_pruneonly.aggregation.mode = 'sparsity_weighted'
    cfg.llm.adapter.adas_pruneonly.aggregation.epsilon = 1e-8  # Numerical stability
    

    # ---------------------------------------------------------------------- #
    # AdaS: Two-stage sparse federated LoRA
    # Stage 1: Structural sparsity over survivor set (same as prune-only)
    # Stage 2: Communication sparsity via residual-aware budgeted selection
    # Gated by federate.method == "adas"
    # ---------------------------------------------------------------------- #
    cfg.llm.adapter.adas = CN()
    cfg.llm.adapter.adas.enabled = False  # Master switch
    
    # Rank bounds
    cfg.llm.adapter.adas.rank_min = 2      # Minimum active components
    cfg.llm.adapter.adas.rank_max = 64     # Maximum rank (global max_rank)
    cfg.llm.adapter.adas.init_rank = 64    # Initial active set size
    
    # Stage 1 configuration (structural sparsity over survivor set)
    cfg.llm.adapter.adas.stage1 = CN()
    cfg.llm.adapter.adas.stage1.gamma = 0.9  # Decay factor (<1), target = gamma * current
    cfg.llm.adapter.adas.stage1.regularizer_weight = 0.01  # Lambda for low-set regularizer
    
    # Stage 2 configuration (communication sparsity over survivors)
    cfg.llm.adapter.adas.stage2 = CN()
    cfg.llm.adapter.adas.stage2.enabled = True  # Enable Stage 2 communication sparsity
    cfg.llm.adapter.adas.stage2.q_up_bits = 8    # Quantization bits for uplink
    cfg.llm.adapter.adas.stage2.q_down_bits = 32  # Quantization bits for downlink
    cfg.llm.adapter.adas.stage2.cmeta_bits = 32  # Bits per metadata element (indices)
    cfg.llm.adapter.adas.stage2.uplink_budget_window_s = 1.0   # Time window for uplink budget (seconds)
    cfg.llm.adapter.adas.stage2.downlink_budget_window_s = 1.0 # Time window for downlink budget (seconds)
    cfg.llm.adapter.adas.stage2.residual_enabled = True  # Enable residual-aware scoring
    
    # Aggregation configuration (index-aware, over transmitted updates only)
    cfg.llm.adapter.adas.aggregation = CN()
    cfg.llm.adapter.adas.aggregation.mode = 'sample_size'  # 'sample_size' or 'sparsity_weighted'
    cfg.llm.adapter.adas.aggregation.epsilon = 1e-8  # Numerical stability
    

    # ---------------------------------------------------------------------- #
    # AdaS-LayerAware: layer-aware component identity + cross-layer competition
    # ---------------------------------------------------------------------- #
    cfg.llm.adapter.adas_layeraware = CN()
    cfg.llm.adapter.adas_layeraware.enabled = False
    cfg.llm.adapter.adas_layeraware.rank_min = 4
    cfg.llm.adapter.adas_layeraware.rank_max = 64
    cfg.llm.adapter.adas_layeraware.init_rank = 64

    cfg.llm.adapter.adas_layeraware.stage1 = CN()
    cfg.llm.adapter.adas_layeraware.stage1.gamma = 0.9
    cfg.llm.adapter.adas_layeraware.stage1.regularizer_weight = 0.01

    cfg.llm.adapter.adas_layeraware.stage2 = CN()
    cfg.llm.adapter.adas_layeraware.stage2.enabled = True
    cfg.llm.adapter.adas_layeraware.stage2.q_up_bits = 8
    cfg.llm.adapter.adas_layeraware.stage2.q_down_bits = 8
    cfg.llm.adapter.adas_layeraware.stage2.cmeta_bits = 32
    cfg.llm.adapter.adas_layeraware.stage2.uplink_budget_window_s = 1.0
    cfg.llm.adapter.adas_layeraware.stage2.downlink_budget_window_s = 1.0
    cfg.llm.adapter.adas_layeraware.stage2.residual_enabled = True

    cfg.llm.adapter.adas_layeraware.aggregation = CN()
    cfg.llm.adapter.adas_layeraware.aggregation.mode = 'sample_size'
    cfg.llm.adapter.adas_layeraware.aggregation.epsilon = 1e-8

    cfg.llm.adapter.adas_layeraware.stage1_global_competition = True
    cfg.llm.adapter.adas_layeraware.stage2_global_competition = True

    # ---------------------------------------------------------------------- #
    # Offsite-tuning related options
    # ---------------------------------------------------------------------- #
    cfg.llm.offsite_tuning = CN()
    cfg.llm.offsite_tuning.use = False
    cfg.llm.offsite_tuning.strategy = 'drop_layer'
    cfg.llm.offsite_tuning.kwargs = [{}]
    cfg.llm.offsite_tuning.emu_l = 1  # Index of emulator layer left
    cfg.llm.offsite_tuning.emu_r = 10  # Index of emulator layer right

    # Used in `eval`
    cfg.llm.offsite_tuning.eval_type = 'emu'  # Choose one of `[emu, full]`

    # Emulator alignment will use dataset in Server
    cfg.llm.offsite_tuning.emu_align = CN()
    cfg.llm.offsite_tuning.emu_align.use = False
    cfg.llm.offsite_tuning.emu_align.restore_from = ''
    cfg.llm.offsite_tuning.emu_align.save_to = ''
    cfg.llm.offsite_tuning.emu_align.exit_after_align = False

    # Server held-out data
    cfg.llm.offsite_tuning.emu_align.data = CN()
    cfg.llm.offsite_tuning.emu_align.data.root = 'data'
    cfg.llm.offsite_tuning.emu_align.data.type = 'alpaca@llm'
    cfg.llm.offsite_tuning.emu_align.data.splits = [0.8, 0.1, 0.1]

    cfg.llm.offsite_tuning.emu_align.train = CN()
    cfg.llm.offsite_tuning.emu_align.train.local_update_steps = 10
    cfg.llm.offsite_tuning.emu_align.train.batch_or_epoch = 'batch'
    cfg.llm.offsite_tuning.emu_align.train.lm_loss_weight = 0.1
    cfg.llm.offsite_tuning.emu_align.train.kd_loss_weight = 0.9

    cfg.llm.offsite_tuning.emu_align.train.optimizer = CN(new_allowed=True)
    cfg.llm.offsite_tuning.emu_align.train.optimizer.type = 'SGD'
    cfg.llm.offsite_tuning.emu_align.train.optimizer.lr = 0.01


def assert_llm_cfg(cfg):
    if cfg.llm.offsite_tuning.emu_align.use:
        if cfg.llm.offsite_tuning.emu_align.restore_from != '':
            logger.warning(
                'Enabling `restore_from` in offsite_tuning emulator '
                'alignment will skip training the emulator.')


register_config("llm", extend_llm_cfg)