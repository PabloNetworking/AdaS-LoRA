"""
Model builder for GLUE tasks with LoRA/QLoRA and FAH-QLoRA support.

Supports DeBERTa, BERT, RoBERTa, and other transformer models for 
sequence classification and regression tasks on GLUE benchmarks.
"""
import logging
import os
import torch
import torch.nn as nn
from collections import OrderedDict
from typing import Optional

logger = logging.getLogger(__name__)


class GLUEAdapterModel(nn.Module):
    """
    A wrapper class for sequence classification models with adapter (LoRA) support.
    
    Similar to AdapterModel in LLM, but designed for sequence classification tasks.
    Runs the forward pass under autocast when the base model uses a bf16/fp16
    compute dtype.
    """
    
    def __init__(
        self, 
        model: nn.Module, 
        use_adapter: bool = False,
        **kwargs
    ):
        """
        Initialize the GLUE adapter model wrapper.
        
        Args:
            model: Base transformer model for sequence classification
            use_adapter: Whether to apply LoRA adapter
            **kwargs: Adapter configuration (adapter_package, adapter_method, etc.)
        """
        super().__init__()
        
        self.model = None
        if use_adapter:
            adapter_package = kwargs.pop('adapter_package', 'peft')
            adapter_method = kwargs.pop('adapter_method', 'lora')
            
            self.model = enable_glue_adapter(
                model, adapter_package, adapter_method, **kwargs
            )
        else:
            self.model = model
    
    def forward(self, *args, **kwargs):
        """Forward pass through the model, with a one-time dtype debug log."""

        # On the first call, log the dtype of a frozen base weight and of a LoRA weight.
        try:
            if not getattr(self, "_fah_dtype_log_done", False):
                setattr(self, "_fah_dtype_log_done", True)
                base = self.model  # PEFT-wrapped model (PeftModelForSequenceClassification)

                # 1) Frozen DeBERTa attention in_proj weight
                name_in = (
                    "base_model.model.deberta.encoder.layer.0."
                    "attention.self.in_proj.weight"
                )
                try:
                    p_in = base.get_parameter(name_in)
                    logger.debug("[DTYPE_CHECK] %s dtype=%s", name_in, p_in.dtype)
                except Exception as e:
                    logger.warning("[DTYPE_CHECK] could not read %s: %s", name_in, e)

                # 2) One LoRA A weight on that same module (if present)
                try:
                    for n, p in base.named_parameters():
                        if (
                            "base_model.model.deberta.encoder.layer.0."
                            "attention.self.in_proj.lora_A" in n
                        ):
                            logger.debug("[DTYPE_CHECK] %s dtype=%s", n, p.dtype)
                            break
                except Exception as e:
                    logger.warning("[DTYPE_CHECK] could not read LoRA A param: %s", e)
        except Exception as e:
            # Never break training just because of debug logging
            logger.warning("[DTYPE_CHECK] forward debug failed: %s", e)

        # Run under autocast when the model was loaded with a bf16/fp16 compute dtype.
        fah_dtype = getattr(self.model, "fah_compute_dtype", None)
        # Prevent mixed-dtype matmuls/linears when some activations become fp32.
        if fah_dtype in (torch.float16, torch.bfloat16) and torch.cuda.is_available():
            with torch.autocast(device_type="cuda", dtype=fah_dtype):
                return self.model.forward(*args, **kwargs)
        return self.model.forward(*args, **kwargs)
    
    def state_dict(self, return_trainable: bool = True, *args, **kwargs):
        """
        Get state dict, optionally filtered to trainable parameters only.
        
        Args:
            return_trainable: If True, return only parameters that require grad
        
        Returns:
            State dictionary
        """
        if return_trainable:
            return self.get_trainable_state_dict()
        return self.model.state_dict(*args, **kwargs)
    
    def load_state_dict(self, state_dict, strict: bool = False):
        """Load state dict into the model."""
        return self.model.load_state_dict(state_dict, strict=False)
    
    def get_trainable_state_dict(self):
        """Get only the trainable parameters from the model."""
        grad_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                grad_params.append(name)
        
        model_state_dict = self.model.state_dict()
        new_state_dict = OrderedDict()
        for k, v in model_state_dict.items():
            if k in grad_params:
                new_state_dict[k] = v
        return new_state_dict
    
    def save_model(self, path: str, state: int = 0):
        """Save the model checkpoint."""
        ckpt = {'cur_round': state, 'model': self.model.state_dict()}
        torch.save(ckpt, path)


def enable_glue_adapter(
    model: nn.Module,
    package: str,
    adapter: str,
    **kwargs
) -> nn.Module:
    """
    Enable a LoRA adapter for a GLUE classification model.
    
    Args:
        model: Pre-trained sequence classification model
        package: Adapter package ('peft')
        adapter: Adapter method ('lora', 'qlora')
        **kwargs: Adapter configuration (r, lora_alpha, target_modules, etc.)
    
    Returns:
        Model with LoRA adapter applied
    """
    adapter = adapter.lower()
    
    if package == 'peft':
        from peft import get_peft_model, TaskType
        
        if adapter in ['lora', 'qlora']:
            from peft import LoraConfig
            
            # Default target modules for DeBERTa
            target_modules = kwargs.pop('target_modules', None)
            if target_modules is None:
                # Auto-detect target modules based on model architecture
                model_type = getattr(model.config, 'model_type', '').lower()
                if 'deberta' in model_type:
                    target_modules = ['query_proj', 'key_proj', 'value_proj', 
                                     'dense', 'output.dense']
                elif 'bert' in model_type:
                    target_modules = ['query', 'key', 'value', 'dense']
                elif 'roberta' in model_type:
                    target_modules = ['query', 'key', 'value', 'dense']
                else:
                    # Generic fallback
                    target_modules = ['query', 'key', 'value']
                logger.info(f"Auto-detected target modules for {model_type}: {target_modules}")
            
            peft_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                r=kwargs.get('r', 8),
                lora_alpha=kwargs.get('lora_alpha', 16),
                lora_dropout=kwargs.get('lora_dropout', 0.05),
                target_modules=target_modules,
                bias="none",
            )
            
            
            model = get_peft_model(model, peft_config)

            model.print_trainable_parameters()
            
        
        else:
            raise NotImplementedError(f"Adapter {adapter} not implemented for GLUE")
    else:
        raise NotImplementedError(f"Package {package} not supported for GLUE")
    
    return model


def get_model_from_huggingface_for_glue(
    model_name: str,
    config,
    num_labels: int = 2,
) -> nn.Module:
    """
    Load a sequence classification model from HuggingFace.
    
    Supports DeBERTa, BERT, RoBERTa, and other AutoModelForSequenceClassification
    compatible models.
    
    Args:
        model_name: HuggingFace model name or path
        config: Configuration object
        num_labels: Number of output labels for classification
    
    Returns:
        Sequence classification model
    """
    from transformers import AutoModelForSequenceClassification
    
    kwargs = {}
    if hasattr(config, 'glue') and hasattr(config.glue, 'cache_dir'):
        kwargs['cache_dir'] = config.glue.cache_dir
    elif hasattr(config.data, 'root'):
        kwargs['cache_dir'] = config.data.root

    if num_labels == 1:
        kwargs['problem_type'] = 'regression'
    
    # Check for quantization (QLoRA for 4-bit/8-bit, or 16-bit)
    if hasattr(config, 'computation_quantization'):
        quant_method = getattr(config.computation_quantization, 'method', None)
        nbits = getattr(config.computation_quantization, 'nbits', 4)
        fah_compute_dtype = None
        
        if quant_method == 'qlora' and nbits in [4, 8]:
            from transformers import BitsAndBytesConfig
            from peft import prepare_model_for_kbit_training
            
            # Get device for models that don't support device_map="auto" (like DeBERTa)
            device = config.device if hasattr(config, 'device') else 0
            if isinstance(device, int) and device < 0:
                device = "cpu"
            device_map = {"": device}
            
            # Optional fallback from 8-bit to 4-bit for DeBERTa. It is off
            # (allow_8bit_deberta=True), so 8-bit DeBERTa clients load with LLM.int8().
            is_deberta = 'deberta' in model_name.lower()
            
            allow_8bit_deberta = True

            if nbits == 8 and is_deberta and not allow_8bit_deberta:
                logger.warning(
                    "[QUANT] 8-bit quantization is not compatible with DeBERTa models "
                    "(bitsandbytes LLM.int8() incompatibility). Falling back to 4-bit."
                )
                nbits = 4
            
            if nbits == 4:
                logger.info(f"[QUANT] Loading {model_name} with 4-bit quantization (NF4) on GPU {device}")
                fah_compute_dtype = torch.bfloat16
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type='nf4'
                )
                                
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_name,
                    num_labels=num_labels,
                    device_map=device_map,
                    quantization_config=quantization_config,
                    torch_dtype=fah_compute_dtype,
                    trust_remote_code=True,
                    **kwargs
                )
            else:  # 8-bit (LLM.int8)
                logger.info(f"[QUANT] Loading {model_name} with 8-bit quantization (LLM.int8) on GPU {device}")
                fah_compute_dtype = torch.bfloat16
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                    llm_int8_skip_modules=['classifier', 'pooler'],  # Skip quantizing newly initialized modules
                )
                
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_name,
                    num_labels=num_labels,
                    device_map=device_map,
                    quantization_config=quantization_config,
                    torch_dtype=fah_compute_dtype,  # non-quantized weights (e.g. the classifier) in bf16
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,  # Helps with initialization order
                    **kwargs
                )
            # Store per-client compute dtype/quantization bits for downstream (e.g., autocast)

            if fah_compute_dtype is not None:
                model.fah_compute_dtype = fah_compute_dtype
                model.fah_quant_bits = nbits

            return prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=True
            )
                
        
        # Full-precision (non k-bit) paths: 16-bit (fp16/bf16) and 32-bit (fp32)
        if nbits in [16, 32] or quant_method in [None, 'none', 'full', 'fp16', 'fp32']:
            # For models like DeBERTa, avoid device_map="auto" and place the full model on a single device.
            device = config.device if hasattr(config, 'device') else 0
            if isinstance(device, int) and device < 0:
                device = "cpu"
            device_map = {"": device}

            if nbits == 16:
                # GPU-aware dtype selection based on compute capability:
                #   sm_80+ (Ampere): bfloat16
                #   otherwise: float32 (fp16 causes NaN in DeBERTa's
                #              disentangled attention — logits overflow fp16 range)
                _cap = (0, 0)
                if torch.cuda.is_available():
                    _cap = torch.cuda.get_device_capability()
                if _cap[0] >= 8 and torch.cuda.is_bf16_supported():
                    fah_compute_dtype = torch.bfloat16
                else:
                    fah_compute_dtype = torch.float32
                logger.info(f"[QUANT] Loading {model_name} in 16-bit mode ({fah_compute_dtype}, no quantization) on GPU {device} (sm_{_cap[0]}{_cap[1]})")
            else:
                fah_compute_dtype = torch.float32
                logger.info(f"[QUANT] Loading {model_name} in 32-bit mode (fp32, no quantization) on GPU {device}")

            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=num_labels,
                device_map=device_map,
                torch_dtype=fah_compute_dtype,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                ignore_mismatched_sizes=True,
                **kwargs
            )

            # Record per-client compute dtype/quant bits for downstream accounting
            model.fah_compute_dtype = fah_compute_dtype
            model.fah_quant_bits = int(nbits)

            # Enable gradient checkpointing for full-precision paths to control activation memory.
            # Set ADAS_DISABLE_GRADIENT_CHECKPOINTING=1 to turn it off.
            _disable_gc = os.environ.get("ADAS_DISABLE_GRADIENT_CHECKPOINTING", "0") == "1"
            if _disable_gc:
                logger.info("[QUANT] Gradient checkpointing disabled via ADAS_DISABLE_GRADIENT_CHECKPOINTING")
            else:
                try:
                    if hasattr(model, "enable_input_require_grads"):
                        model.enable_input_require_grads()
                    if hasattr(model, "gradient_checkpointing_enable"):
                        model.gradient_checkpointing_enable()
                    if hasattr(model, "config"):
                        if hasattr(model.config, "gradient_checkpointing"):
                            model.config.gradient_checkpointing = True
                        if hasattr(model.config, "use_cache"):
                            model.config.use_cache = False
                except Exception:
                    logger.exception("[QUANT] Failed to enable gradient checkpointing for full-precision model")

            return model

# Standard loading without quantization (fp32) fallback when computation_quantization is not set
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
        trust_remote_code=True,
        **kwargs
    )
    
    return model


def get_glue_model(config):
    """
    Build a GLUE model with optional LoRA/QLoRA adapter.
    
    This is the main entry point for creating GLUE models in the federated
    learning framework.
    
    Args:
        config: Configuration object containing model and adapter settings
    
    Returns:
        GLUEAdapterModel with optional LoRA adapter configured
    """
    from federatedscope.glue.dataloader import get_glue_tokenizer

    model_config = config.model
    model_name, model_hub = model_config.type.split('@')
    
    # Determine number of labels from config
    num_labels = getattr(model_config, 'out_channels', 2)
    
    if model_hub in ['huggingface_llm', 'huggingface']:
        model = get_model_from_huggingface_for_glue(
            model_name=model_name,
            config=config,
            num_labels=num_labels,
        )
    else:
        raise NotImplementedError(f"Model hub '{model_hub}' not supported for GLUE")

    if hasattr(model, 'config'):
        if num_labels == 1:
            model.config.problem_type = 'regression'
        else:
            model.config.problem_type = 'single_label_classification'

    # Get adapter configuration
    # Look for adapter config in either glue.adapter or llm.adapter
    adapter_cfg = None
    if hasattr(config, 'glue') and hasattr(config.glue, 'adapter'):
        adapter_cfg = config.glue.adapter
    elif hasattr(config, 'llm') and hasattr(config.llm, 'adapter'):
        adapter_cfg = config.llm.adapter
    
    if adapter_cfg is None or not getattr(adapter_cfg, 'use', False):
        # No adapter, return model wrapped
        return GLUEAdapterModel(model, use_adapter=False)
    
    # Configure adapter arguments
    base_args = adapter_cfg.args[0] if adapter_cfg.args else {}
    args = dict(base_args)
    
    # Add quantization bits if using QLoRA
    if hasattr(config, 'computation_quantization'):
        quant_method = getattr(config.computation_quantization, 'method', None)
        if quant_method == 'qlora':
            args['nbits'] = getattr(config.computation_quantization, 'nbits', 4)
    
    # Handle FAH / HeteroLoRA max_rank configuration
    fah_enabled = hasattr(adapter_cfg, 'fah') and getattr(adapter_cfg.fah, 'enabled', False)
    max_rank = getattr(adapter_cfg, 'max_rank', None)
    
    if max_rank is not None:
        if fah_enabled:
            fah_cfg = adapter_cfg.fah
            if fah_cfg.r_max != max_rank:
                logger.warning(
                    f"[FAH] Mismatched fah.r_max={fah_cfg.r_max} and "
                    f"adapter.max_rank={max_rank}. Using max_rank={max_rank}"
                )
        
        original_r = args.get('r', None)
        if original_r != max_rank:
            logger.warning(
                f"Overriding LoRA rank from {original_r} to {max_rank} "
                f"to match global max_rank"
            )
        args['r'] = max_rank
    elif fah_enabled:
        raise ValueError(
            "[FAH] adapter.max_rank must be set when FAH-QLoRA is enabled"
        )
    
    # Build model with adapter
    model = GLUEAdapterModel(model, use_adapter=adapter_cfg.use, **args)
    
    # FFA-LoRA freezing logic
    if hasattr(config.federate, 'freeze_A') and config.federate.freeze_A:
        for name, param in model.named_parameters():
            if "lora_A" in name:
                param.requires_grad = False
                logger.info(f"[FFA-LoRA] Froze {name}")
    
    return model