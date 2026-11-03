from federatedscope.llm.model.adapter_builder import AdapterModel
import os
import json
import logging

logger = logging.getLogger(__name__)


def get_model_from_huggingface(model_name, config):
    """
    Load a causal language model from HuggingFace transformers library.

    Args:
        model_name (str): The name of the pre-trained model to load.
        config (Config): The configuration object that contains the model
            parameters.

    Returns:
        AutoModelForCausalLM: A causal language model object.
    """
    from transformers import AutoModelForCausalLM
    import torch

    kwargs = {}
    if len(config.llm.cache.model):
        kwargs['cache_dir'] = config.llm.cache.model

    # Get quantization settings
    quant_method = getattr(config.computation_quantization, 'method', 'none')
    nbits = getattr(config.computation_quantization, 'nbits', 4)

    if quant_method == 'qlora' and nbits in [4, 8]:
        from transformers import BitsAndBytesConfig
        from peft import prepare_model_for_kbit_training

        device = config.device if hasattr(config, 'device') else 0
        
        # Configure BitsAndBytesConfig based on quantization bits
        if nbits == 4:
            logger.info(f"[FAH] Loading {model_name} with 4-bit quantization (NF4)")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4'
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                quantization_config=quantization_config,
                torch_dtype=torch.bfloat16,
                trust_remote_code=False,
            )
        else:  # 8-bit
            logger.info(f"[FAH] Loading {model_name} with 8-bit quantization (LLM.int8)")
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=True,  # Keep non-quantized weights in fp16
            )
            # For 8-bit, don't specify torch_dtype - let bitsandbytes handle it
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                quantization_config=quantization_config,
                trust_remote_code=False,
            )
        return prepare_model_for_kbit_training(model,
                                               use_gradient_checkpointing=True)

    elif nbits == 16 or quant_method == 'none':
        # No quantization: nbits == 16 loads the base model in bfloat16
        # (the 16-bit clients of FAH-QLoRA); otherwise it is loaded in fp32.
        if nbits == 16:
            logger.info(f"[FAH] Loading {model_name} in 16-bit (bfloat16) mode")
            return AutoModelForCausalLM.from_pretrained(
                model_name, 
                torch_dtype=torch.bfloat16,
                ignore_mismatched_sizes=True, 
                **kwargs
            )
        else:
            # Standard fp32 loading
            return AutoModelForCausalLM.from_pretrained(
                model_name, 
                ignore_mismatched_sizes=True, 
                **kwargs
            )
    
    else:
        raise ValueError(f'Invalid quantization config: method={quant_method}, '
                        f'nbits={nbits}. Supported: qlora with 4/8 bits, or 16-bit.')



def get_model_from_modelscope(model_name, config):
    """
    Load a causal language model from ModelScope models library.

    Args:
        model_name (str): The name of the pre-trained model to load.
        config (Config): The configuration object that contains the model
            parameters.

    Returns:
        Model: A causal language model object.
    """
    from modelscope import AutoModelForCausalLM

    kwargs = {}
    if len(config.llm.cache.model):
        kwargs['cache_dir'] = config.llm.cache.model

    return AutoModelForCausalLM.from_pretrained(model_name, **kwargs)


def get_llm(config):
    """
    Get a causal language model based on the configuration.

    Args:
        config (Config): The configuration object that contains the model
            parameters.

    Returns:
        AdapterModel: A causal language model object with optional adapter
            layers.
    """
    from federatedscope.llm.dataloader import get_tokenizer

    model_config = config.model
    model_name, model_hub = model_config.type.split('@')
    if model_hub == 'huggingface_llm':
        model = get_model_from_huggingface(model_name=model_name,
                                           config=config)
    elif model_hub == 'modelscope_llm':
        model = get_model_from_modelscope(model_name=model_name, config=config)
    else:
        raise NotImplementedError(f'Not support LLM {model_name} in'
                                  f' {model_hub}.')

    # Resize LLM model based on settings
    tokenizer, num_new_tokens = \
        get_tokenizer(model_name, config.data.root, config.llm.tok_len,
                      model_hub)
    model.resize_token_embeddings(len(tokenizer))
    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

    # Safely copy adapter args
    adapter_cfg = config.llm.adapter
    base_args = adapter_cfg.args[0] if adapter_cfg.args else {}
    args = dict(base_args)  # copy to avoid surprising global mutation

    # Add quantization bits if using QLoRA
    if adapter_cfg.use and config.computation_quantization.method == 'qlora':
        args['nbits'] = config.computation_quantization.nbits

    # Determine whether we are in a FAH / HeteroLoRA regime
    fah_enabled = hasattr(adapter_cfg, 'fah') and adapter_cfg.fah.enabled
    max_rank = getattr(adapter_cfg, 'max_rank', None)

    if adapter_cfg.use and max_rank is not None:
        # If FAH is enabled, enforce r_max == max_rank at config level
        if fah_enabled:
            fah_cfg = adapter_cfg.fah
            if fah_cfg.r_max != max_rank:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(
                    f"[FAH] Mismatched fah.r_max={fah_cfg.r_max} and "
                    f"adapter.max_rank={max_rank} in model_builder. "
                    f"Using max_rank={max_rank} for FAH."
                )

        original_r = args.get('r', None)
        if original_r != max_rank:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(
                f"Overriding LoRA rank from {original_r} to {max_rank} "
                f"to match global max_rank."
            )
        args['r'] = max_rank

    elif adapter_cfg.use and fah_enabled:
        # FAH-QLoRA requires a global max_rank.
        raise ValueError(
            "[FAH] llm.adapter.max_rank must be set when FAH-QLoRA is enabled."
        )

    # Build model with adapter at global max_rank
    model = AdapterModel(model, use_adapter=adapter_cfg.use, **args)

    # FFA-LoRA freezing logic
    if config.federate.freeze_A:
        for name, param in model.named_parameters():
            if "lora_A" in name:
                param.requires_grad = False

    return model
