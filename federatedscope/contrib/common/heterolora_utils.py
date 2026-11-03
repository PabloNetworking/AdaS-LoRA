"""
Utility functions for HeteroLoRA operations.

This module provides functions for:
- Modifying LoRA adapter ranks dynamically
- Distributing aggregated weights to clients with different ranks
- Loading weights for heterogeneous LoRA configurations
"""
import torch
from tqdm import tqdm
import logging
import peft
from typing import Optional 

import federatedscope.contrib.common as fs_common

logger = logging.getLogger(__name__)


def _cfg_debug(cfg) -> bool:
    """Return True iff top-level cfg.debug is True via shared common helpers."""
    if cfg is None:
        return False
    safe_getattr = getattr(fs_common, "_safe_getattr", getattr)
    return bool(safe_getattr(cfg, "debug", False))


def is_qlora_client_cfg(cfg) -> bool:
    """
    Return True if this client configuration uses QLoRA
    (4 or 8 bit base weights via bitsandbytes).
    """
    q = getattr(cfg, "computation_quantization", None)
    if q is None:
        return False
    method = str(getattr(q, "method", "none")).lower()
    try:
        nbits = int(getattr(q, "nbits", 16))
    except Exception:
        nbits = 16
    return (method == "qlora") and (nbits in (4, 8))

def _canonical_lora_key(name: str) -> str:
    """
    Normalize a LoRA parameter key for matching ranks and distribution.
    
    - Strip a trailing '.<digits>' rank suffix if present.
    - Strip a leading 'model.' if present (to align server/client prefixes).
    
    This ensures server keys like:
        base_model.model.model.layers.3.self_attn.q_proj.lora_A.default.weight
    match client keys like:
        model.base_model.model.model.layers.3.self_attn.q_proj.lora_A.default.weight
    
    Args:
        name: Full parameter key name
        
    Returns:
        Canonicalized key (without rank suffix, without leading 'model.')
    """
    # Strip rank suffix if present (e.g., ".8" at the end)
    parts = name.split('.')
    if parts and parts[-1].isdigit():
        parts = parts[:-1]
    base = '.'.join(parts)
    
    # Strip leading 'model.' to match server state dict keys
    if base.startswith('model.'):
        base = base[len('model.'):]
    
    return base


def _get_rank_for_param(name: str, rank_config: dict):
    """
    Given a full parameter name (no rank suffix) and a client rank_config,
    return the rank for this parameter.
    
    rank_config keys are patterns (substrings) like:
      'q_proj', 'layers.3.self_attn.q_proj', etc.
    
    The most specific (longest) matching pattern wins.
    
    Args:
        name: Parameter name (will be canonicalized internally)
        rank_config: Dict mapping pattern strings to ranks
            e.g., {'q_proj': 8, 'layers.3.self_attn.q_proj': 16}
            
    Returns:
        The rank for this parameter, or None if no pattern matches
    """
    canonical = _canonical_lora_key(name)
    best_pattern = None
    best_len = -1
    best_rank = None
    
    for pattern, rank in rank_config.items():
        if pattern in canonical and len(pattern) > best_len:
            best_pattern = pattern
            best_len = len(pattern)
            best_rank = rank
    
    return best_rank


def _recast_trainable_params_after_modify(
    peft_model,
    compute_dtype: Optional[torch.dtype] = None,
    non_lora_trainable_dtype: Optional[torch.dtype] = None,
    log_prefix: str = "[HeteroLoRA]",
    cfg=None,
) -> None:
    """
    Recast trainable parameters after modify_adapter updated LoRA ranks.

    - LoRA parameters (names containing 'lora_') are sent to compute_dtype if provided.
    - Non LoRA trainable parameters are sent to non_lora_trainable_dtype if provided,
      otherwise to compute_dtype if provided.
    """
    debug_mode = _cfg_debug(cfg) if cfg is not None else False

    if compute_dtype is None and non_lora_trainable_dtype is None:
        return

    try:
        device = next(peft_model.parameters()).device
    except StopIteration:
        device = None

    num_cast = 0
    for name, p in peft_model.named_parameters():
        if not getattr(p, "requires_grad", False):
            continue
        if not p.is_floating_point():
            continue

        is_lora = "lora_" in name
        target_dtype = p.dtype

        if is_lora and compute_dtype is not None:
            target_dtype = compute_dtype
        elif (not is_lora) and (non_lora_trainable_dtype is not None):
            target_dtype = non_lora_trainable_dtype
        elif compute_dtype is not None:
            target_dtype = compute_dtype

        if target_dtype is not None and p.dtype is not target_dtype:
            p.data = p.data.to(device=device, dtype=target_dtype)
            num_cast += 1

    if num_cast > 0 and debug_mode:
        logger.debug(
            "%s recast %d trainable params after modify_adapter "
            "(compute_dtype=%s, non_lora_trainable_dtype=%s)",
            log_prefix,
            num_cast,
            compute_dtype,
            non_lora_trainable_dtype,
        )


def modify_adapter(peft_model, adapter_name, modify_module_rank={},
                   layer_dict=[], lora_alpha=16, lora_dropout=0.05,
                   init_lora_weights=True, target_modules=None,
                   compute_dtype: Optional[torch.dtype] = None,
                   non_lora_trainable_dtype: Optional[torch.dtype] = None,
                   recast_trainables: bool = False,
                   recast_log_prefix: str = "[HeteroLoRA]",
                   cfg=None):
    """
    Modify LoRA adapter ranks for specific modules.
    
    Args:
        peft_model: PEFT model with LoRA adapters
        adapter_name: Name of the adapter to modify
        modify_module_rank: Dict mapping module names to their target ranks
            e.g., {'q_proj': 8, 'v_proj': 16, 'k_proj': 8}
        layer_dict: List of layer indices to modify (optional)
        lora_alpha: LoRA alpha parameter (can be per-module if modify_module_rank
            contains alpha values as tuples: {'q_proj': (8, 16)})
        lora_dropout: LoRA dropout rate
        init_lora_weights: Whether to initialize LoRA weights
        target_modules: List of target modules to modify (if None, uses all
            modules in modify_module_rank)
    """
    modules_modified = 0
    
    debug_mode = _cfg_debug(cfg) if cfg is not None else False

    for name, module in peft_model.named_modules():
        # Filter by layer_dict if provided
        if layer_dict and not any(['.' + str(layer) + '.' in name 
                                  for layer in layer_dict]):
            continue
        
        # Check if this module should be modified
        for key, r in modify_module_rank.items():

            # Support both simple rank and (rank, alpha) tuple
            if isinstance(r, (list, tuple)) and len(r) == 2:
                rank, alpha = r
            else:
                rank = r
                alpha = lora_alpha if lora_alpha != 0 else rank
            
            # Filter by target_modules if provided
            if target_modules and key not in target_modules:
                continue
            
            if key in name:
                # Handle all known PEFT LoRA layer types
                # Note: We check for the PEFT LoRA wrapper classes, not inner nn.Linear modules
                if isinstance(module, peft.tuners.lora.Linear):
                    module.update_layer(adapter_name, rank, alpha, 
                                      lora_dropout, init_lora_weights)
                    modules_modified += 1
                elif hasattr(peft.tuners.lora, 'Linear8bitLt') and \
                     isinstance(module, peft.tuners.lora.Linear8bitLt):
                    module.update_layer(adapter_name, rank, alpha,
                                      lora_dropout, init_lora_weights)
                    modules_modified += 1
                elif hasattr(peft.tuners.lora, 'Linear4bit') and \
                     isinstance(module, peft.tuners.lora.Linear4bit):
                    # Support 4-bit quantized LoRA (QLoRA)
                    module.update_layer(adapter_name, rank, alpha, 
                                      lora_dropout, init_lora_weights)
                    modules_modified += 1
                # Inner nn.Linear modules (such as lora_A.default and lora_B.default)
                # also match the key pattern but are not LoRA wrappers; they are skipped.
    

    if modules_modified == 0:
        logger.warning(
            f"modify_adapter: No modules modified! "
            f"Check if target_modules patterns match your model's LoRA module names."
        )
    else:
        if debug_mode:
            logger.debug(
                f"modify_adapter: Modified {modules_modified} modules"
            )

    # Optionally recast trainable parameters for heterogeneous quantization support
    if recast_trainables and (compute_dtype is not None or non_lora_trainable_dtype is not None):
        _recast_trainable_params_after_modify(
            peft_model,
            compute_dtype=compute_dtype,
            non_lora_trainable_dtype=non_lora_trainable_dtype,
            log_prefix=recast_log_prefix,
            cfg=cfg,
        )
def distribute_weight_fast(weighted_single_weights, config_local, 
                          max_rank, debug=False):
    """
    Distribute aggregated weights to clients with different LoRA ranks.
    
    This function truncates the aggregated weights (which are at max_rank)
    to each client's specific rank configuration. No SVD or matrix operations.
    
    Uses canonical key format to ensure server and client keys match exactly.
    
    Args:
        weighted_single_weights: Aggregated LoRA weights from server (at max_rank)
        config_local: Dict mapping client IDs to their LoRA configurations
            Format: {
                'Client_0': {'q_proj': 8, 'v_proj': 8, ...},
                'Client_1': {'q_proj': 16, 'v_proj': 16, ...},
                ...
            }
            Patterns can be per-projection ('q_proj') or per-layer 
            ('layers.3.self_attn.q_proj'). Longest matching pattern wins.
        max_rank: Maximum rank used during aggregation. Must match the
            server-side HeteroLoRAAggregator configuration.
        debug: If True, enable detailed logging
    
    Returns:
        Dict mapping client_id -> {canonical_key.rank: tensor}
        e.g., {
            'Client_0': {
                'base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight.8': tensor,
                ...
            },
            ...
        }
    """
    if max_rank is None:
        raise ValueError("max_rank must be provided to distribute weights safely.")
    
    if debug:
        logger.debug(f"distribute_weight_fast: max_rank={max_rank}")
        logger.debug(f"Input weights: {len(weighted_single_weights)} keys")
        for key in list(weighted_single_weights.keys())[:3]:
            if isinstance(weighted_single_weights[key], torch.Tensor):
                logger.info(f"  {key}: {weighted_single_weights[key].shape}")
    
    # Build per-client distributed weights
    distributed = {}
    
    # Process each LoRA weight in the aggregated state dict
    for orig_key in tqdm(weighted_single_weights.keys(), 
                         desc="Distributing weights"):
        if 'lora_A' not in orig_key and 'lora_B' not in orig_key:
            continue
        
        # Get the aggregated weight at max_rank
        aggregated_weight = weighted_single_weights[orig_key]
        if not isinstance(aggregated_weight, torch.Tensor):
            continue
        
        # Verify max_rank matches
        if 'lora_A' in orig_key and 'lora_B' not in orig_key:
            if aggregated_weight.shape[0] != max_rank:
                raise ValueError(
                    f"Mismatched max_rank for {orig_key}: "
                    f"expected {max_rank}, got {aggregated_weight.shape[0]}"
                )
        elif 'lora_B' in orig_key:
            if aggregated_weight.shape[1] != max_rank:
                raise ValueError(
                    f"Mismatched max_rank for {orig_key}: "
                    f"expected {max_rank}, got {aggregated_weight.shape[1]}"
                )
        
        # Get canonical key (no rank suffix, no leading 'model.')
        base_key = _canonical_lora_key(orig_key)
        
        # Distribute to each client based on their rank_config
        for client_id, rank_config in config_local.items():
            if 'Client' not in client_id:
                continue
            
            # Determine rank for this parameter using pattern matching
            rank = _get_rank_for_param(base_key, rank_config)
            
            if rank is None or rank == 0 or rank > max_rank:
                continue
            
            # Truncate to client's rank
            if 'lora_A' in orig_key and 'lora_B' not in orig_key:
                # lora_A: (max_rank, in_features) -> (rank, in_features)
                truncated = aggregated_weight[:rank, :]
            elif 'lora_B' in orig_key:
                # lora_B: (out_features, max_rank) -> (out_features, rank)
                truncated = aggregated_weight[:, :rank]
            else:
                continue
            
            # Build distributed key: canonical base + rank suffix
            dist_key = f"{base_key}.{rank}"
            
            # Store in per-client dictionary
            if client_id not in distributed:
                distributed[client_id] = {}
            distributed[client_id][dist_key] = truncated
    
    if debug:
        for client_id, weights in distributed.items():
            logger.debug(
                f"distribute_weight_fast: {client_id} has "
                f"{len(weights)} distributed weights"
            )
            sample_keys = list(weights.keys())[:3]
            for key in sample_keys:
                if isinstance(weights[key], torch.Tensor):
                    logger.info(f"  {key}: {weights[key].shape}")
    
    return distributed


def load_weight_local(weighted_single_weights, model, client_rank_config, debug=False):
    """
    Load distributed weights into a model with specific rank configuration.
    
    Uses canonical key format to ensure exact matching with keys produced by
    distribute_weight_fast. No fuzzy matching - keys must match exactly.
    
    Args:
        weighted_single_weights: Distributed weights from server for this client.
            Keys should be in canonical format: 'base_key.rank'
            e.g., 'base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight.8'
        model: Model to load weights into
        client_rank_config: Dict mapping module patterns to ranks for this client
            e.g., {'q_proj': 8, 'v_proj': 16, 'layers.3.self_attn.q_proj': 32}.
            Longest matching pattern wins. Required.
        debug: If True, enable detailed logging
    
    Returns:
        Dict of weights to load into the model (keyed by full model parameter name)
    """
    if client_rank_config is None:
        raise ValueError(
            "client_rank_config must be provided for heterogeneous LoRA loading."
        )
    
    if debug:
        logger.debug(f"load_weight_local: client_rank_config={client_rank_config}")
        logger.debug(f"Available distributed keys: {len(weighted_single_weights)}")
        sample_keys = list(weighted_single_weights.keys())[:5]
        for k in sample_keys:
            logger.debug(f"  {k}")
    
    weight_dict = {}
    missing_keys = []
    
    # Determine if we need to strip 'model.' prefix from parameter names
    # This handles AdapterModel wrapping PEFT models:
    # - AdapterModel.named_parameters() yields 'model.base_model...' keys
    # - But load_state_dict forwards to PEFT model which expects 'base_model...' keys
    strip_model_prefix = (
        hasattr(model, 'model') and 
        hasattr(model.model, 'named_parameters')
    )
    
    with torch.no_grad():
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            
            if 'lora_A' not in name and 'lora_B' not in name:
                continue
            
            # Get canonical key (no rank suffix, no leading 'model.')
            base_key = _canonical_lora_key(name)
            
            # Determine rank for this parameter using pattern matching
            rank = _get_rank_for_param(base_key, client_rank_config)
            
            if rank is None:
                if debug:
                    logger.debug(
                        f"No rank config for {name} "
                        f"(canonical: {base_key}), skipping"
                    )
                continue
            
            # Build expected distributed key: canonical base + rank suffix
            dist_key = f"{base_key}.{rank}"
            
            # Exact lookup - no fuzzy matching
            if dist_key not in weighted_single_weights:
                missing_keys.append((name, dist_key, rank))
                continue
            
            weight_tensor = weighted_single_weights[dist_key]

            if not isinstance(weight_tensor, torch.Tensor):
                weight_tensor = torch.tensor(weight_tensor)

            # Move to the param's device and dtype (the server and client models may
            # use different dtypes, e.g. bfloat16 vs float32).
            weight_tensor = weight_tensor.to(device=param.data.device, dtype=param.data.dtype)

            # Shape adaptation:
            # - distributed tensor has shape (rank, d) or (d, rank)
            # - local param may have a different rank dimension
            t_shape = weight_tensor.shape
            p_shape = param.data.shape

            if t_shape == p_shape:
                # Perfect match, nothing to do
                new_tensor = weight_tensor
            elif 'lora_A' in name:
                # Rank dimension is the first dimension
                if len(t_shape) != 2 or len(p_shape) != 2:
                    raise ValueError(
                        f"Unexpected tensor shape for {name}: "
                        f"distributed={t_shape}, param={p_shape}"
                    )

                if t_shape[1] != p_shape[1]:
                    raise ValueError(
                        f"Mismatched feature dimension for {name}: "
                        f"distributed={t_shape}, param={p_shape}"
                    )

                dist_r, local_r = t_shape[0], p_shape[0]

                if dist_r == local_r:
                    new_tensor = weight_tensor
                elif dist_r < local_r:
                    # Copy into leading rows, zero the tail
                    new_tensor = param.data.clone()
                    new_tensor[:dist_r, :].copy_(weight_tensor)
                    new_tensor[dist_r:, :].zero_()
                    if debug:
                        logger.debug(
                            f"Padded lora_A {name}: "
                            f"dist_rank={dist_r} < local_rank={local_r}"
                        )
                else:
                    # dist_r > local_r: truncate extra rows
                    new_tensor = weight_tensor[:local_r, :].clone()
                    logger.warning(
                        f"Truncating lora_A {name}: "
                        f"dist_rank={dist_r} > local_rank={local_r}"
                    )

            elif 'lora_B' in name:
                # Rank dimension is the second dimension
                if len(t_shape) != 2 or len(p_shape) != 2:
                    raise ValueError(
                        f"Unexpected tensor shape for {name}: "
                        f"distributed={t_shape}, param={p_shape}"
                    )

                if t_shape[0] != p_shape[0]:
                    raise ValueError(
                        f"Mismatched feature dimension for {name}: "
                        f"distributed={t_shape}, param={p_shape}"
                    )

                dist_r, local_r = t_shape[1], p_shape[1]

                if dist_r == local_r:
                    new_tensor = weight_tensor
                elif dist_r < local_r:
                    # Copy into leading columns, zero the tail
                    new_tensor = param.data.clone()
                    new_tensor[:, :dist_r].copy_(weight_tensor)
                    new_tensor[:, dist_r:].zero_()
                    if debug:
                        logger.debug(
                            f"Padded lora_B {name}: "
                            f"dist_rank={dist_r} < local_rank={local_r}"
                        )
                else:
                    # dist_r > local_r: truncate extra columns
                    new_tensor = weight_tensor[:, :local_r].clone()
                    logger.warning(
                        f"Truncating lora_B {name}: "
                        f"dist_rank={dist_r} > local_rank={local_r}"
                    )
            else:
                # Should not happen, but fall back to direct assignment
                new_tensor = weight_tensor

            # Store with appropriate key for state_dict loading
            # If model is AdapterModel wrapper, strip 'model.' prefix so keys match
            # what the inner PEFT model expects in load_state_dict
            store_key = name
            if strip_model_prefix and name.startswith('model.'):
                store_key = name[len('model.'):]
            weight_dict[store_key] = new_tensor
    
    # Report missing keys if any
    if missing_keys:
        # Show a sample of available keys for debugging
        available_sample = list(weighted_single_weights.keys())[:10]
        logger.error(
            f"Missing {len(missing_keys)} distributed LoRA weights. "
            f"First few missing: {missing_keys[:5]}. "
            f"Sample available keys: {available_sample}"
        )
        # Raise error for first missing key to help debugging
        first_missing = missing_keys[0]
        raise KeyError(
            f"Missing distributed LoRA weight for '{first_missing[1]}'. "
            f"Model param: {first_missing[0]}, Expected rank: {first_missing[2]}. "
            f"Available keys (sample): {available_sample}"
        )
    
    if debug:
        logger.info(f"load_weight_local: Loaded {len(weight_dict)} LoRA parameters")
        for name, tensor in list(weight_dict.items())[:5]:
            logger.info(f"  {name}: {tensor.shape}")
    
    return weight_dict


def compute_lora_size(model, max_rank: int, cfg=None) -> tuple:
    """
    Compute LoRA parameter sizes for FAH-QLoRA communication time estimation.
    
    LoRA layers consist of:
        - lora_A: shape (rank, in_features)
        - lora_B: shape (out_features, rank)
    
    For the communication time model (equations 13-14 of the FAH-QLoRA paper):
        - L0_bytes: Size of all LoRA parameters of `model` at its current rank
          (the download size when `model` is the max-rank global model)
        - L(r) = unit_lora_bytes * r: Size at rank r (client -> server upload)
    
    Args:
        model: The model with LoRA adapters (can be PEFT model or AdapterModel wrapper)
        max_rank: Maximum LoRA rank used in the system
        
    Returns:
        Tuple of (L0_bytes, unit_lora_bytes):
            - L0_bytes: Total size of LoRA params at current rank (bytes)
            - unit_lora_bytes: Size per unit rank (bytes), so L(r) ≈ unit_lora_bytes * r
    """
    total_elements = 0
    num_lora_A = 0
    num_lora_B = 0
    detected_rank = 0
    
    # Try to access model parameters (handle AdapterModel wrapper)
    params_iter = None
    if hasattr(model, 'named_parameters'):
        params_iter = model.named_parameters()
    elif hasattr(model, 'model') and hasattr(model.model, 'named_parameters'):
        params_iter = model.model.named_parameters()
    
    if params_iter is None:
        logger.warning("[FAH] Could not access model parameters, using defaults")
        return 50 * 1e6, 1.5 * 1e6  # 50MB total, 1.5MB per rank
    
    # Count LoRA parameters
    for name, param in params_iter:
        if 'lora_A' in name:
            total_elements += param.numel()
            num_lora_A += 1
            # Detect rank from lora_A shape (rank, in_features)
            if param.dim() >= 2:
                detected_rank = max(detected_rank, param.shape[0])
        elif 'lora_B' in name:
            total_elements += param.numel()
            num_lora_B += 1
            # Detect rank from lora_B shape (out_features, rank)
            if param.dim() >= 2:
                detected_rank = max(detected_rank, param.shape[1])
    
    if total_elements == 0:
        logger.warning("[FAH] No LoRA parameters found, using defaults")
        return 50 * 1e6, 1.5 * 1e6
    
    # Bytes per element: assumes 16-bit (float16/bfloat16) LoRA weights
    bytes_per_element = 2
    
    # L0 at current detected rank
    L0_bytes = total_elements * bytes_per_element
    
    # Estimate unit size per rank
    # LoRA size scales linearly with rank: L(r) ∝ r
    # If detected_rank > 0, we can compute the actual per-rank contribution
    effective_rank = detected_rank if detected_rank > 0 else max_rank
    if effective_rank > 0:
        unit_lora_bytes = L0_bytes / effective_rank
    else:
        unit_lora_bytes = L0_bytes
    
    # If model uses different rank than max_rank, scale L0 to max_rank
    if detected_rank > 0 and detected_rank != max_rank:
        L0_bytes = unit_lora_bytes * max_rank
    
    debug_mode = _cfg_debug(cfg) if cfg is not None else False

    if debug_mode:
        logger.debug(
            f"[FAH] Computed LoRA sizes: L0={L0_bytes/1e6:.2f}MB (at rank {max_rank}), "
            f"per_rank={unit_lora_bytes/1e3:.2f}KB, "
            f"detected_rank={detected_rank}, modules={num_lora_A}+{num_lora_B}"
        )
    
    return L0_bytes, unit_lora_bytes


# =============================================================================
# HetLoRA: Rank Self-Pruning Utilities
# =============================================================================

def iter_lora_pairs(model):
    """
    Iterate over all LoRA A/B parameter pairs in the model.
    
    Yields:
        Tuple of (lora_A_param, lora_B_param, base_name) for each LoRA module.
        base_name is the module path without the lora_A/lora_B suffix.
        
    Example:
        For a parameter named 'base_model.model.layers.0.self_attn.q_proj.lora_A.default.weight'
        and 'base_model.model.layers.0.self_attn.q_proj.lora_B.default.weight',
        yields (A_param, B_param, 'base_model.model.layers.0.self_attn.q_proj')
    """
    # Collect all LoRA parameters
    lora_params = {}
    
    # Get the inner model if wrapped
    nn_model = model
    if hasattr(nn_model, 'model') and hasattr(getattr(nn_model, 'model'), 'named_parameters'):
        nn_model = nn_model.model
    
    for name, param in nn_model.named_parameters():
        if 'lora_A' in name:
            # Extract base name by removing lora_A and everything after
            # e.g., 'layers.0.q_proj.lora_A.default.weight' -> 'layers.0.q_proj'
            base = name.split('lora_A')[0].rstrip('.')
            if base not in lora_params:
                lora_params[base] = {'A': None, 'B': None}
            lora_params[base]['A'] = param
        elif 'lora_B' in name:
            base = name.split('lora_B')[0].rstrip('.')
            if base not in lora_params:
                lora_params[base] = {'A': None, 'B': None}
            lora_params[base]['B'] = param
    
    # Yield complete pairs
    for base_name, params in lora_params.items():
        if params['A'] is not None and params['B'] is not None:
            yield params['A'], params['B'], base_name


def tail_penalty(model, decay: float, current_rank: Optional[int] = None) -> torch.Tensor:
    """
    Compute the tail-rank regularizer penalty for HetLoRA self-pruning.

    The penalty pushes the tail ranks (rows/columns beyond decay*r_eff) toward zero,
    encouraging rank reduction during training.

    For each LoRA pair (A, B):
        r_eff = current effective (logical) rank for this client (<= physical max rank)
        tail_start = floor(r_eff * decay)
        penalty += ||B[:, tail_start:r_eff]||_F * ||A[tail_start:r_eff, :]||_F

    Notes:
        - If `current_rank` is None, we also look for `model.hetlora_current_rank`.
          This allows trainers to call `tail_penalty(model, decay)` without threading
          the rank through every call site.
        - We intentionally slice within [0:r_eff] even if LoRA tensors are physically
          allocated at max_rank. This makes pruning logic follow the logical rank.

    Args:
        model: Model with LoRA adapters
        decay: Tail fraction (e.g., 0.9 means last 10% of ranks are "tail")
        current_rank: Logical active rank to use for penalty (optional)

    Returns:
        Scalar tensor containing the total tail penalty
    """
    if current_rank is None:
        current_rank = getattr(model, "hetlora_current_rank", None)

    # Choose device robustly
    try:
        device = next(model.parameters()).device
    except Exception:
        device = torch.device("cpu")

    penalty = torch.tensor(0.0, device=device)

    for A, B, _ in iter_lora_pairs(model):
        # A: (r, in_features), B: (out_features, r)
        r_phys = A.shape[0]

        # Effective rank (logical)
        if current_rank is not None:
            try:
                r_eff = int(current_rank)
                r_eff = min(r_eff, r_phys, B.shape[1])
            except Exception:
                r_eff = r_phys
        else:
            r_eff = r_phys

        if r_eff <= 0:
            continue

        tail_start = int(r_eff * decay)
        if tail_start >= r_eff:
            continue

        # Slice within the effective rank
        A_eff = A[:r_eff, :]
        B_eff = B[:, :r_eff]

        A_tail_norm = torch.norm(A_eff[tail_start:, :], p='fro')
        B_tail_norm = torch.norm(B_eff[:, tail_start:], p='fro')

        # Product of norms (approximates ||B_tail @ A_tail||_F)
        penalty = penalty + A_tail_norm * B_tail_norm

    return penalty


def tail_score(model, decay: float, current_rank: Optional[int] = None) -> float:
    """
    Compute the tail importance score for HetLoRA prune decision.

    This score measures how important the tail ranks are. A decreasing score
    after training indicates the model has learned to push information into
    the leading ranks, making pruning safe.

    The score is the same as tail_penalty but returned as a Python float
    for comparison purposes.

    Args:
        model: Model with LoRA adapters
        decay: Tail fraction (e.g., 0.9 means last 10% of ranks are "tail")
        current_rank: Logical active rank to use for score (optional)

    Returns:
        Float representing the total tail score
    """
    return tail_penalty(model, decay, current_rank=current_rank).item()


def truncate_client_lora_to_rank(
    state_dict: dict,
    new_rank: int,
    current_rank: int = None,
    debug: bool = False
) -> dict:
    """
    Truncate LoRA weights in a state dict to a new (smaller) rank.
    
    Used after HetLoRA prune decision to reduce the outgoing LoRA rank.
    
    Args:
        state_dict: State dict containing LoRA parameters
        new_rank: Target rank to truncate to
        current_rank: Current rank (auto-detected if None)
        debug: Enable debug logging
        
    Returns:
        New state dict with truncated LoRA weights
    """
    truncated = {}
    n_truncated = 0
    
    for key, value in state_dict.items():
        if not isinstance(value, torch.Tensor):
            truncated[key] = value
            continue
        
        if 'lora_A' in key and 'lora_B' not in key:
            # lora_A: (r, in_features) -> truncate first dimension
            old_rank = value.shape[0]
            if old_rank > new_rank:
                truncated[key] = value[:new_rank, :].clone()
                n_truncated += 1
                if debug:
                    logger.debug(
                        f"[HetLoRA] Truncated lora_A {key}: {old_rank} -> {new_rank}"
                    )
            else:
                truncated[key] = value
        elif 'lora_B' in key:
            # lora_B: (out_features, r) -> truncate second dimension
            old_rank = value.shape[1]
            if old_rank > new_rank:
                truncated[key] = value[:, :new_rank].clone()
                n_truncated += 1
                if debug:
                    logger.debug(
                        f"[HetLoRA] Truncated lora_B {key}: {old_rank} -> {new_rank}"
                    )
            else:
                truncated[key] = value
        else:
            truncated[key] = value
    
    if n_truncated > 0:
        logger.info(
            f"[HetLoRA] Truncated {n_truncated} LoRA parameters to rank {new_rank}"
        )
    
    return truncated


def get_current_lora_rank_from_state_dict(state_dict: dict) -> int:
    """
    Detect the current LoRA rank from a state dict.
    
    Args:
        state_dict: Dict of parameters
        
    Returns:
        Current LoRA rank (0 if no LoRA params found)
    """
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor) and 'lora_A' in key and 'lora_B' not in key:
            return value.shape[0]
    return 0

# ===========================================================================
# Apply a rank-annotated LoRA download to the client model. The aggregated global
# adapter arrives with keys "base.rank"; they are canonicalized to the model's
# state_dict keys (load_state_dict(strict=False) would otherwise ignore them). The
# canonicalizer infers each key's rank from the key itself, so both per-module
# (HetLoRA) and per-layer (AdaS grouped/pruned) rank formats are handled, and it
# asserts strict downloaded-key consumption.
# ===========================================================================
def _adapt_lora_rank_shape(tv, pv, is_lora_A):
    """Fit downloaded LoRA tensor `tv` to model-param shape `pv` (pad/truncate rank dim)."""
    tv = tv.to(device=pv.device, dtype=pv.dtype)
    if tuple(tv.shape) == tuple(pv.shape):
        return tv
    if is_lora_A:                      # lora_A: (rank, in_features) -> rank is dim 0
        if tv.shape[1] != pv.shape[1]:
            raise ValueError(f"lora_A feature-dim mismatch: {tuple(tv.shape)} vs {tuple(pv.shape)}")
        dr, lr = tv.shape[0], pv.shape[0]
        if dr < lr:
            out = pv.detach().clone(); out[:dr, :] = tv; out[dr:, :].zero_(); return out
        return tv[:lr, :].clone()
    else:                              # lora_B: (out_features, rank) -> rank is dim 1
        if tv.shape[0] != pv.shape[0]:
            raise ValueError(f"lora_B feature-dim mismatch: {tuple(tv.shape)} vs {tuple(pv.shape)}")
        dr, lr = tv.shape[1], pv.shape[1]
        if dr < lr:
            out = pv.detach().clone(); out[:, :dr] = tv; out[:, dr:].zero_(); return out
        return tv[:, :lr].clone()


def apply_distributed_lora_download(content, model, strict=True, debug=False,
                                    is_partial=False):
    """Re-key a distributed-format LoRA download to the model's trainable state_dict keys.

    Distributed keys look like 'base_model...lora_A.default.weight.<rank>'. For each such
    key we strip the rank suffix (and any 'model.' prefix) via _canonical_lora_key, match it
    to the model's trainable LoRA param with the same canonical key, and pad/truncate the
    rank dimension to fit. Non-LoRA keys and non-distributed payloads pass through unchanged.

    Returns (rekeyed_content, n_consumed, n_model_lora). With strict=True, raises KeyError
    if any distributed LoRA key cannot be matched (such a key would otherwise be ignored).
    """
    has_dist = any(
        ('lora_A' in k or 'lora_B' in k) and isinstance(k, str)
        and '.' in k and k.rsplit('.', 1)[-1].isdigit()
        for k in content
    )
    if not has_dist:
        return content, 0, 0

    sd = model.state_dict()
    model_lora = {}
    for name, t in sd.items():
        if ('lora_A' in name or 'lora_B' in name) and isinstance(t, torch.Tensor):
            model_lora[_canonical_lora_key(name)] = name

    out, consumed, unmapped = {}, set(), []
    for k, v in content.items():
        if 'lora_A' in k or 'lora_B' in k:
            canon = _canonical_lora_key(k)
            tgt = model_lora.get(canon)
            tv = v if isinstance(v, torch.Tensor) else torch.as_tensor(v)
            if tgt is None or not isinstance(tv, torch.Tensor):
                unmapped.append(k); continue
            # Safety: a PARTIAL (sparse/Stage-2) downlink sends a SUBSET of components
            # compacted in download order; the correct placement scatters them to their
            # global positions (via download_indices), which this dense canonicalizer does
            # NOT do. Left-compacting a subset here would mis-place components and zero
            # other survivors. Dense full downlink (rank matches, or unused tail) is fine.
            # Refuse rather than mis-place components: partial+resize needs an index-based scatter.
            if is_partial and tuple(tv.shape) != tuple(sd[tgt].shape):
                raise NotImplementedError(
                    "[HeteroLoRA] a partial (Stage-2 sparse) LoRA downlink that needs a rank "
                    "resize requires an index-based scatter (download_indices), which this "
                    "canonicalizer does not perform. Use a full downlink (e.g. AdaS "
                    "stage2.enabled=False). "
                    f"key={k} download_shape={tuple(tv.shape)} model_shape={tuple(sd[tgt].shape)}")
            out[tgt] = _adapt_lora_rank_shape(tv, sd[tgt], 'lora_A' in k)
            consumed.add(canon)
        else:
            out[k] = v

    if unmapped and strict:
        raise KeyError(
            f"[HeteroLoRA] {len(unmapped)} distributed LoRA download key(s) could not be "
            f"matched to trainable model params -- the aggregated global adapter would be "
            f"dropped. First unmapped: {unmapped[:2]}; model canonical sample: {list(model_lora)[:2]}")
    if debug:
        logger.info("apply_distributed_lora_download: consumed %d/%d model LoRA params (%d unmapped)",
                    len(consumed), len(model_lora), len(unmapped))
    return out, len(consumed), len(model_lora)
