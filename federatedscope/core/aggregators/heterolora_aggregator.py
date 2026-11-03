"""
HeteroLoRA Aggregator for handling LoRA adapters with varying ranks.

This aggregator implements the zero-padding and truncation strategies
for aggregating LoRA weights with heterogeneous ranks, as in HetLoRA
(Cho et al., 2024).
"""
import os
import torch
from federatedscope.core.aggregators import Aggregator
from federatedscope.core.auxiliaries.utils import param2tensor
import logging

import federatedscope.contrib.common as fs_common

logger = logging.getLogger(__name__)


class HeteroLoRAAggregator(Aggregator):
    """
    Aggregator for heterogeneous LoRA adapters with varying ranks.
    
    This aggregator handles LoRA weights with different ranks by:
    1. Zero-padding smaller ranks to a maximum rank
    2. Truncating larger ranks to the maximum rank
    """
    
    def __init__(self, model=None, device='cpu', config=None):
        super().__init__()
        if device is None:
            device = 'cpu'
        if not isinstance(device, torch.device):
            device = torch.device(device)
        self.model = model
        self.device = device
        self.cfg = config
        
        # Get max rank from config or use default
        self.max_rank = fs_common.get_effective_max_rank(self.cfg)
        
    def aggregate(self, agg_info):
        """
        Aggregate LoRA weights with heterogeneous ranks.
        
        Arguments:
            agg_info (dict): the feedbacks from clients containing:
                - client_feedback: list of (sample_size, model_params) tuples
        
        Returns:
            dict: the aggregated results with zero-padded/truncated LoRA weights
        """
        models = agg_info["client_feedback"]
        if not models:
            logger.warning("No models to aggregate, returning empty dict")
            return {}
        
        # Runtime validation: ensure max_rank matches config (config might be updated after init)
        config_max_rank = fs_common.get_effective_max_rank(self.cfg)
        if config_max_rank is not None and config_max_rank != self.max_rank:
            logger.warning(
                f"max_rank mismatch: aggregator has {self.max_rank}, "
                f"config has {config_max_rank}. Updating aggregator to match config."
            )
            self.max_rank = config_max_rank
        
        # Verify max_rank is set correctly
        if self.max_rank is None or self.max_rank <= 0:
            logger.error(
                f"Invalid max_rank={self.max_rank}. "
                f"Using default 64. This may cause shape mismatches!"
            )
            self.max_rank = 64
        
        result = self._hetero_lora_weighted_avg(models)
        
        if not result:
            logger.warning("[HeteroLoRA] Aggregation returned empty result")
            return result
        
        # Final verification: ensure all returned LoRA parameters have max_rank shape
        # This is a second safety check after _hetero_lora_weighted_avg's verification
        shape_errors = []
        shapes_fixed_final = 0
        for key, tensor in list(result.items()):  # Use list() to allow modification during iteration
            if not isinstance(tensor, torch.Tensor):
                continue
                
            if 'lora_A' in key and 'lora_B' not in key:
                if tensor.shape[0] != self.max_rank:
                    shape_errors.append(f"{key}: rank {tensor.shape[0]} != {self.max_rank}")
                    # Fix it by zero-padding to max_rank
                    if tensor.shape[0] < self.max_rank:
                        padded = torch.zeros(
                            self.max_rank, tensor.shape[1],
                            dtype=tensor.dtype,
                            device=tensor.device
                        )
                        padded[:tensor.shape[0], :] = tensor
                        result[key] = padded
                        shapes_fixed_final += 1
                    else:
                        result[key] = tensor[:self.max_rank, :].clone()
                        shapes_fixed_final += 1
            elif 'lora_B' in key:
                if tensor.shape[1] != self.max_rank:
                    shape_errors.append(f"{key}: rank {tensor.shape[1]} != {self.max_rank}")
                    # Fix it by zero-padding to max_rank
                    if tensor.shape[1] < self.max_rank:
                        padded = torch.zeros(
                            tensor.shape[0], self.max_rank,
                            dtype=tensor.dtype,
                            device=tensor.device
                        )
                        padded[:, :tensor.shape[1]] = tensor
                        result[key] = padded
                        shapes_fixed_final += 1
                    else:
                        result[key] = tensor[:, :self.max_rank].clone()
                        shapes_fixed_final += 1
        
        if shape_errors:
            logger.error(
                f"Found {len(shape_errors)} shape mismatches in final check. "
                f"Fixed {shapes_fixed_final}. First few errors: {shape_errors[:3]}"
            )
        
        return result
    
    def _hetero_lora_weighted_avg(self, models):
        """
        Calculate weighted average of LoRA models with heterogeneous ranks.
        
        The server first zero-pads all received LoRA modules to match the 
        maximum rank, then performs simple weighted averaging over the modules.
        """
        federate_cfg = getattr(self.cfg, 'federate', None) if self.cfg else None
        ignore_weight = getattr(federate_cfg, 'ignore_weight', False) \
            if federate_cfg else False
        
        training_set_size = sum(sample_size for sample_size, _ in models)
        uniform_weight = 1.0 / len(models)
        
        # Track which keys are LoRA-related (only track lora_A to avoid duplicates)
        _, reference_model = models[0]
        lora_pairs = {}
        for key in reference_model.keys():
            if 'lora_A' in key and 'lora_B' not in key:
                # Determine the corresponding pair
                b_key = key.replace('lora_A', 'lora_B')
                if b_key in reference_model:
                    lora_pairs[key] = b_key
        
        if not lora_pairs:
            logger.warning("No LoRA parameters found for aggregation.")
            return {}
        
        # Initialize aggregated LoRA tensors with max rank
        aggregated_lora = {}
        valid_pairs = []
        for a_key, b_key in lora_pairs.items():
            a_tensor = param2tensor(reference_model[a_key])
            b_tensor = param2tensor(reference_model[b_key])
            
            if not isinstance(a_tensor, torch.Tensor) or \
               not isinstance(b_tensor, torch.Tensor):
                logger.warning(
                    "Skipping LoRA pair (%s, %s) since reference values are not tensors.",
                    a_key, b_key
                )
                continue
            
            aggregated_lora[a_key] = torch.zeros(
                self.max_rank, a_tensor.shape[1],
                dtype=a_tensor.dtype,
                device=self.device
            )
            aggregated_lora[b_key] = torch.zeros(
                b_tensor.shape[0], self.max_rank,
                dtype=b_tensor.dtype,
                device=self.device
            )
            valid_pairs.append((a_key, b_key))
        
        if not valid_pairs:
            logger.warning("No valid LoRA tensor pairs available for aggregation.")
            return {}
        
        # Debug logging
        debug_mode = bool(getattr(self.cfg, 'debug', False))
        if debug_mode:
            logger.debug(f"Aggregating {len(valid_pairs)} LoRA pairs")
            logger.debug(f"Max rank: {self.max_rank}")
            logger.debug(f"LoRA keys and shapes after initialization:")
            for key in list(aggregated_lora.keys())[:5]:  # Show first 5
                logger.debug(f"  {key}: {aggregated_lora[key].shape}")
        
        # Aggregate each client's model
        for local_sample_size, local_model in models:
            if ignore_weight or training_set_size == 0:
                weight = uniform_weight
            else:
                weight = local_sample_size / training_set_size
            
            for a_key, b_key in valid_pairs:
                local_param = local_model.get(a_key)
                pair_param = local_model.get(b_key)
                
                if local_param is None or pair_param is None:
                    continue
                
                local_param = param2tensor(local_param)
                pair_param = param2tensor(pair_param)
                
                if not isinstance(local_param, torch.Tensor) or \
                   not isinstance(pair_param, torch.Tensor):
                    continue
                
                local_param = local_param.to(self.device)
                pair_param = pair_param.to(self.device)
                
                rank_A = local_param.shape[0]
                rank_B = pair_param.shape[1]
                rank = min(rank_A, rank_B)
                if rank == 0:
                    continue
                
                capped_rank = min(rank, self.max_rank)
                aggregated_lora[a_key][:capped_rank, :] += (
                    local_param[:capped_rank, :] * weight
                )
                aggregated_lora[b_key][:, :capped_rank] += (
                    pair_param[:, :capped_rank] * weight
                )
        
        # Verify all aggregated LoRA parameters have max_rank shape
        # This should never trigger since we initialize with max_rank, but it's a safety net
        shapes_fixed = 0
        for key, tensor in aggregated_lora.items():
            if 'lora_A' in key and 'lora_B' not in key:
                # lora_A should have shape [max_rank, in_features]
                if tensor.shape[0] != self.max_rank:
                    # Fix it by zero-padding or truncating
                    if tensor.shape[0] < self.max_rank:
                        # Zero-pad to max_rank
                        padded = torch.zeros(
                            self.max_rank, tensor.shape[1],
                            dtype=tensor.dtype,
                            device=tensor.device
                        )
                        padded[:tensor.shape[0], :] = tensor
                        aggregated_lora[key] = padded
                        shapes_fixed += 1
                    else:
                        # Truncate to max_rank
                        aggregated_lora[key] = tensor[:self.max_rank, :].clone()
                        shapes_fixed += 1
            elif 'lora_B' in key:
                # lora_B should have shape [out_features, max_rank]
                if tensor.shape[1] != self.max_rank:
                    # Fix it by zero-padding or truncating
                    if tensor.shape[1] < self.max_rank:
                        # Zero-pad to max_rank
                        padded = torch.zeros(
                            tensor.shape[0], self.max_rank,
                            dtype=tensor.dtype,
                            device=tensor.device
                        )
                        padded[:, :tensor.shape[1]] = tensor
                        aggregated_lora[key] = padded
                        shapes_fixed += 1
                    else:
                        # Truncate to max_rank
                        aggregated_lora[key] = tensor[:, :self.max_rank].clone()
                        shapes_fixed += 1
        
        if shapes_fixed > 0:
            logger.warning(
                f"Fixed {shapes_fixed} shape mismatches in aggregated LoRA parameters. "
                f"This should not happen if initialization is correct."
            )
        
        return aggregated_lora
    
    def update(self, model_parameters):
        """
        Update the model with aggregated parameters.
        
        Arguments:
            model_parameters (dict): PyTorch Module object's state_dict.
        """
        self.model.load_state_dict(model_parameters, strict=False)
    
    def save_model(self, path, cur_round=-1):
        """Save the aggregated model."""
        assert self.model is not None
        
        # Handle directory paths: if path is a directory, construct a filename
        if os.path.isdir(path) or (not os.path.splitext(path)[1] and path.endswith('/')):
            # If it's a directory or ends with '/', create a filename
            os.makedirs(path, exist_ok=True)
            filename = f'model_round_{cur_round}.pt' if cur_round >= 0 else 'model.pt'
            path = os.path.join(path, filename)
        else:
            # Ensure parent directory exists
            os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        
        ckpt = {'cur_round': cur_round, 'model': self.model.state_dict()}
        torch.save(ckpt, path)
    
    def load_model(self, path):
        """Load a saved model."""
        assert self.model is not None
        if os.path.exists(path):
            ckpt = torch.load(path, map_location=self.device)
            self.model.load_state_dict(ckpt['model'])
            return ckpt['cur_round']
        else:
            raise ValueError("The file {} does NOT exist".format(path))

