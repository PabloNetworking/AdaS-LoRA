"""
HetLoRA Aggregator with Sparsity-Weighted Aggregation.

This aggregator implements the HetLoRA aggregation strategy:
- Zero-padding of client LoRA weights to max_rank, as in HeteroLoRAAggregator
- Sparsity-weighted aggregation based on Frobenius norm of effective LoRA updates

The aggregation weights are computed as:
    s_k = ||ΔW_k||_F = ||B_k @ A_k||_F  (for each client k)
    p_k = s_k / sum_j(s_j)  (normalized sparsity weight)

This differs from standard FedAvg which uses sample_size weights.
"""
import os
import torch
import logging
from federatedscope.core.aggregators.heterolora_aggregator import HeteroLoRAAggregator
from federatedscope.core.auxiliaries.utils import param2tensor

import federatedscope.contrib.common as fs_common

logger = logging.getLogger(__name__)


class HetLoRAAggregator(HeteroLoRAAggregator):
    """
    Aggregator for HetLoRA with sparsity-weighted aggregation.
    
    Extends HeteroLoRAAggregator to use aggregation weights based on
    the Frobenius norm of each client's effective LoRA update.
    """
    
    def __init__(self, model=None, device='cpu', config=None):
        super().__init__(model=model, device=device, config=config)

        self._cfg = getattr(self, '_cfg', getattr(self, 'cfg', config))

        # HetLoRA-specific defaults
        self.hetlora_cfg = fs_common.get_hetlora_cfg(self._cfg)
        self.aggregation_mode = 'sample_size'  # Default to standard FedAvg
        self.epsilon = 1e-8

        # Top-level debug toggle (config.debug: bool)
        self.debug_mode = bool(getattr(self._cfg, 'debug', False))

        # Align the aggregator rank with the effective adapter root selected by
        # federatedscope.contrib.common.config_resolver.
        effective_max_rank = fs_common.get_effective_max_rank(self._cfg)
        if effective_max_rank is not None and effective_max_rank != self.max_rank:
            logger.info(
                "Updating max_rank from %s to effective max_rank=%s during init",
                self.max_rank,
                effective_max_rank,
            )
            self.max_rank = effective_max_rank

        if self.hetlora_cfg is not None and hasattr(self.hetlora_cfg, 'aggregation'):
            self.aggregation_mode = getattr(
                self.hetlora_cfg.aggregation, 'mode', 'sample_size'
            )
            self.epsilon = getattr(
                self.hetlora_cfg.aggregation, 'epsilon', 1e-8
            )

        logger.info(
            "HetLoRA aggregator initialized with mode='%s', max_rank=%s, hetlora_cfg_found=%s",
            self.aggregation_mode,
            self.max_rank,
            self.hetlora_cfg is not None,
        )
    
    def aggregate(self, agg_info):
        """
        Aggregate LoRA weights with sparsity-weighted or sample-size weights.
        
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
        config_max_rank = fs_common.get_effective_max_rank(self._cfg)
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
        
        # Choose aggregation method based on mode
        if self.aggregation_mode == 'sparsity_weighted':
            result = self._sparsity_weighted_aggregate(models)
        else:
            # Fall back to parent's sample-size weighted aggregation
            result = self._hetero_lora_weighted_avg(models)
        
        if not result:
            logger.warning("Aggregation returned empty result")
            return result
        
        # Final shape check (the same check HeteroLoRAAggregator.aggregate runs)
        result = self._verify_and_fix_shapes(result)

        # Average the non-LoRA trainable params (GLUE task head: classifier + pooler)
        # with sample-size weights, as in FedIT/FedAvg.
        result.update(self._average_non_lora_trainable(models))

        return result

    def _average_non_lora_trainable(self, models):
        """Sample-size-weighted average of the non-LoRA trainable params (e.g. the GLUE
        classification head + pooler), aggregated the same way as in FedAvg/FedIT."""
        total = float(sum(ss for ss, _ in models))
        if total <= 0:
            total = float(len(models))
        _, ref = models[0]
        keys = [k for k in ref.keys()
                if 'lora_A' not in k and 'lora_B' not in k]
        out = {}
        for k in keys:
            acc = None
            for ss, mp in models:
                v = mp.get(k)
                if v is None:
                    continue
                t = param2tensor(v)
                if not isinstance(t, torch.Tensor):
                    continue
                t = t.to(self.device).float()
                w = float(ss) / total
                acc = t * w if acc is None else acc + t * w
            if acc is not None:
                out[k] = acc
        return out

    def _compute_client_sparsity_weights(self, models):
        """
        Compute sparsity weights for each client based on ||B @ A||_F.
        
        Uses the efficient trace trick to avoid materializing full matrices.
        
        Args:
            models: List of (sample_size, model_params) tuples
            
        Returns:
            List of normalized sparsity weights (sum to 1)
        """
        sparsity_norms = []
        
        for _, local_model in models:
            # Compute ||BA||_F^2 for all LoRA pairs in this client
            total_norm_sq = 0.0
            
            # Find all LoRA A/B pairs
            lora_pairs = {}
            for key in local_model.keys():
                if 'lora_A' in key and 'lora_B' not in key:
                    b_key = key.replace('lora_A', 'lora_B')
                    if b_key in local_model:
                        lora_pairs[key] = b_key
            
            for a_key, b_key in lora_pairs.items():
                A = param2tensor(local_model[a_key])
                B = param2tensor(local_model[b_key])
                
                if not isinstance(A, torch.Tensor) or not isinstance(B, torch.Tensor):
                    continue
                
                A = A.to(self.device)
                B = B.to(self.device)
                
                # Efficient computation: ||BA||_F^2 = trace((B^T B)(A A^T))
                # B^T B: (r, out) @ (out, r) -> (r, r)
                # A A^T: (r, in) @ (in, r) -> (r, r)
                BtB = B.T @ B
                AAt = A @ A.T
                # Element-wise product and sum = trace of product
                norm_sq = (BtB * AAt).sum().item()
                total_norm_sq += norm_sq
            
            sparsity_norms.append(total_norm_sq ** 0.5)
        
        # Normalize to get weights summing to 1
        total_norm = sum(sparsity_norms) + self.epsilon
        weights = [s / total_norm for s in sparsity_norms]
        
        # Log sparsity weights for debugging
        if self.debug_mode:
            logger.debug(
                "Sparsity norms: %s",
                [f"{s:.4f}" for s in sparsity_norms],
            )
            logger.debug(
                "Sparsity weights: %s",
                [f"{w:.4f}" for w in weights],
            )
        
        return weights
    
    def _sparsity_weighted_aggregate(self, models):
        """
        Aggregate LoRA weights using sparsity-weighted averaging.
        
        This implements the HetLoRA paper's aggregation strategy where
        weights are based on ||B @ A||_F rather than sample size.
        """
        # Compute sparsity weights
        sparsity_weights = self._compute_client_sparsity_weights(models)
        
        # Track which keys are LoRA-related
        _, reference_model = models[0]
        lora_pairs = {}
        for key in reference_model.keys():
            if 'lora_A' in key and 'lora_B' not in key:
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
            logger.warning("No valid LoRA tensor pairs for aggregation.")
            return {}
        
        # Aggregate using sparsity weights
        for idx, (_, local_model) in enumerate(models):
            weight = sparsity_weights[idx]
            
            for a_key, b_key in valid_pairs:
                local_A = local_model.get(a_key)
                local_B = local_model.get(b_key)
                
                if local_A is None or local_B is None:
                    continue
                
                local_A = param2tensor(local_A)
                local_B = param2tensor(local_B)
                
                if not isinstance(local_A, torch.Tensor) or \
                   not isinstance(local_B, torch.Tensor):
                    continue
                
                local_A = local_A.to(self.device)
                local_B = local_B.to(self.device)
                
                rank_A = local_A.shape[0]
                rank_B = local_B.shape[1]
                rank = min(rank_A, rank_B)
                
                if rank == 0:
                    continue
                
                capped_rank = min(rank, self.max_rank)
                
                # Weighted aggregation with sparsity weights
                aggregated_lora[a_key][:capped_rank, :] += (
                    local_A[:capped_rank, :] * weight
                )
                aggregated_lora[b_key][:, :capped_rank] += (
                    local_B[:, :capped_rank] * weight
                )
        
        # Log aggregation summary
        avg_weight = sum(sparsity_weights) / len(sparsity_weights)
        max_weight = max(sparsity_weights)
        min_weight = min(sparsity_weights)
        if self.debug_mode:
            logger.debug(
                "Sparsity-weighted aggregation complete: n_clients=%d, n_pairs=%d, "
                "weights(min/avg/max)=%.4f/%.4f/%.4f",
                len(models),
                len(valid_pairs),
                min_weight,
                avg_weight,
                max_weight,
            )
        
        return aggregated_lora
    
    def _verify_and_fix_shapes(self, result):
        """
        Verify all LoRA parameters have max_rank shape and fix if needed.
        
        This mirrors the final shape check in HeteroLoRAAggregator.aggregate.
        """
        shapes_fixed = 0
        
        for key, tensor in list(result.items()):
            if not isinstance(tensor, torch.Tensor):
                continue
            
            if 'lora_A' in key and 'lora_B' not in key:
                if tensor.shape[0] != self.max_rank:
                    if tensor.shape[0] < self.max_rank:
                        padded = torch.zeros(
                            self.max_rank, tensor.shape[1],
                            dtype=tensor.dtype,
                            device=tensor.device
                        )
                        padded[:tensor.shape[0], :] = tensor
                        result[key] = padded
                    else:
                        result[key] = tensor[:self.max_rank, :].clone()
                    shapes_fixed += 1
                    
            elif 'lora_B' in key:
                if tensor.shape[1] != self.max_rank:
                    if tensor.shape[1] < self.max_rank:
                        padded = torch.zeros(
                            tensor.shape[0], self.max_rank,
                            dtype=tensor.dtype,
                            device=tensor.device
                        )
                        padded[:, :tensor.shape[1]] = tensor
                        result[key] = padded
                    else:
                        result[key] = tensor[:, :self.max_rank].clone()
                    shapes_fixed += 1
        
        if shapes_fixed > 0:
            logger.warning(
                f"Fixed {shapes_fixed} shape mismatches in aggregated result"
            )
        
        return result