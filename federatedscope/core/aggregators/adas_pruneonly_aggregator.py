"""
AdaS-PruneOnly Aggregator with Index-Aware Per-Component Aggregation.

This aggregator implements index-aware aggregation for AdaS-PruneOnly:
- Each global component is aggregated only from clients that uploaded it
- Supports sample_size and sparsity_weighted aggregation modes
- Avoids "zero-padding dilution" by not averaging in zero contributions

The aggregation is performed per rank-1 component:
    For each global index j:
        - Collect all clients that included j in their indices_list
        - Compute normalized weights among those clients
        - Aggregate A row and B column for component j
"""
import os
import torch
import logging
from typing import List, Dict, Tuple, Optional
from federatedscope.core.aggregators.aggregator import Aggregator
from federatedscope.core.auxiliaries.utils import param2tensor

import federatedscope.contrib.common as fs_common
logger = logging.getLogger(__name__)


class AdaSPruneOnlyAggregator(Aggregator):
    """
    Aggregator for AdaS-PruneOnly with index-aware per-component aggregation.
    
    Input format for client_feedback:
        List of (sample_size, model_dict, indices_list) tuples
        
    Output format:
        Dict containing aggregated LoRA tensors at full max_rank shapes
    """
    
    def __init__(self, model=None, device='cpu', config=None):
        super().__init__()
        self.model = model
        self.device = device
        self.cfg = config
        
        # Get max_rank from config
        self.max_rank = fs_common.get_effective_max_rank(self.cfg)
        
        # Get AdaS-PruneOnly specific config
        self.adas_pruneonly_cfg = None
        self.aggregation_mode = 'sample_size'  # Default
        self.epsilon = 1e-8
        
        if config is not None:
            # Try to get adas_pruneonly config from llm.adapter or glue.adapter
            if hasattr(config, 'llm') and hasattr(config.llm, 'adapter'):
                if hasattr(config.llm.adapter, 'adas_pruneonly'):
                    self.adas_pruneonly_cfg = config.llm.adapter.adas_pruneonly
            if self.adas_pruneonly_cfg is None and hasattr(config, 'glue') and hasattr(config.glue, 'adapter'):
                if hasattr(config.glue.adapter, 'adas_pruneonly'):
                    self.adas_pruneonly_cfg = config.glue.adapter.adas_pruneonly
            
            if self.adas_pruneonly_cfg is not None:
                if hasattr(self.adas_pruneonly_cfg, 'aggregation'):
                    self.aggregation_mode = getattr(
                        self.adas_pruneonly_cfg.aggregation, 'mode', 'sample_size'
                    )
                    self.epsilon = getattr(
                        self.adas_pruneonly_cfg.aggregation, 'epsilon', 1e-8
                    )
        
        logger.info(
            f"Aggregator initialized with mode='{self.aggregation_mode}', "
            f"max_rank={self.max_rank}"
        )
    
    def aggregate(self, agg_info):
        """
        Aggregate LoRA weights with index-aware per-component averaging.
        
        Arguments:
            agg_info (dict): the feedbacks from clients containing:
                - client_feedback: list of (sample_size, model_params, indices_list) tuples
        
        Returns:
            dict: the aggregated results with LoRA weights at max_rank
        """
        models = agg_info.get("client_feedback", [])
        if not models:
            logger.warning("No models to aggregate, returning empty dict")
            return {}
        
        # Validate input format and filter malformed entries
        valid_models = []
        for entry in models:
            if not isinstance(entry, (list, tuple)) or len(entry) < 3:
                logger.warning(
                    f"Skipping malformed entry: expected 3-tuple, got {type(entry)}"
                )
                continue
            
            sample_size, model_dict, indices_list = entry[0], entry[1], entry[2]
            
            if not isinstance(model_dict, dict):
                logger.warning(
                    f"Skipping entry with non-dict model_params"
                )
                continue
            
            if not isinstance(indices_list, (list, tuple)):
                logger.warning(
                    f"Skipping entry with non-list indices: {type(indices_list)}"
                )
                continue
            
            # Validate indices
            try:
                from federatedscope.contrib.common.adas_utils import validate_indices
                validate_indices(list(indices_list), self.max_rank, context="aggregate")
            except ValueError as e:
                logger.warning(f"Skipping entry: {e}")
                continue
            
            valid_models.append((sample_size, model_dict, list(indices_list)))
        
        if not valid_models:
            logger.warning("No valid models after filtering")
            return {}
        
        # Debug logging
        debug_mode = bool(getattr(self.cfg, 'debug', False))
        if debug_mode:
            logger.debug(
                f"Aggregating {len(valid_models)} client updates"
            )
        
        # Perform index-aware aggregation
        return self._index_aware_aggregate(valid_models)
    
    def _index_aware_aggregate(self, models: List[Tuple[int, dict, List[int]]]) -> dict:
        """
        Perform index-aware aggregation over clients.
        
        For each global component index j:
            1. Find all clients that have j in their indices
            2. Compute weights (sample_size or sparsity_weighted)
            3. Aggregate A row and B column for that component
        
        Args:
            models: List of (sample_size, model_dict, indices_list) tuples
            
        Returns:
            Dict of aggregated LoRA tensors at max_rank
        """
        # Initialize aggregated tensors from current global model
        if self.model is not None:
            aggregated = {k: v.clone() for k, v in self.model.state_dict().items()
                         if 'lora_A' in k or 'lora_B' in k}
        else:
            aggregated = {}
        
        # Find all LoRA A/B pairs from reference model (first client)
        _, reference_model, _ = models[0]
        lora_pairs = self._find_lora_pairs(reference_model)
        
        if not lora_pairs:
            logger.warning("No LoRA parameters found for aggregation")
            return aggregated
        
        # Build index -> client mapping
        # For each global index j, track which clients have it
        index_to_clients: Dict[int, List[int]] = {j: [] for j in range(self.max_rank)}
        
        for client_idx, (_, _, indices_list) in enumerate(models):
            for global_idx in indices_list:
                if 0 <= global_idx < self.max_rank:
                    index_to_clients[global_idx].append(client_idx)
        
        # Compute component norms for sparsity_weighted mode
        client_component_norms = None
        if self.aggregation_mode == 'sparsity_weighted':
            client_component_norms = self._compute_all_component_norms(models)
        
        # Aggregate each LoRA pair
        for a_key, b_key in lora_pairs:
            self._aggregate_lora_pair(
                a_key, b_key, models, index_to_clients, 
                client_component_norms, aggregated
            )
        
        # Log aggregation summary
        n_updated = sum(1 for j in range(self.max_rank) if index_to_clients[j])
        ranks = [len(indices) for (_, _, indices) in models]
        rank_min = min(ranks) if ranks else 0
        rank_max = max(ranks) if ranks else 0
        rank_avg = (sum(ranks) / len(ranks)) if ranks else 0.0

        participations = [len(index_to_clients[j]) for j in range(self.max_rank) if index_to_clients[j]]
        part_min = min(participations) if participations else 0
        part_max = max(participations) if participations else 0
        part_avg = (sum(participations) / len(participations)) if participations else 0.0
        
        # Participation distribution: full/partial/single
        n_clients = len(models)
        n_full = sum(1 for j in range(self.max_rank) if len(index_to_clients[j]) == n_clients)
        n_partial = sum(1 for j in range(self.max_rank) if 1 < len(index_to_clients[j]) < n_clients)
        n_single = sum(1 for j in range(self.max_rank) if len(index_to_clients[j]) == 1)


        # Debug logging
        debug_mode = bool(getattr(self.cfg, 'debug', False))
        if debug_mode:
            logger.debug(
                f"Index-aware aggregation complete: "
                f"n_clients={n_clients}, n_pairs={len(lora_pairs)}, "
                f"components_updated={n_updated}/{self.max_rank}, "
                f"rank(min/avg/max)={rank_min}/{rank_avg:.1f}/{rank_max}, "
                f"clients_per_component(min/avg/max)={part_min}/{part_avg:.1f}/{part_max}, "
                f"components_full_participation={n_full}/{n_updated}"
            )
        
        # Participation distribution summary (helps interpret aggregate-only-matching behavior)
        logger.info(
            f"Participation distribution: "
            f"full={n_full}, partial={n_partial}, single={n_single}"
        )
        
        return aggregated
    
    def _find_lora_pairs(self, model_dict: dict) -> List[Tuple[str, str]]:
        """Find all LoRA A/B key pairs in a model dict."""
        pairs = []
        a_keys = set()
        b_keys = set()
        
        for key in model_dict.keys():
            if 'lora_A' in key and 'lora_B' not in key:
                a_keys.add(key)
            elif 'lora_B' in key:
                b_keys.add(key)
        
        for a_key in a_keys:
            b_key = a_key.replace('lora_A', 'lora_B')
            if b_key in b_keys:
                pairs.append((a_key, b_key))
        
        return pairs
    
    def _compute_all_component_norms(
        self, 
        models: List[Tuple[int, dict, List[int]]]
    ) -> List[Dict[int, float]]:
        """
        Compute per-component norms for all clients.
        
        Returns:
            List of dicts, one per client, mapping global_index -> norm_product
        """
        from federatedscope.contrib.common.adas_utils import (
            compute_per_component_norms_from_state_dict
        )
        
        result = []
        for _, model_dict, indices_list in models:
            norms = compute_per_component_norms_from_state_dict(
                model_dict, indices_list
            )
            result.append(norms)
        
        return result
    
    def _aggregate_lora_pair(
        self,
        a_key: str,
        b_key: str,
        models: List[Tuple[int, dict, List[int]]],
        index_to_clients: Dict[int, List[int]],
        client_component_norms: Optional[List[Dict[int, float]]],
        aggregated: dict
    ):
        """
        Aggregate a single LoRA A/B pair with index-aware averaging.
        
        Args:
            a_key: Key for LoRA A tensor
            b_key: Key for LoRA B tensor
            models: List of (sample_size, model_dict, indices_list) tuples
            index_to_clients: Mapping from global index to client indices
            client_component_norms: Per-client component norms (for sparsity_weighted)
            aggregated: Output dict to update
        """
        # Get reference shapes from first client
        _, ref_model, _ = models[0]
        ref_A = param2tensor(ref_model.get(a_key))
        ref_B = param2tensor(ref_model.get(b_key))
        
        if ref_A is None or ref_B is None:
            return
        
        in_features = ref_A.shape[1]
        out_features = ref_B.shape[0]
        
        # Initialize aggregated tensors at max_rank
        if a_key in aggregated:
            agg_A = aggregated[a_key].to(self.device)
        else:
            agg_A = torch.zeros(
                self.max_rank, in_features,
                dtype=ref_A.dtype, device=self.device
            )
        
        if b_key in aggregated:
            agg_B = aggregated[b_key].to(self.device)
        else:
            agg_B = torch.zeros(
                out_features, self.max_rank,
                dtype=ref_B.dtype, device=self.device
            )
        
        # Aggregate each global component
        for global_idx in range(self.max_rank):
            participating_clients = index_to_clients[global_idx]
            
            if not participating_clients:
                # No clients updated this component - keep current value
                continue
            
            # Collect contributions and compute weights
            contributions_A = []
            contributions_B = []
            weights = []
            
            for client_idx in participating_clients:
                sample_size, model_dict, indices_list = models[client_idx]
                
                # Find local position for this global index
                try:
                    local_pos = indices_list.index(global_idx)
                except ValueError:
                    continue
                
                # Get client's A and B tensors
                client_A = param2tensor(model_dict.get(a_key))
                client_B = param2tensor(model_dict.get(b_key))
                
                if client_A is None or client_B is None:
                    continue
                
                client_A = client_A.to(self.device)
                client_B = client_B.to(self.device)
                
                # Extract local component
                if local_pos >= client_A.shape[0] or local_pos >= client_B.shape[1]:
                    continue
                
                a_row = client_A[local_pos, :].clone()
                b_col = client_B[:, local_pos].clone()
                
                contributions_A.append(a_row)
                contributions_B.append(b_col)
                
                # Compute weight for this client
                if self.aggregation_mode == 'sparsity_weighted' and client_component_norms is not None:
                    # Use component norm as weight
                    norm = client_component_norms[client_idx].get(global_idx, 0.0)
                    weights.append(norm + self.epsilon)
                else:
                    # Use sample size as weight
                    weights.append(float(sample_size))
            
            if not contributions_A:
                continue
            
            # Normalize weights
            total_weight = sum(weights) + self.epsilon
            normalized_weights = [w / total_weight for w in weights]
            
            # Weighted average
            agg_A[global_idx, :].zero_()
            agg_B[:, global_idx].zero_()
            
            for a_row, b_col, w in zip(contributions_A, contributions_B, normalized_weights):
                agg_A[global_idx, :] += a_row * w
                agg_B[:, global_idx] += b_col * w
        
        # Store aggregated tensors
        aggregated[a_key] = agg_A
        aggregated[b_key] = agg_B
    
    def update(self, model_parameters, strict=False):
        """
        Update the server model with aggregated parameters.
        
        Arguments:
            model_parameters: dict of aggregated parameters
            strict: Whether to enforce strict matching
        """
        if self.model is not None and model_parameters:
            self.model.load_state_dict(model_parameters, strict=strict)
    
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