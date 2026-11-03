"""
AdaS Aggregator with Sparse Model Update Aggregation.

This aggregator implements Stage 2 aggregation for AdaS:
- Accepts client feedback with model_update_dict, upload_indices, and survivor_indices
- Validates update tensor dimensions match the upload subset
- Aggregates each global component only across clients that actually transmitted it
- Produces aggregated global model updates
- Retains aggregated updates for server-side downlink scoring

Key difference from prune-only:
- prune-only aggregates absolute parameter slices, initializing from global model
- AdaS aggregates model updates (deltas), initializing from zero
"""
import os
import torch
import logging
from typing import List, Dict, Tuple, Optional
from federatedscope.core.aggregators.aggregator import Aggregator
from federatedscope.core.auxiliaries.utils import param2tensor

import federatedscope.contrib.common as fs_common

logger = logging.getLogger(__name__)


class AdaSAggregator(Aggregator):
    """
    Aggregator for AdaS with sparse model update aggregation.
    
    Input format for client_feedback:
        List of dicts or 3-tuples:
        - Dict: {'sample_size': int, 'model_update_dict': dict, 'upload_indices': list, 'survivor_indices': list}
        - Tuple: (sample_size, model_dict, indices_list), as used by AdaS-PruneOnly
        
    Output format:
        Dict containing aggregated model updates at global positions
    """
    
    def __init__(self, model=None, device='cpu', config=None):
        super().__init__()
        self.model = model
        self.device = device
        self.cfg = config
        
        # Get max_rank from config
        self.max_rank = fs_common.get_effective_max_rank(self.cfg)
        
        # Get AdaS specific config
        self.adas_cfg = None
        self.aggregation_mode = 'sample_size'  # Default for AdaS
        self.epsilon = 1e-8
        
        if config is not None:
            # Try to get adas config from llm.adapter or glue.adapter
            if hasattr(config, 'llm') and hasattr(config.llm, 'adapter'):
                if hasattr(config.llm.adapter, 'adas'):
                    self.adas_cfg = config.llm.adapter.adas
            if self.adas_cfg is None and hasattr(config, 'glue') and hasattr(config.glue, 'adapter'):
                if hasattr(config.glue.adapter, 'adas'):
                    self.adas_cfg = config.glue.adapter.adas
            
            if self.adas_cfg is not None:
                if hasattr(self.adas_cfg, 'aggregation'):
                    self.aggregation_mode = getattr(
                        self.adas_cfg.aggregation, 'mode', 'sample_size'
                    )
                    self.epsilon = getattr(
                        self.adas_cfg.aggregation, 'epsilon', 1e-8
                    )
        
        # Store latest aggregated global updates for downlink scoring
        self.latest_aggregated_updates: Optional[dict] = None
        self.latest_updated_components: Optional[List[int]] = None
        
        logger.info(
            f"Aggregator initialized with mode='{self.aggregation_mode}', "
            f"max_rank={self.max_rank}"
        )
    
    def aggregate(self, agg_info):
        """
        Aggregate model updates with index-aware per-component averaging.
        
        Arguments:
            agg_info (dict): the feedbacks from clients containing:
                - client_feedback: list of dicts or 3-tuples
        
        Returns:
            dict: the aggregated model updates at global positions
        """
        models = agg_info.get("client_feedback", [])
        if not models:
            logger.warning("No models to aggregate, returning empty dict")
            return {}
        
        # Parse and validate input
        valid_models = self._parse_client_feedback(models)
        
        if not valid_models:
            logger.warning("No valid models after parsing")
            return {}
        
        if bool(getattr(self.cfg, 'debug', False)):
            logger.debug(
                f"Aggregating {len(valid_models)} client updates"
            )
        
        # Perform sparse update aggregation
        aggregated, updated_components = self._sparse_update_aggregate(valid_models)

        # Synchronized task-head federation: sample-size ABSOLUTE average of
        # the uploaded classifier+pooler, stored SEPARATELY from the LoRA delta path (server
        # REPLACES the head in postprocess; never adds it as a delta).
        try:
            from federatedscope.contrib.common.head_federation import average_head_params
            head_models = [(m.get('sample_size', 1), m.get('head_params', {}))
                           for m in models if isinstance(m, dict)]
            self.latest_head_average = average_head_params(
                head_models, device=getattr(self, 'device', 'cpu'))
        except Exception as e:
            logger.warning(f"[AdaS-head] head average failed: {e}")
            self.latest_head_average = {}

        # Store for downlink scoring
        self.latest_aggregated_updates = aggregated
        self.latest_updated_components = updated_components

        return aggregated
    
    def _parse_client_feedback(self, models) -> List[Dict]:
        """
        Parse and validate client feedback into consistent format.
        
        Returns:
            List of dicts with keys: sample_size, model_update_dict, upload_indices, survivor_indices
        """
        valid_models = []
        
        for entry in models:
            parsed = None
            
            # Handle dict format (AdaS)
            if isinstance(entry, dict):
                if 'model_update_dict' in entry or 'model_para' in entry:
                    model_dict = entry.get('model_update_dict', entry.get('model_para', {}))
                    parsed = {
                        'sample_size': entry.get('sample_size', 1),
                        'model_update_dict': model_dict,
                        'upload_indices': entry.get('upload_indices', []),
                        'survivor_indices': entry.get('survivor_indices', entry.get('upload_indices', [])),
                    }
            
            # Handle tuple format (sample_size, model_dict, indices_list), as used by AdaS-PruneOnly
            elif isinstance(entry, (list, tuple)) and len(entry) >= 3:
                parsed = {
                    'sample_size': entry[0],
                    'model_update_dict': entry[1],
                    'upload_indices': list(entry[2]),
                    'survivor_indices': list(entry[2]),
                }
            elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
                # (sample_size, model_dict) without indices - use full model
                parsed = {
                    'sample_size': entry[0],
                    'model_update_dict': entry[1],
                    'upload_indices': list(range(self.max_rank)),
                    'survivor_indices': list(range(self.max_rank)),
                }
            
            if parsed is None:
                logger.warning(
                    f"Skipping malformed entry: {type(entry)}"
                )
                continue
            
            # Validate model_dict
            if not isinstance(parsed['model_update_dict'], dict):
                logger.warning(
                    f"Skipping entry with non-dict model_update_dict"
                )
                continue
            
            # Validate indices
            try:
                from federatedscope.contrib.common.adas_utils import validate_indices
                validate_indices(parsed['upload_indices'], self.max_rank, context="aggregate")
            except ValueError as e:
                logger.warning(f"Skipping entry: {e}")
                continue
            
            valid_models.append(parsed)
        
        return valid_models
    
    def _sparse_update_aggregate(self, models: List[Dict]) -> Tuple[dict, List[int]]:
        """
        Perform sparse update aggregation over clients.
        
        For each global component index j:
            1. Find all clients that have j in their upload_indices
            2. Compute weights (sample_size or sparsity_weighted)
            3. Aggregate A row update and B column update for that component
        
        Key difference from prune-only: Initialize from zero updates, not from global model.
        
        Args:
            models: List of parsed client dicts
            
        Returns:
            Tuple of (aggregated_updates, updated_components)
        """
        # Find all LoRA A/B pairs from first client (reference)
        reference_model = models[0]['model_update_dict']
        lora_pairs = self._find_lora_pairs(reference_model)
        
        if not lora_pairs:
            logger.warning("No LoRA parameters found for aggregation")
            return {}, []
        
        # Build index -> client mapping
        index_to_clients: Dict[int, List[int]] = {j: [] for j in range(self.max_rank)}
        
        for client_idx, model_data in enumerate(models):
            for global_idx in model_data['upload_indices']:
                if 0 <= global_idx < self.max_rank:
                    index_to_clients[global_idx].append(client_idx)
        
        # Compute component norms for sparsity_weighted mode
        client_component_norms = None
        if self.aggregation_mode == 'sparsity_weighted':
            client_component_norms = self._compute_all_component_norms(models)
        
        # Initialize aggregated updates from zero
        aggregated = {}
        
        for a_key, b_key in lora_pairs:
            ref_A = param2tensor(reference_model.get(a_key))
            ref_B = param2tensor(reference_model.get(b_key))
            
            if ref_A is None or ref_B is None:
                continue
            
            # Infer dimensions
            # For AdaS, client uploads have shape (len(upload_indices), in_features) for A
            # We need to build full max_rank aggregated updates
            in_features = ref_A.shape[-1]
            out_features = ref_B.shape[0]
            
            # Initialize at max_rank with zeros
            aggregated[a_key] = torch.zeros(
                self.max_rank, in_features,
                dtype=ref_A.dtype, device=self.device
            )
            aggregated[b_key] = torch.zeros(
                out_features, self.max_rank,
                dtype=ref_B.dtype, device=self.device
            )
        
        # Aggregate each component
        updated_components = []
        
        for global_idx in range(self.max_rank):
            participating_clients = index_to_clients[global_idx]
            
            if not participating_clients:
                continue
            
            updated_components.append(global_idx)
            
            # Aggregate each LoRA pair for this component
            for a_key, b_key in lora_pairs:
                self._aggregate_component(
                    a_key, b_key, global_idx,
                    models, participating_clients,
                    client_component_norms, aggregated
                )
        
        # Log aggregation summary
        self._log_aggregation_summary(models, index_to_clients, lora_pairs, updated_components)
        
        return aggregated, updated_components
    
    def _aggregate_component(
        self,
        a_key: str,
        b_key: str,
        global_idx: int,
        models: List[Dict],
        participating_clients: List[int],
        client_component_norms: Optional[List[Dict[int, float]]],
        aggregated: dict
    ):
        """
        Aggregate a single component from participating clients.
        """
        contributions_A = []
        contributions_B = []
        weights = []
        
        for client_idx in participating_clients:
            model_data = models[client_idx]
            model_dict = model_data['model_update_dict']
            upload_indices = model_data['upload_indices']
            sample_size = model_data['sample_size']
            
            # Find local position for this global index
            try:
                local_pos = upload_indices.index(global_idx)
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
            
            # Compute weight
            if self.aggregation_mode == 'sparsity_weighted' and client_component_norms is not None:
                norm = client_component_norms[client_idx].get(global_idx, 0.0)
                weights.append(norm + self.epsilon)
            else:
                weights.append(float(sample_size))
        
        if not contributions_A:
            return
        
        # Normalize weights
        total_weight = sum(weights) + self.epsilon
        normalized_weights = [w / total_weight for w in weights]
        
        # Weighted average
        agg_A = aggregated[a_key]
        agg_B = aggregated[b_key]
        
        for a_row, b_col, w in zip(contributions_A, contributions_B, normalized_weights):
            agg_A[global_idx, :] += a_row * w
            agg_B[:, global_idx] += b_col * w
    
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
        models: List[Dict]
    ) -> List[Dict[int, float]]:
        """
        Compute per-component norms for all clients.
        """
        from federatedscope.contrib.common.adas_utils import (
            compute_per_component_norms_from_state_dict
        )
        
        result = []
        for model_data in models:
            norms = compute_per_component_norms_from_state_dict(
                model_data['model_update_dict'],
                model_data['upload_indices']
            )
            result.append(norms)
        
        return result
    
    def _log_aggregation_summary(
        self,
        models: List[Dict],
        index_to_clients: Dict[int, List[int]],
        lora_pairs: List[Tuple[str, str]],
        updated_components: List[int]
    ):
        """Log aggregation summary."""
        n_clients = len(models)
        n_updated = len(updated_components)
        
        upload_sizes = [len(m['upload_indices']) for m in models]
        upload_min = min(upload_sizes) if upload_sizes else 0
        upload_max = max(upload_sizes) if upload_sizes else 0
        upload_avg = sum(upload_sizes) / len(upload_sizes) if upload_sizes else 0.0
        
        participations = [len(index_to_clients[j]) for j in updated_components] if updated_components else []
        part_min = min(participations) if participations else 0
        part_max = max(participations) if participations else 0
        part_avg = sum(participations) / len(participations) if participations else 0.0
        
        n_full = sum(1 for j in updated_components if len(index_to_clients[j]) == n_clients)
        n_partial = sum(1 for j in updated_components if 1 < len(index_to_clients[j]) < n_clients)
        n_single = sum(1 for j in updated_components if len(index_to_clients[j]) == 1)
        
        if bool(getattr(self.cfg, 'debug', False)):
            logger.debug(
                f"Aggregation complete: "
                f"n_clients={n_clients}, n_pairs={len(lora_pairs)}, "
                f"components_updated={n_updated}/{self.max_rank}, "
                f"upload_sizes(min/avg/max)={upload_min}/{upload_avg:.1f}/{upload_max}, "
                f"clients_per_component(min/avg/max)={part_min}/{part_avg:.1f}/{part_max}"
            )
        
        logger.info(
            f"Participation distribution: "
            f"full={n_full}, partial={n_partial}, single={n_single}"
        )
    
    def get_latest_aggregated_updates(self) -> Optional[dict]:
        """Get the latest aggregated global updates for downlink scoring."""
        return self.latest_aggregated_updates
    
    def get_latest_updated_components(self) -> Optional[List[int]]:
        """Get the list of components that were updated in the last round."""
        return self.latest_updated_components
    
    def update(self, model_parameters, strict=False):
        """
        Update the server model with aggregated parameters.
        
        For AdaS, this applies aggregated updates to the global model bank.
        """
        if self.model is not None and model_parameters:
            # For AdaS, we add updates to the current model rather than replacing
            current_state = self.model.state_dict()
            
            for key, update_tensor in model_parameters.items():
                if key in current_state and isinstance(update_tensor, torch.Tensor):
                    current_state[key] = current_state[key] + update_tensor.to(current_state[key].device)
            
            self.model.load_state_dict(current_state, strict=strict)
    
    def save_model(self, path, cur_round=-1):
        """Save the aggregated model."""
        assert self.model is not None
        
        if os.path.isdir(path) or (not os.path.splitext(path)[1] and path.endswith('/')):
            os.makedirs(path, exist_ok=True)
            filename = f'model_round_{cur_round}.pt' if cur_round >= 0 else 'model.pt'
            path = os.path.join(path, filename)
        else:
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