"""
AdaS-LayerAware aggregator with layer-aware component aggregation.

This aggregator implements layer-aware aggregation for AdaS-LayerAware:
- Accepts client feedback with grouped metadata (by layer) or flat metadata
- Uses ComponentID = (layer_key, global_idx) for participation tracking
- Routes each ComponentID to its exact LoRA pair
- Produces dense max-rank-shaped updates per LoRA key

Key difference from AdaS:
- AdaS assumes one shared component index across all LoRA layers
- layer-aware uses exact layer-specific ComponentID for aggregation routing
"""
import os
import torch
import logging
from typing import List, Dict, Tuple, Optional, Set

from federatedscope.core.aggregators.aggregator import Aggregator
from federatedscope.core.auxiliaries.utils import param2tensor

import federatedscope.contrib.common as fs_common
from federatedscope.contrib.common.adas_layeraware_utils import (
    ComponentID,
    canonicalize_lora_layer_key,
    infer_layer_keys_from_state_dict,
    get_lora_keys_for_layer,
    flatten_grouped_indices_by_layer,
    normalize_indices_to_grouped,
)

logger = logging.getLogger(__name__)


class AdaSLayerAwareAggregator(Aggregator):
    """
    Aggregator for AdaS-LayerAware with per-layer (ComponentID) component aggregation.
    
    Input format for client_feedback:
        List of dicts containing:
        - 'sample_size': int
        - 'model_update_dict': dict of LoRA updates
        - 'upload_indices': grouped dict {layer_key: [indices]} or flat list
        - 'survivor_indices': grouped dict {layer_key: [indices]} or flat list
        
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
        
        # Get AdaS-LayerAware specific config
        self.adas_layeraware_cfg = None
        self.aggregation_mode = 'sample_size'  # Default for layer-aware
        self.epsilon = 1e-8
        
        if config is not None:
            # Try to get adas_layeraware config from glue.adapter or llm.adapter
            if hasattr(config, 'glue') and hasattr(config.glue, 'adapter'):
                if hasattr(config.glue.adapter, 'adas_layeraware'):
                    self.adas_layeraware_cfg = config.glue.adapter.adas_layeraware
            if self.adas_layeraware_cfg is None and hasattr(config, 'llm') and hasattr(config.llm, 'adapter'):
                if hasattr(config.llm.adapter, 'adas_layeraware'):
                    self.adas_layeraware_cfg = config.llm.adapter.adas_layeraware
            
            if self.adas_layeraware_cfg is not None:
                if hasattr(self.adas_layeraware_cfg, 'aggregation'):
                    self.aggregation_mode = getattr(
                        self.adas_layeraware_cfg.aggregation, 'mode', 'sample_size'
                    )
                    self.epsilon = getattr(
                        self.adas_layeraware_cfg.aggregation, 'epsilon', 1e-8
                    )
        
        # Store latest aggregated global updates for downlink scoring
        self.latest_aggregated_updates: Optional[dict] = None
        self.latest_updated_components: Optional[List[ComponentID]] = None
        
        logger.info(
            f"AdaSLayerAwareAggregator initialized with mode='{self.aggregation_mode}', "
            f"max_rank={self.max_rank}"
        )
    
    def aggregate(self, agg_info):
        """
        Aggregate model updates with layer-aware per-ComponentID averaging.
        
        Arguments:
            agg_info (dict): the feedbacks from clients containing:
                - client_feedback: list of dicts with model updates and indices
        
        Returns:
            dict: the aggregated model updates at global positions
        """
        models = agg_info.get("client_feedback", [])
        if not models:
            logger.warning("No models to aggregate, returning empty dict")
            return {}
        
        # Parse and validate input (handles both grouped and flat formats)
        valid_models = self._parse_client_feedback(models)
        
        if not valid_models:
            logger.warning("No valid models after parsing")
            return {}
        
        if bool(getattr(self.cfg, 'debug', False)):
            logger.debug(
                f"Aggregating {len(valid_models)} client updates"
            )
        
        # Perform layer-aware sparse update aggregation
        aggregated, updated_components = self._sparse_update_aggregate_layeraware(valid_models)

        # Synchronized task-head federation: sample-size ABSOLUTE average of
        # the uploaded classifier+pooler, stored SEPARATELY from the LoRA delta path. The
        # server REPLACES the head with this in postprocess (never adds it as a delta).
        try:
            from federatedscope.contrib.common.head_federation import average_head_params
            head_models = [(m.get('sample_size', 1), m.get('head_params', {}))
                           for m in models if isinstance(m, dict)]
            self.latest_head_average = average_head_params(
                head_models, device=getattr(self, 'device', 'cpu'))
        except Exception as e:
            logger.warning(f"[layer-aware-head] head average failed: {e}")
            self.latest_head_average = {}

        # Store for downlink scoring
        self.latest_aggregated_updates = aggregated
        self.latest_updated_components = updated_components

        return aggregated
    
    def _parse_client_feedback(self, models) -> List[Dict]:
        """
        Parse and validate client feedback into consistent format.
        
        Handles both:
        - layer-aware grouped format: {'upload_indices': {layer_key: [indices]}}
        - Flat format: {'upload_indices': [flat_list]}
        
        Returns:
            List of dicts with keys: sample_size, model_update_dict, 
                                     upload_indices_grouped, survivor_indices_grouped
        """
        valid_models = []
        
        for entry in models:
            parsed = None
            
            # Handle dict format
            if isinstance(entry, dict):
                if 'model_update_dict' in entry or 'model_para' in entry:
                    model_dict = entry.get('model_update_dict', entry.get('model_para', {}))
                    
                    # Parse indices - could be grouped or flat
                    upload_raw = entry.get('upload_indices', {})
                    survivor_raw = entry.get('survivor_indices', entry.get('upload_indices', {}))
                    
                    # Normalize to grouped format (normalize_indices_to_grouped)
                    upload_grouped = self._normalize_indices_to_grouped_internal(upload_raw, model_dict)
                    survivor_grouped = self._normalize_indices_to_grouped_internal(survivor_raw, model_dict)
                    
                    parsed = {
                        'sample_size': entry.get('sample_size', 1),
                        'model_update_dict': model_dict,
                        'upload_indices_grouped': upload_grouped,
                        'survivor_indices_grouped': survivor_grouped,
                    }
            
            # Handle tuple format (sample_size, model_dict, flat_indices), as used by AdaS-PruneOnly
            elif isinstance(entry, (list, tuple)) and len(entry) >= 3:
                model_dict = entry[1]
                flat_indices = list(entry[2])
                
                # Convert flat indices to grouped format (normalize_indices_to_grouped)
                upload_grouped = self._normalize_indices_to_grouped_internal(flat_indices, model_dict)
                
                parsed = {
                    'sample_size': entry[0],
                    'model_update_dict': model_dict,
                    'upload_indices_grouped': upload_grouped,
                    'survivor_indices_grouped': upload_grouped,
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
            
            valid_models.append(parsed)
        
        return valid_models
    
    def _normalize_indices_to_grouped_internal(
        self, 
        indices_raw,
        model_dict: dict
    ) -> Dict[str, List[int]]:
        """
        Normalize indices to grouped format with normalize_indices_to_grouped().
        
        Flat indices are expanded to all layers (the AdaS shared-index semantics).
        After normalization, the aggregator works only with grouped metadata.
        
        Args:
            indices_raw: Raw indices (grouped dict or flat list)
            model_dict: Model state dict to infer layer keys from
            
        Returns:
            Grouped indices dict with exact layer keys
        """
        # Infer layer keys for potential flat->grouped expansion
        layer_keys = infer_layer_keys_from_state_dict(model_dict)
        
        # Normalize with normalize_indices_to_grouped
        return normalize_indices_to_grouped(
            payload=indices_raw,
            layer_keys=layer_keys,
            max_rank=self.max_rank
        )
    
    def _sparse_update_aggregate_layeraware(self, models: List[Dict]) -> Tuple[dict, List[ComponentID]]:
        """
        Perform layer-aware sparse update aggregation over clients.
        
        For each ComponentID = (layer_key, global_idx):
            1. Find all clients that have this ComponentID in their upload_indices_grouped
            2. Compute weights (sample_size or sparsity_weighted)
            3. Aggregate only the specific A row / B column for that exact layer
        
        layer-aware key difference: Component (layer_X, idx) and (layer_Y, idx) are treated
        as distinct components, not the same logical component.
        
        Args:
            models: List of parsed client dicts
            
        Returns:
            Tuple of (aggregated_updates, updated_components)
        """
        # Find all LoRA pairs from first client (reference)
        reference_model = models[0]['model_update_dict']
        layer_keys = infer_layer_keys_from_state_dict(reference_model)
        
        if not layer_keys:
            logger.warning("No LoRA parameters found for aggregation")
            return {}, []
        
        # Build ComponentID -> client list mapping
        component_to_clients: Dict[ComponentID, List[int]] = {}
        
        for client_idx, model_data in enumerate(models):
            upload_grouped = model_data['upload_indices_grouped']
            
            for layer_key, indices in upload_grouped.items():
                for global_idx in indices:
                    cid = (layer_key, global_idx)
                    if cid not in component_to_clients:
                        component_to_clients[cid] = []
                    component_to_clients[cid].append(client_idx)
        
        # Compute component norms for sparsity_weighted mode
        client_component_norms = None
        if self.aggregation_mode == 'sparsity_weighted':
            client_component_norms = self._compute_all_component_norms_layeraware(models)
        logger.info(
            f"AdaSLayerAwareAggregator.aggregate: mode='{self.aggregation_mode}' "
            f"(n_clients={len(models)}, n_components={len(component_to_clients)})"
        )
        
        # Initialize aggregated updates from zero
        aggregated = {}
        
        for layer_key in layer_keys:
            a_key, b_key = get_lora_keys_for_layer(reference_model, layer_key)
            
            if a_key is None or b_key is None:
                continue
            
            ref_A = param2tensor(reference_model.get(a_key))
            ref_B = param2tensor(reference_model.get(b_key))
            
            if ref_A is None or ref_B is None:
                continue
            
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
        
        # Aggregate each ComponentID
        updated_components = []
        
        for cid, participating_clients in component_to_clients.items():
            layer_key, global_idx = cid
            
            if not participating_clients:
                continue
            
            updated_components.append(cid)
            
            # Aggregate this specific ComponentID
            self._aggregate_component_layeraware(
                layer_key, global_idx,
                models, participating_clients,
                client_component_norms, aggregated
            )
        
        # Log aggregation summary
        self._log_aggregation_summary_layeraware(models, component_to_clients, updated_components)
        
        return aggregated, updated_components
    
    def _aggregate_component_layeraware(
        self,
        layer_key: str,
        global_idx: int,
        models: List[Dict],
        participating_clients: List[int],
        client_component_norms: Optional[List[Dict[ComponentID, float]]],
        aggregated: dict
    ):
        """
        Aggregate a single ComponentID from participating clients.
        
        Routes to the exact LoRA pair identified by layer_key.
        """
        a_key, b_key = get_lora_keys_for_layer(aggregated, layer_key)
        
        if a_key is None or b_key is None:
            return
        
        contributions_A = []
        contributions_B = []
        weights = []
        
        cid = (layer_key, global_idx)
        
        for client_idx in participating_clients:
            model_data = models[client_idx]
            model_dict = model_data['model_update_dict']
            upload_grouped = model_data['upload_indices_grouped']
            sample_size = model_data['sample_size']
            
            # Get layer-specific upload indices
            layer_upload_indices = upload_grouped.get(layer_key, [])
            
            # Find local position for this global index
            try:
                local_pos = layer_upload_indices.index(global_idx)
            except ValueError:
                continue
            
            # Get client's A and B tensors for this layer
            client_a_key, client_b_key = get_lora_keys_for_layer(model_dict, layer_key)
            
            if client_a_key is None or client_b_key is None:
                continue
            
            client_A = param2tensor(model_dict.get(client_a_key))
            client_B = param2tensor(model_dict.get(client_b_key))
            
            if client_A is None or client_B is None:
                continue
            
            client_A = client_A.to(self.device)
            client_B = client_B.to(self.device)
            
            # Extract local component
            if local_pos >= client_A.shape[0] or local_pos >= client_B.shape[1]:
                continue
            
            # Normalize to fp32: clients may train in mixed precision (e.g. bf16),
            # so cast contributions to the fp32 global model dtype before the
            # weighted average.
            a_row = client_A[local_pos, :].clone().float()
            b_col = client_B[:, local_pos].clone().float()

            contributions_A.append(a_row)
            contributions_B.append(b_col)
            
            # Compute weight
            if self.aggregation_mode == 'sparsity_weighted' and client_component_norms is not None:
                norm = client_component_norms[client_idx].get(cid, 0.0)
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
    
    def _compute_all_component_norms_layeraware(
        self, 
        models: List[Dict]
    ) -> List[Dict[ComponentID, float]]:
        """
        Compute per-ComponentID norms for all clients.
        """
        result = []
        
        for model_data in models:
            model_dict = model_data['model_update_dict']
            upload_grouped = model_data['upload_indices_grouped']
            
            norms = {}
            
            for layer_key, indices in upload_grouped.items():
                a_key, b_key = get_lora_keys_for_layer(model_dict, layer_key)
                
                if a_key is None or b_key is None:
                    continue
                
                A = param2tensor(model_dict.get(a_key))
                B = param2tensor(model_dict.get(b_key))
                
                if A is None or B is None:
                    continue
                
                for local_pos, global_idx in enumerate(indices):
                    if local_pos >= A.shape[0] or local_pos >= B.shape[1]:
                        continue
                    
                    a_norm = torch.norm(A[local_pos, :], p=2).item()
                    b_norm = torch.norm(B[:, local_pos], p=2).item()
                    
                    cid = (layer_key, global_idx)
                    norms[cid] = a_norm * b_norm
            
            result.append(norms)
        
        return result
    
    def _log_aggregation_summary_layeraware(
        self,
        models: List[Dict],
        component_to_clients: Dict[ComponentID, List[int]],
        updated_components: List[ComponentID]
    ):
        """Log layer-aware aggregation summary."""
        n_clients = len(models)
        n_updated = len(updated_components)
        
        # Count unique layers
        unique_layers = set(cid[0] for cid in updated_components)
        n_layers = len(unique_layers)
        
        # Participation statistics
        participations = [len(clients) for clients in component_to_clients.values()]
        if participations:
            part_min = min(participations)
            part_max = max(participations)
            part_avg = sum(participations) / len(participations)
        else:
            part_min = part_max = 0
            part_avg = 0.0
        
        n_full = sum(1 for cid in updated_components if len(component_to_clients.get(cid, [])) == n_clients)
        n_partial = sum(1 for cid in updated_components if 1 < len(component_to_clients.get(cid, [])) < n_clients)
        n_single = sum(1 for cid in updated_components if len(component_to_clients.get(cid, [])) == 1)
        
        if bool(getattr(self.cfg, 'debug', False)):
            logger.debug(
                f"Layer-aware aggregation complete: "
                f"n_clients={n_clients}, n_layers={n_layers}, "
                f"components_updated={n_updated}, "
                f"clients_per_component(min/avg/max)={part_min}/{part_avg:.1f}/{part_max}"
            )
        
        logger.info(
            f"layer-aware Participation distribution: "
            f"full={n_full}, partial={n_partial}, single={n_single}, "
            f"layers={n_layers}"
        )
    
    def get_latest_aggregated_updates(self) -> Optional[dict]:
        """Get the latest aggregated global updates for downlink scoring."""
        return self.latest_aggregated_updates
    
    def get_latest_updated_components(self) -> Optional[List[ComponentID]]:
        """Get the list of ComponentIDs that were updated in the last round."""
        return self.latest_updated_components
    
    def update(self, model_parameters, strict=False):
        """
        Update the server model with aggregated parameters.
        
        For layer-aware, this applies aggregated updates to the global model bank.
        """
        if self.model is not None and model_parameters:
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
