"""Method-specific AdaS-LayerAware client.

This client implements layer-aware AdaS with:
- ComponentID = (layer_key, global_idx) internal representation
- Grouped survivor/upload/download indices per layer
- Layer-aware Stage 1 structural pruning
- Layer-aware Stage 2 communication sparsity
- ComponentID-keyed residual buffers

Key differences from AdaS:
- AdaS: flat integer component index shared across all layers
- layer-aware: exact layer-specific ComponentID for all operations
"""

import logging
import torch
from typing import Dict, List, Optional, Tuple

import federatedscope.contrib.common as fs_common
from federatedscope.core.workers import Client
from federatedscope.contrib.common.adas_layeraware_utils import (
    ComponentID,
    canonicalize_lora_layer_key,
    infer_layer_keys_from_state_dict,
    get_lora_keys_for_layer,
    flatten_grouped_indices_by_layer,
    group_component_ids_by_layer,
    normalize_indices_to_grouped,
    compute_component_scores_grouped,
    compute_lowset_grouped,
    compute_stage1_adjusted_scores_grouped,
    compute_stage2_upload_scores_grouped,
    compute_component_upload_cost_grouped,
    greedy_select_by_score_cost_ratio_grouped,
    slice_model_update_by_component_ids,
    compute_model_update_from_snapshot_grouped,
    apply_residual_to_update_grouped,
    update_residual_buffers_after_upload_grouped,
    compute_residual_norm_summary_grouped,
    prune_residual_buffers_grouped,
)

logger = logging.getLogger(__name__)


class AdaSLayerAwareClient(Client):
    """AdaS-LayerAware client with per-layer component identity (ComponentID = (layer_key, global_idx))."""
    
    METHOD_NAME = 'adas_layeraware'

    def _init_adas_layeraware(self):
        """
        Initialize AdaS-LayerAware attributes for layer-aware federated LoRA.
        
        layer-aware maintains layer-grouped state:
        - survivors_by_layer: Dict[layer_key, List[int]] - structural survivors per layer
        - upload_components_last: List[ComponentID] - uploaded components last round
        - download_components_last: List[ComponentID] - downloaded components last round
        - residual_buffers: Dict[ComponentID, Dict[str, Tensor]] - keyed by (layer_key, global_idx)
        """
        method = getattr(self._cfg.federate, 'method', '').lower()
    
        self.adas_layeraware_enabled = (
            method == 'adas_layeraware' and
            hasattr(self._cfg, 'llm') and
            hasattr(self._cfg.llm.adapter, 'adas_layeraware') and
            getattr(self._cfg.llm.adapter.adas_layeraware, 'enabled', False)
        )
    
        # Also check glue adapter config
        if not self.adas_layeraware_enabled:
            self.adas_layeraware_enabled = (
                method == 'adas_layeraware' and
                hasattr(self._cfg, 'glue') and
                hasattr(self._cfg.glue.adapter, 'adas_layeraware') and
                getattr(self._cfg.glue.adapter.adas_layeraware, 'enabled', False)
            )
    
        if not self.adas_layeraware_enabled:
            # Initialize empty state for disabled layer-aware
            self._init_disabled_layeraware_state()
            return

        # Whether this run is a GLUE task (used in the startup log)
        is_glue_task = (
            hasattr(self._cfg, 'data') and
            hasattr(self._cfg.data, 'type') and
            '@glue' in self._cfg.data.type.lower()
        )

        # Get adas_layeraware config
        layeraware_cfg = fs_common.get_adas_layeraware_cfg(self._cfg)
        if layeraware_cfg is None:
            self._init_disabled_layeraware_state()
            return
    
        # Store config values
        self.adas_layeraware_rank_min = getattr(layeraware_cfg, 'rank_min', 2)
        self.adas_layeraware_rank_max = getattr(layeraware_cfg, 'rank_max', 64)
        self.adas_layeraware_init_rank = getattr(layeraware_cfg, 'init_rank', 64)
    
        # Stage 1 config (structural pruning)
        stage1_cfg = getattr(layeraware_cfg, 'stage1', None)
        if stage1_cfg:
            self.adas_layeraware_gamma = getattr(stage1_cfg, 'gamma', 0.9)
            self.adas_layeraware_reg_weight = getattr(stage1_cfg, 'regularizer_weight', 0.01)
        else:
            self.adas_layeraware_gamma = 0.9
            self.adas_layeraware_reg_weight = 0.01
    
        # Stage 2 config (communication sparsity)
        stage2_cfg = getattr(layeraware_cfg, 'stage2', None)
        if stage2_cfg:
            self.adas_layeraware_stage2_enabled = getattr(stage2_cfg, 'enabled', True)
            self.adas_layeraware_q_up_bits = getattr(stage2_cfg, 'q_up_bits', 8)
            self.adas_layeraware_q_down_bits = getattr(stage2_cfg, 'q_down_bits', 8)
            self.adas_layeraware_cmeta_bits = getattr(stage2_cfg, 'cmeta_bits', 32)
            self.adas_layeraware_uplink_window = getattr(stage2_cfg, 'uplink_budget_window_s', 1.0)
            self.adas_layeraware_residual_enabled = getattr(stage2_cfg, 'residual_enabled', True)
        else:
            self.adas_layeraware_stage2_enabled = True
            self.adas_layeraware_q_up_bits = 8
            self.adas_layeraware_q_down_bits = 8
            self.adas_layeraware_cmeta_bits = 32
            self.adas_layeraware_uplink_window = 1.0
            self.adas_layeraware_residual_enabled = True
        
        # layer-aware-specific config
        self.adas_layeraware_stage1_global_competition = getattr(layeraware_cfg, 'stage1_global_competition', False)
        self.adas_layeraware_stage2_global_competition = getattr(layeraware_cfg, 'stage2_global_competition', False)
        
        # Stage 1 score-normalization / layer-importance prior
        self.adas_layeraware_stage1_norm_eps = 1e-12
        self.adas_layeraware_stage1_importance_strength = 1.0
        self.adas_layeraware_stage1_importance_beta = 0.2
        self.adas_layeraware_stage1_importance_min = 0.25
        self.adas_layeraware_stage1_importance_max = 1.0
        
        # Layer-importance state (updated after each pruning round)
        self.adas_layeraware_layer_importance: Dict[str, float] = {}
        
        # Initialize layer-aware layer-grouped state
        # Will be populated on first model access
        self.adas_layeraware_survivors_by_layer: Dict[str, List[int]] = {}
        self.adas_layeraware_survivor_components: List[ComponentID] = []
        self.adas_layeraware_upload_components_last: Optional[List[ComponentID]] = None
        self.adas_layeraware_download_components_last: Optional[List[ComponentID]] = None
        
        # Pre-round state for Stage 1 pruning
        self.adas_layeraware_low_candidates_before: Optional[List[ComponentID]] = None
        self.adas_layeraware_scores_before_low: Dict[ComponentID, float] = {}
        self.adas_layeraware_survivors_before_stage1: Optional[Dict[str, List[int]]] = None
        
        # Snapshot and residual state for Stage 2
        self.adas_layeraware_pre_round_lora_snapshot: Optional[dict] = None
        self.adas_layeraware_residual_buffers: Dict[ComponentID, Dict[str, torch.Tensor]] = {}
        
        # Bandwidth and budget state
        self.adas_layeraware_uplink_budget_last = None
        # Flag to track if we've initialized layer keys
        self.adas_layeraware_layer_keys_initialized = False

        # Startup log
        if bool(getattr(self._cfg, 'debug', False)):
            logger.debug(
                f"Client {self.ID} startup: "
                f"method=adas_layeraware, enabled={self.adas_layeraware_enabled}, "
                f"config_subtree={'glue' if is_glue_task else 'llm'}.adapter.adas_layeraware"
            )
        logger.info(
            f"Client {self.ID}: enabled, "
            f"init_rank={self.adas_layeraware_init_rank}, "
            f"rank_bounds=[{self.adas_layeraware_rank_min}, {self.adas_layeraware_rank_max}], "
            f"stage1_gamma={self.adas_layeraware_gamma}, stage1_reg_weight={self.adas_layeraware_reg_weight},"
            f"stage2_enabled={self.adas_layeraware_stage2_enabled}, "
            f"global_competition(stage1/stage2)={self.adas_layeraware_stage1_global_competition}/{self.adas_layeraware_stage2_global_competition}"
        )
    
    def _init_disabled_layeraware_state(self):
        """Initialize empty state for disabled layer-aware."""
        self.adas_layeraware_survivors_by_layer = {}
        self.adas_layeraware_survivor_components = []
        self.adas_layeraware_upload_components_last = None
        self.adas_layeraware_download_components_last = None
        self.adas_layeraware_low_candidates_before = None
        self.adas_layeraware_scores_before_low = {}
        self.adas_layeraware_survivors_before_stage1 = None
        self.adas_layeraware_pre_round_lora_snapshot = None
        self.adas_layeraware_residual_buffers = {}
        self.adas_layeraware_uplink_budget_last = None
        self.adas_layeraware_layer_keys_initialized = False
        # Stage 1 importance state (disabled defaults)
        self.adas_layeraware_stage1_norm_eps = 1e-12
        self.adas_layeraware_stage1_importance_strength = 1.0
        self.adas_layeraware_stage1_importance_beta = 0.2
        self.adas_layeraware_stage1_importance_min = 0.25
        self.adas_layeraware_stage1_importance_max = 1.0
        self.adas_layeraware_layer_importance = {}
    def _init_layer_keys_from_model(self):
        """
        Initialize layer keys and survivors from the model.
        Called lazily when model is first available.
        
        Each layer gets its own initial survivor set, sized by pattern matching
        the layer key against the client's per-layer rank config.
        """
        if self.adas_layeraware_layer_keys_initialized:
            return
        
        if not hasattr(self, 'trainer') or not hasattr(self.trainer, 'ctx'):
            return
        
        if not hasattr(self.trainer.ctx, 'model'):
            return
        
        try:
            state_dict = self.trainer.ctx.model.state_dict()
            layer_keys = infer_layer_keys_from_state_dict(state_dict)
            
            if not layer_keys:
                logger.warning(f"Client {self.ID}: No LoRA layers found in model")
                return
            
            # Get the full per-layer rank config for this client
            client_rank_config = self._get_client_rank_config_layeraware()
            default_rank = self.adas_layeraware_init_rank
            
            # Initialize survivors per layer using exact layer-aware config
            rank_summary = {}  # For logging
            for layer_key in layer_keys:
                # Determine rank for this specific layer using pattern matching
                layer_rank = self._get_rank_for_layer(layer_key, client_rank_config, default_rank)
                self.adas_layeraware_survivors_by_layer[layer_key] = list(range(layer_rank))
                
                # Initialize layer importance to 1.0 for new layers (do not reset existing)
                if layer_key not in self.adas_layeraware_layer_importance:
                    self.adas_layeraware_layer_importance[layer_key] = 1.0
                
                # Track for logging
                rank_summary[layer_rank] = rank_summary.get(layer_rank, 0) + 1
            
            # Build flattened ComponentID list
            self.adas_layeraware_survivor_components = flatten_grouped_indices_by_layer(
                self.adas_layeraware_survivors_by_layer
            )
            
            self.adas_layeraware_layer_keys_initialized = True
            
            # Log initialization with per-layer detail
            if bool(getattr(self._cfg, 'debug', False)):
                rank_dist_str = ", ".join(f"r{r}:{c}" for r, c in sorted(rank_summary.items()))
                logger.debug(
                    f"Client {self.ID}: Initialized layer-aware layer state with "
                    f"{len(layer_keys)} layers, rank_distribution=[{rank_dist_str}], "
                    f"total {len(self.adas_layeraware_survivor_components)} ComponentIDs"
                )
                
        except Exception as e:
            logger.warning(f"Client {self.ID}: Failed to initialize layer keys: {e}")
            import traceback
            logger.debug(traceback.format_exc())
    
    def _get_client_rank_config_layeraware(self) -> Optional[Dict[str, int]]:
        """
        Get the full per-layer rank config for this client.
        
        Returns the rank config dict mapping patterns to ranks, or None if not available.
        """
        config_local = fs_common.get_active_hetero_config_local(self._cfg)
        if not config_local:
            return None
        
        # Try both 1-indexed and 0-indexed client keys
        client_key_1indexed = f'Client_{self.ID}'
        client_key_0indexed = f'Client_{self.ID - 1}'
        
        if client_key_1indexed in config_local and config_local[client_key_1indexed]:
            return dict(config_local[client_key_1indexed])
        elif client_key_0indexed in config_local and config_local[client_key_0indexed]:
            return dict(config_local[client_key_0indexed])
        
        return None
    
    def _get_rank_for_layer(
        self, 
        layer_key: str, 
        rank_config: Optional[Dict[str, int]], 
        default_rank: int
    ) -> int:
        """
        Determine the initial rank for a specific layer using pattern matching.
        
        This uses the same longest-pattern matching as heterolora_utils._get_rank_for_param,
        so per-layer ranks are read from the same rank config format.
        
        Args:
            layer_key: Canonical layer key (e.g., 'model.layers.0.self_attn.q_proj')
            rank_config: Dict mapping patterns to ranks (e.g., {'q_proj': 8, 'v_proj': 16})
            default_rank: Default rank if no pattern matches
            
        Returns:
            The rank for this layer
        """
        if rank_config is None:
            return default_rank
        
        # Use longest matching pattern (same logic as _get_rank_for_param)
        best_len = -1
        best_rank = default_rank
        
        for pattern, rank in rank_config.items():
            if pattern in layer_key and len(pattern) > best_len:
                best_len = len(pattern)
                try:
                    best_rank = int(rank)
                except (TypeError, ValueError):
                    pass
        
        return best_rank

    def _adas_layeraware_log_round_start(self):
        """Log layer-aware state at the start of each round."""
        if not self.adas_layeraware_enabled:
            return
        
        self._init_layer_keys_from_model()
        
        n_layers = len(self.adas_layeraware_survivors_by_layer)
        n_components = len(self.adas_layeraware_survivor_components)
        n_upload = len(self.adas_layeraware_upload_components_last) if self.adas_layeraware_upload_components_last else 0
        n_download = len(self.adas_layeraware_download_components_last) if self.adas_layeraware_download_components_last else 0
        
        # Validate layer-aware state in debug mode
        if bool(getattr(self._cfg, 'debug', False)):
            from federatedscope.contrib.common.adas_layeraware_utils import validate_layeraware_state
            validate_layeraware_state(
                self.adas_layeraware_survivors_by_layer,
                self.adas_layeraware_survivor_components,
                self.adas_layeraware_upload_components_last,
                self.adas_layeraware_download_components_last,
                context=f"Client {self.ID} round start",
                raise_on_error=False
            )
        
        logger.info(
            f"Client {self.ID} round start: "
            f"n_layers={n_layers}, n_survivor_components={n_components}, "
            f"prior_upload_count={n_upload}, prior_download_count={n_download}"
        )

    def _adas_layeraware_save_pre_round_lora_snapshot(self):
        """Save a snapshot of current LoRA tensors before local training."""
        if not self.adas_layeraware_enabled:
            return
    
        try:
            model_state_dict = self.trainer.ctx.model.state_dict()
        
            # Extract, clone, and move to CPU for storage (reduces GPU memory pressure)
            lora_snapshot = {}
            sample_device = None
            sample_dtype = None
            for key, tensor in model_state_dict.items():
                if isinstance(tensor, torch.Tensor) and ('lora_A' in key or 'lora_B' in key):
                    lora_snapshot[key] = tensor.clone().cpu()
                    if sample_device is None:
                        sample_device = tensor.device
                        sample_dtype = tensor.dtype
        
            self.adas_layeraware_pre_round_lora_snapshot = lora_snapshot
        
            if bool(getattr(self._cfg, 'debug', False)):
                logger.debug(
                    f"Client {self.ID}: Saved layer-aware pre-round LoRA snapshot "
                    f"with {len(lora_snapshot)} tensors on CPU "
                    f"(source model was {sample_device}, {sample_dtype})"
                )
        
        except Exception as e:
            logger.error(f"Client {self.ID}: Failed to save pre-round LoRA snapshot: {e}")
            self.adas_layeraware_pre_round_lora_snapshot = None

    def _adas_layeraware_record_lowset_before(self):
        """
        Record low-set ComponentIDs and per-candidate scores before training.
        
        layer-aware: Scores are computed per ComponentID = (layer_key, global_idx),
        not collapsed across layers. Low-set selection uses adjusted scores
        (median-normalized + layer-importance prior).
        """
        if not self.adas_layeraware_enabled:
            return
        
        self._init_layer_keys_from_model()
    
        try:
            model = self.trainer.ctx.model
            
            # Compute raw per-ComponentID scores
            raw_scores = compute_component_scores_grouped(
                model, self.adas_layeraware_survivors_by_layer
            )
            
            if not raw_scores:
                self.adas_layeraware_low_candidates_before = []
                self.adas_layeraware_scores_before_low = {}
                self.adas_layeraware_survivors_before_stage1 = {
                    k: list(v) for k, v in self.adas_layeraware_survivors_by_layer.items()
                }
                return
            
            # Compute adjusted scores with layer normalization + importance prior
            adjusted_scores, normalized_scores, layer_medians = compute_stage1_adjusted_scores_grouped(
                raw_scores,
                layer_importance=self.adas_layeraware_layer_importance,
                eps=self.adas_layeraware_stage1_norm_eps,
                importance_strength=self.adas_layeraware_stage1_importance_strength,
            )
            
            # Compute low-set from ADJUSTED scores
            low_candidates, low_score_sum = compute_lowset_grouped(
                adjusted_scores,
                self.adas_layeraware_gamma,
                self.adas_layeraware_rank_min,
                global_competition=self.adas_layeraware_stage1_global_competition
            )
            
            # Store per-candidate baseline scores from RAW scores for prune comparison
            self.adas_layeraware_low_candidates_before = low_candidates
            self.adas_layeraware_scores_before_low = {
                cid: raw_scores[cid] for cid in low_candidates if cid in raw_scores
            }
            
            self.adas_layeraware_survivors_before_stage1 = {
                k: list(v) for k, v in self.adas_layeraware_survivors_by_layer.items()
            }
            
            # Log low-set info with both raw and adjusted scores
            if low_candidates:
                raw_vals = [raw_scores[cid] for cid in low_candidates if cid in raw_scores]
                adj_vals = [adjusted_scores[cid] for cid in low_candidates if cid in adjusted_scores]
                
                if raw_vals and adj_vals:
                    raw_min, raw_avg, raw_max = min(raw_vals), sum(raw_vals) / len(raw_vals), max(raw_vals)
                    adj_min, adj_avg, adj_max = min(adj_vals), sum(adj_vals) / len(adj_vals), max(adj_vals)
                    
                    if bool(getattr(self._cfg, 'debug', False)):
                        sorted_layers = sorted(self.adas_layeraware_layer_importance.keys())
                        sample_layers = sorted_layers[:5]

                        importance_sample = ", ".join(
                            f"{layer.split('.')[-1]}:{self.adas_layeraware_layer_importance.get(layer, 1.0):.3f}"
                            for layer in sample_layers
                        ) if sample_layers else "none"

                        median_sample = ", ".join(
                            f"{layer.split('.')[-1]}:{layer_medians.get(layer, 0.0):.6f}"
                            for layer in sample_layers
                        ) if sample_layers else "none"

                        median_values = list(layer_medians.values())
                        if median_values:
                            med_min = min(median_values)
                            med_avg = sum(median_values) / len(median_values)
                            med_max = max(median_values)
                            median_summary = f"{med_min:.6f}/{med_avg:.6f}/{med_max:.6f}"
                        else:
                            median_summary = "0.000000/0.000000/0.000000"

                        logger.debug(
                            f"Client {self.ID}: Stage1 layer-aware low_candidates={len(low_candidates)}, "
                            f"raw(min/avg/max)={raw_min:.4f}/{raw_avg:.4f}/{raw_max:.4f}, "
                            f"adjusted(min/avg/max)={adj_min:.4f}/{adj_avg:.4f}/{adj_max:.4f}, "
                            f"layer_medians(min/avg/max)={median_summary}, "
                            f"importance_sample=[{importance_sample}], "
                            f"median_sample=[{median_sample}]"
                        )
            
            # Set model attributes for trainer regularizer
            setattr(model, "adas_layeraware_low_candidates", low_candidates)
            setattr(model, "adas_layeraware_survivors_by_layer", self.adas_layeraware_survivors_by_layer)
        
        except Exception as e:
            logger.warning(f"Client {self.ID}: Failed to compute Stage1 layer-aware low-set: {e}")
            self.adas_layeraware_low_candidates_before = None
            self.adas_layeraware_scores_before_low = {}
            self.adas_layeraware_survivors_before_stage1 = None

    def _update_adas_layeraware_layer_importance_from_pruning(
        self,
        low_candidates: List[ComponentID],
        prune_candidates: List[ComponentID]
    ):
        """
        Update layer importance based on per-layer prune rate using EMA.
        
        If a layer produces a higher prune rate, its importance decreases.
        Lower importance means more prune pressure in future rounds.
        
        Args:
            low_candidates: List of low-set ComponentIDs before pruning
            prune_candidates: List of ComponentIDs that were actually pruned
        """
        low_by_layer = group_component_ids_by_layer(low_candidates)
        pruned_by_layer = group_component_ids_by_layer(prune_candidates)
        
        beta = self.adas_layeraware_stage1_importance_beta
        imp_min = self.adas_layeraware_stage1_importance_min
        imp_max = self.adas_layeraware_stage1_importance_max
        
        updated_layers = []
        
        for layer_key in self.adas_layeraware_survivors_by_layer.keys():
            prev_importance = self.adas_layeraware_layer_importance.get(layer_key, 1.0)
            low_count = len(low_by_layer.get(layer_key, []))
            pruned_count = len(pruned_by_layer.get(layer_key, []))
            
            if low_count <= 0:
                new_importance = prev_importance
            else:
                prune_rate = pruned_count / max(1, low_count)
                target_importance = 1.0 - prune_rate
                new_importance = (1 - beta) * prev_importance + beta * target_importance
                new_importance = max(imp_min, min(imp_max, new_importance))
            
            self.adas_layeraware_layer_importance[layer_key] = new_importance
            
            if prev_importance != new_importance:
                updated_layers.append((layer_key, low_count, pruned_count, prev_importance, new_importance))
        
        # Debug log: sample a few layers
        if bool(getattr(self._cfg, 'debug', False)) and updated_layers:
            sample = updated_layers[:3]
            for layer_key, lc, pc, prev, new in sample:
                pr = pc / max(1, lc) if lc > 0 else 0.0
                logger.debug(
                    f"Client {self.ID}: layer_importance '{layer_key}': "
                    f"low={lc}, pruned={pc}, rate={pr:.2f}, "
                    f"importance: {prev:.3f} -> {new:.3f}"
                )

    def _adas_layeraware_stage1_prune(self):
        """
        Perform Stage 1 structural pruning for layer-aware after training.
        
        Compares per-ComponentID scores before/after training.
        Prunes only components whose score decreased.
        """
        if not self.adas_layeraware_enabled:
            return self.adas_layeraware_survivors_by_layer
        
        if (self.adas_layeraware_low_candidates_before is None or 
            not hasattr(self, 'adas_layeraware_scores_before_low') or
            self.adas_layeraware_survivors_before_stage1 is None):
            logger.info(f"Client {self.ID}: Stage1 layer-aware no pruning (missing pre-training state)")
            return self.adas_layeraware_survivors_by_layer
    
        try:
            model = self.trainer.ctx.model
            low_candidates = self.adas_layeraware_low_candidates_before
            m = len(low_candidates)
            
            if m == 0:
                logger.info(
                    f"Client {self.ID}: Stage1 layer-aware no pruning (m=0)"
                )
                return self.adas_layeraware_survivors_by_layer
            
            # Compute scores after training
            scores_after = compute_component_scores_grouped(
                model, self.adas_layeraware_survivors_by_layer
            )
            
            if not scores_after:
                return self.adas_layeraware_survivors_by_layer
            
            # Per-candidate comparison: find candidates whose score decreased
            prune_candidates = []
            
            for cid in low_candidates:
                if cid not in scores_after:
                    continue
                before = self.adas_layeraware_scores_before_low.get(cid, 0.0)
                after = scores_after[cid]
                
                if after < before:
                    prune_candidates.append(cid)
            
            n_decreased = len(prune_candidates)
            
            # Enforce rank_min as a single minimum on the total number of survivors
            # across all layers (not a per-layer minimum). This holds in both modes:
            # stage1_global_competition only changes how the low-set is selected
            # (per layer or from one global pool).
            total_survivors = sum(len(v) for v in self.adas_layeraware_survivors_by_layer.values())
            max_prunable = max(0, total_survivors - self.adas_layeraware_rank_min)
            if len(prune_candidates) > max_prunable:
                prune_candidates = prune_candidates[:max_prunable]
            
            n_pruned = len(prune_candidates)
            
            if n_pruned == 0:
                # Still update importance even with no actual pruning
                self._update_adas_layeraware_layer_importance_from_pruning(
                    low_candidates,
                    prune_candidates
                )
                
                # Compute importance summary for logging
                imp_values = list(self.adas_layeraware_layer_importance.values())
                if imp_values:
                    imp_min, imp_avg, imp_max = min(imp_values), sum(imp_values) / len(imp_values), max(imp_values)
                    imp_summary = f", importance(min/avg/max)={imp_min:.2f}/{imp_avg:.2f}/{imp_max:.2f}"
                else:
                    imp_summary = ""
                
                logger.info(
                    f"Client {self.ID}: Stage1 layer-aware candidates={m}, "
                    f"decreased={n_decreased}, pruned=0{imp_summary}"
                )
                return self.adas_layeraware_survivors_by_layer
            
            # Perform pruning: remove pruned ComponentIDs from survivors
            pruned_set = set(prune_candidates)
            
            new_survivors_by_layer = {}
            for layer_key, indices in self.adas_layeraware_survivors_by_layer.items():
                new_indices = [
                    idx for idx in indices
                    if (layer_key, idx) not in pruned_set
                ]
                new_survivors_by_layer[layer_key] = sorted(new_indices)
            
            # Update state
            old_count = len(self.adas_layeraware_survivor_components)
            self.adas_layeraware_survivors_by_layer = new_survivors_by_layer
            self.adas_layeraware_survivor_components = flatten_grouped_indices_by_layer(new_survivors_by_layer)
            new_count = len(self.adas_layeraware_survivor_components)
            
            # Drop residual buffers of pruned components (debug mode logs how many).

            if bool(getattr(self._cfg, 'debug', False)):
                residuals = getattr(self, 'adas_layeraware_residual_buffers', {}) or {}
                removed_count = len(set(prune_candidates) & set(residuals.keys()))
                logger.debug(
                    f"Client {self.ID}: Removed residual buffers for pruned component {removed_count}/{len(residuals)} "
                )

            self.adas_layeraware_residual_buffers = prune_residual_buffers_grouped(
                self.adas_layeraware_residual_buffers, prune_candidates
            )
            
            # Update layer importance based on prune rates
            self._update_adas_layeraware_layer_importance_from_pruning(
                low_candidates,
                prune_candidates
            )
            
            # Compute importance summary for logging
            imp_values = list(self.adas_layeraware_layer_importance.values())
            if imp_values:
                imp_min, imp_avg, imp_max = min(imp_values), sum(imp_values) / len(imp_values), max(imp_values)
                imp_summary = f", importance(min/avg/max)={imp_min:.2f}/{imp_avg:.2f}/{imp_max:.2f}"
            else:
                imp_summary = ""
            
            logger.info(
                f"Client {self.ID}: Stage1 layer-aware candidates={m}, "
                f"decreased={n_decreased}, pruned={n_pruned}, "
                f"total_components: {old_count} -> {new_count}{imp_summary}"
            )
            
            return self.adas_layeraware_survivors_by_layer
        
        except Exception as e:
            logger.warning(f"Client {self.ID}: Stage1 layer-aware pruning failed: {e}")
            import traceback
            if bool(getattr(self._cfg, 'debug', False)):
                logger.debug(traceback.format_exc())
            return self.adas_layeraware_survivors_by_layer

    def _sync_adas_layeraware_state_from_message(self,
                                              survivor_indices=None,
                                              download_indices=None,
                                              bandwidth_info=None,
                                              client_rank_config=None):
        """
        Keep AdaS-LayerAware state aligned with server payloads.
        
        Index payloads are normalized with normalize_indices_to_grouped:
        - Grouped format: survivor_indices = {layer_key: [indices]}
        - Flat format: survivor_indices = [flat_list] - expanded to all layers
        """
        if not self.adas_layeraware_enabled:
            return
        
        self._init_layer_keys_from_model()
        
        # Get layer keys for flat->grouped expansion
        layer_keys = list(self.adas_layeraware_survivors_by_layer.keys())
        
        # Update structural survivor set
        if survivor_indices is not None:
            # Normalize to grouped indices (a flat list expands to all layers)
            normalized_survivors = normalize_indices_to_grouped(
                survivor_indices, layer_keys=layer_keys
            )
            
            if normalized_survivors:
                self.adas_layeraware_survivors_by_layer = {
                    k: list(v) for k, v in normalized_survivors.items()
                }
                self.adas_layeraware_survivor_components = flatten_grouped_indices_by_layer(
                    self.adas_layeraware_survivors_by_layer
                )
                
                logger.info(
                    f"Client {self.ID} received layer-aware survivor_indices: "
                    f"n_layers={len(self.adas_layeraware_survivors_by_layer)}, "
                    f"count={len(self.adas_layeraware_survivor_components)}"
                )
        
        # Update download indices
        if download_indices is not None:
            # Normalize to grouped indices (a flat list expands to all layers)
            normalized_download = normalize_indices_to_grouped(
                download_indices, layer_keys=layer_keys
            )
            
            if normalized_download:
                self.adas_layeraware_download_components_last = flatten_grouped_indices_by_layer(
                    normalized_download
                )
            
            if bool(getattr(self._cfg, 'debug', False)):
                logger.debug(
                    f"Client {self.ID} received layer-aware download_indices: "
                    f"count={len(self.adas_layeraware_download_components_last)}"
                )
        
        # Update bandwidth info for Stage 2 budgets
        if bandwidth_info is not None:
            ul_kbits = bandwidth_info.get('upload_kbits', 0)
            
            if 'uplink_budget_bits' in bandwidth_info:
                self.adas_layeraware_uplink_budget_last = bandwidth_info['uplink_budget_bits']
            else:
                self.adas_layeraware_uplink_budget_last = ul_kbits * self.adas_layeraware_uplink_window * 1000
        
        # Fall back to client_rank_config if survivor_indices not provided
        # Each layer's rank comes from pattern matching against client_rank_config
        if survivor_indices is None and client_rank_config is not None:
            try:
                for layer_key in list(self.adas_layeraware_survivors_by_layer.keys()):
                    # Use pattern matching to get rank for this specific layer
                    layer_rank = self._get_rank_for_layer(
                        layer_key, 
                        client_rank_config, 
                        self.adas_layeraware_init_rank
                    )
                    self.adas_layeraware_survivors_by_layer[layer_key] = list(range(layer_rank))
                self.adas_layeraware_survivor_components = flatten_grouped_indices_by_layer(
                    self.adas_layeraware_survivors_by_layer
                )
            except Exception:
                pass

    def _adas_layeraware_stage2_select_upload(self):
        """
        Perform Stage 2 upload selection for layer-aware.
        
        Returns:
            Tuple of (selected_components, upload_dict, upload_indices_grouped)
        """
        if not self.adas_layeraware_enabled:
            return None, None, None

        if self.adas_layeraware_pre_round_lora_snapshot is None:
            logger.warning(f"Client {self.ID}: layer-aware Stage 2 skipped (no snapshot)")
            return None, None, None

        # Stage 2 disabled (stage2.enabled=False): upload all Stage 1 survivors through
        # the normal aggregation path, with no residual accounting, no bandwidth budget
        # and no component selection.
        if not self.adas_layeraware_stage2_enabled:
            try:
                model_state = self.trainer.ctx.model.state_dict()
                delta_dict = compute_model_update_from_snapshot_grouped(
                    model_state,
                    self.adas_layeraware_pre_round_lora_snapshot,
                    self.adas_layeraware_survivors_by_layer,
                    cfg=self._cfg
                )
                # ALL survivors selected; upload_dict is the full survivor-set delta.
                selected_components = list(self.adas_layeraware_survivor_components)
                upload_dict = slice_model_update_by_component_ids(
                    delta_dict,
                    self.adas_layeraware_survivors_by_layer,
                    selected_components
                )
                upload_indices_grouped = group_component_ids_by_layer(selected_components)
                self.adas_layeraware_upload_components_last = selected_components
                # No residual buffers are touched (nothing is withheld => residual == 0).
                logger.info(
                    f"Client {self.ID}: layer-aware Stage 2 OFF -> uploading all "
                    f"{len(selected_components)} survivors (no budget/residual)"
                )
                return selected_components, upload_dict, upload_indices_grouped
            except Exception as e:
                logger.error(
                    f"Client {self.ID}: layer-aware stage-2-off upload failed: {e}")
                return None, None, None
        
        try:
            model_state = self.trainer.ctx.model.state_dict()
            
            # Compute model update from snapshot
            delta_dict = compute_model_update_from_snapshot_grouped(
                model_state,
                self.adas_layeraware_pre_round_lora_snapshot,
                self.adas_layeraware_survivors_by_layer,
                cfg=self._cfg
            )
            
            # Apply residuals to get effective update
            if self.adas_layeraware_residual_enabled and self.adas_layeraware_residual_buffers:
                effective_update = apply_residual_to_update_grouped(
                    delta_dict,
                    self.adas_layeraware_residual_buffers,
                    self.adas_layeraware_survivors_by_layer,
                    cfg=self._cfg
                )
            else:
                effective_update = delta_dict
            
            # Compute upload scores and costs
            upload_scores = compute_stage2_upload_scores_grouped(
                effective_update,
                self.adas_layeraware_survivors_by_layer
            )
            
            upload_costs = compute_component_upload_cost_grouped(
                effective_update,
                self.adas_layeraware_survivors_by_layer,
                q_bits=self.adas_layeraware_q_up_bits,
                cmeta_bits=self.adas_layeraware_cmeta_bits
            )
            
            # Select components under budget
            budget = self.adas_layeraware_uplink_budget_last if self.adas_layeraware_uplink_budget_last else float('inf')
            
            selected_components = greedy_select_by_score_cost_ratio_grouped(
                upload_scores,
                upload_costs,
                budget,
                global_competition=self.adas_layeraware_stage2_global_competition
            )
            
            if not selected_components:
                # No component fits the budget: upload all survivors
                selected_components = self.adas_layeraware_survivor_components
            
            # Slice updates to selected components
            upload_dict = slice_model_update_by_component_ids(
                effective_update,
                self.adas_layeraware_survivors_by_layer,
                selected_components
            )
            
            # Update residual buffers
            self.adas_layeraware_residual_buffers = update_residual_buffers_after_upload_grouped(
                self.adas_layeraware_residual_buffers,
                effective_update,
                selected_components,
                self.adas_layeraware_survivors_by_layer,
                cfg=self._cfg
            )

            # Log residual state summary.
            residual_summary = compute_residual_norm_summary_grouped(
                self.adas_layeraware_residual_buffers
            )
            if bool(getattr(self._cfg, 'debug', False)):
                logger.debug(
                    f"Client {self.ID}: residual state after upload: "
                    f"count={residual_summary.get('count', 0)}, "
                    f"total={residual_summary.get('total', 0.0):.6f}, "
                    f"avg={residual_summary.get('avg', 0.0):.6f}, "
                    f"max={residual_summary.get('max', 0.0):.6f}"
                )

            # Store for logging
            self.adas_layeraware_upload_components_last = selected_components

            # Convert to grouped format for wire
            upload_indices_grouped = group_component_ids_by_layer(selected_components)
            
            used_budget = sum(upload_costs.get(cid, 0) for cid in selected_components)
            budget_ratio = used_budget / budget if budget > 0 and budget != float('inf') else 0.0
            
            logger.info(
                f"Client {self.ID}: layer-aware Stage 2 upload: "
                f"survivors={len(self.adas_layeraware_survivor_components)}, "
                f"selected={len(selected_components)}, "
                f"budget_ratio={budget_ratio*100:.1f}%"
            )
            
            return selected_components, upload_dict, upload_indices_grouped
        
        except Exception as e:
            logger.error(f"Client {self.ID}: layer-aware Stage 2 upload selection failed: {e}")
            import traceback
            if bool(getattr(self._cfg, 'debug', False)):
                logger.debug(traceback.format_exc())
            return None, None, None
