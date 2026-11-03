"""Method-specific AdaS-LayerAware server.

This server implements layer-aware AdaS with:
- Per-layer survivor tracking for each client
- Grouped payload metadata on the wire
- Layer-aware downlink selection
- ComponentID-based routing

Key differences from AdaS:
- AdaS: flat shared component index across all layers
- layer-aware: exact layer-specific ComponentID for all operations
"""

import logging
import torch
from typing import Dict, List, Optional

import federatedscope.contrib.common as fs_common
from federatedscope.core.workers import Server
from federatedscope.core.message import Message
from federatedscope.contrib.common.adas_layeraware_utils import (
    ComponentID,
    infer_layer_keys_from_state_dict,
    get_lora_keys_for_layer,
    flatten_grouped_indices_by_layer,
    group_component_ids_by_layer,
    normalize_indices_to_grouped,
    compute_stage2_downlink_scores_grouped,
    compute_component_downlink_cost_grouped,
    greedy_select_by_score_cost_ratio_grouped,
    distribute_weights_by_layer_indices,
)

logger = logging.getLogger(__name__)


class AdaSLayerAwareServer(Server):
    """AdaS-LayerAware server with per-layer component identity (ComponentID)."""
    
    METHOD_NAME = 'adas_layeraware'

    def _init_adas_layeraware(self):
        """Initialize AdaS-LayerAware server attributes."""
        layeraware_cfg = fs_common.get_adas_layeraware_cfg(self._cfg)
        self.adas_layeraware_enabled = layeraware_cfg is not None

        if not self.adas_layeraware_enabled:
            self._init_disabled_layeraware_state()
            return

        logger.info("Initializing AdaS-LayerAware server attributes")

        init_rank = getattr(layeraware_cfg, 'init_rank', 64)
        rank_min = getattr(layeraware_cfg, 'rank_min', 2)
        rank_max = getattr(layeraware_cfg, 'rank_max', 64)

        stage2_cfg = getattr(layeraware_cfg, 'stage2', None)
        self.adas_layeraware_stage2_enabled = getattr(stage2_cfg, 'enabled', True) if stage2_cfg else True
        self.adas_layeraware_uplink_window_s = getattr(stage2_cfg, 'uplink_budget_window_s', 1.0) if stage2_cfg else 1.0
        self.adas_layeraware_downlink_window_s = getattr(stage2_cfg, 'downlink_budget_window_s', 1.0) if stage2_cfg else 1.0
        self.adas_layeraware_q_down_bits = getattr(stage2_cfg, 'q_down_bits', 8) if stage2_cfg else 8
        self.adas_layeraware_cmeta_bits = getattr(stage2_cfg, 'cmeta_bits', 32) if stage2_cfg else 32

        # layer-aware-specific config
        self.adas_layeraware_stage2_global_competition = getattr(layeraware_cfg, 'stage2_global_competition', False)

        agg_cfg = getattr(layeraware_cfg, 'aggregation', None)
        self.adas_layeraware_agg_mode = getattr(agg_cfg, 'mode', 'sample_size') if agg_cfg else 'sample_size'

        # layer-aware uses per-layer survivor tracking
        # client_survivors_by_layer[client_id] = {layer_key: [indices]}
        self.adas_layeraware_client_survivors_by_layer: Dict[int, Dict[str, List[int]]] = {}
        self.adas_layeraware_client_last_upload_components: Dict[int, Optional[List[ComponentID]]] = {}
        self.adas_layeraware_client_last_download_components: Dict[int, Optional[List[ComponentID]]] = {}

        # Per-client rank configs, used to initialize per-layer survivor state
        config_local = fs_common.get_active_hetero_config_local(self._cfg)
        
        # Layer keys will be populated on first model access
        self.adas_layeraware_layer_keys: List[str] = []
        self.adas_layeraware_init_rank = init_rank
        self.adas_layeraware_rank_min = rank_min
        self.adas_layeraware_rank_max = rank_max
        
        # Full per-client rank configs (pattern -> rank); once layer keys are known they
        # are used to initialize per-layer survivors
        self.adas_layeraware_client_rank_configs: Dict[int, Optional[Dict[str, int]]] = {}
        
        for client_id in range(1, self._client_num + 1):
            client_rank_config = None
            if config_local:
                client_key = fs_common.resolve_client_key(config_local, client_id)
                if client_key is not None and client_key in config_local:
                    # Store the full rank config (pattern -> rank)
                    client_rank_config = dict(config_local[client_key]) if config_local[client_key] else None
            
            self.adas_layeraware_client_rank_configs[client_id] = client_rank_config
            
            # Sentinel until layer keys are known; _init_layer_keys_from_model() replaces it
            # with per-layer survivors built from the client's rank config
            self.adas_layeraware_client_survivors_by_layer[client_id] = {"__pending_config__": True}
            self.adas_layeraware_client_last_upload_components[client_id] = None
            self.adas_layeraware_client_last_download_components[client_id] = None

        self.adas_layeraware_aggregated_global_updates = None
        self.adas_layeraware_layer_keys_initialized = False

        logger.info(
            f"Server startup: "
            f"method=adas_layeraware, enabled={self.adas_layeraware_enabled}, "
            f"config_subtree={'glue' if fs_common.is_glue_task(self._cfg) else 'llm'}.adapter.adas_layeraware. "
            f"Bandwidth via shared RoundBandwidthManager."
        )

        logger.info(
            f"Initialized server state: "
            f"n_clients={len(self.adas_layeraware_client_survivors_by_layer)}, "
            f"init_rank={init_rank}, rank_bounds=[{rank_min}, {rank_max}], "
            f"stage2_enabled={self.adas_layeraware_stage2_enabled}, "
            f"aggregation_mode={self.adas_layeraware_agg_mode}, "
            f"global_competition={self.adas_layeraware_stage2_global_competition}"
        )

    def _init_disabled_layeraware_state(self):
        """Initialize empty state for disabled layer-aware."""
        self.adas_layeraware_client_survivors_by_layer = {}
        self.adas_layeraware_client_last_upload_components = {}
        self.adas_layeraware_client_last_download_components = {}
        self.adas_layeraware_aggregated_global_updates = None
        self.adas_layeraware_layer_keys = []
        self.adas_layeraware_layer_keys_initialized = False

    def _init_layer_keys_from_model(self):
        """
        Initialize layer keys from the server model.
        
        Initializes each client's per-layer survivor state from its full rank config,
        resolving the rank of every layer separately.
        """
        if self.adas_layeraware_layer_keys_initialized:
            return
        
        if not hasattr(self, 'models') or not self.models:
            return
        
        try:
            state_dict = self.models[0].state_dict()
            self.adas_layeraware_layer_keys = infer_layer_keys_from_state_dict(state_dict)
            
            if not self.adas_layeraware_layer_keys:
                logger.warning("Server: No LoRA layers found in model")
                return
            
            # Initialize per-client per-layer survivors using exact layer-aware config
            for client_id in list(self.adas_layeraware_client_survivors_by_layer.keys()):
                current = self.adas_layeraware_client_survivors_by_layer[client_id]
                
                # Check if this client needs initialization
                if "__pending_config__" not in current:
                    continue
                
                # Get this client's full rank config
                client_rank_config = self.adas_layeraware_client_rank_configs.get(client_id)
                
                # Build per-layer survivors using pattern matching
                new_survivors = {}
                for layer_key in self.adas_layeraware_layer_keys:
                    layer_rank = self._get_rank_for_layer(
                        layer_key, client_rank_config, self.adas_layeraware_init_rank
                    )
                    new_survivors[layer_key] = list(range(layer_rank))
                
                self.adas_layeraware_client_survivors_by_layer[client_id] = new_survivors
            
            self.adas_layeraware_layer_keys_initialized = True
            
            # Log initialization summary
            total_components_per_client = []
            for client_id, survivors in self.adas_layeraware_client_survivors_by_layer.items():
                total = sum(len(indices) for indices in survivors.values())
                total_components_per_client.append(total)
            
            if total_components_per_client:
                min_c = min(total_components_per_client)
                max_c = max(total_components_per_client)
                avg_c = sum(total_components_per_client) / len(total_components_per_client)
                if bool(getattr(self._cfg, 'debug', False)):
                    logger.debug(
                        f"Server: Initialized layer-aware layer keys with {len(self.adas_layeraware_layer_keys)} layers, "
                        f"per-client total components (min/avg/max)={min_c}/{avg_c:.1f}/{max_c}"
                    )
            else:
                logger.info(
                    f"Server: Initialized layer-aware layer keys with {len(self.adas_layeraware_layer_keys)} layers"
                )
            
        except Exception as e:
            logger.warning(f"Server: Failed to initialize layer keys: {e}")
            import traceback
            logger.debug(traceback.format_exc())
    
    def _get_rank_for_layer(
        self, 
        layer_key: str, 
        rank_config: Optional[Dict[str, int]], 
        default_rank: int
    ) -> int:
        """
        Determine the initial rank for a specific layer using pattern matching.
        
        This mirrors the pattern-matching logic used in heterolora_utils._get_rank_for_param
        so per-layer ranks follow the same rank config format.
        
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

    def _build_per_layer_rank_config(
        self,
        survivors_by_layer: Dict[str, List[int]],
        target_modules: List[str]
    ) -> Dict[str, int]:
        """
        Build a per-layer rank config from exact layer-aware survivor state.
        
        Keys are exact layer keys only; no module-pattern entries are added.
        
        Args:
            survivors_by_layer: Dict mapping layer_key -> list of survivor indices
            target_modules: Unused; the rank config is keyed by exact layer keys
            
        Returns:
            Dict mapping exact layer keys to their survivor counts
        """
        if not survivors_by_layer:
            return {}
        
        # One entry per exact layer key: its survivor count
        rank_config = {}
        for layer_key, indices in survivors_by_layer.items():
            rank_config[layer_key] = len(indices)
        
        return rank_config

    def _adas_layeraware_log_round_start(self):
        """Log layer-aware state at the start of each round."""
        if not self.adas_layeraware_enabled:
            return
        
        self._init_layer_keys_from_model()

        survivor_counts = []
        for client_id, survivors_by_layer in self.adas_layeraware_client_survivors_by_layer.items():
            if "__pending_config__" in survivors_by_layer:
                # Not yet initialized - skip
                continue
            count = sum(len(indices) for indices in survivors_by_layer.values())
            survivor_counts.append((client_id, count))

        if survivor_counts:
            counts_only = [c for _, c in survivor_counts]
            min_c = min(counts_only)
            max_c = max(counts_only)
            avg_c = sum(counts_only) / len(counts_only)

            logger.info(
                f"Server round {self.state} start: "
                f"n_layers={len(self.adas_layeraware_layer_keys)}, "
                f"survivor_component_counts(min/avg/max)={min_c}/{avg_c:.1f}/{max_c}"
            )

    def _adas_layeraware_validate_upload_indices(self, client_id: int, upload_indices) -> bool:
        """Validate that upload indices are subset of survivors."""
        if not self.adas_layeraware_enabled or upload_indices is None:
            return True

        survivors_by_layer = self.adas_layeraware_client_survivors_by_layer.get(client_id, {})
        
        if isinstance(upload_indices, dict):
            # layer-aware grouped format
            for layer_key, indices in upload_indices.items():
                survivor_set = set(survivors_by_layer.get(layer_key, []))
                upload_set = set(indices)
                not_in_survivors = upload_set - survivor_set
                if not_in_survivors:
                    logger.warning(
                        f"Client {client_id} upload_indices for layer '{layer_key}' contain "
                        f"{len(not_in_survivors)} indices not in survivor set"
                    )
                    return False
        elif isinstance(upload_indices, (list, tuple)):
            # Flat format - validate against all layers
            for layer_key, survivor_indices in survivors_by_layer.items():
                if layer_key.startswith("__pending"):
                    continue
                survivor_set = set(survivor_indices)
                upload_set = set(upload_indices)
                not_in_survivors = upload_set - survivor_set
                if not_in_survivors:
                    logger.warning(
                        f"Client {client_id} flat upload_indices contain "
                        f"{len(not_in_survivors)} indices not in survivor set for layer '{layer_key}'"
                    )
                    return False
        
        return True

    def _postprocess_method_aggregated_result(self, aggregator, model, result):
        """Post-process aggregated result for layer-aware."""
        if not (hasattr(aggregator, '__class__') and
                aggregator.__class__.__name__ == 'AdaSLayerAwareAggregator'):
            return result

        max_rank = fs_common.get_effective_max_rank(self._cfg)
        if bool(getattr(self._cfg, 'debug', False)):
            logger.debug(
                f"[Server] AdaSLayerAwareAggregator: applying aggregated updates "
                f"with max_rank={max_rank}"
            )

        self.adas_layeraware_aggregated_global_updates = result

        # Synchronized task-head federation: REPLACE the model's classifier+pooler with the
        # sample-size ABSOLUTE average computed by the aggregator, BEFORE the base server
        # merges the LoRA-only deltas (which never touch these keys). Strict key-consumption.
        try:
            from federatedscope.contrib.common.head_federation import replace_model_head
            head_avg = getattr(aggregator, 'latest_head_average', None)
            if head_avg:
                n = replace_model_head(model, head_avg, strict=True)
                logger.info(f"[layer-aware-head] federated {n} task-head params (sample-size avg, absolute)")
            else:
                logger.warning("[layer-aware-head] no head average available -- task head NOT federated")
        except Exception as e:
            logger.warning(f"[layer-aware-head] head replace failed: {e}")

        n_updated_components = 0
        if hasattr(aggregator, 'get_latest_updated_components'):
            updated_comps = aggregator.get_latest_updated_components()
            n_updated_components = len(updated_comps) if updated_comps else 0

        model_state = model.state_dict()
        updated_result = {}

        for key in result.keys():
            if key not in model_state:
                continue

            result_tensor = result[key]
            model_tensor = model_state[key]

            if not isinstance(result_tensor, torch.Tensor) or not isinstance(model_tensor, torch.Tensor):
                continue

            if result_tensor.shape != model_tensor.shape:
                # Handle shape mismatch (same as AdaS)
                if 'lora_A' in key and 'lora_B' not in key:
                    if result_tensor.shape[0] < model_tensor.shape[0]:
                        padded = torch.zeros(
                            model_tensor.shape[0], result_tensor.shape[1],
                            dtype=result_tensor.dtype,
                            device=result_tensor.device
                        )
                        padded[:result_tensor.shape[0], :] = result_tensor
                        result_tensor = padded
                    elif result_tensor.shape[0] > model_tensor.shape[0]:
                        result_tensor = result_tensor[:model_tensor.shape[0], :].clone()
                elif 'lora_B' in key:
                    if result_tensor.shape[1] < model_tensor.shape[1]:
                        padded = torch.zeros(
                            result_tensor.shape[0], model_tensor.shape[1],
                            dtype=result_tensor.dtype,
                            device=result_tensor.device
                        )
                        padded[:, :result_tensor.shape[1]] = result_tensor
                        result_tensor = padded
                    elif result_tensor.shape[1] > model_tensor.shape[1]:
                        result_tensor = result_tensor[:, :model_tensor.shape[1]].clone()

            updated_result[key] = model_tensor + result_tensor.to(model_tensor.device)

        if bool(getattr(self._cfg, 'debug', False)):
            update_norms = [torch.norm(v).item() for v in result.values() if isinstance(v, torch.Tensor)]
            if update_norms:
                logger.debug(
                    f"Applied {len(updated_result)} parameter updates, "
                    f"n_components_updated={n_updated_components}"
                )

        return updated_result

    def _broadcast_method_model_para(self, msg_type='model_para', receiver=None, rnd=0,
                                     skip_broadcast=False, filter_unseen_clients=True):
        """Broadcast model parameters with layer-aware payloads."""
        use_adas_layeraware = (
            self.adas_layeraware_enabled and
            (msg_type == 'model_para' or msg_type == 'evaluate') and
            not skip_broadcast
        )
        if not use_adas_layeraware:
            return False

        self._adas_layeraware_log_round_start()
        self._init_layer_keys_from_model()

        max_rank = fs_common.get_effective_max_rank(self._cfg)
        target_modules = fs_common.get_effective_target_modules(self._cfg)

        download_counts = []
        download_budget_ratios = []

        for client_id in receiver:
            if client_id not in self.adas_layeraware_client_survivors_by_layer:
                logger.warning(f"Missing survivor indices for client {client_id}")
                continue

            client_survivors_by_layer = self.adas_layeraware_client_survivors_by_layer[client_id]
            
            # Skip if not properly initialized
            if "__pending_config__" in client_survivors_by_layer:
                logger.warning(f"Client {client_id} survivors not yet initialized with layer keys")
                continue
            
            total_survivors = sum(len(indices) for indices in client_survivors_by_layer.values())
            if total_survivors == 0:
                logger.warning(f"Empty survivor indices for client {client_id}")
                continue

            bandwidth_info = None
            downlink_budget = float('inf')

            # Get bandwidth from shared manager
            if hasattr(self, 'bandwidth_manager') and self.bandwidth_manager is not None:
                bandwidth_info = self.bandwidth_manager.get_bandwidth_info(client_id, self.state)
                if bandwidth_info:
                    dl_kbits = bandwidth_info.get('download_kbits', 50000.0)
                    downlink_window_s = getattr(self, 'adas_layeraware_downlink_window_s', 1.0)
                    downlink_budget = dl_kbits * downlink_window_s * 1000
                    bandwidth_info['downlink_budget_bits'] = downlink_budget
                    bandwidth_info['uplink_budget_bits'] = (
                        bandwidth_info.get('upload_kbits', 5000.0) *
                        getattr(self, 'adas_layeraware_uplink_window_s', 1.0) * 1000
                    )

            is_bootstrap_round = (self.state == 0)

            if is_bootstrap_round:
                # Bootstrap: download all survivor components
                download_components = flatten_grouped_indices_by_layer(client_survivors_by_layer)
                logger.info(
                    f"Client {client_id} bootstrap downlink (round 0): "
                    f"full refresh with n_components={len(download_components)}"
                )
            elif (self.adas_layeraware_stage2_enabled and
                  self.adas_layeraware_aggregated_global_updates is not None and
                  downlink_budget < float('inf')):
                # Stage 2 downlink selection
                downlink_scores = compute_stage2_downlink_scores_grouped(
                    self.adas_layeraware_aggregated_global_updates,
                    client_survivors_by_layer
                )
                downlink_costs = compute_component_downlink_cost_grouped(
                    self.adas_layeraware_aggregated_global_updates,
                    client_survivors_by_layer,
                    q_bits=self.adas_layeraware_q_down_bits,
                    cmeta_bits=self.adas_layeraware_cmeta_bits
                )
                download_components = greedy_select_by_score_cost_ratio_grouped(
                    downlink_scores,
                    downlink_costs,
                    downlink_budget,
                    global_competition=self.adas_layeraware_stage2_global_competition
                )
                
                used_budget = sum(downlink_costs.get(cid, 0) for cid in download_components)
                budget_ratio = used_budget / downlink_budget if downlink_budget > 0 else 0.0
                download_budget_ratios.append(budget_ratio)

                if bool(getattr(self._cfg, 'debug', False)):
                    logger.debug(
                        f"Client {client_id} Stage 2 layer-aware downlink selection: "
                        f"survivors={total_survivors}, selected={len(download_components)}, "
                        f"budget={downlink_budget:.0f}bits, used={used_budget:.0f}bits"
                    )
            elif self.adas_layeraware_aggregated_global_updates is None:
                download_components = []
                logger.warning(
                    f"Client {client_id} round {self.state}: "
                    f"aggregated global updates unavailable, using empty downlink"
                )
            else:
                download_components = flatten_grouped_indices_by_layer(client_survivors_by_layer)
                if bool(getattr(self._cfg, 'debug', False)):
                    logger.debug(
                        f"Client {client_id} full downlink: n_components={len(download_components)}"
                    )

            download_counts.append(len(download_components))

            # Store last download
            self.adas_layeraware_client_last_download_components[client_id] = download_components

            is_partial_downlink = (
                not is_bootstrap_round and
                len(download_components) < total_survivors
            )

            # Distribute weights
            server_state = self.models[0].state_dict()
            server_lora_only = {
                k: v for k, v in server_state.items()
                if 'lora_A' in k or 'lora_B' in k
            }
            
            client_model_para = distribute_weights_by_layer_indices(
                server_lora_only,
                client_survivors_by_layer,
                download_components,
                max_rank,
                debug=bool(getattr(self._cfg, 'debug', False))
            )

            # Synchronized task-head federation: broadcast the (now federated-averaged)
            # classifier+pooler so every client starts the round with the SAME head
            # (round-start tensor-equality). The client's apply passes non-LoRA keys through.
            try:
                from federatedscope.contrib.common.head_federation import head_keys_from_model
                head_state = {k: server_state[k]
                              for k in head_keys_from_model(self.models[0])
                              if k in server_state}
                client_model_para = {**client_model_para, **head_state}
            except Exception as e:
                logger.warning(f"[layer-aware-head] broadcast head failed: {e}")

            # Convert to grouped format for wire
            download_indices_grouped = group_component_ids_by_layer(download_components)

            # The payload is keyed by exact layer keys: grouped survivor_indices and
            # download_indices describe the client's components, and client_rank_config
            # maps each exact layer key to its survivor count (full downlinks only).
            client_rank_config_layeraware = None
            if not is_partial_downlink and client_survivors_by_layer:
                client_rank_config_layeraware = self._build_per_layer_rank_config(
                    client_survivors_by_layer, target_modules
                )
            
            msg_content = {
                'model_para': client_model_para,
                'client_rank_config': client_rank_config_layeraware,  # layer-aware: exact layer keys only
                'survivor_indices': client_survivors_by_layer,  # layer-aware: grouped dict (authoritative)
                'download_indices': download_indices_grouped,   # layer-aware: grouped dict
                'bandwidth_info': bandwidth_info,
                'is_partial_downlink': is_partial_downlink,
            }

            self.comm_manager.send(
                Message(msg_type=msg_type,
                        sender=self.ID,
                        receiver=[client_id],
                        state=min(rnd, self.total_round_num),
                        timestamp=self.cur_timestamp,
                        content=msg_content))

        if download_counts:
            avg_download = sum(download_counts) / len(download_counts)
            avg_budget_ratio = (
                sum(download_budget_ratios) / len(download_budget_ratios)
                if download_budget_ratios else 0.0
            )
            logger.info(
                f"Round {self.state} layer-aware broadcast: "
                f"n_clients={len(download_counts)}, "
                f"avg_download_count={avg_download:.1f}, "
                f"avg_budget_ratio={avg_budget_ratio*100:.1f}%"
            )

        if filter_unseen_clients:
            self.sampler.change_state(self.unseen_clients_id, 'seen')
        return True

    def _parse_method_model_para_content(self, sender, content):
        """Parse layer-aware model parameter content from client."""
        if not (self.adas_layeraware_enabled and isinstance(content, dict) and 'survivor_indices' in content):
            return False, content

        sample_size = content.get('sample_size', 0)
        model_dict = content.get('model_update_dict', {})
        upload_indices = content.get('upload_indices', {})
        survivor_indices = content.get('survivor_indices', {})

        if self._cfg.quantization.method == 'uniform':
            from federatedscope.core.compression import symmetric_uniform_dequantization
            if isinstance(model_dict, list):
                model_dict = [symmetric_uniform_dequantization(x) for x in model_dict]
            else:
                model_dict = symmetric_uniform_dequantization(model_dict)

        self._adas_layeraware_validate_upload_indices(sender, upload_indices)

        # Update client survivors (normalized to grouped indices)
        # This handles both grouped and flat formats consistently
        normalized_survivors = normalize_indices_to_grouped(
            survivor_indices, layer_keys=self.adas_layeraware_layer_keys
        )
        if normalized_survivors:
            self.adas_layeraware_client_survivors_by_layer[sender] = {
                k: list(v) for k, v in normalized_survivors.items()
            }
        
        # Store last upload (normalized to grouped indices)
        normalized_upload = normalize_indices_to_grouped(
            upload_indices, layer_keys=self.adas_layeraware_layer_keys
        )
        if normalized_upload:
            self.adas_layeraware_client_last_upload_components[sender] = flatten_grouped_indices_by_layer(
                normalized_upload
            )

        new_total = sum(
            len(indices) for indices in self.adas_layeraware_client_survivors_by_layer.get(sender, {}).values()
        )
        upload_total = len(self.adas_layeraware_client_last_upload_components.get(sender, []) or [])

        logger.info(
            f"Received from client {sender}: "
            f"total_survivor_components={new_total}, upload_count={upload_total}"
        )

        parsed = {
            'sample_size': sample_size,
            'model_update_dict': model_dict,
            'upload_indices': upload_indices,  # Keep original format for aggregator
            'survivor_indices': survivor_indices,
            # Synchronized head federation: carry the uploaded absolute head through to the
            # aggregator.
            'head_params': content.get('head_params', {}),
        }
        return True, parsed
