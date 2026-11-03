"""Method-specific AdaS server.

AdaS consumes bandwidth_info from the shared RoundBandwidthManager.
"""

import logging

import torch

import federatedscope.contrib.common as fs_common
from federatedscope.core.workers import Server
from federatedscope.core.message import Message

logger = logging.getLogger(__name__)


class AdaSServer(Server):
    METHOD_NAME = 'adas'

    def _init_adas(self):
        adas_method_cfg = fs_common.get_adas_cfg(self._cfg)
        self.adas_enabled = adas_method_cfg is not None

        if not self.adas_enabled:
            self.adas_client_survivors = {}
            self.adas_client_last_upload_indices = {}
            self.adas_client_last_download_indices = {}
            self.adas_aggregated_global_updates = None
            return

        logger.info("Initializing AdaS server attributes")

        init_rank = getattr(adas_method_cfg, 'init_rank', 64)
        rank_min = getattr(adas_method_cfg, 'rank_min', 2)
        rank_max = getattr(adas_method_cfg, 'rank_max', 64)

        stage2_cfg = getattr(adas_method_cfg, 'stage2', None)
        self.adas_stage2_enabled = getattr(stage2_cfg, 'enabled', True) if stage2_cfg else True
        
        # Store Stage 2 config for budget computation
        self.adas_uplink_window_s = getattr(stage2_cfg, 'uplink_budget_window_s', 1.0) if stage2_cfg else 1.0
        self.adas_downlink_window_s = getattr(stage2_cfg, 'downlink_budget_window_s', 1.0) if stage2_cfg else 1.0

        agg_cfg = getattr(adas_method_cfg, 'aggregation', None)
        self.adas_agg_mode = getattr(agg_cfg, 'mode', 'sample_size') if agg_cfg else 'sample_size'

        self.adas_client_survivors = {}
        self.adas_client_last_upload_indices = {}
        self.adas_client_last_download_indices = {}

        config_local = fs_common.get_active_hetero_config_local(self._cfg)
        for client_id in range(1, self._client_num + 1):
            client_init_rank = init_rank
            if config_local:
                client_key = fs_common.resolve_client_key(config_local, client_id)
                module_ranks = config_local.get(client_key, {}) if client_key is not None else {}
                if module_ranks:
                    try:
                        client_init_rank = int(next(iter(module_ranks.values())))
                    except Exception:
                        pass

            self.adas_client_survivors[client_id] = fs_common.indices_from_rank(client_init_rank)
            self.adas_client_last_upload_indices[client_id] = None
            self.adas_client_last_download_indices[client_id] = None

        self.adas_aggregated_global_updates = None

        survivor_counts = [len(v) for v in self.adas_client_survivors.values()]
        avg_survivors = sum(survivor_counts) / len(survivor_counts) if survivor_counts else 0.0

        logger.info(
            f"Server startup: "
            f"method=adas, enabled={self.adas_enabled}, "
            f"config_subtree={'glue' if fs_common.is_glue_task(self._cfg) else 'llm'}.adapter.adas. "
            f"Bandwidth via shared RoundBandwidthManager."
        )
        logger.info(
            f"Initialized server state: "
            f"n_clients={len(self.adas_client_survivors)}, "
            f"init_rank={init_rank}, rank_bounds=[{rank_min}, {rank_max}], "
            f"survivor_counts(min/avg/max)={min(survivor_counts) if survivor_counts else 0}/"
            f"{avg_survivors:.1f}/{max(survivor_counts) if survivor_counts else 0}, "
            f"stage2_enabled={self.adas_stage2_enabled}, "
            f"aggregation_mode={self.adas_agg_mode}"
        )

    def _adas_log_round_start(self):
        if not self.adas_enabled:
            return

        survivor_counts = []
        empty_survivor_clients = []

        for client_id, survivors in self.adas_client_survivors.items():
            count = len(survivors) if survivors else 0
            survivor_counts.append((client_id, count))
            if count == 0:
                empty_survivor_clients.append(client_id)

        survivor_counts_sorted = sorted(survivor_counts, key=lambda x: x[0])
        sample_str = ", ".join([f"C{cid}:{cnt}" for cid, cnt in survivor_counts_sorted[:6]])
        if len(survivor_counts_sorted) > 6:
            sample_str += "..."

        counts_only = [c for _, c in survivor_counts]
        min_c = min(counts_only) if counts_only else 0
        max_c = max(counts_only) if counts_only else 0
        avg_c = sum(counts_only) / len(counts_only) if counts_only else 0.0

        logger.info(
            f"Server round {self.state} start: "
            f"survivor_counts=[{sample_str}], "
            f"stats(min/avg/max)={min_c}/{avg_c:.1f}/{max_c}"
        )

        if empty_survivor_clients:
            logger.warning(
                f"Clients with empty survivor set: {empty_survivor_clients}"
            )

    def _adas_validate_upload_indices(self, client_id: int, upload_indices) -> bool:
        if not self.adas_enabled or upload_indices is None:
            return True

        survivors = self.adas_client_survivors.get(client_id, []) or []
        survivor_set = set(survivors)
        upload_set = set(upload_indices)
        not_in_survivors = upload_set - survivor_set
        if not_in_survivors:
            logger.warning(
                f"Client {client_id} upload_indices contain "
                f"{len(not_in_survivors)} indices not in survivor set: {list(not_in_survivors)[:5]}..."
            )
            return False
        return True

    def _postprocess_method_aggregated_result(self, aggregator, model, result):
        if not (hasattr(aggregator, '__class__') and
                aggregator.__class__.__name__ == 'AdaSAggregator'):
            return result

        max_rank = fs_common.get_effective_max_rank(self._cfg)
        if bool(getattr(self._cfg, 'debug', False)):
            logger.debug(
                f"[Server] AdaSAggregator: applying aggregated updates "
                f"with max_rank={max_rank}"
            )

        self.adas_aggregated_global_updates = result

        # Synchronized task-head federation: REPLACE the classifier+pooler with the sample-size
        # ABSOLUTE average before the LoRA-only delta merge (which never touches head keys).
        try:
            from federatedscope.contrib.common.head_federation import replace_model_head
            head_avg = getattr(aggregator, 'latest_head_average', None)
            if head_avg:
                n = replace_model_head(model, head_avg, strict=True)
                logger.info(f"[AdaS-head] federated {n} task-head params (sample-size avg, absolute)")
            else:
                logger.warning("[AdaS-head] no head average available -- task head NOT federated")
        except Exception as e:
            logger.warning(f"[AdaS-head] head replace failed: {e}")

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

        update_norms = [torch.norm(v).item() for v in result.values() if isinstance(v, torch.Tensor)]
        if update_norms and bool(getattr(self._cfg, 'debug', False)):
            logger.debug(
                f"Applied {len(updated_result)} parameter updates, "
                f"n_components_updated={n_updated_components}, "
                f"update_norms(min/avg/max)={min(update_norms):.4f}/"
                f"{sum(update_norms)/len(update_norms):.4f}/{max(update_norms):.4f}"
            )

        return updated_result

    def _broadcast_method_model_para(self, msg_type='model_para', receiver=None, rnd=0,
                                     skip_broadcast=False, filter_unseen_clients=True):
        use_adas = (
            self.adas_enabled and
            (msg_type == 'model_para' or msg_type == 'evaluate') and
            not skip_broadcast
        )
        if not use_adas:
            return False

        self._adas_log_round_start()

        from federatedscope.contrib.common.adas_utils import (
            distribute_weights_by_indices,
            compute_stage2_downlink_scores,
            compute_component_downlink_cost,
            greedy_select_by_score_cost_ratio
        )

        max_rank = fs_common.get_effective_max_rank(self._cfg)
        target_modules = fs_common.get_effective_target_modules(self._cfg)

        adas_method_cfg = fs_common.get_adas_cfg(self._cfg)
        stage2_enabled = self.adas_stage2_enabled if hasattr(self, 'adas_stage2_enabled') else True
        q_down_bits = 32
        cmeta_bits = 32

        if adas_method_cfg and hasattr(adas_method_cfg, 'stage2'):
            stage2_cfg = adas_method_cfg.stage2
            q_down_bits = getattr(stage2_cfg, 'q_down_bits', 32)
            cmeta_bits = getattr(stage2_cfg, 'cmeta_bits', 32)

        download_counts = []
        download_budget_ratios = []

        for client_id in receiver:
            if client_id not in self.adas_client_survivors:
                logger.warning(f"Missing survivor indices for client {client_id}")
                continue

            client_indices = list(self.adas_client_survivors[client_id])
            if len(client_indices) == 0:
                logger.warning(f"Empty survivor indices for client {client_id}")
                continue

            bandwidth_info = None
            downlink_budget = float('inf')

            # Get bandwidth from shared manager
            if hasattr(self, 'bandwidth_manager') and self.bandwidth_manager is not None:
                bandwidth_info = self.bandwidth_manager.get_bandwidth_info(client_id, self.state)
                if bandwidth_info:
                    # Compute downlink budget from bandwidth rate
                    dl_kbits = bandwidth_info.get('download_kbits', 50000.0)
                    downlink_window_s = getattr(self, 'adas_downlink_window_s', 1.0)
                    downlink_budget = dl_kbits * downlink_window_s * 1000  # kbit/s * s * 1000 = bits
                    bandwidth_info['downlink_budget_bits'] = downlink_budget
                    bandwidth_info['uplink_budget_bits'] = (
                        bandwidth_info.get('upload_kbits', 5000.0) *
                        getattr(self, 'adas_uplink_window_s', 1.0) * 1000
                    )

            is_bootstrap_round = (self.state == 0)

            if is_bootstrap_round:
                download_indices = client_indices
                logger.info(
                    f"Client {client_id} bootstrap downlink (round 0): "
                    f"full refresh with n_components={len(download_indices)}"
                )
            elif (stage2_enabled and
                  self.adas_aggregated_global_updates is not None and
                  downlink_budget < float('inf')):
                downlink_scores = compute_stage2_downlink_scores(
                    self.adas_aggregated_global_updates,
                    client_indices
                )
                downlink_costs = compute_component_downlink_cost(
                    self.adas_aggregated_global_updates,
                    client_indices,
                    q_bits=q_down_bits,
                    cmeta_bits=cmeta_bits
                )
                download_indices = greedy_select_by_score_cost_ratio(
                    scores=downlink_scores,
                    costs=downlink_costs,
                    budget=downlink_budget,
                    survivor_indices=client_indices
                )
                used_budget = sum(downlink_costs.get(idx, 0) for idx in download_indices)
                budget_ratio = used_budget / downlink_budget if downlink_budget > 0 else 0.0
                download_budget_ratios.append(budget_ratio)

                if bool(getattr(self._cfg, 'debug', False)):
                    logger.debug(
                    f"Client {client_id} Stage 2 downlink selection: "
                    f"survivors={len(client_indices)}, selected={len(download_indices)}, "
                    f"budget={downlink_budget:.0f}bits, used={used_budget:.0f}bits ({budget_ratio*100:.1f}%)"
                )
            elif self.adas_aggregated_global_updates is None:
                download_indices = []
                logger.warning(
                    f"Client {client_id} round {self.state}: "
                    f"aggregated global updates unavailable, using empty downlink (no refresh)"
                )
            else:
                download_indices = client_indices
                if bool(getattr(self._cfg, 'debug', False)):
                    logger.debug(
                    f"Client {client_id} full downlink (no budget constraint): "
                    f"n_components={len(download_indices)}"
                )

            download_counts.append(len(download_indices))

            if not set(download_indices).issubset(set(client_indices)):
                invalid_indices = set(download_indices) - set(client_indices)
                logger.warning(
                    f"Download selection includes {len(invalid_indices)} "
                    f"indices outside survivor set for client {client_id}"
                )
                download_indices = [idx for idx in download_indices if idx in client_indices]

            self.adas_client_last_download_indices[client_id] = list(download_indices)

            is_partial_downlink = (
                not is_bootstrap_round and
                len(download_indices) < len(client_indices)
            )

            server_state = self.models[0].state_dict()
            server_lora_only = {
                k: v for k, v in server_state.items()
                if 'lora_A' in k or 'lora_B' in k
            }
            client_model_para = distribute_weights_by_indices(
                server_lora_only, download_indices, max_rank, debug=False
            )

            # Synchronized task-head federation: broadcast the federated-averaged head so
            # every client starts the round with the SAME classifier+pooler (tensor-equality).
            try:
                from federatedscope.contrib.common.head_federation import head_keys_from_model
                head_state = {k: server_state[k]
                              for k in head_keys_from_model(self.models[0])
                              if k in server_state}
                client_model_para = {**client_model_para, **head_state}
            except Exception as e:
                logger.warning(f"[AdaS-head] broadcast head failed: {e}")

            client_rank = len(client_indices)
            msg_content = {
                'model_para': client_model_para,
                'client_rank_config': None if is_partial_downlink else {m: client_rank for m in target_modules},
                'survivor_indices': client_indices,
                'download_indices': download_indices,
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
                f"Round {self.state} broadcast: "
                f"n_clients={len(download_counts)}, "
                f"avg_download_count={avg_download:.1f}, "
                f"avg_budget_ratio={avg_budget_ratio*100:.1f}%"
            )

        if filter_unseen_clients:
            self.sampler.change_state(self.unseen_clients_id, 'seen')
        return True

    def _parse_method_model_para_content(self, sender, content):
        if not (self.adas_enabled and isinstance(content, dict) and 'survivor_indices' in content):
            return False, content

        sample_size = content.get('sample_size', 0)
        model_dict = content.get('model_update_dict', {})
        upload_indices = content.get('upload_indices', [])
        survivor_indices = content.get('survivor_indices', [])

        if self._cfg.quantization.method == 'uniform':
            from federatedscope.core.compression import symmetric_uniform_dequantization
            if isinstance(model_dict, list):
                model_dict = [symmetric_uniform_dequantization(x) for x in model_dict]
            else:
                model_dict = symmetric_uniform_dequantization(model_dict)

        self._adas_validate_upload_indices(sender, upload_indices)

        old_survivors = self.adas_client_survivors.get(sender, [])
        self.adas_client_survivors[sender] = list(survivor_indices)
        self.adas_client_last_upload_indices[sender] = list(upload_indices)

        new_survivor_count = len(survivor_indices)
        old_survivor_count = len(old_survivors) if old_survivors else 0
        upload_count = len(upload_indices)

        logger.info(
            f"Received from client {sender}: "
            f"survivor_count={new_survivor_count}, upload_count={upload_count}, "
            f"survivor_indices (sample)={survivor_indices[:5]}..."
        )

        if old_survivor_count != new_survivor_count:
            if bool(getattr(self._cfg, 'debug', False)):
                logger.debug(
                f"Client {sender} survivor count changed: "
                f"{old_survivor_count} -> {new_survivor_count}"
            )

        if new_survivor_count == 0:
            logger.warning(
                f"Client {sender} has empty survivor set after upload"
            )

        parsed = {
            'sample_size': sample_size,
            'model_update_dict': model_dict,
            'upload_indices': list(upload_indices),
            'survivor_indices': list(survivor_indices),
            # Synchronized head federation: carry the uploaded absolute head to the aggregator.
            'head_params': content.get('head_params', {}),
        }
        return True, parsed