"""Method-specific AdaS-PruneOnly server."""

import logging

import torch

import federatedscope.contrib.common as fs_common
from federatedscope.core.workers import Server
from federatedscope.core.message import Message

logger = logging.getLogger(__name__)


class AdaSPruneOnlyServer(Server):
    METHOD_NAME = 'adas_pruneonly'

    def _init_adas_pruneonly(self):
        adas_pruneonly_cfg = fs_common.get_adas_pruneonly_cfg(self._cfg)
        self.adas_pruneonly_enabled = adas_pruneonly_cfg is not None

        if not self.adas_pruneonly_enabled:
            self.adas_pruneonly_client_indices = {}
            self.adas_pruneonly_client_ranks = {}
            return

        logger.info("Initializing AdaS-PruneOnly server attributes")

        init_rank = getattr(adas_pruneonly_cfg, 'init_rank', 64)
        rank_min = getattr(adas_pruneonly_cfg, 'rank_min', 2)
        rank_max = getattr(adas_pruneonly_cfg, 'rank_max', 64)

        self.adas_pruneonly_client_indices = {}
        self.adas_pruneonly_client_ranks = {}

        for client_id in range(1, self._client_num + 1):
            self.adas_pruneonly_client_indices[client_id] = list(range(init_rank))
            self.adas_pruneonly_client_ranks[client_id] = init_rank

        config_local = fs_common.get_active_hetero_config_local(self._cfg)
        if config_local:
            for client_key, module_ranks in config_local.items():
                if 'Client_' not in str(client_key) or not module_ranks:
                    continue
                try:
                    client_id = int(str(client_key).split('_')[1])
                    rank = int(next(iter(module_ranks.values())))
                except Exception:
                    continue
                if 1 <= client_id <= self._client_num:
                    self.adas_pruneonly_client_ranks[client_id] = rank
                    self.adas_pruneonly_client_indices[client_id] = list(range(rank))

        rank_values = list(self.adas_pruneonly_client_ranks.values())
        avg_rank = sum(rank_values) / len(rank_values) if rank_values else 0.0
        distinct_ranks = sorted(set(rank_values))

        logger.info(
            f"Initialized client indices: "
            f"n_clients={len(self.adas_pruneonly_client_indices)}, "
            f"default_init_rank={init_rank}, rank_min={rank_min}, rank_max={rank_max}, "
            f"rank(min/avg/max)={min(rank_values) if rank_values else 0}/{avg_rank:.1f}/{max(rank_values) if rank_values else 0}, "
            f"distinct_ranks={distinct_ranks[:8]}{'...' if len(distinct_ranks) > 8 else ''}"
        )

    def _prepare_method_initial_payloads(self):
        if not self.adas_pruneonly_enabled:
            return False

        from federatedscope.contrib.common.adas_utils import distribute_weights_by_indices

        max_rank = fs_common.get_effective_max_rank(self._cfg)
        target_modules = fs_common.get_effective_target_modules(self._cfg)

        server_state = self.models[0].state_dict()
        server_lora_only = {
            k: v for k, v in server_state.items()
            if 'lora_A' in k or 'lora_B' in k
        }
        filtered_base = {
            k: v for k, v in server_state.items()
            if 'lora_A' not in k and 'lora_B' not in k
        }

        for client_id in range(1, self._client_num + 1):
            if client_id not in self.adas_pruneonly_client_indices:
                raise RuntimeError(
                    f"Missing initial indices for client {client_id} before the initial broadcast."
                )

            client_indices = list(self.adas_pruneonly_client_indices[client_id])
            if len(client_indices) == 0:
                raise RuntimeError(
                    f"Empty initial indices for client {client_id} before the initial broadcast."
                )

            client_rank = len(client_indices)
            client_lora_weights = distribute_weights_by_indices(
                server_lora_dict=server_lora_only,
                client_indices=client_indices,
                max_rank=max_rank,
                debug=bool(getattr(self._cfg, 'debug', False)),
            )

            self._initial_adas_pruneonly_payloads[client_id] = {
                'model_para': {**filtered_base, **client_lora_weights},
                'client_rank_config': {m: client_rank for m in target_modules},
                'adas_indices': client_indices,
            }
        return True

    def _postprocess_method_aggregated_result(self, aggregator, model, result):
        if not (hasattr(aggregator, '__class__') and
                aggregator.__class__.__name__ == 'AdaSPruneOnlyAggregator'):
            return result

        max_rank = fs_common.get_effective_max_rank(self._cfg)
        if bool(getattr(self._cfg, 'debug', False)):
            logger.debug(
            f"AdaSPruneOnlyAggregator: updating server model with "
            f"max_rank={max_rank} aggregated weights"
        )

        model_state = model.state_dict()
        for key in list(result.keys()):
            if key not in model_state:
                continue
            result_tensor = result[key]
            model_tensor = model_state[key]

            if not isinstance(result_tensor, torch.Tensor) or not isinstance(model_tensor, torch.Tensor):
                continue

            if result_tensor.shape == model_tensor.shape:
                continue

            if 'lora_A' in key and 'lora_B' not in key:
                if result_tensor.shape[0] < model_tensor.shape[0]:
                    padded = torch.zeros(
                        model_tensor.shape[0], result_tensor.shape[1],
                        dtype=result_tensor.dtype,
                        device=result_tensor.device
                    )
                    padded[:result_tensor.shape[0], :] = result_tensor
                    result[key] = padded
                elif result_tensor.shape[0] > model_tensor.shape[0]:
                    result[key] = result_tensor[:model_tensor.shape[0], :].clone()
            elif 'lora_B' in key:
                if result_tensor.shape[1] < model_tensor.shape[1]:
                    padded = torch.zeros(
                        result_tensor.shape[0], model_tensor.shape[1],
                        dtype=result_tensor.dtype,
                        device=result_tensor.device
                    )
                    padded[:, :result_tensor.shape[1]] = result_tensor
                    result[key] = padded
                elif result_tensor.shape[1] > model_tensor.shape[1]:
                    result[key] = result_tensor[:, :model_tensor.shape[1]].clone()
        return result

    def _broadcast_method_model_para(self, msg_type='model_para', receiver=None, rnd=0,
                                     skip_broadcast=False, filter_unseen_clients=True):
        use_adas_pruneonly = (
            self.adas_pruneonly_enabled and
            (msg_type == 'model_para' or msg_type == 'evaluate') and
            not skip_broadcast
        )
        if not use_adas_pruneonly:
            return False

        from federatedscope.contrib.common.adas_utils import distribute_weights_by_indices

        max_rank = fs_common.get_effective_max_rank(self._cfg)
        target_modules = fs_common.get_effective_target_modules(self._cfg)

        for client_id in receiver:
            if msg_type == 'model_para' and self.state == 0 and client_id in self._initial_adas_pruneonly_payloads:
                msg_content = self._initial_adas_pruneonly_payloads[client_id]
            else:
                if client_id not in self.adas_pruneonly_client_indices:
                    raise RuntimeError(
                        f"Missing indices for client {client_id} at round {self.state}. Refusing to broadcast a plain max-rank payload."
                    )

                client_indices = list(self.adas_pruneonly_client_indices[client_id])
                if len(client_indices) == 0:
                    raise RuntimeError(
                        f"Empty indices for client {client_id} at round {self.state}. Refusing to broadcast a plain max-rank payload."
                    )

                client_rank = len(client_indices)
                if bool(getattr(self._cfg, 'debug', False)):
                    logger.debug(
                        f"Broadcasting to client {client_id}: "
                        f"indices={client_indices[:5]}..., rank={client_rank}"
                    )

                server_state = self.models[0].state_dict()
                server_lora_only = {
                    k: v for k, v in server_state.items()
                    if 'lora_A' in k or 'lora_B' in k
                }
                filtered_base = {
                    k: v for k, v in server_state.items()
                    if 'lora_A' not in k and 'lora_B' not in k
                }

                client_lora_weights = distribute_weights_by_indices(
                    server_lora_dict=server_lora_only,
                    client_indices=client_indices,
                    max_rank=max_rank,
                    debug=bool(getattr(self._cfg, 'debug', False))
                )

                msg_content = {
                    'model_para': {**filtered_base, **client_lora_weights},
                    'client_rank_config': {m: client_rank for m in target_modules},
                    'adas_indices': client_indices,
                }

            client_model_para = msg_content['model_para']

            if self._cfg.quantization.method == 'uniform':
                from federatedscope.core.compression import symmetric_uniform_quantization
                client_model_para = symmetric_uniform_quantization(
                    client_model_para, self._cfg.quantization.nbits)

            send_content = {
                **{k: v for k, v in msg_content.items() if k != 'model_para'},
                'model_para': client_model_para,
            }

            self.comm_manager.send(
                Message(msg_type=msg_type,
                        sender=self.ID,
                        receiver=[client_id],
                        state=min(rnd, self.total_round_num),
                        timestamp=self.cur_timestamp,
                        content=send_content))

        if filter_unseen_clients:
            self.sampler.change_state(self.unseen_clients_id, 'seen')
        return True

    def _parse_method_model_para_content(self, sender, content):
        if not (self.adas_pruneonly_enabled and isinstance(content, (list, tuple)) and len(content) >= 3):
            return False, content

        sample_size = content[0]
        model_dict = content[1]
        indices_list = content[2]

        if self._cfg.quantization.method == 'uniform':
            from federatedscope.core.compression import symmetric_uniform_dequantization
            if isinstance(model_dict, list):
                model_dict = [symmetric_uniform_dequantization(x) for x in model_dict]
            else:
                model_dict = symmetric_uniform_dequantization(model_dict)

        if isinstance(indices_list, (list, tuple)):
            old_indices = self.adas_pruneonly_client_indices.get(sender, None)
            old_rank = self.adas_pruneonly_client_ranks.get(sender, None)
            self.adas_pruneonly_client_indices[sender] = list(indices_list)
            new_rank = len(indices_list)
            self.adas_pruneonly_client_ranks[sender] = new_rank

            if old_indices is not None:
                _new_set = set(indices_list)
                missing_prev_indices = [i for i in old_indices if i not in _new_set]
            else:
                missing_prev_indices = []

            if bool(getattr(self._cfg, 'debug', False)):
                logger.debug(
                    f"Received from client {sender}: "
                    f"indices={list(indices_list)[:5]}..., rank={new_rank}"
                )
            if missing_prev_indices:
                logger.info(
                    f"Missing since last aggregation for client {sender}: "
                    f"indices={missing_prev_indices[:]}, n_missing={len(missing_prev_indices)}"
                )
            else:
                logger.info(
                    f"Missing since last aggregation for client {sender}: none"
                )

            if old_rank is not None and old_rank != new_rank:
                if bool(getattr(self._cfg, 'debug', False)):
                    logger.debug(
                        f"Client {sender} rank changed: {old_rank} -> {new_rank}"
                    )

        return True, (sample_size, model_dict, list(indices_list))