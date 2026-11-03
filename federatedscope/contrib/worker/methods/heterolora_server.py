"""HeteroLoRA server capability and worker classes.

The concrete worker path is selected by `federate.method`. HeteroLoRA
capability helpers are also reused underneath FAH-QLoRA and HetLoRA.
"""

import copy
import logging

import torch

import federatedscope.contrib.common as fs_common

from federatedscope.core.message import Message
from federatedscope.core.workers import Server

logger = logging.getLogger(__name__)


class HeteroLoRAServer(Server):
    METHOD_NAME = 'heterolora'

    def _build_personalized_hetero_payloads(
        self,
        config_local,
        receiver,
        rnd,
        include_fah_ranks=True,
    ):
        if config_local is None:
            return {}

        receiver = list(receiver) if receiver is not None else []
        if not receiver:
            return {}

        from federatedscope.contrib.common.heterolora_utils import distribute_weight_fast

        debug_mode = bool(getattr(self._cfg, 'debug', False))
        server_state = self.models[0].state_dict()
        server_lora_only = {
            k: v for k, v in server_state.items()
            if 'lora_A' in k or 'lora_B' in k
        }
        filtered_base = {
            k: v for k, v in server_state.items()
            if 'lora_A' not in k and 'lora_B' not in k
        }
        max_rank = fs_common.get_effective_max_rank(self._cfg)

        distributed_weights = distribute_weight_fast(
            weighted_single_weights=server_lora_only,
            config_local=config_local,
            max_rank=max_rank,
            debug=debug_mode,
        )

        payloads = {}
        for client_id in receiver:
            client_key = fs_common.resolve_client_key(config_local, client_id)
            if client_key is None:
                raise RuntimeError(
                    f"Missing hetero rank config for client {client_id} at round {rnd}."
                )

            client_lora_weights = distributed_weights.get(client_key, {})
            if not client_lora_weights:
                raise RuntimeError(
                    f"No distributed LoRA weights for {client_key} at round {rnd}."
                )

            msg_content = {
                'model_para': {**filtered_base, **client_lora_weights},
                'client_rank_config': copy.deepcopy(config_local.get(client_key, {})),
            }

            if include_fah_ranks and getattr(self, 'fah_enabled', False) and getattr(self, 'fah_client_ranks', None):
                fah_ranks = self.fah_client_ranks.get(client_id, None)
                if fah_ranks is not None:
                    msg_content['fah_ranks'] = fah_ranks

            bandwidth_info = self._get_client_bandwidth_info(client_id, rnd)
            if bandwidth_info is not None:
                msg_content['bandwidth_info'] = bandwidth_info

            payloads[client_id] = msg_content

        return payloads

    def _prepare_capability_initial_payloads(self):
        config_local = fs_common.get_active_hetero_config_local(self._cfg)
        if config_local is None:
            return False

        receiver = list(range(1, self._client_num + 1))
        self._initial_hetero_payloads = self._build_personalized_hetero_payloads(
            config_local=config_local,
            receiver=receiver,
            rnd=0,
            include_fah_ranks=True,
        )
        return True

    def _postprocess_capability_aggregated_result(self, aggregator, model, result):
        aggregator_mro_names = {
            cls.__name__ for cls in getattr(aggregator.__class__, '__mro__', ())
        }
        if not aggregator_mro_names.intersection({'HeteroLoRAAggregator', 'HetLoRAAggregator'}):
            return result

        max_rank = fs_common.get_effective_max_rank(self._cfg)
        debug_mode = bool(getattr(self._cfg, 'debug', False))
        if debug_mode:
            logger.debug(
                f"HeteroLoRAAggregator: updating server model with max_rank={max_rank} aggregated weights"
            )
        model_state = model.state_dict()
        shape_mismatches = []
        for key in result.keys():
            if key in model_state:
                result_shape = result[key].shape if isinstance(result[key], torch.Tensor) else None
                model_shape = model_state[key].shape if isinstance(model_state[key], torch.Tensor) else None
                if result_shape and model_shape and result_shape != model_shape:
                    shape_mismatches.append(f"{key}: result={result_shape}, model={model_shape}")
        if shape_mismatches:
            logger.error(
                f"Found {len(shape_mismatches)} shape mismatches before merge. "
                f"Fixing by zero-padding/truncating to match model shapes."
            )
            for key in list(result.keys()):
                if key not in model_state:
                    continue
                result_tensor = result[key]
                model_tensor = model_state[key]
                if (
                    not isinstance(result_tensor, torch.Tensor)
                    or not isinstance(model_tensor, torch.Tensor)
                    or result_tensor.shape == model_tensor.shape
                ):
                    continue
                if 'lora_A' in key and 'lora_B' not in key:
                    if result_tensor.shape[0] < model_tensor.shape[0]:
                        padded = torch.zeros(
                            model_tensor.shape[0],
                            result_tensor.shape[1],
                            dtype=result_tensor.dtype,
                            device=result_tensor.device,
                        )
                        padded[:result_tensor.shape[0], :] = result_tensor
                        result[key] = padded
                    elif result_tensor.shape[0] > model_tensor.shape[0]:
                        result[key] = result_tensor[:model_tensor.shape[0], :].clone()
                elif 'lora_B' in key:
                    if result_tensor.shape[1] < model_tensor.shape[1]:
                        padded = torch.zeros(
                            result_tensor.shape[0],
                            model_tensor.shape[1],
                            dtype=result_tensor.dtype,
                            device=result_tensor.device,
                        )
                        padded[:, :result_tensor.shape[1]] = result_tensor
                        result[key] = padded
                    elif result_tensor.shape[1] > model_tensor.shape[1]:
                        result[key] = result_tensor[:, :model_tensor.shape[1]].clone()
        return result

    def _update_hetero_ranks_config(self, r_per_client: dict):
        """
        Update hetero_ranks config with server-authoritative per-client ranks.

        This helper is capability-level, not FAH-specific. It is used by both
        FAH-QLoRA and HetLoRA rank updates.
        """
        target_modules = fs_common.get_effective_target_modules(self._cfg)
        current_config_local = fs_common.get_active_hetero_config_local(self._cfg) or {}

        updated_config_local = {}
        for key, value in current_config_local.items():
            if 'Client_' not in str(key):
                updated_config_local[key] = copy.deepcopy(value)

        for client_id, rank in r_per_client.items():
            rank = int(rank)

            client_key_1indexed = f'Client_{client_id}'
            client_key_0indexed = f'Client_{client_id - 1}'

            if client_key_1indexed in current_config_local:
                client_key = client_key_1indexed
            elif client_key_0indexed in current_config_local:
                client_key = client_key_0indexed
            else:
                client_key = client_key_1indexed

            updated_config_local[client_key] = {mod: rank for mod in target_modules}

        debug_mode = bool(getattr(self._cfg, 'debug', False))
        try:
            self._cfg.defrost()

            if hasattr(self._cfg, 'llm') and hasattr(self._cfg.llm, 'adapter') and                     hasattr(self._cfg.llm.adapter, 'hetero_ranks'):
                self._cfg.llm.adapter.hetero_ranks.config_local = copy.deepcopy(updated_config_local)

            if hasattr(self._cfg, 'glue') and hasattr(self._cfg.glue, 'adapter') and                     hasattr(self._cfg.glue.adapter, 'hetero_ranks'):
                self._cfg.glue.adapter.hetero_ranks.config_local = copy.deepcopy(updated_config_local)

            self._cfg.freeze(inform=False)

            if debug_mode:
                distinct_ranks = sorted(set(int(v) for v in r_per_client.values()))
                logger.debug(
                    f"Updated hetero_ranks config with "
                    f"{len(r_per_client)} client configs, distinct_ranks={distinct_ranks}"
                )
        except Exception as e:
            logger.warning(f"Failed to update config: {e}")


    def _broadcast_capability_model_para(
        self,
        msg_type='model_para',
        receiver=None,
        rnd=0,
        skip_broadcast=False,
        filter_unseen_clients=True,
    ):
        config_local = fs_common.get_active_hetero_config_local(self._cfg)
        if not ((msg_type == 'model_para' or msg_type == 'evaluate') and not skip_broadcast and config_local is not None):
            return False

        receiver = list(receiver) if receiver is not None else []

        if (
            msg_type == 'model_para'
            and self.state == 0
            and rnd == 0
            and receiver
            and all(client_id in self._initial_hetero_payloads for client_id in receiver)
        ):
            payloads = {client_id: self._initial_hetero_payloads[client_id] for client_id in receiver}
        else:
            payloads = self._build_personalized_hetero_payloads(
                config_local=config_local,
                receiver=receiver,
                rnd=rnd,
                include_fah_ranks=True,
            )

        for client_id in receiver:
            msg_content = payloads[client_id]
            client_model_para = msg_content['model_para']

            if self._cfg.quantization.method == 'uniform':
                from federatedscope.core.compression import symmetric_uniform_quantization
                client_model_para = symmetric_uniform_quantization(
                    client_model_para, self._cfg.quantization.nbits
                )

            send_content = {
                **{k: v for k, v in msg_content.items() if k != 'model_para'},
                'model_para': client_model_para,
            }
            self.comm_manager.send(
                Message(
                    msg_type=msg_type,
                    sender=self.ID,
                    receiver=[client_id],
                    state=min(rnd, self.total_round_num),
                    timestamp=self.cur_timestamp,
                    content=send_content,
                )
            )

        if filter_unseen_clients:
            self.sampler.change_state(self.unseen_clients_id, 'seen')
        return True