"""FAH-QLoRA server mixin and server class.

This module contains the FAH-specific server logic (FahQLoRAServerMixin)
and defines the FAH-QLoRA server class.

FAH-QLoRA is selected with `federate.method: fah_qlora`; the aggregator
builder maps this method onto the shared HeteroLoRAAggregator.

FAH-QLoRA consumes bandwidth_info from the shared RoundBandwidthManager.
"""

import copy
import logging

import numpy as np

import federatedscope.contrib.common as fs_common
from federatedscope.core.message import Message

logger = logging.getLogger(__name__)


class FahQLoRAServerMixin:
    def _init_fah_qloRA(self):
        """Initialize FAH-QLoRA attributes if enabled."""
        self.fah_enabled = (
            hasattr(self._cfg, 'llm') and
            hasattr(self._cfg.llm.adapter, 'fah') and
            getattr(self._cfg.llm.adapter.fah, 'enabled', False)
        )
    
        if not self.fah_enabled:
            self.fah_scheduler = None
            self.fah_stats_buffer = {}
            self.fah_round_config_local = {}
            self._initial_fah_payloads = {}
            return
    
        from federatedscope.contrib.worker.methods.fah_rank_scheduler import FahRankScheduler
    
        fah_cfg = self._cfg.llm.adapter.fah
        adapter_cfg = self._cfg.llm.adapter
        max_rank_adapter = getattr(adapter_cfg, "max_rank", None)

        if max_rank_adapter is None:
            logger.warning(
                "llm.adapter.max_rank must be set when FAH-QLoRA is enabled."
            )

        if fah_cfg.r_max != max_rank_adapter:
            logger.warning(
                f"Mismatched r_max={fah_cfg.r_max} and adapter.max_rank={max_rank_adapter}. "
                f"Setting both to {max_rank_adapter} (FAH-QLoRA uses one global maximum rank)."
            )
            self._cfg.defrost()
            self._cfg.llm.adapter.fah.r_max = max_rank_adapter
            self._cfg.freeze(inform=False)
            fah_cfg = self._cfg.llm.adapter.fah

        if not (fah_cfg.r_min <= fah_cfg.init_rank <= fah_cfg.r_max):
            logger.warning(
                f"init_rank={fah_cfg.init_rank} not in "
                f"[r_min={fah_cfg.r_min}, r_max={fah_cfg.r_max}]."
            )

        # Immutable FAH capacity comes from runner-provided client caps / config, not from warmup.
        if not getattr(self, 'fah_client_rank_caps', None):
            self.fah_client_rank_caps = {}
        if not isinstance(getattr(self, 'fah_cap_config_local', None), dict) or not self.fah_cap_config_local:
            active_config_local = fs_common.get_active_hetero_config_local(self._cfg)
            self.fah_cap_config_local = copy.deepcopy(active_config_local) if active_config_local is not None else {}

        self.fah_scheduler = FahRankScheduler(
            init_rank=fah_cfg.init_rank,
            r_min=fah_cfg.r_min,
            r_max=fah_cfg.r_max,
            lambda_dec=fah_cfg.lambda_dec,
            lambda_inc=fah_cfg.lambda_inc,
            warmup_rounds=fah_cfg.warmup_rounds,
            alpha_fraction=fah_cfg.alpha_fraction,
            client_num=self._client_num,
            client_rank_caps=self.fah_client_rank_caps,
        )
    
        self.fah_stats_buffer = {}
        self.fah_global_loss_history = []
        self.fah_rank_history = []
        self.fah_client_ranks = {}
        self.fah_warmup_times = {}
        self.fah_round_config_local = {}
        self._initial_fah_payloads = {}

        for client_id in range(1, self._client_num + 1):
            client_cap = self.fah_scheduler._get_client_rank_cap(client_id)
            rank = min(fah_cfg.init_rank, client_cap)
            hat_rank = max(fah_cfg.r_min, min(client_cap, rank - 1))
            self.fah_client_ranks[client_id] = (rank, hat_rank)

        self.fah_round_config_local = self._build_fah_warmup_config_local()
    
        logger.info(
            f"Server: FAH-QLoRA enabled with init_rank={fah_cfg.init_rank}, "
            f"warmup={fah_cfg.warmup_rounds}, r_range=[{fah_cfg.r_min}, {fah_cfg.r_max}], "
            f"immutable_caps={self.fah_client_rank_caps}. "
            f"Bandwidth via shared RoundBandwidthManager."
        )

    def _get_fah_cap_config_local(self):
        if isinstance(getattr(self, 'fah_cap_config_local', None), dict) and self.fah_cap_config_local:
            return self.fah_cap_config_local

        active_config_local = fs_common.get_active_hetero_config_local(self._cfg)
        if isinstance(active_config_local, dict):
            return active_config_local
        return {}

    def _build_fah_config_local_from_rank_map(self, rank_map: dict):
        target_modules = fs_common.get_effective_target_modules(self._cfg)
        meta_source = self._get_fah_cap_config_local()

        config_local = {}
        for meta_key in ('alpha', 'lora_dropout'):
            if meta_key in meta_source:
                config_local[meta_key] = meta_source[meta_key]

        if not target_modules:
            return config_local

        fah_cfg = self._cfg.llm.adapter.fah
        for client_id in range(1, self._client_num + 1):
            client_cap = self.fah_scheduler._get_client_rank_cap(client_id) if self.fah_scheduler else self.fah_client_rank_caps.get(client_id, fah_cfg.r_max)
            rank = int(rank_map.get(client_id, fah_cfg.init_rank))
            rank = max(fah_cfg.r_min, min(client_cap, rank))
            config_local[f'Client_{client_id}'] = {
                mod: rank for mod in target_modules
            }

        return config_local

    def _build_fah_warmup_config_local(self):
        fah_cfg = self._cfg.llm.adapter.fah
        warmup_rank_map = {}
        for client_id in range(1, self._client_num + 1):
            client_cap = self.fah_scheduler._get_client_rank_cap(client_id) if self.fah_scheduler else self.fah_client_rank_caps.get(client_id, fah_cfg.r_max)
            warmup_rank_map[client_id] = min(fah_cfg.init_rank, client_cap)
        return self._build_fah_config_local_from_rank_map(warmup_rank_map)

    def _get_fah_runtime_config_local(self, round_idx: int):
        if self.fah_scheduler is not None and self.fah_scheduler.is_in_warmup(round_idx):
            return self._build_fah_warmup_config_local()

        if getattr(self, 'fah_round_config_local', None):
            return copy.deepcopy(self.fah_round_config_local)

        if getattr(self, 'fah_client_ranks', None):
            rank_map = {cid: pair[0] for cid, pair in self.fah_client_ranks.items()}
            if rank_map:
                return self._build_fah_config_local_from_rank_map(rank_map)

        return self._build_fah_warmup_config_local()

    def _prepare_method_initial_payloads(self):
        if not self.fah_enabled:
            return False

        receiver = list(range(1, self._client_num + 1))
        warmup_config_local = self._build_fah_warmup_config_local()
        self._initial_fah_payloads = self._build_personalized_hetero_payloads(
            config_local=warmup_config_local,
            receiver=receiver,
            rnd=0,
            include_fah_ranks=True,
        )
        return True

    def _broadcast_method_model_para(
        self,
        msg_type='model_para',
        receiver=None,
        rnd=0,
        skip_broadcast=False,
        filter_unseen_clients=True,
    ):
        if not self.fah_enabled:
            return False

        if not ((msg_type == 'model_para' or msg_type == 'evaluate') and not skip_broadcast):
            return False

        receiver = list(receiver) if receiver is not None else []
        if not receiver:
            return True

        if (
            msg_type == 'model_para'
            and self.state == 0
            and rnd == 0
            and receiver
            and all(client_id in self._initial_fah_payloads for client_id in receiver)
        ):
            payloads = {client_id: self._initial_fah_payloads[client_id] for client_id in receiver}
        else:
            config_local = self._get_fah_runtime_config_local(rnd)
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

    def _process_fah_round(self, round_idx: int):
        """
        Process FAH statistics after receiving from all clients.
    
        This implements the FAH-QLoRA rank adaptation logic (Algorithm 1):
        1. Aggregate loss statistics from clients
        2. Compute round times with _compute_client_time (equations 12-14): T_i from the
           measured training time plus analytical communication time, T̂_i analytically
        3. Update average rank via gradient sign (Stage 1, equations 10-11)
        4. Solve P1 for per-client ranks (Stage 2)
        5. Update hetero_ranks config for next round
    
        Note: T_i uses the measured client training time plus the analytical
        communication time of equations (13)-(14); T̂_i is fully analytical.
        Bandwidth values are read from the shared RoundBandwidthManager.
        """
        if not self.fah_enabled or self.fah_scheduler is None:
            return
    
        stats = self.fah_stats_buffer.get(round_idx, {})
        if not stats:
            return
    
        local_loss_pairs = {}
    
        for client_id, client_stats in stats.items():
            F_n = client_stats.get('F_n', 0.0)
            F_hat_n = client_stats.get('F_hat_n', 0.0)
            local_loss_pairs[client_id] = (F_n, F_hat_n)
        
            if self.fah_scheduler.is_in_warmup(round_idx):
                training_time = client_stats.get('training_time', 1.0)
                if client_id not in self.fah_warmup_times:
                    self.fah_warmup_times[client_id] = []
                self.fah_warmup_times[client_id].append(training_time)
    
        if self.fah_scheduler.is_in_warmup(round_idx):
            # Sync scheduler profiles from shared bandwidth manager
            self._sync_fah_bandwidth_from_manager(round_idx)
            
            if local_loss_pairs:
                current_F = np.mean([p[0] for p in local_loss_pairs.values()])
                self.fah_global_loss_history.append(current_F)
        
            if round_idx == self.fah_scheduler.warmup_rounds - 1:
                self._finalize_fah_profiling()
        else:
            previous_global_loss = None
            if len(self.fah_global_loss_history) >= 1:
                previous_global_loss = self.fah_global_loss_history[-1]
        
            time_pairs = {}
            for client_id, client_stats in stats.items():
                training_time = client_stats.get("training_time", None)
                if training_time is None:
                    logger.warning(f"Round {round_idx} client {client_id} had None training_time.")
            
                t_cmp = float(training_time) if training_time else 1.0
                time_pairs[client_id] = (t_cmp, 0.0, t_cmp, 0.0)
        
            # Sync scheduler profiles from shared bandwidth manager
            self._sync_fah_bandwidth_from_manager(round_idx)

            self.fah_scheduler.update_round_stats(
                round_idx=round_idx,
                local_loss_pairs=local_loss_pairs,
                time_pairs=time_pairs,
                previous_global_loss=previous_global_loss
            )
        
            if local_loss_pairs:
                current_F = np.mean([p[0] for p in local_loss_pairs.values()])
                self.fah_global_loss_history.append(current_F)
    
        if not self.fah_scheduler.is_in_warmup(round_idx):
            r_i, r_hat_i = self.fah_scheduler.update_average_rank(round_idx)
        
            r_per_client, r_hat_per_client = self.fah_scheduler.solve_p1_for_round(
                round_idx, r_i
            )
        
            self.fah_client_ranks = {
                cid: (r_per_client[cid], r_hat_per_client[cid])
                for cid in r_per_client
            }
        
            self.fah_round_config_local = self._build_fah_config_local_from_rank_map(r_per_client)

            self.fah_rank_history.append(r_i)
        
            rank_dict = {
                "c"+str(cid): r for cid, r in zip(r_per_client, r_per_client.values())
            }
            logger.info(f"Round {round_idx} complete: avg_rank={r_i:.1f}, "
                        f"client_ranks={rank_dict}"
            )

    def _finalize_fah_profiling(self):
        """
        Finalize client profiling after warmup rounds.
    
        This method:
        1. Computes LoRA size parameters (L0_bytes, unit_lora_bytes)
        2. Estimates alpha_n and t_lora_n from measured warmup training times
        3. Gets bandwidth from the shared RoundBandwidthManager
        4. Registers all client profiles with the FAH scheduler

        """
        if not self.fah_enabled or self.fah_scheduler is None:
            return
    
        fah_cfg = self._cfg.llm.adapter.fah
        init_rank = fah_cfg.init_rank
        r_max = fah_cfg.r_max
        alpha_fraction = fah_cfg.alpha_fraction
    
        # Step 1: Compute LoRA sizes from model
        if hasattr(self, 'models') and self.models:
            from federatedscope.contrib.common.heterolora_utils import compute_lora_size
            try:
                L0_bytes, unit_lora_bytes = compute_lora_size(self.models[0], r_max)
                self.fah_scheduler.set_lora_size(L0_bytes, unit_lora_bytes)
            except Exception as e:
                logger.warning(f"Failed to compute LoRA size: {e}")
                self.fah_scheduler.set_lora_size(50 * 1e6, 1.5 * 1e6)
    
        # Step 2: Register clients with profiled data using shared bandwidth manager
        for client_id, times in self.fah_warmup_times.items():
            if times:
                T_train_n = np.mean(times)
                alpha_n = alpha_fraction * T_train_n
                t_lora_n = (T_train_n - alpha_n) * (r_max / init_rank)
            
                # Get bandwidth from the shared RoundBandwidthManager
                b_up_n, b_dn_n, b_up_units, b_dn_units = self._get_client_bandwidth_for_scheduler(client_id)
            
                self.fah_scheduler.register_client(
                    client_id=client_id,
                    alpha_n=alpha_n,
                    t_lora_n=t_lora_n,
                    b_up_n=b_up_n,
                    b_dn_n=b_dn_n,
                    b_up_units=b_up_units,
                    b_dn_units=b_dn_units
                )
            
                if bool(getattr(self._cfg, 'debug', False)):
                    logger.debug(
                        f"Client {client_id} profile: T_train={T_train_n:.2f}s, "
                        f"t_lora={t_lora_n:.2f}s, b_up={b_up_n:.2f}{b_up_units}, b_dn={b_dn_n:.2f}{b_dn_units}."
                    )
    
        logger.info(
            f"Profiling complete: {len(self.fah_warmup_times)} clients registered."
        )

        # Prepare the first post-warmup personalized ranks immediately after profiling.
        # Warmup itself is homogeneous; the first broadcast after warmup uses these ranks.
        if self.fah_scheduler.client_profiles:
            seed_avg_rank = self.fah_scheduler.get_current_average_rank()
            r_per_client, r_hat_per_client = self.fah_scheduler.solve_p1_for_round(
                round_idx=self.fah_scheduler.warmup_rounds,
                r_i=seed_avg_rank,
            )
            self.fah_client_ranks = {
                cid: (r_per_client[cid], r_hat_per_client[cid])
                for cid in r_per_client
            }
            self.fah_round_config_local = self._build_fah_config_local_from_rank_map(r_per_client)
            logger.info(
                f"Prepared first post-warmup FAH personalized ranks at avg={seed_avg_rank:.2f}: "
                f"{r_per_client}"
            )
    
    def _get_client_bandwidth_for_scheduler(self, client_id: int):
        """
        Get bandwidth values for FAH scheduler profile registration.

        FAH profiling reads bandwidth from the shared RoundBandwidthManager.

        Returns:
            Tuple of (b_up_n, b_dn_n, b_up_units, b_dn_units)
        """
        if not hasattr(self, 'bandwidth_manager') or self.bandwidth_manager is None:
            raise RuntimeError(
                "FAH-QLoRA profiling requires the shared RoundBandwidthManager. "
                "Enable federate.communication and initialize the shared server "
                "bandwidth manager before FAH profiling."
            )

        bw_info = self.bandwidth_manager.get_bandwidth_info(client_id)
        if not bw_info:
            raise RuntimeError(
                f"Missing shared bandwidth info for client {client_id} during "
                "FAH-QLoRA profiling."
            )

        return (
            bw_info['upload_kbits'],
            bw_info['download_kbits'],
            'kbit/s',
            'kbit/s',
        )

    def _sync_fah_bandwidth_from_manager(self, round_idx: int):
        """
        Sync FAH scheduler profiles with current bandwidth from shared manager.
        
        This updates the scheduler's client profiles with the latest bandwidth
        values from the shared RoundBandwidthManager.
        """
        if not self.fah_enabled or self.fah_scheduler is None:
            return
        
        if not hasattr(self, 'bandwidth_manager') or self.bandwidth_manager is None:
            return
        
        for client_id in self.fah_scheduler.client_profiles:
            bw_info = self.bandwidth_manager.get_bandwidth_info(client_id, round_idx)
            if bw_info:
                profile = self.fah_scheduler.client_profiles[client_id]
                profile['b_up'] = bw_info.get('upload_kbits', profile.get('b_up', 5000.0))
                profile['b_dn'] = bw_info.get('download_kbits', profile.get('b_dn', 50000.0))
                profile['b_up_units'] = 'kbit/s'
                profile['b_dn_units'] = 'kbit/s'

    def get_fah_client_ranks(self, client_id: int) -> tuple:
        """
        Get FAH rank pair for a specific client.
    
        Returns:
            Tuple (r_i^n, r̂_i^n) or None if not using FAH
        """
        if not self.fah_enabled:
            return None
        return self.fah_client_ranks.get(client_id, None)


from federatedscope.contrib.worker.methods.heterolora_server import HeteroLoRAServer


class FahQLoRAServer(FahQLoRAServerMixin, HeteroLoRAServer):
    METHOD_NAME = 'fah_qlora'
