"""Method-specific AdaS client.

Implements AdaS initialization, Stage 1 structural pruning and AdaS state
synchronization for the ``adas`` method on top of the shared Client lifecycle.

Bandwidth info is received from the shared RoundBandwidthManager via server messages.
"""

import logging
import torch

import federatedscope.contrib.common as fs_common
from federatedscope.core.workers import Client

logger = logging.getLogger(__name__)


class AdaSClient(Client):
    METHOD_NAME = 'adas'

    def _init_adas(self):
        """
        Initialize AdaS attributes for two-stage sparse federated LoRA.
    
        Stage 1: Structural sparsity over survivor set (low-set pruning)
        Stage 2: Communication sparsity via residual-aware budgeted selection
    
        AdaS maintains three distinct index sets:
        - survivor_indices: components that survived Stage 1 pruning
        - upload_indices: subset of survivors selected for upload this round
        - download_indices: subset of survivors selected for download this round
        """
        method = getattr(self._cfg.federate, 'method', '').lower()
    
        self.adas_enabled = (
            method == 'adas' and
            hasattr(self._cfg, 'llm') and
            hasattr(self._cfg.llm.adapter, 'adas') and
            getattr(self._cfg.llm.adapter.adas, 'enabled', False)
        )
    
        # Also check glue adapter config
        if not self.adas_enabled:
            self.adas_enabled = (
                method == 'adas' and
                hasattr(self._cfg, 'glue') and
                hasattr(self._cfg.glue.adapter, 'adas') and
                getattr(self._cfg.glue.adapter.adas, 'enabled', False)
            )
    
        if not self.adas_enabled:
            # Initialize empty state for disabled AdaS
            self.adas_survivor_indices_current = None
            self.adas_upload_indices_last = None
            self.adas_download_indices_last = None
            self.adas_indices_before_stage1 = None
            self.adas_pre_round_lora_snapshot = None
            self.adas_uplink_budget_last = None
            self.adas_downlink_budget_last = None
            self.adas_residual_buffers = None
            return
    
        # Get adas config
        is_glue_task = (
            hasattr(self._cfg, 'data') and
            hasattr(self._cfg.data, 'type') and
            '@glue' in self._cfg.data.type.lower()
        )
        if is_glue_task and hasattr(self._cfg, 'glue') and hasattr(self._cfg.glue.adapter, 'adas'):
            adas_method_cfg = self._cfg.glue.adapter.adas
        elif hasattr(self._cfg, 'llm') and hasattr(self._cfg.llm.adapter, 'adas'):
            adas_method_cfg = self._cfg.llm.adapter.adas
        else:
            adas_method_cfg = self._cfg.glue.adapter.adas
    
        # Store config values
        self.adas_rank_min = getattr(adas_method_cfg, 'rank_min', 2)
        self.adas_rank_max = getattr(adas_method_cfg, 'rank_max', 64)
        self.adas_init_rank = getattr(adas_method_cfg, 'init_rank', 64)
    
        # Stage 1 config (structural pruning)
        stage1_cfg = getattr(adas_method_cfg, 'stage1', None)
        if stage1_cfg:
            self.adas_gamma = getattr(stage1_cfg, 'gamma', 0.9)
            self.adas_reg_weight = getattr(stage1_cfg, 'regularizer_weight', 0.01)
        else:
            self.adas_gamma = 0.9
            self.adas_reg_weight = 0.01
    
        # Stage 2 config (communication sparsity)
        stage2_cfg = getattr(adas_method_cfg, 'stage2', None)
        if stage2_cfg:
            self.adas_stage2_enabled = getattr(stage2_cfg, 'enabled', True)
            self.adas_q_up_bits = getattr(stage2_cfg, 'q_up_bits', 8)
            self.adas_q_down_bits = getattr(stage2_cfg, 'q_down_bits', 32)
            self.adas_cmeta_bits = getattr(stage2_cfg, 'cmeta_bits', 32)
            self.adas_uplink_window = getattr(stage2_cfg, 'uplink_budget_window_s', 1.0)
            self.adas_downlink_window = getattr(stage2_cfg, 'downlink_budget_window_s', 1.0)
            self.adas_residual_enabled = getattr(stage2_cfg, 'residual_enabled', True)
        else:
            self.adas_stage2_enabled = True
            self.adas_q_up_bits = 8
            self.adas_q_down_bits = 32
            self.adas_cmeta_bits = 32
            self.adas_uplink_window = 1.0
            self.adas_downlink_window = 1.0
            self.adas_residual_enabled = True
    
        # Initialize AdaS state: three distinct index sets
        # Respect any pre-generated heterogeneous LoRA rank config
        initial_rank = self.adas_init_rank
        init_source = "adas_init_rank"
        config_local = fs_common.get_active_hetero_config_local(self._cfg)
    
        if config_local:
            client_key_1indexed = f'Client_{self.ID}'
            client_key_0indexed = f'Client_{self.ID - 1}'
            if client_key_1indexed in config_local and config_local[client_key_1indexed]:
                try:
                    initial_rank = int(next(iter(config_local[client_key_1indexed].values())))
                    init_source = client_key_1indexed
                except Exception:
                    pass
            elif client_key_0indexed in config_local and config_local[client_key_0indexed]:
                try:
                    initial_rank = int(next(iter(config_local[client_key_0indexed].values())))
                    init_source = client_key_0indexed
                except Exception:
                    pass
    
        # The three AdaS states
        self.adas_survivor_indices_current = list(range(initial_rank))
        self.adas_upload_indices_last = None  # No upload yet
        self.adas_download_indices_last = None  # No download yet
    
        # Pre-round state for Stage 1 pruning
        self.adas_indices_before_stage1 = None
        self.adas_low_positions_before = None
        self.adas_scores_before_low = {}
    
        # Snapshot and residual state for Stage 2
        self.adas_pre_round_lora_snapshot = None
        self.adas_residual_buffers = {}  # {global_idx: residual_tensor}
    
        # Bandwidth and budget state
        self.adas_uplink_budget_last = None
        self.adas_downlink_budget_last = None
    
        # Startup log
        if bool(getattr(self._cfg, 'debug', False)):
            logger.debug(
                f"Client {self.ID} startup: "
                f"method=adas, enabled={self.adas_enabled}, "
                f"config_subtree={'glue' if is_glue_task else 'llm'}.adapter.adas"
            )
        logger.info(
            f"Client {self.ID}: enabled, "
            f"init_rank={initial_rank}, init_source={init_source}, "
            f"survivor_count={len(self.adas_survivor_indices_current)}, "
            f"rank_bounds=[{self.adas_rank_min}, {self.adas_rank_max}], "
            f"stage1_gamma={self.adas_gamma}, stage1_reg_weight={self.adas_reg_weight}, "
            f"stage2_enabled={self.adas_stage2_enabled}, "
            f"q_bits_up/down={self.adas_q_up_bits}/{self.adas_q_down_bits}"
        )

    def _adas_log_round_start(self):
        """
        Log AdaS state at the start of each round:
        - log current survivor count at the start of each round
        - log prior upload and downlink subset sizes
        """
        if not self.adas_enabled:
            return
    
        survivor_count = len(self.adas_survivor_indices_current) if self.adas_survivor_indices_current else 0
        upload_count = len(self.adas_upload_indices_last) if self.adas_upload_indices_last else 0
        download_count = len(self.adas_download_indices_last) if self.adas_download_indices_last else 0
    
        logger.info(
            f"Client {self.ID} round start: "
            f"survivor_count={survivor_count}, "
            f"prior_upload_count={upload_count}, "
            f"prior_download_count={download_count}"
        )

    def _adas_save_pre_round_lora_snapshot(self):
        """
        Save a snapshot of current LoRA tensors before local training.
    
        This snapshot is used in Stage 2 to compute the fresh local model update
        by comparing post-training tensors against this pre-round state.
    
        The snapshot is:
        - A dict keyed like model.state_dict()
        - Limited to LoRA tensors only (keys containing 'lora_A' or 'lora_B')
        - Cloned and stored on CPU to reduce GPU memory pressure
        - Stage 2 helpers handle device alignment at computation time
        """
        if not self.adas_enabled:
            return
    
        try:
            model_state_dict = self.trainer.ctx.model.state_dict()
        
            # Extract, clone, and move to CPU for storage (reduces GPU memory pressure)
            lora_snapshot = {}
            sample_device = None
            sample_dtype = None
            for key, tensor in model_state_dict.items():
                if isinstance(tensor, torch.Tensor) and ('lora_A' in key or 'lora_B' in key):
                    # Clone and move to CPU for storage
                    lora_snapshot[key] = tensor.clone().cpu()
                    if sample_device is None:
                        sample_device = tensor.device
                        sample_dtype = tensor.dtype
        
            self.adas_pre_round_lora_snapshot = lora_snapshot
        
            # Log storage policy: CPU-backed auxiliary tensors
            if bool(getattr(self._cfg, 'debug', False)):
                logger.debug(
                f"Client {self.ID}: Saved pre-round LoRA snapshot "
                f"with {len(lora_snapshot)} tensors on CPU "
                f"(source model was {sample_device}, {sample_dtype})"
            )
        
        except Exception as e:
            logger.error(
                f"Client {self.ID}: Failed to save pre-round LoRA snapshot: {e}"
            )
            self.adas_pre_round_lora_snapshot = None

    def _adas_record_lowset_before(self):
        """
        Record low-set positions and per-candidate scores before training for Stage 1 pruning.
        """
        if not self.adas_enabled:
            return
    
        try:
            from federatedscope.contrib.common.adas_utils import (
                compute_component_scores, compute_lowset_and_score
            )
        
            model = self.trainer.ctx.model
            current_rank = len(self.adas_survivor_indices_current) if self.adas_survivor_indices_current else 0
        
            # Compute per-component scores
            scores = compute_component_scores(model, current_rank=current_rank)
        
            if len(scores) == 0:
                self.adas_low_positions_before = []
                self.adas_scores_before_low = {}
                self.adas_indices_before_stage1 = list(self.adas_survivor_indices_current) if self.adas_survivor_indices_current else []
                return
        
            # Compute low-set (candidates for pruning)
            low_positions, low_score_sum = compute_lowset_and_score(
                scores, self.adas_gamma, self.adas_rank_min
            )
        
            # Store per-candidate baseline scores
            self.adas_low_positions_before = low_positions
            self.adas_scores_before_low = {
                pos: scores[pos].item() for pos in low_positions if pos < len(scores)
            }
            self.adas_indices_before_stage1 = list(self.adas_survivor_indices_current) if self.adas_survivor_indices_current else []
        
            # Log low-set info
            if low_positions:
                low_indices = [self.adas_survivor_indices_current[p] for p in low_positions if p < len(self.adas_survivor_indices_current)]
                low_scores = [scores[p].item() for p in low_positions if p < len(scores)]
                min_s = min(low_scores) if low_scores else 0.0
                avg_s = sum(low_scores) / len(low_scores) if low_scores else 0.0
                max_s = max(low_scores) if low_scores else 0.0
            
                if bool(getattr(self._cfg, 'debug', False)):
                    logger.debug(
                    f"Client {self.ID}: "
                    f"Stage1 low_positions (sample)={low_positions[:5]}..., "
                    f"low_indices (sample)={low_indices[:5]}..., "
                    f"score(min/avg/max)={min_s:.4f}/{avg_s:.4f}/{max_s:.4f}"
                )
        
            # Set model attributes for trainer regularizer
            if hasattr(self.trainer, 'ctx') and hasattr(self.trainer.ctx, 'model'):
                setattr(model, "adas_low_positions", low_positions)
                setattr(model, "adas_current_rank", current_rank)
                setattr(model, "adas_indices", self.adas_survivor_indices_current)
        
        except Exception as e:
            logger.warning(
                f"Client {self.ID}: Failed to compute Stage1 low-set: {e}"
            )
            self.adas_low_positions_before = None
            self.adas_scores_before_low = {}
            self.adas_indices_before_stage1 = None

    def _adas_stage1_prune(self):
        """
        Perform Stage 1 structural pruning for AdaS after training.
    
        This is a pure Stage 1 prune step that:
        1. Computes post-training scores
        2. Compares candidates' before/after scores
        3. Prunes only candidates whose score decreased
        4. Updates the structural survivor set
    
        Returns:
            The new survivor_indices list after pruning
        """
        if not self.adas_enabled:
            return self.adas_survivor_indices_current
    
        # Check for required state from before training
        if (self.adas_low_positions_before is None or 
            not hasattr(self, 'adas_scores_before_low') or
            self.adas_indices_before_stage1 is None):
            logger.info(
                f"Client {self.ID}: Stage1 no pruning (missing pre-training state)"
            )
            return self.adas_survivor_indices_current
    
        try:
            from federatedscope.contrib.common.adas_utils import compute_component_scores
        
            model = self.trainer.ctx.model
            current_rank = len(self.adas_survivor_indices_current) if self.adas_survivor_indices_current else 0
            low_positions = self.adas_low_positions_before
            m = len(low_positions)  # number of candidates
        
            # Handle case: no candidates (m=0)
            if m == 0:
                logger.info(
                    f"Client {self.ID}: Stage1 "
                    f"candidates={m}/{current_rank}, decreased=0, pruned=0, new_survivor_count={current_rank} | "
                    f"No pruning: m=0 (already at k_target or rank_min)"
                )
                return self.adas_survivor_indices_current
        
            # Compute scores after training
            scores_after = compute_component_scores(model, current_rank=current_rank)
        
            if len(scores_after) == 0:
                return self.adas_survivor_indices_current
        
            # Per-candidate comparison: find candidates whose score decreased
            prune_positions = []
            candidate_deltas = []  # for debug logging
        
            for pos in low_positions:
                if pos >= len(scores_after):
                    continue
                before = self.adas_scores_before_low.get(pos, 0.0)
                after = scores_after[pos].item()
                decreased = after < before
            
                # Store delta info for logging
                gidx = self.adas_indices_before_stage1[pos] if pos < len(self.adas_indices_before_stage1) else -1
                candidate_deltas.append((gidx, before, after, decreased))
            
                if decreased:
                    prune_positions.append(pos)
        
            n_decreased = len(prune_positions)
        
            # Convert pruned local positions to global component IDs
            pruned_global_indices = []
            for pos in prune_positions:
                if pos < len(self.adas_indices_before_stage1):
                    gidx = self.adas_indices_before_stage1[pos]
                    pruned_global_indices.append(gidx)
        
            # Enforce rank_min constraint
            r_current = len(self.adas_survivor_indices_current)
            max_prunable = max(0, r_current - self.adas_rank_min)
        
            reason_suffix = ""
            if len(pruned_global_indices) > max_prunable:
                pruned_global_indices = pruned_global_indices[:max_prunable]
                reason_suffix = " (truncated by rank_min)"
        
            n_pruned = len(pruned_global_indices)
            r_new = r_current - n_pruned
        
            # Log per-candidate delta sample (first 5) at DEBUG level
            if candidate_deltas:
                sample = candidate_deltas[:5]
                delta_str = ", ".join(
                    f"(gidx={g}, before={b:.4f}, after={a:.4f}, dec={d})"
                    for g, b, a, d in sample
                )
                if bool(getattr(self._cfg, 'debug', False)):
                    logger.debug(
                    f"Client {self.ID}: Stage1 candidate_deltas (sample): {delta_str}"
                )
        
            # Main decision logging
            if n_pruned == 0:
                if n_decreased == 0:
                    reason = "No pruning: no candidates decreased within-round"
                elif max_prunable == 0:
                    reason = "No pruning: candidates decreased but rank_min prevents pruning"
                else:
                    reason = "No pruning: all decreased candidates truncated by rank_min"
            
                logger.info(
                    f"Client {self.ID}: Stage1 "
                    f"candidates={m}/{current_rank}, decreased={n_decreased}, pruned=0, "
                    f"new_survivor_count={r_new} | {reason}"
                )
                return self.adas_survivor_indices_current
        
            # Perform pruning: build new survivor list
            pruned_set = set(pruned_global_indices)
            new_survivor_indices = [
                gidx for gidx in self.adas_survivor_indices_current
                if gidx not in pruned_set
            ]
        
            # Sort survivors to maintain canonical order
            new_survivor_indices = sorted(new_survivor_indices)
        
            # Update AdaS state
            old_survivor_count = len(self.adas_survivor_indices_current)
            self.adas_survivor_indices_current = new_survivor_indices
            new_survivor_count = len(new_survivor_indices)
        
            # Log pruning decision
            logger.info(
                f"Client {self.ID}: Stage1 "
                f"candidates={m}/{current_rank}, decreased={n_decreased}, pruned={n_pruned}, "
                f"new_survivor_count={new_survivor_count}, "
                f"pruned_indices={pruned_global_indices[:5]}{'...' if len(pruned_global_indices) > 5 else ''}"
                f"{reason_suffix}"
            )
        
            # Clean up residual buffers for pruned components
            if hasattr(self, 'adas_residual_buffers') and self.adas_residual_buffers:
                for gidx in pruned_global_indices:
                    if gidx in self.adas_residual_buffers:
                        del self.adas_residual_buffers[gidx]
                        if bool(getattr(self._cfg, 'debug', False)):
                            logger.debug(
                            f"Client {self.ID}: Removed residual buffer for pruned component {gidx}"
                        )
        
            # Log sample of final survivor indices
            if bool(getattr(self._cfg, 'debug', False)):
                logger.debug(
                f"Client {self.ID}: Stage1 final survivor_indices (sample): "
                f"{new_survivor_indices[:10]}..."
            )
        
            return new_survivor_indices
        
        except Exception as e:
            logger.warning(
                f"Client {self.ID}: Stage1 pruning failed: {e}"
            )
            import traceback
            if bool(getattr(self._cfg, 'debug', False)):
                logger.debug(traceback.format_exc())
            return self.adas_survivor_indices_current


    def _sync_adas_state_from_message(self,
                                               survivor_indices=None,
                                               download_indices=None,
                                               bandwidth_info=None,
                                               client_rank_config=None):
        """
        Keep AdaS three-state bookkeeping aligned with server payloads.
    
        Message contract:
        - survivor_indices defines the client's current structural set
        - download_indices defines which survivor components are actually refreshed this round
        - bandwidth_info contains upload/download rates for Stage 2 budgets
        """
        if not self.adas_enabled:
            return
    
        # Update structural survivor set
        if survivor_indices is not None:
            old_survivor_count = len(self.adas_survivor_indices_current) if self.adas_survivor_indices_current else 0
            self.adas_survivor_indices_current = list(survivor_indices)
            new_survivor_count = len(self.adas_survivor_indices_current)
        
            logger.info(
                f"Client {self.ID} received survivor_indices: "
                f"count={new_survivor_count}, sample={self.adas_survivor_indices_current[:5]}..."
            )
        
            if old_survivor_count != new_survivor_count:
                if bool(getattr(self._cfg, 'debug', False)):
                    logger.debug(
                    f"Client {self.ID} survivor count changed: "
                    f"{old_survivor_count} -> {new_survivor_count}"
                )
    
        # Update download indices (what was actually sent this round)
        if download_indices is not None:
            self.adas_download_indices_last = list(download_indices)
            if bool(getattr(self._cfg, 'debug', False)):
                logger.debug(
                f"Client {self.ID} received download_indices: "
                f"count={len(download_indices)}, sample={download_indices[:5]}..."
            )
    
        # Update bandwidth info for Stage 2 budgets
        if bandwidth_info is not None:
            ul_kbits = bandwidth_info.get('upload_kbits', 0)
            dl_kbits = bandwidth_info.get('download_kbits', 0)
        
            # Use pre-computed budgets from bandwidth_info (in bits)
            # The bandwidth helper computes these as: kbit/s * window_s * 1000 = bits
            if 'uplink_budget_bits' in bandwidth_info:
                self.adas_uplink_budget_last = bandwidth_info['uplink_budget_bits']
            else:
                # Fallback: derive from kbit/s rate (convert to bits)
                self.adas_uplink_budget_last = ul_kbits * self.adas_uplink_window * 1000
        
            if 'downlink_budget_bits' in bandwidth_info:
                self.adas_downlink_budget_last = bandwidth_info['downlink_budget_bits']
            else:
                # Fallback: derive from kbit/s rate (convert to bits)
                self.adas_downlink_budget_last = dl_kbits * self.adas_downlink_window * 1000
        
            if bool(getattr(self._cfg, 'debug', False)):
                logger.debug(
                f"Client {self.ID} received bandwidth_info: "
                f"ul={ul_kbits:.1f}kbit/s, dl={dl_kbits:.1f}kbit/s, "
                f"derived budgets: U={self.adas_uplink_budget_last:.0f}bits, "
                f"D={self.adas_downlink_budget_last:.0f}bits"
            )
    
        # Fall back to client_rank_config if survivor_indices not provided
        if survivor_indices is None and client_rank_config is not None:
            try:
                rank = int(next(iter(client_rank_config.values())))
                self.adas_survivor_indices_current = list(range(rank))
                if bool(getattr(self._cfg, 'debug', False)):
                    logger.debug(
                    f"Client {self.ID} inferred survivor_indices from rank: {rank}"
                )
            except Exception:
                pass
