"""Method-specific AdaS-PruneOnly client.

Implements AdaS-PruneOnly initialization, rank/index bookkeeping and per-candidate
pruning for the ``adas_pruneonly`` method on top of the shared Client lifecycle.
"""

import logging

import federatedscope.contrib.common as fs_common
from federatedscope.core.workers import Client

logger = logging.getLogger(__name__)


class AdaSPruneOnlyClient(Client):
    METHOD_NAME = 'adas_pruneonly'

    def _resolve_adas_pruneonly_initial_rank(self):
        initial_rank = self.adas_pruneonly_init_rank
        init_source = 'adas_pruneonly_init_rank'
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

        return initial_rank, init_source

    def _init_adas_pruneonly(self):
        """
        Initialize AdaS-PruneOnly attributes for component-based pruning.

        AdaS-PruneOnly differs from HetLoRA in that it:
        - Uses global component indices, not just a rank
        - Performs component-based importance scoring
        - Prunes based on low-set score comparison
        """
        adas_pruneonly_cfg = fs_common.get_adas_pruneonly_cfg(self._cfg)
        self.adas_pruneonly_enabled = adas_pruneonly_cfg is not None

        if not self.adas_pruneonly_enabled:
            self.adas_pruneonly_indices_current = None
            self.adas_pruneonly_low_positions_before = None
            self.adas_pruneonly_scores_before_low = {}
            self.adas_pruneonly_indices_before = None
            self.adas_pruneonly_pruning_enabled = False
            self.adas_pruneonly_rank_min = None
            self.adas_pruneonly_rank_max = None
            self.adas_pruneonly_init_rank = None
            self.adas_pruneonly_gamma = None
            self.adas_pruneonly_reg_weight = None
            return

        self.adas_pruneonly_rank_min = getattr(adas_pruneonly_cfg, 'rank_min', 2)
        self.adas_pruneonly_rank_max = getattr(adas_pruneonly_cfg, 'rank_max', 64)
        self.adas_pruneonly_init_rank = getattr(adas_pruneonly_cfg, 'init_rank', 64)

        pruning_cfg = getattr(adas_pruneonly_cfg, 'pruning', None)
        if pruning_cfg:
            self.adas_pruneonly_pruning_enabled = getattr(pruning_cfg, 'enabled', True)
            self.adas_pruneonly_gamma = getattr(pruning_cfg, 'gamma', 0.9)
            self.adas_pruneonly_reg_weight = getattr(pruning_cfg, 'regularizer_weight', 0.01)
        else:
            self.adas_pruneonly_pruning_enabled = True
            self.adas_pruneonly_gamma = 0.9
            self.adas_pruneonly_reg_weight = 0.01

        initial_rank, init_source = self._resolve_adas_pruneonly_initial_rank()
        self.adas_pruneonly_indices_current = list(range(initial_rank))
        self.adas_pruneonly_low_positions_before = None
        self.adas_pruneonly_scores_before_low = {}
        self.adas_pruneonly_indices_before = None
        logger.info(
            f"Client {self.ID}: enabled, "
            f"configured_init_rank={initial_rank}, "
            f"current_rank={len(self.adas_pruneonly_indices_current)}, "
            f"init_source={init_source}, "
            f"rank_bounds=[{self.adas_pruneonly_rank_min}, {self.adas_pruneonly_rank_max}], "
            f"pruning={'ON' if self.adas_pruneonly_pruning_enabled else 'OFF'}, "
            f"gamma={self.adas_pruneonly_gamma}, reg_weight={self.adas_pruneonly_reg_weight}."
        )

    def _adas_pruneonly_record_lowset_before(self):
        """
        Record low-set positions and per-candidate scores before training.
        """
        if not self.adas_pruneonly_enabled or not self.adas_pruneonly_pruning_enabled:
            return

        try:
            from federatedscope.contrib.common.adas_utils import (
                compute_component_scores, compute_lowset_and_score
            )

            model = self.trainer.ctx.model
            current_rank = len(self.adas_pruneonly_indices_current)
            scores = compute_component_scores(model, current_rank=current_rank)

            if len(scores) == 0:
                self.adas_pruneonly_low_positions_before = []
                self.adas_pruneonly_scores_before_low = {}
                self.adas_pruneonly_indices_before = list(self.adas_pruneonly_indices_current)
                return

            low_positions, low_score_sum = compute_lowset_and_score(
                scores, self.adas_pruneonly_gamma, self.adas_pruneonly_rank_min
            )
            _ = low_score_sum

            self.adas_pruneonly_low_positions_before = low_positions
            self.adas_pruneonly_scores_before_low = {
                pos: scores[pos].item() for pos in low_positions
            }
            self.adas_pruneonly_indices_before = list(self.adas_pruneonly_indices_current)

            try:
                low_indices = [self.adas_pruneonly_indices_current[p] for p in low_positions]
            except Exception:
                low_indices = []

            if low_positions:
                scores_vals = [scores[p].item() for p in low_positions]
                score_min = min(scores_vals)
                score_max = max(scores_vals)
                score_avg = sum(scores_vals) / len(scores_vals)
            else:
                score_min = score_max = score_avg = 0.0

            logger.info(
                f"Client {self.ID}: "
                f"candidates={len(low_positions)}/{current_rank}, "
                f"candidate_indices={low_indices[:5]}{'...' if len(low_indices) > 5 else ''}, "
                f"candidate_score_stats_before=(min={score_min:.4f}/avg={score_avg:.4f}/max={score_max:.4f})"
            )

            try:
                setattr(model, 'adas_low_positions', low_positions)
                setattr(model, 'adas_current_rank', current_rank)
                setattr(model, 'adas_indices', self.adas_pruneonly_indices_current)
            except Exception:
                pass

        except Exception as e:
            logger.warning(
                f"Client {self.ID}: Failed to compute low-set: {e}"
            )
            self.adas_pruneonly_low_positions_before = None
            self.adas_pruneonly_scores_before_low = {}
            self.adas_pruneonly_indices_before = None

    def _adas_pruneonly_prune_and_prepare_upload(self, model_para: dict):
        """
        Perform per-candidate AdaS-PruneOnly pruning after training and prepare upload.
        """
        if not self.adas_pruneonly_enabled:
            return model_para, self.adas_pruneonly_indices_current

        if not self.adas_pruneonly_pruning_enabled:
            return model_para, self.adas_pruneonly_indices_current

        if (self.adas_pruneonly_low_positions_before is None or
            not hasattr(self, 'adas_pruneonly_scores_before_low') or
            self.adas_pruneonly_indices_before is None):
            logger.info(
                f"Client {self.ID}: No pruning (missing pre-training state)"
            )
            return model_para, self.adas_pruneonly_indices_current

        try:
            from federatedscope.contrib.common.adas_utils import (
                compute_component_scores, slice_update_by_keep_positions
            )

            model = self.trainer.ctx.model
            current_rank = len(self.adas_pruneonly_indices_current)
            low_positions = self.adas_pruneonly_low_positions_before
            m = len(low_positions)

            if m == 0:
                logger.info(
                    f"Client {self.ID}: "
                    f"candidates={m}/{current_rank}, decreased=0, pruned=0, new_rank={current_rank} | "
                    f"No pruning: m=0 (already at k_target or rank_min)"
                )
                return model_para, self.adas_pruneonly_indices_current

            scores_after = compute_component_scores(model, current_rank=current_rank)
            if len(scores_after) == 0:
                return model_para, self.adas_pruneonly_indices_current

            prune_positions = []
            candidate_deltas = []

            for pos in low_positions:
                if pos >= len(scores_after):
                    continue
                before = self.adas_pruneonly_scores_before_low.get(pos, 0.0)
                after = scores_after[pos].item()
                decreased = after < before
                gidx = self.adas_pruneonly_indices_before[pos] if pos < len(self.adas_pruneonly_indices_before) else -1
                candidate_deltas.append((gidx, before, after, decreased))
                if decreased:
                    prune_positions.append(pos)

            n_decreased = len(prune_positions)
            pruned_global_indices = []
            for pos in prune_positions:
                if pos < len(self.adas_pruneonly_indices_before):
                    pruned_global_indices.append(self.adas_pruneonly_indices_before[pos])

            r_current = len(self.adas_pruneonly_indices_current)
            max_prunable = max(0, r_current - self.adas_pruneonly_rank_min)

            reason_suffix = ''
            if len(pruned_global_indices) > max_prunable:
                pruned_global_indices = pruned_global_indices[:max_prunable]
                reason_suffix = ' (truncated by rank_min)'

            n_pruned = len(pruned_global_indices)
            r_new = r_current - n_pruned

            if candidate_deltas and bool(getattr(self._cfg, 'debug', False)):
                sample = candidate_deltas[:5]
                delta_str = ', '.join(
                    f"(gidx={g}, before={b:.4f}, after={a:.4f}, dec={d})"
                    for g, b, a, d in sample
                )
                
                logger.debug(
                    f"Client {self.ID}: candidate_deltas (sample): {delta_str}"
                )

            if n_pruned == 0:
                if n_decreased == 0:
                    reason = 'No pruning: no candidates decreased within-round'
                elif max_prunable == 0:
                    reason = 'No pruning: candidates decreased but rank_min prevents pruning'
                else:
                    reason = 'No pruning: all decreased candidates truncated by rank_min'

                logger.info(
                    f"Client {self.ID}: "
                    f"candidates={m}/{current_rank}, decreased={n_decreased}, pruned=0, new_rank={r_new} | "
                    f"{reason}"
                )
                return model_para, self.adas_pruneonly_indices_current

            pruned_set = set(pruned_global_indices)
            keep_positions = [
                pos for pos in range(current_rank)
                if self.adas_pruneonly_indices_before[pos] not in pruned_set
            ]

            new_model_para, new_indices = slice_update_by_keep_positions(
                model_para, self.adas_pruneonly_indices_current, keep_positions
            )

            self.adas_pruneonly_indices_current = new_indices

            logger.info(
                f"Client {self.ID}: "
                f"candidates={m}/{current_rank}, decreased={n_decreased}, pruned={n_pruned}, "
                f"new_rank={r_new}, pruned_indices={pruned_global_indices[:5]}{'...' if len(pruned_global_indices) > 5 else ''}"
                f"{reason_suffix}"
            )
            if r_new != r_current:
                logger.info(
                    f"Client {self.ID}: rank updated: {r_current} -> {r_new}"
                )

            return new_model_para, new_indices

        except Exception as e:
            logger.warning(
                f"Client {self.ID}: Pruning failed: {e}"
            )
            import traceback
            if bool(getattr(self._cfg, 'debug', False)):
                logger.debug(traceback.format_exc())
            return model_para, self.adas_pruneonly_indices_current

    def _sync_adas_pruneonly_state_from_message(self,
                                           client_rank_config=None,
                                           adas_indices=None):
        """Keep AdaS local rank/index bookkeeping aligned with server payloads."""
        if not self.adas_pruneonly_enabled:
            return

        if adas_indices is not None:
            self.adas_pruneonly_indices_current = fs_common.normalize_indices(adas_indices)
            return

        rank = fs_common.infer_rank_from_client_rank_config(client_rank_config)
        if rank is not None:
            self.adas_pruneonly_indices_current = fs_common.indices_from_rank(rank)