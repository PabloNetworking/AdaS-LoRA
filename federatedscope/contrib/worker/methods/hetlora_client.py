"""Method-specific HetLoRA client.

Implements HetLoRA initialization and pruning for the ``hetlora`` method on top of
the shared Client lifecycle.
"""

import logging

import federatedscope.contrib.common as fs_common

from federatedscope.core.workers import Client

logger = logging.getLogger(__name__)


class HetLoRAClient(Client):
    METHOD_NAME = 'hetlora'

    def _resolve_hetlora_initial_rank(self):
        initial_rank = self.hetlora_init_rank
        init_source = 'hetlora_init_rank'
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

    def _init_hetlora(self):
        """
        Initialize HetLoRA attributes for rank self-pruning.
        """
        hetlora_cfg = fs_common.get_hetlora_cfg(self._cfg)
        self.hetlora_enabled = hetlora_cfg is not None

        if not self.hetlora_enabled:
            self.hetlora_current_rank = None
            self.hetlora_tail_score_before = None
            self.hetlora_pruning_enabled = False
            return

        self.hetlora_rank_min = getattr(hetlora_cfg, 'rank_min', 2)
        self.hetlora_rank_max = getattr(hetlora_cfg, 'rank_max', 64)
        self.hetlora_init_rank = getattr(hetlora_cfg, 'init_rank', 64)

        pruning_cfg = getattr(hetlora_cfg, 'pruning', None)
        if pruning_cfg:
            self.hetlora_pruning_enabled = getattr(pruning_cfg, 'enabled', True)
            self.hetlora_decay = getattr(pruning_cfg, 'decay', 0.99)
            self.hetlora_reg_weight = getattr(pruning_cfg, 'regularizer_weight', 0.01)
        else:
            self.hetlora_pruning_enabled = True
            self.hetlora_decay = 0.99
            self.hetlora_reg_weight = 0.01

        initial_rank, init_source = self._resolve_hetlora_initial_rank()
        self.hetlora_current_rank = initial_rank
        self.hetlora_tail_score_before = None

        try:
            setattr(self.trainer.ctx.model, 'hetlora_current_rank', int(initial_rank))
        except Exception:
            pass

        logger.info(
            f"Client {self.ID}: HetLoRA enabled, "
            f"configured_init_rank={initial_rank}, "
            f"current_rank={self.hetlora_current_rank}, "
            f"init_source={init_source}, "
            f"rank_bounds=[{self.hetlora_rank_min}, {self.hetlora_rank_max}], "
            f"pruning={'ON' if self.hetlora_pruning_enabled else 'OFF'}, "
            f"decay={self.hetlora_decay}, reg_weight={self.hetlora_reg_weight}."
        )

    def _hetlora_record_tail_score_before(self):
        """Record the HetLoRA tail importance score before training."""
        if not self.hetlora_enabled or not self.hetlora_pruning_enabled:
            return

        try:
            from federatedscope.contrib.common.heterolora_utils import tail_score

            model = self.trainer.ctx.model
            try:
                if self.hetlora_current_rank is not None:
                    setattr(model, 'hetlora_current_rank', int(self.hetlora_current_rank))
            except Exception:
                pass

            self.hetlora_tail_score_before = tail_score(
                model,
                self.hetlora_decay,
                current_rank=self.hetlora_current_rank,
            )
            if bool(getattr(self._cfg, 'debug', False)):
                logger.debug(
                    f"Client {self.ID}: score_before={self.hetlora_tail_score_before} "
                    f"with decay={self.hetlora_decay}"
                )
        except Exception as e:
            logger.warning(
                f"Client {self.ID}: Failed to compute tail_score_before: {e}"
            )
            self.hetlora_tail_score_before = None

    def _hetlora_prune_and_send_rank(self, model_para: dict, sender, round_idx, timestamp):
        """
        Perform HetLoRA rank self-pruning after training.

        Design note:
        HetLoRA clients keep a logical current rank that may be smaller than the
        physical local tensor shapes after repeated heterogeneous downlink loads.
        Consistent with the HetLoRA design, the client should always upload only
        the LoRA parameters corresponding to its current logical rank, and only
        notify the server when the logical rank itself changes due to self-pruning.
        """
        if not self.hetlora_enabled:
            logger.warning(
                'Unable to perform HetLoRA upload truncation, hetlora is disabled.'
            )
            return model_para

        try:
            from federatedscope.core.message import Message
            from federatedscope.contrib.common.heterolora_utils import (
                tail_score,
                truncate_client_lora_to_rank,
                get_current_lora_rank_from_state_dict,
            )

            def _truncate_payload_to_rank(payload, target_rank, prior_rank=None):
                if isinstance(payload, list):
                    return [
                        truncate_client_lora_to_rank(mp, target_rank, prior_rank, debug=False)
                        for mp in payload
                    ]
                return truncate_client_lora_to_rank(
                    payload,
                    target_rank,
                    prior_rank,
                    debug=False,
                )

            model = self.trainer.ctx.model
            current_rank = self.hetlora_current_rank
            if current_rank is None:
                _mp = model_para[0] if isinstance(model_para, list) and len(model_para) > 0 else model_para
                current_rank = get_current_lora_rank_from_state_dict(_mp)
                if current_rank == 0:
                    current_rank = self.hetlora_init_rank
                    logger.warning(
                        f"Client {self.ID}: Could not infer current rank from state_dict; "
                        f"falling back to init_rank={current_rank}."
                    )

            current_rank = int(max(self.hetlora_rank_min, min(self.hetlora_rank_max, current_rank)))
            target_rank = current_rank

            try:
                setattr(model, 'hetlora_current_rank', int(current_rank))
            except Exception:
                pass

            if not self.hetlora_pruning_enabled:
                if bool(getattr(self._cfg, 'debug', False)):
                    logger.debug(
                        f"Client {self.ID}: HetLoRA pruning disabled, preserving current_rank={current_rank}"
                    )
            elif self.hetlora_tail_score_before is None:
                logger.warning(
                    f"Client {self.ID}: tail score before training is None; "
                    f"skipping prune decision and uploading at current_rank={current_rank}."
                )
            else:
                score_after = tail_score(model, self.hetlora_decay, current_rank=int(current_rank))
                score_before = self.hetlora_tail_score_before
                should_prune = score_after < score_before

                if should_prune:
                    new_rank = max(self.hetlora_rank_min, int(current_rank * self.hetlora_decay))

                    if new_rank < current_rank:
                        target_rank = int(new_rank)
                        logger.info(
                            f"Client {self.ID}: Pruning rank {current_rank} -> {target_rank} "
                            f"(score_before={score_before:.4f}, score_after={score_after:.4f})"
                        )

                        self.hetlora_current_rank = target_rank
                        try:
                            setattr(self.trainer.ctx.model, 'hetlora_current_rank', int(target_rank))
                        except Exception:
                            pass

                        self.comm_manager.send(
                            Message(
                                msg_type='hetlora_rank',
                                sender=self.ID,
                                receiver=[sender],
                                state=round_idx,
                                timestamp=timestamp,
                                content={'rank': target_rank},
                            )
                        )
                    else:
                        logger.info(
                            f"Client {self.ID}: No pruning "
                            f"(new_rank={new_rank} >= current={current_rank})"
                        )
                else:
                    logger.info(
                        f"Client {self.ID}: No pruning "
                        f"(score_after={score_after:.4f} >= score_before={score_before:.4f})"
                    )

            if self.hetlora_current_rank is None:
                self.hetlora_current_rank = int(target_rank)
            else:
                self.hetlora_current_rank = int(max(self.hetlora_rank_min, min(self.hetlora_rank_max, self.hetlora_current_rank)))
                target_rank = int(self.hetlora_current_rank)

            try:
                setattr(self.trainer.ctx.model, 'hetlora_current_rank', int(target_rank))
            except Exception:
                pass

            model_para = _truncate_payload_to_rank(
                model_para,
                int(target_rank),
                prior_rank=int(current_rank),
            )

            if bool(getattr(self._cfg, 'debug', False)):
                logger.debug(
                    f"Client {self.ID}: Prepared HetLoRA upload at logical rank={target_rank} "
                    f"(pre-truncation reference rank={current_rank})"
                )
        except Exception as e:
            logger.warning(f"Client {self.ID}: Failed during pruning/upload truncation: {e}")

        return model_para