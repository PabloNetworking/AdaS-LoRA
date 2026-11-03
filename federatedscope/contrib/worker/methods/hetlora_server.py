"""Method-specific HetLoRA server."""

import logging

import federatedscope.contrib.common as fs_common
from federatedscope.contrib.worker.methods.heterolora_server import HeteroLoRAServer

logger = logging.getLogger(__name__)


class HetLoRAServer(HeteroLoRAServer):
    METHOD_NAME = 'hetlora'

    def _init_hetlora(self):
        """Initialize HetLoRA attributes if enabled."""
        hetlora_cfg = fs_common.get_hetlora_cfg(self._cfg)
        self.hetlora_enabled = hetlora_cfg is not None

        if not self.hetlora_enabled:
            self.hetlora_client_ranks = {}
            return

        logger.info("Initializing HetLoRA server attributes")
        self.hetlora_client_ranks = {}

        init_rank = getattr(hetlora_cfg, 'init_rank', 64)
        rank_min = getattr(hetlora_cfg, 'rank_min', 2)
        rank_max = getattr(hetlora_cfg, 'rank_max', 64)

        for client_id in range(1, self._client_num + 1):
            self.hetlora_client_ranks[client_id] = init_rank

        config_local = fs_common.get_active_hetero_config_local(self._cfg)
        if config_local:
            for client_id in range(1, self._client_num + 1):
                client_key = fs_common.resolve_client_key(config_local, client_id)
                module_ranks = config_local.get(client_key, {}) if client_key is not None else {}
                try:
                    if module_ranks:
                        first_rank = list(module_ranks.values())[0]
                        self.hetlora_client_ranks[client_id] = first_rank
                except (ValueError, IndexError):
                    pass

        logger.info(
            f"Initialized client ranks: "
            f"n_clients={len(self.hetlora_client_ranks)}, "
            f"init_rank={init_rank}, rank_min={rank_min}, rank_max={rank_max}"
        )

    def callback_funcs_for_hetlora_rank(self, message):
        if not self.hetlora_enabled:
            return

        if self.fah_enabled:
            if bool(getattr(self._cfg, 'debug', False)):
                logger.debug("Ignoring hetlora_rank message because FAH is enabled")
            return

        sender = message.sender
        content = message.content

        if not isinstance(content, dict) or 'rank' not in content:
            logger.warning(
                f"Invalid hetlora_rank message from client {sender}: {content}"
            )
            return

        new_rank = content['rank']
        old_rank = self.hetlora_client_ranks.get(sender, None)

        hetlora_cfg = fs_common.get_hetlora_cfg(self._cfg)
        if hetlora_cfg:
            rank_min = getattr(hetlora_cfg, 'rank_min', 2)
            rank_max = getattr(hetlora_cfg, 'rank_max', 64)
            new_rank = max(rank_min, min(rank_max, new_rank))

        self.hetlora_client_ranks[sender] = new_rank

        if old_rank != new_rank:
            logger.info(f"Client {sender} rank updated: {old_rank} -> {new_rank}")

        self._update_hetero_ranks_config(self.hetlora_client_ranks)
