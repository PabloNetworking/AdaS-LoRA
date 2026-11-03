"""HeteroLoRA client capability and worker classes.

The concrete worker path is selected by `federate.method`. HeteroLoRA
capability helpers are also reused underneath FAH-QLoRA and HetLoRA.
"""

import logging

import torch

import federatedscope.contrib.common as fs_common

from federatedscope.core.workers import Client

logger = logging.getLogger(__name__)


class HeteroLoRAClient(Client):
    METHOD_NAME = 'heterolora'

    def _expects_client_specific_hetero_payload(self):
        method = fs_common.normalize_method_name(getattr(self._cfg.federate, 'method', ''))
        config_local = fs_common.get_active_hetero_config_local(self._cfg)
        return config_local is not None and method in ['hetlora', 'adas_pruneonly', 'heterolora', 'fah_qlora']

    def _resolve_client_rank_config(self, client_rank_config_from_msg=None, context='train'):
        debug_mode = bool(getattr(self._cfg, 'debug', False))
        if client_rank_config_from_msg is not None:
            if debug_mode:
                logger.debug(f"Client {self.ID}: Using client_rank_config from server message ({context})")
            return client_rank_config_from_msg
        config_local = fs_common.get_active_hetero_config_local(self._cfg)
        client_key = fs_common.resolve_client_key(config_local, self.ID)
        if client_key is not None:
            return config_local[client_key]
        logger.warning(f"Client {self.ID}: No rank config found in local config ({context})")
        return None

    def _apply_client_specific_heterolora_payload(self, content, client_rank_config_from_msg=None, context='train'):
        has_distributed_format = any('.' in key and key.split('.')[-1].isdigit() and ('lora_A' in key or 'lora_B' in key) for key in content.keys())
        if fs_common.get_active_hetero_config_local(self._cfg) is None:
            return content
        if context == 'finish' and not has_distributed_format:
            logger.info(f"Client {self.ID}: Filtering out max-rank LoRA weights from finish message to preserve local hetero configuration")
            return {k: v for k, v in content.items() if 'lora_A' not in k and 'lora_B' not in k}
        if not has_distributed_format:
            return content
        from federatedscope.contrib.common.heterolora_utils import load_weight_local
        client_rank_config = self._resolve_client_rank_config(client_rank_config_from_msg, context=context)
        if client_rank_config is None:
            return content
        if getattr(self, 'hetlora_enabled', False):
            try:
                current_rank = max(int(v) for v in client_rank_config.values())
                self.hetlora_current_rank = current_rank
                setattr(self.trainer.ctx.model, 'hetlora_current_rank', current_rank)
            except Exception as error:
                logger.warning(f"[HetLoRA] Client {self.ID}: Failed to infer current rank from client_rank_config: {error}")
        debug_mode = bool(getattr(self._cfg, 'debug', False))
        self._apply_heterolora_rank_config(client_rank_config=client_rank_config, debug=debug_mode)
        lora_state_dict = load_weight_local(weighted_single_weights=content, model=self.trainer.ctx.model, client_rank_config=client_rank_config, debug=debug_mode)
        non_lora_content = {k: v for k, v in content.items() if not ((('lora_A' in k or 'lora_B' in k) and '.' in k and k.split('.')[-1].isdigit()))}
        return {**non_lora_content, **lora_state_dict}

    def _apply_heterolora_rank_config(self, client_rank_config: dict, debug: bool = False):
        if client_rank_config is None:
            return
        try:
            from federatedscope.contrib.common.heterolora_utils import modify_adapter, is_qlora_client_cfg
        except ImportError as error:
            logger.error(f"Client {self.ID}: Failed to import modify_adapter: {error}")
            return
        model = self.trainer.ctx.model
        if hasattr(model, 'model') and hasattr(model.model, 'peft_config'):
            peft_model = model.model
        elif hasattr(model, 'peft_config'):
            peft_model = model
        else:
            peft_model = model
            if debug:
                logger.debug(f"Client {self.ID}: Could not identify PEFT model structure, using model as-is (type: {type(model).__name__})")
        if hasattr(peft_model, 'peft_config') and peft_model.peft_config:
            adapter_name = next(iter(peft_model.peft_config.keys()))
        elif hasattr(peft_model, 'active_adapters') and getattr(peft_model, 'active_adapters', None):
            active = peft_model.active_adapters; adapter_name = list(active)[0] if isinstance(active, (list, tuple, set)) else active
        else:
            adapter_name = 'default'
            if debug:
                logger.debug(f"Client {self.ID}: Could not infer adapter name, falling back to '{adapter_name}'")
        adapter_cfg = self._cfg.llm.adapter
        base_args = adapter_cfg.args[0] if getattr(adapter_cfg, 'args', None) else {}
        lora_alpha = base_args.get('lora_alpha', 16)
        lora_dropout = base_args.get('lora_dropout', 0.05)
        compute_dtype = self._fah_resolve_compute_dtype()
        is_qlora = is_qlora_client_cfg(self._cfg)
        non_lora_dtype = torch.float32 if is_qlora else None
        try:
            modify_adapter(peft_model=peft_model, adapter_name=adapter_name, modify_module_rank=client_rank_config, lora_alpha=lora_alpha, lora_dropout=lora_dropout, init_lora_weights=False, target_modules=list(client_rank_config.keys()), compute_dtype=compute_dtype, non_lora_trainable_dtype=non_lora_dtype, recast_trainables=True, recast_log_prefix=f"[FAH] Client {self.ID}:")
            if debug:
                logger.debug(f"Client {self.ID}: Applied rank config {client_rank_config} to adapter '{adapter_name}'")
        except Exception as error:
            logger.error(f"Client {self.ID}: modify_adapter failed: {error}")