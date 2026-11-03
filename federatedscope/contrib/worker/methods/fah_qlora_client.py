"""FAH-QLoRA client mixin and client class.

The FAH-specific helper logic lives in :class:`FahQLoRAClientMixin`.
:class:`FahQLoRAClient` is selected with `federate.method: fah_qlora`.
"""

import logging

import torch

logger = logging.getLogger(__name__)


class FahQLoRAClientMixin:
    def _init_fah_qloRA(self):
        self.fah_enabled = (
            hasattr(self._cfg, 'llm') and
            hasattr(self._cfg.llm.adapter, 'fah') and
            getattr(self._cfg.llm.adapter.fah, 'enabled', False)
        )
        if not self.fah_enabled:
            self.fah_current_rank = None
            self.fah_current_hat_rank = None
            self.fah_last_training_time = 0.0
            return
        fah_cfg = self._cfg.llm.adapter.fah
        adapter_cfg = self._cfg.llm.adapter
        max_rank_adapter = getattr(adapter_cfg, 'max_rank', None)
        if max_rank_adapter is None:
            logger.warning(
                f"[FAH] Client {self.ID}: llm.adapter.max_rank must be set when FAH is enabled."
            )
        if fah_cfg.r_max != max_rank_adapter:
            logger.warning(
                f"[FAH] Client {self.ID}: fah.r_max={fah_cfg.r_max} != "
                f"adapter.max_rank={max_rank_adapter}. Using {max_rank_adapter} on this client."
            )
            self._cfg.defrost()
            self._cfg.llm.adapter.fah.r_max = max_rank_adapter
            self._cfg.freeze(inform=False)
            fah_cfg = self._cfg.llm.adapter.fah
        self.fah_current_rank = fah_cfg.init_rank
        self.fah_current_hat_rank = max(fah_cfg.r_min, fah_cfg.init_rank - 1)
        self.fah_last_training_time = 0.0
        logger.info(
            f"[FAH] Client {self.ID}: FAH-QLoRA enabled, init_rank={fah_cfg.init_rank}"
        )

    def _fah_resolve_compute_dtype(self):
        cq = getattr(self._cfg, 'computation_quantization', None)
        if cq is not None:
            cd_str = getattr(cq, 'compute_dtype', None)
            if isinstance(cd_str, str) and cd_str:
                s = cd_str.lower()
                if s in ['fp16', 'float16', 'half']:
                    return torch.float16
                if s in ['bf16', 'bfloat16']:
                    return torch.bfloat16
                if s in ['fp32', 'float32']:
                    return torch.float32
            if getattr(cq, 'method', None) == 'qlora' and getattr(cq, 'nbits', None) in [4, 8]:
                return torch.bfloat16
        try:
            model = getattr(getattr(self.trainer, 'ctx', None), 'model', None)
            if model is not None:
                for name, param in model.named_parameters():
                    if param.is_floating_point() and 'lora_' not in name:
                        return param.dtype
                for _, param in model.named_parameters():
                    if param.is_floating_point():
                        return param.dtype
        except Exception:
            pass
        return torch.float16

    def _fah_cast_trainable_params(self, dtype: torch.dtype, model=None, log_prefix='[FAH]'):
        model = self.trainer.ctx.model if model is None else model
        peft_or_model = model.model if hasattr(model, 'model') and hasattr(getattr(model, 'model'), 'named_parameters') else model
        num_cast = 0
        for _, param in peft_or_model.named_parameters():
            if not getattr(param, 'requires_grad', False) or not param.is_floating_point() or param.dtype == dtype:
                continue
            try:
                param.data = param.data.to(dtype=dtype)
                num_cast += 1
            except Exception:
                continue
        if num_cast > 0 and bool(getattr(self._cfg, 'debug', False)):
            logger.debug('%s Recast %d trainable params to compute dtype=%s', log_prefix, num_cast, dtype)

    def _fah_evaluate_and_send_stats(self, model_content, server_id, round_idx, timestamp):
        if not self.fah_enabled:
            return
        from federatedscope.core.message import Message
        fah_cfg = self._cfg.llm.adapter.fah
        r_current = self.fah_current_rank or fah_cfg.init_rank
        r_hat = self.fah_current_hat_rank or max(fah_cfg.r_min, r_current - 1)
        try:
            F_n = self._fah_evaluate_loss()
            F_hat_n = self._fah_evaluate_loss_at_rank(r_hat)
        except Exception as error:
            logger.warning(f"[FAH] Client {self.ID} evaluation failed: {error}")
            F_n = 1.0
            F_hat_n = 1.0
        self.comm_manager.send(
            Message(
                msg_type='fah_stats',
                sender=self.ID,
                receiver=[server_id],
                state=round_idx,
                timestamp=timestamp,
                content={
                    'F_n': F_n,
                    'F_hat_n': F_hat_n,
                    'training_time': self.fah_last_training_time,
                    'rank': r_current,
                    'hat_rank': r_hat,
                },
            )
        )
        logger.info(
            f"[FAH] Client {self.ID} sent FAH stats: F={F_n:.4f}, "
            f"F̂={F_hat_n:.4f}, r={r_current}, r̂={r_hat}, "
            f"t_train={self.fah_last_training_time:.2f}s"
        )

    def _fah_evaluate_loss(self):
        return self._fah_eval_val_loss()

    def _fah_eval_val_loss(self):
        try:
            metrics = self.trainer.evaluate(target_data_split_name='val')
            for key in ['val_avg_loss', 'val_loss', 'test_loss']:
                if key in metrics:
                    return float(metrics[key])
            for key, value in metrics.items():
                if 'loss' in key.lower():
                    return float(value)
            return 1.0
        except Exception as error:
            logger.warning(f"[FAH] Client {self.ID} validation evaluation failed: {error}")
            return 1.0

    def _fah_evaluate_loss_at_rank(self, rank: int):
        if not self.fah_enabled:
            return self._fah_eval_val_loss()
        fah_cfg = self._cfg.llm.adapter.fah
        target_rank = max(fah_cfg.r_min, min(int(rank), fah_cfg.r_max))
        model = self.model
        orig_state = model.state_dict()
        masked_state = {}
        for name, param in orig_state.items():
            if not isinstance(param, torch.Tensor) or param.dtype not in (torch.float16, torch.float32, torch.bfloat16):
                masked_state[name] = param
                continue
            if 'lora_A' in name and 'lora_B' not in name:
                current_r = param.shape[0]
                r_eff = min(target_rank, current_r)
                masked = param.clone()
                if r_eff < current_r:
                    masked[r_eff:, :] = 0.0
                masked_state[name] = masked
            elif 'lora_B' in name:
                current_r = param.shape[1]
                r_eff = min(target_rank, current_r)
                masked = param.clone()
                if r_eff < current_r:
                    masked[:, r_eff:] = 0.0
                masked_state[name] = masked
            else:
                masked_state[name] = param
        model.load_state_dict(masked_state, strict=False)
        loss = self._fah_eval_val_loss()
        model.load_state_dict(orig_state, strict=False)
        return loss


from federatedscope.contrib.worker.methods.heterolora_client import HeteroLoRAClient


class FahQLoRAClient(FahQLoRAClientMixin, HeteroLoRAClient):
    METHOD_NAME = 'fah_qlora'