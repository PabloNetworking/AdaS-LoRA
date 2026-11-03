"""Worker registration for FAH-QLoRA.

- Use `federate.method: fah_qlora` to select the FAH-QLoRA client and server.
- The FAH config block under `cfg.llm.adapter.fah` provides the
  FAH-specific hyperparameters and feature gating.
"""

import logging

from federatedscope.register import register_worker
from federatedscope.contrib.worker.methods.fah_qlora_client import FahQLoRAClient
from federatedscope.contrib.worker.methods.fah_qlora_server import FahQLoRAServer

logger = logging.getLogger(__name__)

_METHOD_ALIASES = {'fah_qlora', 'fah-qlora', 'fahqlora'}


def call_fah_qlora_worker(method):
    if method is None:
        return None
    if method.lower() in {m.lower() for m in _METHOD_ALIASES}:
        logger.info('Using FAH-QLoRA client and server')
        return {
            'client': FahQLoRAClient,
            'server': FahQLoRAServer,
        }
    return None


register_worker('fah_qlora', call_fah_qlora_worker)