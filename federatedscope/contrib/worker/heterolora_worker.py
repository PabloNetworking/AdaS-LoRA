"""Capability-based worker subclasses for HeteroLoRA.

Use `federate.method: heterolora` to select this concrete worker path.
"""

import logging

from federatedscope.register import register_worker
from federatedscope.contrib.worker.methods.heterolora_client import HeteroLoRAClient
from federatedscope.contrib.worker.methods.heterolora_server import HeteroLoRAServer

logger = logging.getLogger(__name__)

_METHOD_ALIASES = {'heterolora', 'hetero_lora', 'hetero-lora'}


def call_heterolora_worker(method):
    if method is None:
        return None
    if method.lower() in _METHOD_ALIASES:
        logger.info('Using HeteroLoRA client and server')
        return {
            'client': HeteroLoRAClient,
            'server': HeteroLoRAServer
        }
    return None


register_worker('heterolora', call_heterolora_worker)
