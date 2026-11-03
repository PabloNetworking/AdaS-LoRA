"""Worker wiring for AdaS."""

import logging

from federatedscope.register import register_worker
from federatedscope.contrib.worker.methods.adas_client import AdaSClient
from federatedscope.contrib.worker.methods.adas_server import AdaSServer

logger = logging.getLogger(__name__)

_METHOD_ALIASES = {'adas'}


def call_adas_worker(method):
    if method is None:
        return None
    if method.lower() in _METHOD_ALIASES:
        logger.info('Using AdaS client and server')
        return {
            'client': AdaSClient,
            'server': AdaSServer,
        }
    return None


register_worker('adas', call_adas_worker)
