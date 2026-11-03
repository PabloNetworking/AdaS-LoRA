"""Worker wiring for AdaS-PruneOnly."""

import logging

from federatedscope.register import register_worker
from federatedscope.contrib.worker.methods.adas_pruneonly_client import AdaSPruneOnlyClient
from federatedscope.contrib.worker.methods.adas_pruneonly_server import AdaSPruneOnlyServer

logger = logging.getLogger(__name__)

_METHOD_ALIASES = {'adas_pruneonly', 'adas-pruneonly'}


def call_adas_pruneonly_worker(method):
    if method is None:
        return None
    if method.lower() in _METHOD_ALIASES:
        logger.info('Using AdaS-PruneOnly client and server')
        return {
            'client': AdaSPruneOnlyClient,
            'server': AdaSPruneOnlyServer,
        }
    return None


register_worker('adas_pruneonly', call_adas_pruneonly_worker)