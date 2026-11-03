"""Worker wiring for AdaS-LayerAware."""

import logging

from federatedscope.register import register_worker
from federatedscope.contrib.worker.methods.adas_layeraware_client import AdaSLayerAwareClient
from federatedscope.contrib.worker.methods.adas_layeraware_server import AdaSLayerAwareServer

logger = logging.getLogger(__name__)

_METHOD_ALIASES = {'adas_layeraware', 'adas-layeraware'}


def call_adas_layeraware_worker(method):
    if method is None:
        return None
    if method.lower() in _METHOD_ALIASES:
        logger.info('Using AdaS-LayerAware client and server')
        return {
            'client': AdaSLayerAwareClient,
            'server': AdaSLayerAwareServer,
        }
    return None


register_worker('adas_layeraware', call_adas_layeraware_worker)
