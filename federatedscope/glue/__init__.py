"""
GLUE module for federated LoRA fine-tuning.

This module provides support for GLUE benchmark datasets (CoLA, SST-2, MRPC,
STS-B, QQP, MNLI, QNLI, RTE).
WNLI is not supported.
"""

from federatedscope.glue.dataloader import load_glue_dataset, GLUEDataCollator
from federatedscope.glue.model import get_glue_model
from federatedscope.glue.trainer import GLUETrainer

__all__ = [
    'load_glue_dataset',
    'GLUEDataCollator', 
    'get_glue_model',
    'GLUETrainer',
]

