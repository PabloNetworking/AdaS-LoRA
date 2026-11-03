"""FAH-QLoRA evaluation helper: validation loss and evaluation time over the first
validation batches, shared by the GLUE and LLM trainers."""

import logging
import time
from typing import Tuple

import torch

logger = logging.getLogger(__name__)


def evaluate_for_fah_impl(trainer, rank: int, validation_fraction: float = 0.1, validation_steps: int = 10) -> Tuple[float, float]:
    del rank, validation_fraction
    ctx = trainer.ctx
    ctx.model.eval()

    val_loader = ctx.get('val_loader', None)
    if val_loader is None:
        logger.warning('[FAH] No validation loader for FAH evaluation')
        return 0.0, 0.0

    total_loss = 0.0
    total_samples = 0
    start_time = time.time()

    with torch.no_grad():
        for step, batch in enumerate(val_loader):
            if step >= validation_steps:
                break

            input_ids = batch['input_ids'].to(ctx.device)
            attention_mask = batch['attention_mask'].to(ctx.device)
            labels = batch['labels'].to(ctx.device)
            token_type_ids = batch.get('token_type_ids', None)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(ctx.device)

            outputs = ctx.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels,
            )
            loss = outputs.loss
            if not torch.isnan(loss):
                total_loss += loss.item() * len(labels)
                total_samples += len(labels)

    compute_time = time.time() - start_time
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    ctx.model.train()
    return avg_loss, compute_time