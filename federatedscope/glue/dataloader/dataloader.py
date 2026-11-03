"""
GLUE dataloader utilities for federated sequence classification and regression.

Supported GLUE tasks in this module:
- CoLA
- SST-2
- MRPC
- STS-B
- QQP
- MNLI
- QNLI
- RTE

WNLI is not supported.
"""
import logging
import torch
import transformers
from dataclasses import dataclass
from typing import Dict, List
from torch.utils.data import Dataset, Subset

logger = logging.getLogger(__name__)


GLUE_TASK_INFO = {
    'cola': {
        'hf_subset_name': 'cola',
        'num_labels': 2,
        'task_type': 'single_sentence',
        'text_fields': ['sentence'],
        'label_field': 'label',
        'metric': 'mcc',
        'model_task': 'SequenceClassification',
        'criterion': 'CrossEntropyLoss',
    },
    'sst2': {
        'hf_subset_name': 'sst2',
        'num_labels': 2,
        'task_type': 'single_sentence',
        'text_fields': ['sentence'],
        'label_field': 'label',
        'metric': 'acc',
        'model_task': 'SequenceClassification',
        'criterion': 'CrossEntropyLoss',
    },
    'mrpc': {
        'hf_subset_name': 'mrpc',
        'num_labels': 2,
        'task_type': 'sentence_pair',
        'text_fields': ['sentence1', 'sentence2'],
        'label_field': 'label',
        'metric': 'f1',
        'model_task': 'SequenceClassification',
        'criterion': 'CrossEntropyLoss',
    },
    'stsb': {
        'hf_subset_name': 'stsb',
        'num_labels': 1,
        'task_type': 'sentence_pair',
        'text_fields': ['sentence1', 'sentence2'],
        'label_field': 'label',
        'metric': 'spearman',
        'model_task': 'SequenceRegression',
        'criterion': 'MSELoss',
    },
    'qqp': {
        'hf_subset_name': 'qqp',
        'num_labels': 2,
        'task_type': 'sentence_pair',
        'text_fields': ['question1', 'question2'],
        'label_field': 'label',
        'metric': 'f1',
        'model_task': 'SequenceClassification',
        'criterion': 'CrossEntropyLoss',
    },
    'mnli': {
        'hf_subset_name': 'mnli',
        'num_labels': 3,
        'task_type': 'sentence_pair',
        'text_fields': ['premise', 'hypothesis'],
        'label_field': 'label',
        'metric': 'acc',
        'model_task': 'SequenceClassification',
        'criterion': 'CrossEntropyLoss',
    },
    'qnli': {
        'hf_subset_name': 'qnli',
        'num_labels': 2,
        'task_type': 'sentence_pair',
        'text_fields': ['question', 'sentence'],
        'label_field': 'label',
        'metric': 'acc',
        'model_task': 'SequenceClassification',
        'criterion': 'CrossEntropyLoss',
    },
    'rte': {
        'hf_subset_name': 'rte',
        'num_labels': 2,
        'task_type': 'sentence_pair',
        'text_fields': ['sentence1', 'sentence2'],
        'label_field': 'label',
        'metric': 'acc',
        'model_task': 'SequenceClassification',
        'criterion': 'CrossEntropyLoss',
    },
}


class GLUEDataset(Dataset):
    """
    Dataset wrapper for GLUE tasks.

    Supports both single sentence tasks and sentence pair tasks, including
    STS-B regression.
    """

    def __init__(
        self,
        examples: List[Dict],
        tokenizer: transformers.PreTrainedTokenizer,
        task_name: str,
        max_length: int = 128,
    ):
        super().__init__()

        self.examples = examples
        self.tokenizer = tokenizer
        self.task_name = normalize_glue_task_name(task_name)
        self.max_length = max_length

        if self.task_name not in GLUE_TASK_INFO:
            raise ValueError(f"Unknown GLUE task: {task_name}")

        self.task_info = GLUE_TASK_INFO[self.task_name]
        self.is_regression = self.task_info['num_labels'] == 1
        self.encodings = self._tokenize_examples()

    def _tokenize_examples(self) -> Dict[str, torch.Tensor]:
        text_fields = self.task_info['text_fields']

        if len(text_fields) == 1:
            texts = [ex[text_fields[0]] for ex in self.examples]
            encodings = self.tokenizer(
                texts,
                truncation=True,
                max_length=self.max_length,
                padding='max_length',
                return_tensors='pt',
            )
        else:
            texts_a = [ex[text_fields[0]] for ex in self.examples]
            texts_b = [ex[text_fields[1]] for ex in self.examples]
            encodings = self.tokenizer(
                texts_a,
                texts_b,
                truncation=True,
                max_length=self.max_length,
                padding='max_length',
                return_tensors='pt',
            )

        label_field = self.task_info['label_field']
        if self.is_regression:
            labels = torch.tensor(
                [float(ex[label_field]) for ex in self.examples],
                dtype=torch.float32,
            )
        else:
            labels = torch.tensor(
                [int(ex[label_field]) for ex in self.examples],
                dtype=torch.long,
            )

        return {
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'token_type_ids': encodings.get(
                'token_type_ids', torch.zeros_like(encodings['input_ids'])
            ),
            'labels': labels,
        }

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'token_type_ids': self.encodings['token_type_ids'][idx],
            'labels': self.encodings['labels'][idx],
        }


@dataclass
class GLUEDataCollator:
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: List[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = torch.stack([inst['input_ids'] for inst in instances])
        attention_mask = torch.stack([inst['attention_mask'] for inst in instances])
        token_type_ids = torch.stack([inst['token_type_ids'] for inst in instances])
        labels = torch.stack([inst['labels'] for inst in instances])

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'labels': labels,
        }


def normalize_glue_task_name(task_name: str) -> str:
    return task_name.lower().replace('-', '').strip()


def get_glue_tokenizer(
    model_name: str,
    cache_dir: str,
    max_length: int = 128,
) -> transformers.PreTrainedTokenizer:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        model_max_length=max_length,
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    return tokenizer


def _get_mnli_split_names(config):
    mnli_eval_split = getattr(config.glue, 'mnli_eval_split', 'matched').lower()
    if mnli_eval_split not in ('matched', 'mismatched'):
        raise ValueError(
            f"Unsupported glue.mnli_eval_split={mnli_eval_split}. "
            f"Use 'matched' or 'mismatched'."
        )

    if mnli_eval_split == 'mismatched':
        return 'validation_mismatched', 'test_mismatched'
    return 'validation_matched', 'test_matched'


def _filter_labeled_examples(examples, label_field='label'):
    filtered = []
    for ex in examples:
        if label_field not in ex:
            continue
        label = ex[label_field]
        if label is None:
            continue
        if isinstance(label, int) and label < 0:
            continue
        filtered.append(ex)
    return filtered


def _apply_glue_task_defaults(config, task_name: str):
    task_info = GLUE_TASK_INFO[task_name]

    config.defrost()
    config.glue.task_name = task_name
    config.model.out_channels = task_info['num_labels']
    config.model.task = task_info['model_task']
    if hasattr(config, 'criterion'):
        config.criterion.type = task_info['criterion']
    config.freeze()


def load_glue_dataset(config=None, **kwargs):
    """
    Load a GLUE dataset for federated training.

    Supported tasks: CoLA, SST-2, MRPC, STS-B, QQP, MNLI, QNLI, RTE.
    WNLI is not supported.
    """
    from datasets import load_dataset as hf_load_dataset

    model_name, _ = config.model.type.split('@')
    dataset_name, _ = config.data.type.split('@')

    task_name = normalize_glue_task_name(dataset_name)
    if task_name == 'wnli':
        raise ValueError(
            "WNLI is not supported. "
            "Use another GLUE task instead."
        )
    if task_name not in GLUE_TASK_INFO:
        raise ValueError(
            f"Unknown GLUE task: {dataset_name}. "
            f"Supported tasks: {list(GLUE_TASK_INFO.keys())}"
        )

    task_info = GLUE_TASK_INFO[task_name]

    max_length = getattr(config.glue, 'max_length', 128)
    if hasattr(config.data, 'args') and config.data.args:
        for arg in config.data.args:
            if isinstance(arg, dict) and 'max_len' in arg:
                max_length = arg['max_len']

    tokenizer = get_glue_tokenizer(model_name, config.data.root, max_length)

    logger.info(
        "Loading GLUE task '%s' from HuggingFace datasets...",
        task_info['hf_subset_name'],
    )
    try:
        raw_datasets = hf_load_dataset(
            'glue',
            task_info['hf_subset_name'],
            cache_dir=config.data.root,
        )
    except Exception as e:
        logger.error(f"Failed to load GLUE dataset: {e}")
        raise

    train_examples = list(raw_datasets['train'])

    if task_name == 'mnli':
        val_split, test_split = _get_mnli_split_names(config)
    else:
        val_split, test_split = 'validation', 'test'

    if val_split in raw_datasets:
        val_examples = list(raw_datasets[val_split])
    else:
        val_examples = []
        logger.warning("No validation split found for %s", task_name)

    if test_split in raw_datasets:
        test_examples = list(raw_datasets[test_split])
    else:
        test_examples = []

    logger.info(
        "Loaded GLUE %s: train=%d, val=%d, test=%d",
        task_name,
        len(train_examples),
        len(val_examples),
        len(test_examples),
    )

    train_dataset = GLUEDataset(train_examples, tokenizer, task_name, max_length)

    # Keep val/test as Dataset objects even when they are empty. The
    # FederatedScope base translator calls len(test), so returning None here
    # breaks on GLUE tasks whose HuggingFace test split has hidden labels.
    empty_dataset = Subset(train_dataset, [])

    val_dataset = (
        GLUEDataset(val_examples, tokenizer, task_name, max_length)
        if val_examples else empty_dataset
    )

    labeled_test_examples = _filter_labeled_examples(
        test_examples,
        label_field=task_info['label_field'],
    )
    test_dataset = (
        GLUEDataset(labeled_test_examples, tokenizer, task_name, max_length)
        if labeled_test_examples else empty_dataset
    )

    _apply_glue_task_defaults(config, task_name)

    dataset = (train_dataset, val_dataset, test_dataset)
    return dataset, config