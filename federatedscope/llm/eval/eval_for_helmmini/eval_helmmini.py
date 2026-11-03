"""Lightweight HELM-Mini evaluator for FederatedScope-LLM checkpoints.

Reproduces the 4-subtask HELM-Mini proxy from FederatedScope-LLM paper A.2:
    MMLU, NaturalQuestions (open), OpenbookQA, IMDB
without the crfm-helm Docker harness. It reuses FSChatBot (same checkpoint-loading
path as eval_dolly.py / humaneval.py) plus the next-token-logit multiple-choice
scoring pattern from eval_for_mmlu/eval.py.

Scoring:
  - MMLU, OpenbookQA : 4-way MC via argmax over the A/B/C/D next-token logits
  - IMDB             : 2-way via logits of " Positive" vs " Negative"
  - NaturalQA (open) : generative, SQuAD-normalized EM + token-F1
Held-out: uses each dataset's TEST/VALIDATION split (never the training data),
capped at HELMMINI_N examples per subtask (default 100, matching the paper).

Caveat: the letter-logit MC method tracks the trend but is not numerically
identical to HELM's length-normalized calibrated method — valid for RELATIVE
comparison across checkpoints, not paper-absolute.

Usage:
    [HELMMINI_N=100] python -m federatedscope.llm.eval.eval_for_helmmini.eval_helmmini \
        --cfg <run_config.yaml> \
        federate.save_to <ckpt> device 0
Writes: <cfg.federate.save_to>_helmmini.json
"""
import os
import re
import json
import string
import collections
from itertools import islice

import torch
import numpy as np
import transformers
from datasets import load_dataset

from federatedscope.core.configs.config import global_cfg
from federatedscope.core.cmd_args import parse_args, parse_client_cfg
from federatedscope.core.auxiliaries.utils import setup_seed
from federatedscope.core.auxiliaries.logging import update_logger
from federatedscope.llm.misc.fschat import FSChatBot

transformers.logging.set_verbosity(40)

N_DEFAULT = int(os.environ.get('HELMMINI_N', '100'))


# ------------------------------------------------------------------ utilities
@torch.no_grad()
def _mc_logit_pred(model, tokenizer, device, prompt, option_words):
    """Return argmax index over the first-token logits of each option word.

    Robust to BPE space-prefix: for each option we score the max logit over both
    the bare form ("A") and the space-prefixed form (" A"), since after a prompt
    ending in "Answer:" the model emits the space-prefixed token (different id).
    """
    # Left-truncate at a generous cap so the question and the trailing "Answer:"
    # are always kept; 5-shot MMLU prompts are longer than the training tok_len.
    _prev_side = tokenizer.truncation_side
    tokenizer.truncation_side = 'left'
    input_ids = tokenizer(prompt, return_tensors="pt",
                          truncation=True, max_length=2048
                          ).input_ids.to(device)
    tokenizer.truncation_side = _prev_side
    logits = model(input_ids=input_ids).logits[0, -1]
    scores = []
    for w in option_words:
        cand_ids = set()
        for form in (w, ' ' + w.lstrip()):
            ids = tokenizer(form, add_special_tokens=False).input_ids
            if ids:
                cand_ids.add(ids[0])   # first sub-token of the option
        scores.append(max(logits[t].item() for t in cand_ids))
    return int(np.argmax(scores))


def _normalize_answer(s):
    """SQuAD normalization: lowercase, strip punctuation/articles/extra ws."""
    s = s.lower()
    s = ''.join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    return ' '.join(s.split())


def _em(pred, golds):
    p = _normalize_answer(pred)
    return float(any(p == _normalize_answer(g) for g in golds))


def _f1(pred, golds):
    p_toks = _normalize_answer(pred).split()
    best = 0.0
    for g in golds:
        g_toks = _normalize_answer(g).split()
        common = collections.Counter(p_toks) & collections.Counter(g_toks)
        num_same = sum(common.values())
        if num_same == 0:
            continue
        prec = num_same / len(p_toks) if p_toks else 0
        rec = num_same / len(g_toks) if g_toks else 0
        best = max(best, 2 * prec * rec / (prec + rec) if (prec + rec) else 0)
    return best


# ------------------------------------------------------------------ subtasks
def eval_mmlu(bot, n):
    """MMLU (all), 4-way MC. 5-shot prompt from dev, scored on test."""
    model, tok, dev = bot.model, bot.tokenizer, bot.device
    dev_ds = list(islice(load_dataset('cais/mmlu', 'all', split='dev',
                                      streaming=True), 5))
    # Build a small fixed 5-shot preamble
    letters = ['A', 'B', 'C', 'D']
    def fmt(ex, with_ans):
        s = ex['question']
        for j, c in enumerate(ex['choices']):
            s += f"\n{letters[j]}. {c}"
        s += "\nAnswer:"
        if with_ans:
            s += f" {letters[ex['answer']]}\n\n"
        return s
    preamble = ("The following are multiple choice questions "
                "(with answers).\n\n")
    for ex in dev_ds:
        preamble += fmt(ex, True)
    cors = []
    for ex in islice(load_dataset('cais/mmlu', 'all', split='test',
                                  streaming=True), n):
        prompt = preamble + fmt(ex, False)
        pred = _mc_logit_pred(model, tok, dev, prompt, letters)
        cors.append(pred == ex['answer'])
    return float(np.mean(cors)), len(cors)


def eval_openbookqa(bot, n):
    model, tok, dev = bot.model, bot.tokenizer, bot.device
    letters = ['A', 'B', 'C', 'D']
    cors = []
    for ex in islice(load_dataset('openbookqa', 'main', split='test',
                                  streaming=True), n):
        opts = ex['choices']['text']
        labels = ex['choices']['label']
        prompt = ex['question_stem']
        for lab, txt in zip(labels, opts):
            prompt += f"\n{lab}. {txt}"
        prompt += "\nAnswer:"
        gold = labels.index(ex['answerKey']) if ex['answerKey'] in labels else 0
        pred = _mc_logit_pred(model, tok, dev, prompt, letters[:len(opts)])
        cors.append(pred == gold)
    return float(np.mean(cors)), len(cors)


def eval_imdb(bot, n):
    """IMDB sentiment, 2-way via ' Positive' vs ' Negative' logits."""
    model, tok, dev = bot.model, bot.tokenizer, bot.device
    cors = []
    # balance: take n/2 from each end of the test split (test is neg-first)
    stream = load_dataset('imdb', split='test', streaming=True)
    rows = list(islice(stream, n // 2))          # negatives
    # positives are at the tail; stream a shuffled buffer for the other half
    stream2 = load_dataset('imdb', split='test', streaming=True).shuffle(seed=42, buffer_size=2000)
    rows += list(islice((r for r in stream2 if r['label'] == 1), n - len(rows)))
    for ex in rows:
        text = ex['text'][:1500]  # truncate long reviews
        prompt = (f"Review: {text}\n"
                  f"Is the sentiment of this review positive or negative?\n"
                  f"Answer:")
        pred = _mc_logit_pred(model, tok, dev, prompt, [' Positive', ' Negative'])
        pred_label = 1 if pred == 0 else 0  # option 0 = Positive
        cors.append(pred_label == ex['label'])
    return float(np.mean(cors)), len(cors)


def eval_nq_open(bot, n):
    """NaturalQuestions-open (closed-book proxy via nq_open): generative EM/F1."""
    ems, f1s = [], []
    gen_kwargs = dict(max_new_tokens=32, do_sample=False)
    for ex in islice(load_dataset('nq_open', split='validation',
                                  streaming=True), n):
        q = ex['question']
        golds = ex['answer']
        prompt = f"Question: {q}\nAnswer:"
        out = bot.generate(prompt, generate_kwargs=gen_kwargs)
        if isinstance(out, list):
            out = out[0]
        # keep only the first line/sentence of the answer
        out = out.strip().split('\n')[0]
        ems.append(_em(out, golds))
        f1s.append(_f1(out, golds))
    return float(np.mean(ems)), float(np.mean(f1s)), len(ems)


def main():
    init_cfg = global_cfg.clone()
    args = parse_args()
    if args.cfg_file:
        init_cfg.merge_from_file(args.cfg_file)
    cfg_opt, _ = parse_client_cfg(args.opts)
    init_cfg.merge_from_list(cfg_opt)
    update_logger(init_cfg, clear_before_add=True)
    setup_seed(init_cfg.seed)

    n = int(os.environ.get('HELMMINI_N', N_DEFAULT))
    bot = FSChatBot(init_cfg)

    report = {}
    print("[helmmini] MMLU ...", flush=True)
    report['mmlu_acc'], report['mmlu_n'] = eval_mmlu(bot, n)
    print(f"  mmlu_acc={report['mmlu_acc']:.4f}", flush=True)

    print("[helmmini] OpenbookQA ...", flush=True)
    report['openbookqa_acc'], report['openbookqa_n'] = eval_openbookqa(bot, n)
    print(f"  openbookqa_acc={report['openbookqa_acc']:.4f}", flush=True)

    print("[helmmini] IMDB ...", flush=True)
    report['imdb_acc'], report['imdb_n'] = eval_imdb(bot, n)
    print(f"  imdb_acc={report['imdb_acc']:.4f}", flush=True)

    print("[helmmini] NQ-open ...", flush=True)
    report['nq_em'], report['nq_f1'], report['nq_n'] = eval_nq_open(bot, n)
    print(f"  nq_em={report['nq_em']:.4f} nq_f1={report['nq_f1']:.4f}", flush=True)

    # HELM-Mini aggregate = unweighted mean of the 4 primary scores
    #   (MMLU acc, OpenbookQA acc, IMDB acc, NQ f1)
    report['helmmini_avg'] = float(np.mean([
        report['mmlu_acc'], report['openbookqa_acc'],
        report['imdb_acc'], report['nq_f1']]))

    out_path = init_cfg.federate.save_to + '_helmmini.json'
    with open(out_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"[helmmini] wrote {out_path} | avg={report['helmmini_avg']:.4f}",
          flush=True)


if __name__ == "__main__":
    main()
