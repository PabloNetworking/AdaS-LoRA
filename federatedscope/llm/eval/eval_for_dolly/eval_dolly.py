from typing import Any


import os

import collections
import json
import numpy as np
import transformers
from tqdm import tqdm
from rouge import Rouge

from federatedscope.core.configs.config import global_cfg
from federatedscope.core.cmd_args import parse_args, parse_client_cfg
from federatedscope.core.auxiliaries.utils import setup_seed
from federatedscope.core.auxiliaries.logging import update_logger
from federatedscope.core.data.utils import download_url
from federatedscope.llm.dataloader.dataloader import load_jsonl
from federatedscope.llm.misc.fschat import FSChatBot

transformers.logging.set_verbosity(40)

DEBUG = True

rouge = Rouge()


def rouge_score(hyps, refs):
    try:
        rouge_score = rouge.get_scores(hyps, refs)[0]['rouge-l']['f']
    except ValueError:
        return 0.0
    return rouge_score


def main():
    init_cfg = global_cfg.clone()
    args = parse_args()

    if args.cfg_file:
        init_cfg.merge_from_file(args.cfg_file)
    cfg_opt, client_cfg_opt = parse_client_cfg(args.opts)
    init_cfg.merge_from_list(cfg_opt)

    update_logger(init_cfg, clear_before_add=True)
    setup_seed(init_cfg.seed)

    # load your finetuned model (saved as xxx.ckpt)
    #    in yaml file federate.save_to
    fschatbot = FSChatBot(init_cfg)

    # Dolly-15K data file (every sample is evaluated)
    fp = os.path.join(init_cfg.data.root, "databricks-dolly-15k.jsonl")
    if not os.path.exists(fp):
        download_url(
            'https://raw.githubusercontent.com/databrickslabs'
            '/dolly/d000e3030970379aabbf6d291f50ffdd3b715b64'
            '/data/databricks-dolly-15k.jsonl', init_cfg.data.root)
        os.rename(os.path.join(init_cfg.data.root, 'test.jsonl'), fp)

    list_data_dict = load_jsonl(fp,
                                instruction='instruction',
                                input='context',
                                output='response',
                                category='category')

    # Score every Dolly category and report Rouge-L per category. If the
    # DEBUG_SAMPLES env var is set, evaluate only the first N samples.
    debug_samples = os.environ.get('DEBUG_SAMPLES')
    if debug_samples:
        list_data_dict = list_data_dict[:int(debug_samples)]

    answers = collections.defaultdict(list)
    for sample in tqdm(list_data_dict):
        input_text = sample['instruction']
        context = sample.get('input', '')
        generate_kwargs = dict(max_new_tokens=256, top_p=0.95, temperature=0.8)
        model_answer = fschatbot.generate(input_text, context, generate_kwargs)
        rougel_cor = rouge_score(model_answer, sample['output'])
        answers[sample['category']].append(rougel_cor)
        if DEBUG:
            print(f'[{sample["category"]}] rouge-L={rougel_cor:.4f} | '
                  f'A: {model_answer[:160]}')

    per_cat = {c: {'mean': float(np.mean(v)), 'n': len(v)}
               for c, v in answers.items()}
    overall_unweighted = float(np.mean([per_cat[c]['mean'] for c in per_cat])) \
        if per_cat else 0.0
    all_scores = [s for v in answers.values() for s in v]
    overall_weighted = float(np.mean(all_scores)) if all_scores else 0.0

    out_path = init_cfg.federate.save_to + '_dolly_rougel_per_cat.json'
    with open(out_path, 'w') as fh:
        json.dump({'per_category': per_cat,
                   'overall_unweighted': overall_unweighted,
                   'overall_weighted': overall_weighted,
                   'n_samples': len(all_scores)}, fh, indent=2)
    print(f'[eval_dolly] wrote {out_path} | '
          f'overall_unweighted={overall_unweighted:.4f} '
          f'weighted={overall_weighted:.4f} n={len(all_scores)}')


if __name__ == "__main__":
    main()