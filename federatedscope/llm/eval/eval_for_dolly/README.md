# Rouge-L

## Dolly-15K
`eval_dolly.py` generates an answer for every Dolly-15K sample with the fine-tuned model
(loaded from `federate.save_to`) and scores it against the reference response with Rouge-L.
It reports the mean Rouge-L per category and overall, and writes them to
`{federate.save_to}_dolly_rougel_per_cat.json`.

To do the evaluation, run
```bash
python federatedscope/llm/eval/eval_for_dolly/eval_dolly.py --cfg <config.yaml>
```