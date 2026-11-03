#!/usr/bin/env bash
# Run one method on one GLUE task with simulated (standalone) federated clients.
#
#   bash scripts/run_glue.sh <method> <task> [key value ...]
#
#   method  fedit | hetlora | fah_qlora | adas | adas_layeraware | adas_pruneonly
#   task    rte | mrpc | cola | stsb | sst2 | qnli | qqp | mnli
#
# Trailing "key value" pairs are passed to federatedscope/main.py and override
# everything set here, e.g.  bash scripts/run_glue.sh adas rte seed 51
#
# Environment variables (defaults in brackets):
#   DEVICE   GPU index [0]
#   NET      client bandwidth: trace (4G traces in data/4Gnetwork_trace) or generated [trace]
#   RANK     largest client LoRA rank [200]
#   GAMMA    pruning intensity: HetLoRA decay, AdaS eta [0.65]
#   RW       pruning regularizer weight (HetLoRA, AdaS) [0.05]
#   UL_S     AdaS uplink window in seconds [230]
#   DL_S     AdaS downlink window in seconds [63]
#   INIT_RANK, LAMBDA   FAH-QLoRA initial rank and lambda_inc = lambda_dec [64, 5]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

usage="usage: bash scripts/run_glue.sh <method> <task> [key value ...]"
method=${1:?$usage}
task=${2:?$usage}
shift 2

DEVICE=${DEVICE:-0}
NET=${NET:-trace}
RANK=${RANK:-200}
GAMMA=${GAMMA:-0.65}
RW=${RW:-0.05}
UL_S=${UL_S:-230}
DL_S=${DL_S:-63}
INIT_RANK=${INIT_RANK:-64}
LAMBDA=${LAMBDA:-5}

# Per-task federation settings: 6 clients for the small tasks, 12 for the large ones.
case "$task" in
    rte|mrpc|cola|stsb) clients=6 ;;
    sst2|qnli|qqp|mnli) clients=12 ;;
    *) echo "unknown GLUE task '$task'" >&2; echo "$usage" >&2; exit 1 ;;
esac
case "$task" in qqp|mnli) rounds=30 ;; *) rounds=20 ;; esac
case "$task" in mrpc|rte) steps=15 ;; *) steps=30 ;; esac
case "$task" in
    stsb) key=val_pearson ;;
    cola) key=val_mcc ;;
    mrpc|qqp) key=val_f1 ;;
    *) key=val_acc ;;
esac

args=(
    device "$DEVICE"
    data.type "${task}@glue"
    federate.client_num "$clients"
    federate.sample_client_num "$clients"
    federate.total_round_num "$rounds"
    train.local_update_steps "$steps"
    eval.best_res_update_round_wise_key "$key"
)

case "$NET" in
    trace)
        args+=(
            federate.communication.source trace
            federate.communication.network_trace_path data/4Gnetwork_trace/
            federate.communication.network_trace_distribution.static_extended 75.0
            federate.communication.network_trace_distribution.pedestrian_extended 25.0
        ) ;;
    generated) args+=(federate.communication.source generated) ;;
    *) echo "NET must be 'trace' or 'generated'" >&2; exit 1 ;;
esac

# Adapter settings are read from both glue.adapter and llm.adapter, so set both.
adapter() {
    args+=("glue.adapter.$1" "$2" "llm.adapter.$1" "$2")
}

adapter max_rank "$RANK"
case "$method" in
    fedit)
        config=configs/fedit.yaml
        adapter hetero_strategy homo ;;
    hetlora)
        config=configs/hetlora.yaml
        adapter hetero_strategy custom
        adapter hetlora.rank_max "$RANK"
        adapter hetlora.pruning.decay "$GAMMA"
        adapter hetlora.pruning.regularizer_weight "$RW" ;;
    fah_qlora)
        config=configs/fah_qlora.yaml
        adapter hetero_strategy custom
        adapter fah.r_max "$RANK"
        adapter fah.init_rank "$INIT_RANK"
        adapter fah.lambda_inc "$LAMBDA"
        adapter fah.lambda_dec "$LAMBDA" ;;
    adas|adas_layeraware)
        config=configs/$method.yaml
        adapter hetero_strategy custom
        adapter "$method.rank_max" "$RANK"
        adapter "$method.stage1.gamma" "$GAMMA"
        adapter "$method.stage1.regularizer_weight" "$RW"
        adapter "$method.stage2.uplink_budget_window_s" "$UL_S"
        adapter "$method.stage2.downlink_budget_window_s" "$DL_S"
        adapter "$method.stage2.residual_enabled" True ;;
    adas_pruneonly)
        config=configs/adas_pruneonly.yaml
        adapter hetero_strategy custom
        adapter adas_pruneonly.rank_max "$RANK"
        adapter adas_pruneonly.pruning.gamma "$GAMMA"
        adapter adas_pruneonly.pruning.regularizer_weight "$RW" ;;
    *) echo "unknown method '$method'" >&2; echo "$usage" >&2; exit 1 ;;
esac

set -x
python federatedscope/main.py --cfg "$config" "${args[@]}" "$@"
