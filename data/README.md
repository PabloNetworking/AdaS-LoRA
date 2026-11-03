# Data

Datasets are not redistributed in this repository. Everything in `data/`
except this file is ignored by git.

## GLUE

The GLUE tasks (RTE, MRPC, CoLA, STS-B, SST-2, QNLI, QQP, MNLI) are downloaded
from the Hugging Face Hub with `datasets` on first use and cached in `data/`.
WNLI is not supported. The training split is partitioned IID across clients,
and evaluation uses the official validation split (`validation_matched` for
MNLI). GLUE test labels are not public, so no test split is used.

## 4G network traces

With `NET=trace` (the default in `scripts/run_glue.sh`), client bandwidth is
read from the 4G LTE traces of Raca et al. (MMSys 2018). The loader expects one
directory per client class, each holding trace CSV files with `DL_bitrate`
and `UL_bitrate` columns in kbit/s. `scripts/run_glue.sh` uses two classes,
`static_extended` and `pedestrian_extended`, holding the dataset's static and
pedestrian traces; class names are free and are referenced by
`federate.communication.network_trace_distribution`:

```
data/4Gnetwork_trace/
├── static_extended/       # *.csv
└── pedestrian_extended/   # *.csv
```

Uplink rates read from the traces are multiplied by
`federate.communication.trace_uplink_scale` (default 10).

Clients are assigned to classes in the proportions given by
`federate.communication.network_trace_distribution` (75% `static_extended`,
25% `pedestrian_extended` in `scripts/run_glue.sh`). To run without traces, use
`NET=generated`, which samples uplink bandwidth uniformly between
`federate.communication.uplink_min_mbps` and `uplink_max_mbps`.
