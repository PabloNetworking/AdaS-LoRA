import copy
import json
import logging
import os
import gzip
import shutil
import datetime
from collections import defaultdict
from importlib import import_module

import numpy as np

from federatedscope.core.auxiliaries.logging import logline_2_wandb_dict
from federatedscope.core.monitors.metric_calculator import MetricCalculator

try:
    import torch
except ImportError:
    torch = None

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

global_all_monitors = [
]  # used in standalone mode, to merge sys metric results for all workers


class Monitor(object):
    """
    Provide the monitoring functionalities such as formatting the \
    evaluation results into diverse metrics. \
    Besides the prediction related performance, the monitor also can \
    track efficiency related metrics for a worker

    Args:
        cfg: a cfg node object
        monitored_object: object to be monitored

    Attributes:
        log_res_best: best ever seen results
        outdir: output directory
        use_wandb: whether use ``wandb``
        wandb_online_track: whether use ``wandb`` to track online
        monitored_object: object to be monitored
        metric_calculator: metric calculator, /
            see ``core.monitors.metric_calculator``
        round_wise_update_key: key to decide which result of evaluation \
            round is better
    """
    SUPPORTED_FORMS = ['weighted_avg', 'avg', 'fairness', 'raw']

    def __init__(self, cfg, monitored_object=None):
        self.cfg = cfg
        self.log_res_best = {}
        self.outdir = cfg.outdir
        self.use_wandb = cfg.wandb.use
        self.wandb_online_track = cfg.wandb.online_track if cfg.wandb.use \
            else False
        # self.use_tensorboard = cfg.use_tensorboard

        self.monitored_object = monitored_object
        self.metric_calculator = MetricCalculator(cfg.eval.metrics)

        # Obtain the whether the larger the better
        self.round_wise_update_key = cfg.eval.best_res_update_round_wise_key
        for mode in ['train', 'val', 'test']:
            if mode in self.round_wise_update_key:
                update_key = self.round_wise_update_key.split(f'{mode}_')[1]
        assert update_key in self.metric_calculator.eval_metric, \
            f'{update_key} not found in metrics.'
        self.the_larger_the_better = self.metric_calculator.eval_metric[
            update_key][1]

        # =======  efficiency indicators of the worker to be monitored =======
        # leveraged the flops counter provided by [fvcore](
        # https://github.com/facebookresearch/fvcore)
        self.total_model_size = 0  # model size used in the worker, in terms
        # of number of parameters
        self.flops_per_sample = 0  # average flops for forwarding each data
        # sample
        self.flop_count = 0  # used to calculated the running mean for
        # flops_per_sample
        self.total_flops = 0  # total computation flops to convergence until
        # current fl round
        self.total_upload_bytes = 0  # total upload space cost in bytes
        # until current fl round
        self.total_download_bytes = 0  # total download space cost in bytes
        # until current fl round
        self.fl_begin_wall_time = datetime.datetime.now()
        self.fl_end_wall_time = 0
        # for the metrics whose names includes "convergence", 0 indicates
        # the worker does not converge yet
        # Note:
        # 1) the convergence wall time is prone to fluctuations due to
        # possible resource competition during FL courses
        # 2) the global/local indicates whether the early stopping triggered
        # with global-aggregation/local-training
        self.global_convergence_round = 0  # total fl rounds to convergence
        self.global_convergence_wall_time = 0
        self.local_convergence_round = 0  # total fl rounds to convergence
        self.local_convergence_wall_time = 0

        # =======  Extended per-round metrics tracking  =======
        # system_metrics_mode from config (registered default "extended"; "legacy" if unset)
        self.system_metrics_mode = getattr(
            getattr(cfg, 'monitor', None), 'system_metrics_mode', 'legacy'
        ) if hasattr(cfg, 'monitor') else 'legacy'
        
        # Per-round time metrics (seconds)
        # {round_id: {compute_seconds, upload_seconds, download_seconds}}
        self.sys_time_per_round = {}
        
        # Per-round communication metrics (bytes)
        # {round_id: {upload_bytes, download_bytes}}
        self.sys_comm_per_round = {}
        
        # Per-round memory metrics (bytes)
        # {round_id: {frozen_model_bytes, lora_bytes, activations_bytes,
        #             optimizer_bytes, gradients_bytes, final_memory_bytes}}
        self.sys_mem_per_round = {}
        
        # Total number of trainable LoRA parameters (constant per client)
        self.sys_trained_params = 0
        
        # Store per-round bandwidth for time calculations (kbit/s)
        # {round_id: {upload_bandwidth_kbits, download_bandwidth_kbits}}
        self.sys_bandwidth_per_round = {}

        if self.wandb_online_track:
            global_all_monitors.append(self)
        if self.use_wandb:
            try:
                import wandb
            except ImportError:
                logger.error(
                    "cfg.wandb.use=True but not install the wandb package")
                exit()

    def eval(self, ctx):
        """
        Evaluates the given context with ``metric_calculator``.

        Args:
            ctx: context of trainer, see ``core.trainers.context``

        Returns:
            Evaluation results.
        """
        results = self.metric_calculator.eval(ctx)
        return results

    def global_converged(self):
        """
        Calculate wall time and round when global convergence has been reached.
        """
        self.global_convergence_wall_time = datetime.datetime.now(
        ) - self.fl_begin_wall_time
        self.global_convergence_round = self.monitored_object.state

    def local_converged(self):
        """
        Calculate wall time and round when local convergence has been reached.
        """
        self.local_convergence_wall_time = datetime.datetime.now(
        ) - self.fl_begin_wall_time
        self.local_convergence_round = self.monitored_object.state

    def finish_fl(self):
        """
        When FL finished, write system metrics to file.
        Supports both "legacy" and "extended" modes.
        """
        self.fl_end_wall_time = datetime.datetime.now(
        ) - self.fl_begin_wall_time

        sys_metric_f_name = os.path.join(self.outdir, "system_metrics.log")
        
        if self.system_metrics_mode == 'extended':
            # Use extended generic format
            system_metrics = self.get_extended_sys_metrics()
        else:
            # Use legacy format
            system_metrics = self.get_sys_metrics()
        
        with open(sys_metric_f_name, "a") as f:
            f.write(json.dumps(system_metrics) + "\n")

    def get_sys_metrics(self, verbose=True):
        system_metrics = {
            "id": self.monitored_object.ID,
            "fl_end_time_minutes": self.fl_end_wall_time.total_seconds() /
            60 if isinstance(self.fl_end_wall_time, datetime.timedelta) else 0,
            "total_model_size": self.total_model_size,
            "total_flops": self.total_flops,
            "total_upload_bytes": self.total_upload_bytes,
            "total_download_bytes": self.total_download_bytes,
            "global_convergence_round": self.global_convergence_round,
            "local_convergence_round": self.local_convergence_round,
            "global_convergence_time_minutes": self.
            global_convergence_wall_time.total_seconds() / 60 if isinstance(
                self.global_convergence_wall_time, datetime.timedelta) else 0,
            "local_convergence_time_minutes": self.local_convergence_wall_time.
            total_seconds() / 60 if isinstance(
                self.local_convergence_wall_time, datetime.timedelta) else 0,
        }
        if verbose:
            logger.info(
                f"In worker #{self.monitored_object.ID}, the system-related "
                f"metrics are: {str(system_metrics)}")
        return system_metrics

    def get_extended_sys_metrics(self, verbose=True):
        """
        Generate extended system metrics in generic extended format.
        
        This produces a JSON entry with:
        - Per-client entry: total time, total communication, memory breakdown,
          per-round metrics, trained LoRA parameter count
        
        All times in minutes, all sizes in MB, rounded to 4 decimal places.
        
        Returns:
            dict: Extended system metrics for this worker
        """
        BYTES_TO_MB = 1024 * 1024
        SECONDS_TO_MINUTES = 60
        DECIMAL_PLACES = 4
        
        def _round(val):
            return round(val, DECIMAL_PLACES)
        
        # Aggregate per-round time metrics
        total_compute_seconds = 0.0
        total_upload_seconds = 0.0
        total_download_seconds = 0.0
        per_round_time = {}
        
        for round_id, time_data in sorted(self.sys_time_per_round.items()):
            A_r = time_data.get('compute_seconds', 0.0)
            B_r = time_data.get('upload_seconds', 0.0)
            C_r = time_data.get('download_seconds', 0.0)
            X_r = A_r + B_r + C_r
            
            total_compute_seconds += A_r
            total_upload_seconds += B_r
            total_download_seconds += C_r
            
            per_round_time[f"round{round_id}"] = {
                "training_time": _round(X_r / SECONDS_TO_MINUTES),
                "computing_time": _round(A_r / SECONDS_TO_MINUTES),
                "uploading_time": _round(B_r / SECONDS_TO_MINUTES),
                "downloading_time": _round(C_r / SECONDS_TO_MINUTES),
            }
        
        total_training_seconds = (total_compute_seconds + 
                                  total_upload_seconds + 
                                  total_download_seconds)
        
        # Aggregate per-round communication metrics
        total_upload_bytes = 0
        total_download_bytes = 0
        per_round_comm = {}
        
        for round_id, comm_data in sorted(self.sys_comm_per_round.items()):
            L_r = comm_data.get('upload_bytes', 0)
            M_r = comm_data.get('download_bytes', 0)
            K_r = L_r + M_r

            total_upload_bytes += L_r
            total_download_bytes += M_r

            round_comm = {
                "communicated_megabytes": _round(K_r / BYTES_TO_MB),
                "uploaded_megabytes": _round(L_r / BYTES_TO_MB),
                "downloaded_megabytes": _round(M_r / BYTES_TO_MB),
            }
            per_round_comm[f"round{round_id}"] = round_comm
        
        total_communicated_bytes = total_upload_bytes + total_download_bytes
        
        # Get memory metrics from the last round (final state)
        per_round_mem = {}
        final_memory = {}
        
        if self.sys_mem_per_round:
            last_round_id = max(self.sys_mem_per_round.keys())
            
            for round_id, mem_data in sorted(self.sys_mem_per_round.items()):
                D_r = mem_data.get('frozen_model_bytes', 0)
                E_r = mem_data.get('lora_bytes', 0)
                F_r = mem_data.get('activations_bytes', 0)
                G_r = mem_data.get('optimizer_bytes', 0)
                H_r = mem_data.get('gradients_bytes', 0)
                Y_r = mem_data.get('final_memory_bytes', D_r + E_r + F_r + G_r + H_r)
                
                per_round_mem[f"round{round_id}"] = {
                    "final_memory": _round(Y_r / BYTES_TO_MB),
                    "final_memory_frozenModel": _round(D_r / BYTES_TO_MB),
                    "final_memory_LoRAWeights": _round(E_r / BYTES_TO_MB),
                    "final_memory_Activations": _round(F_r / BYTES_TO_MB),
                    "final_memory_OptimizerStates": _round(G_r / BYTES_TO_MB),
                    "final_memory_Gradients": _round(H_r / BYTES_TO_MB),
                }

                # Optional CUDA debug fields (MB)
                cuda_map = {
                    "cuda_baseline_allocated_mb": "cuda_baseline_allocated",
                    "cuda_peak_allocated_mb": "cuda_peak_allocated",
                    "cuda_peak_delta_allocated_mb": "cuda_peak_delta_allocated",
                    "cuda_baseline_reserved_mb": "cuda_baseline_reserved",
                    "cuda_peak_reserved_mb": "cuda_peak_reserved",
                    "cuda_peak_delta_reserved_mb": "cuda_peak_delta_reserved",
                    "cuda_accounted_dynamic_mb": "cuda_accounted_dynamic_bytes",
                    "cuda_activations_est_mb": "cuda_activations_est_bytes",
                }
                for out_k, in_k in cuda_map.items():
                    if in_k in mem_data:
                        per_round_mem[f"round{round_id}"][out_k] = _round(mem_data[in_k] / BYTES_TO_MB)
            
            # Final memory from last round
            last_mem = self.sys_mem_per_round[last_round_id]
            D = last_mem.get('frozen_model_bytes', 0)
            E = last_mem.get('lora_bytes', 0)
            F = last_mem.get('activations_bytes', 0)
            G = last_mem.get('optimizer_bytes', 0)
            H = last_mem.get('gradients_bytes', 0)
            Y = last_mem.get('final_memory_bytes', D + E + F + G + H)
            
            final_memory = {
                "final_memory": _round(Y / BYTES_TO_MB),
                "final_memory_frozenModel": _round(D / BYTES_TO_MB),
                "final_memory_LoRAWeights": _round(E / BYTES_TO_MB),
                "final_memory_Activations": _round(F / BYTES_TO_MB),
                "final_memory_OptimizerStates": _round(G / BYTES_TO_MB),
                "final_memory_Gradients": _round(H / BYTES_TO_MB),
            }

            # Optional CUDA debug fields (MB) from last round
            cuda_map_final = {
                "cuda_baseline_allocated_mb": "cuda_baseline_allocated",
                "cuda_peak_allocated_mb": "cuda_peak_allocated",
                "cuda_peak_delta_allocated_mb": "cuda_peak_delta_allocated",
                "cuda_baseline_reserved_mb": "cuda_baseline_reserved",
                "cuda_peak_reserved_mb": "cuda_peak_reserved",
                "cuda_peak_delta_reserved_mb": "cuda_peak_delta_reserved",
                "cuda_accounted_dynamic_mb": "cuda_accounted_dynamic_bytes",
                "cuda_activations_est_mb": "cuda_activations_est_bytes",
            }
            for out_k, in_k in cuda_map_final.items():
                if in_k in last_mem:
                    final_memory[out_k] = _round(last_mem[in_k] / BYTES_TO_MB)
        else:
            final_memory = {
                "final_memory": 0.0,
                "final_memory_frozenModel": 0.0,
                "final_memory_LoRAWeights": 0.0,
                "final_memory_Activations": 0.0,
                "final_memory_OptimizerStates": 0.0,
                "final_memory_Gradients": 0.0,
            }
        
        # Build the extended metrics dict
        extended_metrics = {
            "id": self.monitored_object.ID,
            "total_time_minutes": {
                "total_training_time": _round(total_training_seconds / SECONDS_TO_MINUTES),
                "total_computing_time": _round(total_compute_seconds / SECONDS_TO_MINUTES),
                "total_uploading_time": _round(total_upload_seconds / SECONDS_TO_MINUTES),
                "total_downloading_time": _round(total_download_seconds / SECONDS_TO_MINUTES),
            },
            "total_communication_megabytes": {
                "total_communicated_megabytes": _round(total_communicated_bytes / BYTES_TO_MB),
                "total_uploaded_megabytes": _round(total_upload_bytes / BYTES_TO_MB),
                "total_downloaded_megabytes": _round(total_download_bytes / BYTES_TO_MB),
            },
            "total_memory_megabytes": final_memory,
            "total_trained_parameters": self.sys_trained_params,
            "per_round_time_minutes": per_round_time,
            "per_round_communication_megabytes": per_round_comm,
            "per_round_memory_megabytes": per_round_mem,
        }
        
        if verbose:
            logger.info(
                f"In worker #{self.monitored_object.ID}, the extended system-related "
                f"metrics are: id={self.monitored_object.ID}, "
                f"total_training_time={extended_metrics['total_time_minutes']['total_training_time']:.4f}min, "
                f"total_communicated={extended_metrics['total_communication_megabytes']['total_communicated_megabytes']:.4f}MB, "
                f"trained_params={self.sys_trained_params}")
        
        return extended_metrics

    def merge_system_metrics_simulation_mode(self,
                                             file_io=True,
                                             from_global_monitors=False):
        """
        Average the system metrics recorded in ``system_metrics.json`` by \
        all workers.

        For ``system_metrics_mode="extended"``, instead of computing avg/std,
        generate the final server (id 0) entry with three separated concepts:
        - ``fl_endtime_minutes``: observed end-to-end FL wallclock
        - ``wallclock_time_minutes``: estimated wallclock breakdown computed
          by summing per-round client maxima for compute/upload/download
        - ``aggregate_client_time_minutes`` and
          ``total_communication_megabytes``: aggregate client-consumed totals
        """
        # Check if we're in extended mode
        is_extended_mode = self.system_metrics_mode == 'extended'
        
        if is_extended_mode:
            self._merge_extended_metrics_simulation_mode(file_io, from_global_monitors)
            return
        
        # Legacy mode: compute avg/std metrics
        all_sys_metrics = defaultdict(list)
        avg_sys_metrics = defaultdict()
        std_sys_metrics = defaultdict()

        if file_io:
            sys_metric_f_name = os.path.join(self.outdir, "system_metrics.log")
            if not os.path.exists(sys_metric_f_name):
                logger.warning(
                    "You have not tracked the workers' system metrics in "
                    "$outdir$/system_metrics.log, "
                    "we will skip the merging. Plz check whether you do not "
                    "want to call monitor.finish_fl()")
                return
            with open(sys_metric_f_name, "r") as f:
                for line in f:
                    res = json.loads(line)
                    if all_sys_metrics is None:
                        all_sys_metrics = res
                        all_sys_metrics["id"] = "all"
                    else:
                        for k, v in res.items():
                            all_sys_metrics[k].append(v)
            id_to_be_merged = all_sys_metrics["id"]
            if len(id_to_be_merged) != len(set(id_to_be_merged)):
                logger.warning(
                    f"The sys_metric_file ({sys_metric_f_name}) contains "
                    f"duplicated tracked sys-results with these ids: "
                    f"f{id_to_be_merged} "
                    f"We will skip the merging as the merge is invalid. "
                    f"Plz check whether you specify the 'outdir' "
                    f"as the same as the one of another older experiment.")
                return
        elif from_global_monitors:
            for monitor in global_all_monitors:
                res = monitor.get_sys_metrics(verbose=False)
                if all_sys_metrics is None:
                    all_sys_metrics = res
                    all_sys_metrics["id"] = "all"
                else:
                    for k, v in res.items():
                        all_sys_metrics[k].append(v)
        else:
            raise ValueError("file_io or from_monitors should be True: "
                             f"but got file_io={file_io}, from_monitors"
                             f"={from_global_monitors}")

        for k, v in all_sys_metrics.items():
            if k == "id":
                avg_sys_metrics[k] = "sys_avg"
                std_sys_metrics[k] = "sys_std"
            else:
                v = np.array(v).astype("float")
                mean_res = np.mean(v)
                std_res = np.std(v)
                if "flops" in k or "bytes" in k or "size" in k:
                    mean_res = self.convert_size(mean_res)
                    std_res = self.convert_size(std_res)
                avg_sys_metrics[f"sys_avg/{k}"] = mean_res
                std_sys_metrics[f"sys_std/{k}"] = std_res

        logger.info(
            f"After merging the system metrics from all works, we got avg:"
            f" {avg_sys_metrics}")

        if file_io:
            with open(sys_metric_f_name, "a") as f:
                f.write(json.dumps(avg_sys_metrics) + "\n")
                f.write(json.dumps(std_sys_metrics) + "\n")

        if self.use_wandb and self.wandb_online_track:
            try:
                import wandb
                # wandb.log(avg_sys_metrics)
                # wandb.log(std_sys_metrics)
                for k, v in avg_sys_metrics.items():
                    wandb.summary[k] = v
                for k, v in std_sys_metrics.items():
                    wandb.summary[k] = v
            except ImportError:
                logger.error(
                    "cfg.wandb.use=True but not install the wandb package")
                exit()

    def _merge_extended_metrics_simulation_mode(self, file_io=True, from_global_monitors=False):
        """
        Generate server (id 0) metrics for extended mode.

        The final server entry separates three concepts:
        - ``fl_endtime_minutes``: observed end-to-end FL wallclock
        - ``wallclock_time_minutes``: estimated simulation wallclock obtained
          by summing per-round client maxima for compute/upload/download
        - ``aggregate_client_time_minutes`` and
          ``total_communication_megabytes``: aggregate client-consumed totals
        """
        sys_metric_f_name = os.path.join(self.outdir, "system_metrics.log")

        client_metrics = []
        if file_io:
            if not os.path.exists(sys_metric_f_name):
                logger.warning(
                    "You have not tracked the workers' system metrics in "
                    "$outdir$/system_metrics.log, "
                    "we will skip generating server metrics.")
                return
            with open(sys_metric_f_name, "r") as f:
                for line in f:
                    res = json.loads(line)
                    if res.get("id", 0) == 0 or res.get("id") in ["sys_avg", "sys_std"]:
                        continue
                    client_metrics.append(res)
        elif from_global_monitors:
            for monitor in global_all_monitors:
                if monitor is self:
                    continue
                try:
                    res = monitor.get_extended_sys_metrics(verbose=False)
                except Exception:
                    continue
                if res.get("id", 0) == 0:
                    continue
                client_metrics.append(res)
        else:
            raise ValueError("file_io or from_global_monitors should be True: "
                             f"but got file_io={file_io}, from_global_monitors={from_global_monitors}")

        if not client_metrics:
            logger.warning("No client metrics found for server aggregation")
            return

        if "total_time_minutes" not in client_metrics[0] or "total_communication_megabytes" not in client_metrics[0]:
            logger.warning(
                "Client metrics are not in extended format, "
                "skipping server metrics generation"
            )
            return

        DECIMAL_PLACES = 4

        def _round(val):
            return round(val, DECIMAL_PLACES)

        # Aggregate client-consumed totals across all clients
        aggregate_training_time = 0.0
        aggregate_computing_time = 0.0
        aggregate_uploading_time = 0.0
        aggregate_downloading_time = 0.0
        total_communicated_mb = 0.0
        total_uploaded_mb = 0.0
        total_downloaded_mb = 0.0

        for cm in client_metrics:
            time_metrics = cm.get("total_time_minutes", {})
            aggregate_training_time += float(time_metrics.get("total_training_time", 0.0))
            aggregate_computing_time += float(time_metrics.get("total_computing_time", 0.0))
            aggregate_uploading_time += float(time_metrics.get("total_uploading_time", 0.0))
            aggregate_downloading_time += float(time_metrics.get("total_downloading_time", 0.0))

            comm_metrics = cm.get("total_communication_megabytes", {})
            total_communicated_mb += float(comm_metrics.get("total_communicated_megabytes", 0.0))
            total_uploaded_mb += float(comm_metrics.get("total_uploaded_megabytes", 0.0))
            total_downloaded_mb += float(comm_metrics.get("total_downloaded_megabytes", 0.0))

        # Estimated wallclock from per-round client maxima
        all_rounds = set()
        for cm in client_metrics:
            all_rounds.update(cm.get("per_round_time_minutes", {}).keys())

        def _round_sort_key(round_key):
            try:
                return int(str(round_key).replace("round", ""))
            except Exception:
                return str(round_key)

        per_round_wallclock_time = {}
        wallclock_compute_total = 0.0
        wallclock_upload_total = 0.0
        wallclock_download_total = 0.0

        for round_key in sorted(all_rounds, key=_round_sort_key):
            max_compute = 0.0
            max_upload = 0.0
            max_download = 0.0

            for cm in client_metrics:
                round_time = cm.get("per_round_time_minutes", {}).get(round_key, {})
                max_compute = max(max_compute, float(round_time.get("computing_time", 0.0)))
                max_upload = max(max_upload, float(round_time.get("uploading_time", 0.0)))
                max_download = max(max_download, float(round_time.get("downloading_time", 0.0)))

            round_training = max_compute + max_upload + max_download
            per_round_wallclock_time[round_key] = {
                "training_time": _round(round_training),
                "computing_time": _round(max_compute),
                "uploading_time": _round(max_upload),
                "downloading_time": _round(max_download),
            }

            wallclock_compute_total += max_compute
            wallclock_upload_total += max_upload
            wallclock_download_total += max_download

        wallclock_training_total = (
            wallclock_compute_total +
            wallclock_upload_total +
            wallclock_download_total
        )

        fl_endtime_minutes = 0.0
        if isinstance(self.fl_end_wall_time, datetime.timedelta):
            fl_endtime_minutes = self.fl_end_wall_time.total_seconds() / 60

        server_metrics = {
            "id": 0,
            "fl_endtime_minutes": _round(fl_endtime_minutes),
            "wallclock_time_minutes": {
                "total_training_time": _round(wallclock_training_total),
                "total_computing_time": _round(wallclock_compute_total),
                "total_uploading_time": _round(wallclock_upload_total),
                "total_downloading_time": _round(wallclock_download_total),
            },
            "per_round_wallclock_time_minutes": per_round_wallclock_time,
            "aggregate_client_time_minutes": {
                "total_training_time": _round(aggregate_training_time),
                "total_computing_time": _round(aggregate_computing_time),
                "total_uploading_time": _round(aggregate_uploading_time),
                "total_downloading_time": _round(aggregate_downloading_time),
            },
            "total_communication_megabytes": {
                "total_communicated_megabytes": _round(total_communicated_mb),
                "total_uploaded_megabytes": _round(total_uploaded_mb),
                "total_downloaded_megabytes": _round(total_downloaded_mb),
            },
        }

        logger.info(
            "Server (id 0) metrics: "
            f"fl_endtime={server_metrics['fl_endtime_minutes']:.4f}min, "
            f"wallclock_training={server_metrics['wallclock_time_minutes']['total_training_time']:.4f}min, "
            f"wallclock_compute={server_metrics['wallclock_time_minutes']['total_computing_time']:.4f}min, "
            f"wallclock_upload={server_metrics['wallclock_time_minutes']['total_uploading_time']:.4f}min, "
            f"wallclock_download={server_metrics['wallclock_time_minutes']['total_downloading_time']:.4f}min, "
            f"aggregate_communicated={server_metrics['total_communication_megabytes']['total_communicated_megabytes']:.4f}MB"
        )

        if file_io:
            with open(sys_metric_f_name, "a") as f:
                f.write(json.dumps(server_metrics) + "\n")

        if self.use_wandb and self.wandb_online_track:
            try:
                import wandb
                wandb.summary["server/id"] = server_metrics["id"]
                wandb.summary["server/fl_endtime_minutes"] = server_metrics["fl_endtime_minutes"]
                for group_name, group_metrics in server_metrics.items():
                    if isinstance(group_metrics, dict):
                        for k, v in group_metrics.items():
                            wandb.summary[f"server/{group_name}/{k}"] = v
            except ImportError:
                logger.error(
                    "cfg.wandb.use=True but not install the wandb package")
                exit()

    def save_formatted_results(self,
                               formatted_res,
                               save_file_name="eval_results.log"):
        """
        Save formatted results to a file.
        """
        line = str(formatted_res) + "\n"
        if save_file_name != "":
            with open(os.path.join(self.outdir, save_file_name),
                      "a") as outfile:
                outfile.write(line)
        if self.use_wandb and self.wandb_online_track:
            try:
                import wandb
                exp_stop_normal = False
                exp_stop_normal, log_res = logline_2_wandb_dict(
                    exp_stop_normal, line, self.log_res_best, raw_out=False)
                wandb.log(log_res)
            except ImportError:
                logger.error(
                    "cfg.wandb.use=True but not install the wandb package")
                exit()

    def finish_fed_runner(self, fl_mode=None):
        """
        Finish the Fed runner.
        """
        self.compress_raw_res_file()
        if fl_mode == "standalone":
            self.merge_system_metrics_simulation_mode()

        if self.use_wandb and not self.wandb_online_track:
            try:
                import wandb
            except ImportError:
                logger.error(
                    "cfg.wandb.use=True but not install the wandb package")
                exit()

            from federatedscope.core.auxiliaries.logging import \
                logfile_2_wandb_dict
            with open(os.path.join(self.outdir, "eval_results.log"),
                      "r") as exp_log_f:
                # track the prediction related performance
                all_log_res, exp_stop_normal, last_line, log_res_best = \
                    logfile_2_wandb_dict(exp_log_f, raw_out=False)
                for log_res in all_log_res:
                    wandb.log(log_res)
                wandb.log(log_res_best)

                # track the system related performance
                sys_metric_f_name = os.path.join(self.outdir,
                                                 "system_metrics.log")
                with open(sys_metric_f_name, "r") as f:
                    for line in f:
                        res = json.loads(line)
                        if res["id"] in ["sys_avg", "sys_std"]:
                            # wandb.log(res)
                            for k, v in res.items():
                                wandb.summary[k] = v

    def compress_raw_res_file(self):
        """
        Compress the raw res file to be written to disk.
        """
        old_f_name = os.path.join(self.outdir, "eval_results.raw")
        if os.path.exists(old_f_name):
            logger.info(
                "We will compress the file eval_results.raw into a .gz file "
                "(the original file is kept)")
            with open(old_f_name, 'rb') as f_in:
                with gzip.open(old_f_name + ".gz", 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

    def format_eval_res(self,
                        results,
                        rnd,
                        role=-1,
                        forms=None,
                        return_raw=False):
        """
        Format the evaluation results from ``trainer.ctx.eval_results``

        Args:
            results (dict): a dict to store the evaluation results {metric:
            value}
            rnd (int|string): FL round
            role (int|string): the output role
            forms (list): format type
            return_raw (bool): return either raw results, or other results

        Returns:
            dict: round_formatted_results, a formatted results with \
            different forms and roles

        Note:
          Example of return value:
            ```
            {                                                                 \
            'Role': 'Server #',                                               \
            'Round': 200,                                                     \
            'Results_weighted_avg': {                                         \
                'test_avg_loss': 0.58, 'test_acc': 0.67, 'test_correct':      \
                3356, 'test_loss': 2892, 'test_total': 5000                   \
                },                                                            \
            'Results_avg': {                                                  \
                'test_avg_loss': 0.57, 'test_acc': 0.67, 'test_correct':      \
                3356, 'test_loss': 2892, 'test_total': 5000                   \
                },                                                            \
            'Results_fairness': {                                             \
             'test_total': 33.99, 'test_correct': 27.185,                     \
             'test_avg_loss_std': 0.433551,                                   \
             'test_avg_loss_bottom_decile': 0.356503,                         \
             'test_avg_loss_top_decile': 1.212492,                            \
             'test_avg_loss_min': 0.198317, 'test_avg_loss_max': 3.603567,    \
             'test_avg_loss_bottom10%': 0.276681, 'test_avg_loss_top10%':     \
             1.686649,                                                        \
             'test_avg_loss_cos1': 0.8679, 'test_avg_loss_entropy': 5.1641,   \
             'test_loss_std': 13.686828, 'test_loss_bottom_decile': 11.8220,  \
             'test_loss_top_decile': 39.727236, 'test_loss_min': 7.337724,    \
             'test_loss_max': 100.899873, 'test_loss_bottom10%': 9.618685,    \
             'test_loss_top10%': 54.96769, 'test_loss_cos1': 0.880356,        \
             'test_loss_entropy': 5.175803, 'test_acc_std': 0.123823,         \
             'test_acc_bottom_decile': 0.676471, 'test_acc_top_decile':       \
             0.916667,                                                        \
             'test_acc_min': 0.071429, 'test_acc_max': 0.972973,              \
             'test_acc_bottom10%': 0.527482, 'test_acc_top10%': 0.94486,      \
             'test_acc_cos1': 0.988134, 'test_acc_entropy': 5.283755          \
                },                                                            \
            }
            ```
        """
        if forms is None:
            forms = ['weighted_avg', 'avg', 'fairness', 'raw']
        round_formatted_results = {'Role': role, 'Round': rnd}
        round_formatted_results_raw = {'Role': role, 'Round': rnd}

        if 'group_avg' in forms:  # have different format
            # ({client_id: metrics})
            new_results = {}
            num_of_client_for_data = self.cfg.data.num_of_client_for_data
            client_start_id = 1
            for group_id, num_clients in enumerate(num_of_client_for_data):
                if client_start_id > len(results):
                    break
                group_res = copy.deepcopy(results[client_start_id])
                num_div = num_clients - max(
                    0, client_start_id + num_clients - len(results) - 1)
                for client_id in range(client_start_id,
                                       client_start_id + num_clients):
                    if client_id > len(results):
                        break
                    for k, v in group_res.items():
                        if isinstance(v, dict):
                            for kk in v:
                                if client_id == client_start_id:
                                    group_res[k][kk] /= num_div
                                else:
                                    group_res[k][kk] += results[client_id][k][
                                        kk] / num_div
                        else:
                            if client_id == client_start_id:
                                group_res[k] /= num_div
                            else:
                                group_res[k] += results[client_id][k] / num_div
                new_results[group_id + 1] = group_res
                client_start_id += num_clients
                round_formatted_results['Results_group_avg'] = new_results

        else:
            for form in forms:
                new_results = copy.deepcopy(results)
                if not role.lower().startswith('server') or form == 'raw':
                    round_formatted_results_raw['Results_raw'] = new_results
                elif form not in Monitor.SUPPORTED_FORMS:
                    continue
                else:
                    for key in results.keys():
                        dataset_name = key.split("_")[0]
                        if f'{dataset_name}_total' not in results:
                            raise ValueError(
                                "Results to be formatted should be include "
                                "the dataset_num in the dict,"
                                f"with key = {dataset_name}_total")
                        else:
                            dataset_num = np.array(
                                results[f'{dataset_name}_total'])
                            if key in [
                                    f'{dataset_name}_total',
                                    f'{dataset_name}_correct'
                            ]:
                                new_results[key] = np.mean(new_results[key])

                        if key in [
                                f'{dataset_name}_total',
                                f'{dataset_name}_correct'
                        ]:
                            new_results[key] = np.mean(new_results[key])
                        else:
                            all_res = np.array(copy.copy(results[key]))
                            metric_values = np.array(new_results[key])
                            if form == 'weighted_avg':
                                # Handle shape mismatch: if metric has fewer 
                                # values than dataset_num (some clients didn't
                                # report this metric), fall back to simple mean
                                if metric_values.shape == dataset_num.shape:
                                    new_results[key] = np.sum(
                                        metric_values *
                                        dataset_num) / np.sum(dataset_num)
                                else:
                                    # Fallback to simple mean for mismatched shapes
                                    new_results[key] = np.mean(metric_values)
                            if form == "avg":
                                new_results[key] = np.mean(new_results[key])
                            if form == "fairness" and all_res.size > 1:
                                # by default, log the std and decile
                                new_results.pop(
                                    key,
                                    None)  # delete the redundant original one
                                all_res.sort()
                                new_results[f"{key}_std"] = np.std(
                                    np.array(all_res))
                                new_results[f"{key}_bottom_decile"] = all_res[
                                    all_res.size // 10]
                                new_results[f"{key}_top_decile"] = all_res[
                                    all_res.size * 9 // 10]
                                # log more fairness metrics
                                # min and max
                                new_results[f"{key}_min"] = all_res[0]
                                new_results[f"{key}_max"] = all_res[-1]
                                # bottom and top 10%
                                new_results[f"{key}_bottom10%"] = np.mean(
                                    all_res[:all_res.size // 10])
                                new_results[f"{key}_top10%"] = np.mean(
                                    all_res[all_res.size * 9 // 10:])
                                # cosine similarity between the performance
                                # distribution and 1
                                new_results[f"{key}_cos1"] = np.mean(
                                    all_res) / (np.sqrt(np.mean(all_res**2)))
                                # entropy of performance distribution
                                all_res_preprocessed = all_res + 1e-9
                                new_results[f"{key}_entropy"] = np.sum(
                                    -all_res_preprocessed /
                                    np.sum(all_res_preprocessed) * (np.log(
                                        (all_res_preprocessed) /
                                        np.sum(all_res_preprocessed))))
                    round_formatted_results[f'Results_{form}'] = new_results

        with open(os.path.join(self.outdir, "eval_results.raw"),
                  "a") as outfile:
            outfile.write(str(round_formatted_results_raw) + "\n")

        return round_formatted_results_raw if return_raw else \
            round_formatted_results

    def calc_model_metric(self, last_model, local_updated_models, rnd):
        """
        Arguments:
            last_model (dict): the state of last round.
            local_updated_models (list): each element is (data_size, model).

        Returns:
            dict: model_metric_dict
        """
        model_metric_dict = {}
        for metric in self.cfg.eval.monitoring:
            func_name = f'calc_{metric}'
            calc_metric = getattr(
                import_module(
                    'federatedscope.core.monitors.metric_calculator'),
                func_name)
            metric_value = calc_metric(last_model, local_updated_models)
            model_metric_dict[f'train_{metric}'] = metric_value
        formatted_log = {
            'Role': 'Server #',
            'Round': rnd,
            'Results_model_metric': model_metric_dict
        }
        if len(model_metric_dict.keys()):
            logger.info(formatted_log)

        return model_metric_dict

    def convert_size(self, size_bytes):
        """
        Convert bytes to human-readable size.
        """
        import math
        if size_bytes <= 0:
            return str(size_bytes)
        size_name = ("", "K", "M", "G", "T", "P", "E", "Z", "Y")
        i = int(math.floor(math.log(size_bytes, 1024)))
        p = math.pow(1024, i)
        s = round(size_bytes / p, 2)
        return f"{s}{size_name[i]}"

    def track_model_size(self, models):
        """
        calculate the total model size given the models hold by the \
        worker/trainer

        Args
            models: torch.nn.Module or list of torch.nn.Module
        """
        if self.total_model_size != 0:
            logger.warning(
                "the total_model_size is not zero. You may have been "
                "calculated the total_model_size before")

        if not hasattr(models, '__iter__'):
            models = [models]
        for model in models:
            assert isinstance(model, torch.nn.Module), \
                f"the `model` should be type torch.nn.Module when " \
                f"calculating its size, but got {type(model)}"
            for name, para in model.named_parameters():
                self.total_model_size += para.numel()

    def track_avg_flops(self, flops, sample_num=1):
        """
        update the average flops for forwarding each data sample, \
        for most models and tasks, \
        the averaging is not needed as the input shape is fixed
        """

        self.flops_per_sample = (self.flops_per_sample * self.flop_count +
                                 flops) / (self.flop_count + sample_num)
        self.flop_count += 1

    def track_upload_bytes(self, bytes):
        """
        Track the number of bytes uploaded.
        """
        self.total_upload_bytes += bytes

    def track_download_bytes(self, bytes):
        """
        Track the number of bytes downloaded.
        """
        self.total_download_bytes += bytes

    # =====================================================================
    # Extended metrics tracking methods
    # =====================================================================
    
    def track_round_compute_time(self, round_id, compute_seconds):
        """
        Track compute time for a specific round.
        
        Args:
            round_id: FL round index
            compute_seconds: Time spent in local training (seconds)
        """
        if round_id not in self.sys_time_per_round:
            self.sys_time_per_round[round_id] = {
                'compute_seconds': 0.0,
                'upload_seconds': 0.0,
                'download_seconds': 0.0,
            }
        self.sys_time_per_round[round_id]['compute_seconds'] = compute_seconds
    
    def track_round_bandwidth(self, round_id, upload_bw_kbits, download_bw_kbits):
        """
        Track per-round bandwidth for communication time calculations.
        
        Args:
            round_id: FL round index
            upload_bw_kbits: Upload bandwidth in kbit/s
            download_bw_kbits: Download bandwidth in kbit/s
        """
        self.sys_bandwidth_per_round[round_id] = {
            'upload_bandwidth_kbits': upload_bw_kbits,
            'download_bandwidth_kbits': download_bw_kbits,
        }
    
    def track_round_communication(self, round_id, upload_bytes, download_bytes):
        """
        Track communication bytes for a specific round.
        Communication time is computed from bytes and the per-round
        configured bandwidth recorded in ``sys_bandwidth_per_round``.

        Args:
            round_id: FL round index
            upload_bytes: Bytes uploaded this round
            download_bytes: Bytes downloaded this round
        """
        self.sys_comm_per_round[round_id] = {
            'upload_bytes': upload_bytes,
            'download_bytes': download_bytes,
        }

        # Time computation uses the per-round configured bandwidth only:
        # upload/download seconds = bytes / configured_bw.
        upload_seconds = 0.0
        download_seconds = 0.0
        if round_id in self.sys_bandwidth_per_round:
            bw = self.sys_bandwidth_per_round[round_id]
            ul_bw = bw.get('upload_bandwidth_kbits', 0)
            if ul_bw > 0 and upload_bytes > 0:
                upload_seconds = (upload_bytes * 8) / (ul_bw * 1000)
            dl_bw = bw.get('download_bandwidth_kbits', 0)
            if dl_bw > 0 and download_bytes > 0:
                download_seconds = (download_bytes * 8) / (dl_bw * 1000)

        if round_id not in self.sys_time_per_round:
            self.sys_time_per_round[round_id] = {
                'compute_seconds': 0.0,
                'upload_seconds': 0.0,
                'download_seconds': 0.0,
            }

        self.sys_time_per_round[round_id]['upload_seconds'] = upload_seconds
        self.sys_time_per_round[round_id]['download_seconds'] = download_seconds
    
    def track_round_memory(self,
                         round_id,
                         model,
                         optimizer=None,
                         device=None,
                         cuda_baseline_allocated=None,
                         cuda_baseline_reserved=None):
        """
        Track memory breakdown for a specific round.

        This function computes parameter/optimizer/gradient bytes analytically from tensor dtypes.
        For the "dynamic" portion, it uses CUDA peak memory stats and an explicit baseline captured
        right before the local training loop starts (ideally after model.to(device) and optimizer init).

        Args:
            round_id: FL round index
            model: The model being trained
            optimizer: The optimizer (optional, for optimizer state tracking)
            device: Device for CUDA memory tracking (optional)
            cuda_baseline_allocated: CUDA baseline memory_allocated (bytes) captured before local steps
            cuda_baseline_reserved: CUDA baseline memory_reserved (bytes) captured before local steps
        """
        if torch is None:
            return

        frozen_model_bytes = 0
        lora_bytes = 0
        gradients_bytes = 0
        optimizer_bytes = 0

        # Get the actual model (handle wrappers like AdapterModel)
        actual_model = model
        if hasattr(model, 'model') and hasattr(model.model, 'named_parameters'):
            actual_model = model.model

        # Compute frozen_model_bytes and lora_bytes from parameters
        for name, p in actual_model.named_parameters():
            param_bytes = p.numel() * p.element_size()
            if not p.requires_grad:
                frozen_model_bytes += param_bytes
            else:
                # LoRA parameters
                if ('lora_' in name.lower()) or ('lora_A' in name) or ('lora_B' in name):
                    lora_bytes += param_bytes

        # Compute gradients_bytes
        for p in actual_model.parameters():
            if p.grad is not None:
                gradients_bytes += p.grad.numel() * p.grad.element_size()

        # Compute optimizer_bytes from optimizer state
        if optimizer is not None:
            for state in optimizer.state.values():
                for v in state.values():
                    if torch.is_tensor(v):
                        optimizer_bytes += v.numel() * v.element_size()

        # CUDA peak stats
        peak_alloc = None
        peak_reserved = None
        delta_alloc = None
        delta_reserved = None

        activations_bytes = 0
        accounted_dynamic = optimizer_bytes + gradients_bytes

        if device is not None and torch.cuda.is_available():
            try:
                dev = device
                if isinstance(dev, int):
                    dev = torch.device(f'cuda:{dev}')
                elif isinstance(dev, str):
                    dev = torch.device(dev)

                if isinstance(dev, torch.device) and dev.type == 'cuda':
                    peak_alloc = torch.cuda.max_memory_allocated(dev)
                    # max_memory_reserved exists in recent PyTorch; fallback to current reserved if missing
                    if hasattr(torch.cuda, 'max_memory_reserved'):
                        peak_reserved = torch.cuda.max_memory_reserved(dev)
                    else:
                        peak_reserved = torch.cuda.memory_reserved(dev)

                    if cuda_baseline_allocated is not None:
                        delta_alloc = max(0, peak_alloc - int(cuda_baseline_allocated))
                    if cuda_baseline_reserved is not None:
                        delta_reserved = max(0, peak_reserved - int(cuda_baseline_reserved))

                    # If we have a baseline, interpret delta_alloc as the dynamic peak during local steps.
                    if delta_alloc is not None:
                        # Estimate activations as remaining dynamic memory after optimizer state and gradients.
                        activations_bytes = max(0, delta_alloc - accounted_dynamic)
                    else:
                        # Fallback heuristic if baseline was not provided
                        dynamic_bytes = max(0, peak_alloc - frozen_model_bytes - lora_bytes)
                        activations_bytes = max(0, dynamic_bytes - accounted_dynamic)
            except Exception:
                pass

        final_memory_bytes = (frozen_model_bytes + lora_bytes +
                              activations_bytes + optimizer_bytes + gradients_bytes)

        # Store all details for later JSON serialization
        entry = {
            'frozen_model_bytes': frozen_model_bytes,
            'lora_bytes': lora_bytes,
            'activations_bytes': activations_bytes,
            'optimizer_bytes': optimizer_bytes,
            'gradients_bytes': gradients_bytes,
            'final_memory_bytes': final_memory_bytes,
        }

        # Optional CUDA debug fields (bytes)
        if cuda_baseline_allocated is not None:
            entry['cuda_baseline_allocated'] = int(cuda_baseline_allocated)
        if cuda_baseline_reserved is not None:
            entry['cuda_baseline_reserved'] = int(cuda_baseline_reserved)
        if peak_alloc is not None:
            entry['cuda_peak_allocated'] = int(peak_alloc)
        if peak_reserved is not None:
            entry['cuda_peak_reserved'] = int(peak_reserved)
        if delta_alloc is not None:
            entry['cuda_peak_delta_allocated'] = int(delta_alloc)
        if delta_reserved is not None:
            entry['cuda_peak_delta_reserved'] = int(delta_reserved)

        entry['cuda_accounted_dynamic_bytes'] = int(accounted_dynamic)
        entry['cuda_activations_est_bytes'] = int(activations_bytes)

        self.sys_mem_per_round[round_id] = entry

    def compute_model_bytes(self, state_dict):
        """
        Compute total bytes from a model state dict, respecting actual dtypes.
        
        Args:
            state_dict: Model state dict or parameter dict
            
        Returns:
            Total bytes
        """
        if torch is None:
            return 0
        
        total_bytes = 0
        for key, tensor in state_dict.items():
            if torch.is_tensor(tensor):
                total_bytes += tensor.numel() * tensor.element_size()
        return total_bytes
    
    def track_trained_parameters(self, model):
        """
        Track the total number of trainable LoRA parameters.
        
        Args:
            model: The model being trained
        """
        if torch is None:
            return
        
        # Get the actual model (handle wrappers)
        actual_model = model
        if hasattr(model, 'model') and hasattr(model.model, 'named_parameters'):
            actual_model = model.model
        
        total_trainable = 0
        for name, p in actual_model.named_parameters():
            if p.requires_grad:
                total_trainable += p.numel()
        
        self.sys_trained_params = total_trainable

    def update_best_result(self, best_results, new_results, results_type):
        """
        Update best evaluation results. \
        by default, the update is based on validation loss with \
        ``round_wise_update_key="val_loss" ``
        """
        update_best_this_round = False
        if not isinstance(new_results, dict):
            raise ValueError(
                f"update best results require `results` a dict, but got"
                f" {type(new_results)}")
        else:
            if results_type not in best_results:
                best_results[results_type] = dict()
            best_result = best_results[results_type]
            # update different keys separately: the best values can be in
            # different rounds
            if self.round_wise_update_key is None:
                for key in new_results:
                    cur_result = new_results[key]
                    if 'loss' in key or 'std' in key:  # the smaller,
                        # the better
                        if results_type in [
                                "client_best_individual",
                                "unseen_client_best_individual"
                        ]:
                            cur_result = min(cur_result)
                        if key not in best_result or cur_result < best_result[
                                key]:
                            best_result[key] = cur_result
                            update_best_this_round = True

                    elif 'acc' in key:  # the larger, the better
                        if results_type in [
                                "client_best_individual",
                                "unseen_client_best_individual"
                        ]:
                            cur_result = max(cur_result)
                        if key not in best_result or cur_result > best_result[
                                key]:
                            best_result[key] = cur_result
                            update_best_this_round = True
                    else:
                        # unconcerned metric
                        pass
            # update different keys round-wise: if find better
            # round_wise_update_key, update others at the same time
            else:
                found_round_wise_update_key = False
                sorted_keys = []
                for key in new_results:
                    # TODO: fix `in` condition
                    if self.round_wise_update_key in key:
                        sorted_keys.insert(0, key)
                        found_round_wise_update_key = key
                    else:
                        sorted_keys.append(key)
                if not found_round_wise_update_key:
                    raise ValueError(
                        "Your specified eval.best_res_update_round_wise_key "
                        "is not in target results, "
                        "use another key or check the name. \n"
                        f"Got eval.best_res_update_round_wise_key"
                        f"={self.round_wise_update_key}, "
                        f"the keys of results are {list(new_results.keys())}")

                # the first key must be the `round_wise_update_key`,
                # `update_best_this_round` should be set while evaluating the
                # first key, so we can check whether `update_best_this_round`
                # firstly
                cur_result = new_results[found_round_wise_update_key]

                if self.the_larger_the_better:
                    # The larger, the better
                    if results_type in [
                            "client_best_individual",
                            "unseen_client_best_individual"
                    ]:
                        cur_result = max(cur_result)
                    try:
                        cur_is_nan = np.isnan(cur_result)
                    except (TypeError, ValueError):
                        cur_is_nan = False
                    if not cur_is_nan:
                        prev = best_result.get(
                            found_round_wise_update_key, None)
                        try:
                            prev_is_nan = prev is not None and np.isnan(prev)
                        except (TypeError, ValueError):
                            prev_is_nan = False
                        if prev is None or prev_is_nan or \
                                cur_result > prev:
                            best_result[
                                found_round_wise_update_key] = cur_result
                            update_best_this_round = True
                else:
                    # The smaller, the better
                    if results_type in [
                            "client_best_individual",
                            "unseen_client_best_individual"
                    ]:
                        cur_result = min(cur_result)
                    try:
                        cur_is_nan = np.isnan(cur_result)
                    except (TypeError, ValueError):
                        cur_is_nan = False
                    if not cur_is_nan:
                        prev = best_result.get(
                            found_round_wise_update_key, None)
                        try:
                            prev_is_nan = prev is not None and np.isnan(prev)
                        except (TypeError, ValueError):
                            prev_is_nan = False
                        if prev is None or prev_is_nan or \
                                cur_result < prev:
                            best_result[
                                found_round_wise_update_key] = cur_result
                            update_best_this_round = True

                # update other metrics only if update_best_this_round is True
                if update_best_this_round:
                    for key in sorted_keys[1:]:
                        cur_result = new_results[key]
                        if results_type in [
                                "client_best_individual",
                                "unseen_client_best_individual"
                        ]:
                            # Obtain the whether the larger the better
                            for mode in ['train', 'val', 'test']:
                                if mode in key:
                                    _key = key.split(f'{mode}_')[1]
                                    # Check if metric is in eval_metric before accessing
                                    if _key in self.metric_calculator.eval_metric:
                                        if self.metric_calculator.eval_metric[
                                                _key][1]:
                                            cur_result = max(cur_result)
                                        else:
                                            cur_result = min(cur_result)
                                    else:
                                        # For unknown metrics, assume larger is better
                                        # for common positive metrics like 'correct', 'f1', 'mcc'
                                        # and smaller is better for 'loss'-like metrics
                                        if any(neg in _key.lower() for neg in ['loss', 'error', 'mse', 'mae']):
                                            cur_result = min(cur_result)
                                        else:
                                            cur_result = max(cur_result)
                        best_result[key] = cur_result

        if update_best_this_round:
            line = f"Find new best result: {best_results}"
            logging.info(line)
            if self.use_wandb and self.wandb_online_track:
                try:
                    import wandb
                    exp_stop_normal = False
                    exp_stop_normal, log_res = logline_2_wandb_dict(
                        exp_stop_normal,
                        line,
                        self.log_res_best,
                        raw_out=False)
                    # wandb.log(self.log_res_best)
                    for k, v in self.log_res_best.items():
                        wandb.summary[k] = v
                except ImportError:
                    logger.error(
                        "cfg.wandb.use=True but not install the wandb package")
                    exit()
        return update_best_this_round

    def add_items_to_best_result(self, best_results, new_results,
                                 results_type):
        """
        Add a new key: value item (results-type: new_results) to best_result
        """
        best_results[results_type] = new_results
