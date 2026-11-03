import abc
import logging

from collections import deque
import heapq

import numpy as np
from peft import get_peft_model
from federatedscope.core.workers import Server, Client
from federatedscope.core.gpu_manager import GPUManager
from federatedscope.core.auxiliaries.model_builder import get_model
from federatedscope.core.auxiliaries.utils import get_resource_info, \
    get_ds_rank
from federatedscope.core.auxiliaries.feat_engr_builder import \
    get_feat_engr_wrapper

import federatedscope.contrib.common as fs_common

logger = logging.getLogger(__name__)

import torch

def _client_compute_dtype(client_cfg):
    """Resolve the local *compute dtype* used for forward/backward on a given client.

    We prefer an explicit per-client `computation_quantization.compute_dtype` (set by the runner)
    and fall back to a conservative default policy when missing.
    """
    if hasattr(client_cfg, "computation_quantization"):
        cd = getattr(client_cfg.computation_quantization, "compute_dtype", None)
        if isinstance(cd, str):
            cd_l = cd.strip().lower()
            if cd_l in {"bf16", "bfloat16"}:
                if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                    return torch.bfloat16
                return torch.float16
            if cd_l in {"fp16", "float16", "half"}:
                return torch.float16
            if cd_l in {"fp32", "float32", "float"}:
                return torch.float32

        method = getattr(client_cfg.computation_quantization, "method", None)
        nbits = getattr(client_cfg.computation_quantization, "nbits", None)

        # QLoRA (k-bit base): prefer bf16 if supported, else fp16
        if method == "qlora":
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                return torch.bfloat16
            return torch.float16

    # Default fallback
    return torch.float32


class BaseRunner(object):
    """
    This class is a base class to construct an FL course, which includes \
    ``_set_up()`` and ``run()``.

    Args:
        data: The data used in the FL courses, which are formatted as \
        ``{'ID':data}`` for standalone mode. More details can be found in \
        federatedscope.core.auxiliaries.data_builder .
        server_class: The server class is used for instantiating a ( \
        customized) server.
        client_class: The client class is used for instantiating a ( \
        customized) client.
        config: The configurations of the FL course.
        client_configs: The clients' configurations.

    Attributes:
        data: The data used in the FL courses, which are formatted as \
        ``{'ID':data}`` for standalone mode. More details can be found in \
        federatedscope.core.auxiliaries.data_builder .
        server: The instantiated server.
        client: The instantiate client(s).
        cfg : The configurations of the FL course.
        client_cfgs: The clients' configurations.
        mode: The run mode for FL, ``distributed`` or ``standalone``
        gpu_manager: manager of GPU resource
        resource_info: information of resource
    """
    def __init__(self,
                 data,
                 server_class=Server,
                 client_class=Client,
                 config=None,
                 client_configs=None):
        self.data = data
        self.server_class = server_class
        self.client_class = client_class
        assert config is not None, \
            "When using Runner, you should specify the `config` para"
        if not config.is_ready_for_run:
            config.ready_for_run()
        self.cfg = config
        self.client_cfgs = client_configs
        self.serial_num_for_msg = 0

        self.mode = self.cfg.federate.mode.lower()
        self.gpu_manager = GPUManager(gpu_available=self.cfg.use_gpu,
                                      specified_device=self.cfg.device)

        self.unseen_clients_id = []
        self.feat_engr_wrapper_client, self.feat_engr_wrapper_server = \
            get_feat_engr_wrapper(config)
        if self.cfg.federate.unseen_clients_rate > 0:
            self.unseen_clients_id = np.random.choice(
                np.arange(1, self.cfg.federate.client_num + 1),
                size=max(
                    1,
                    int(self.cfg.federate.unseen_clients_rate *
                        self.cfg.federate.client_num)),
                replace=False).tolist()
        # get resource information
        self.resource_info = get_resource_info(
            config.federate.resource_info_file)

        # Check if FAH-QLoRA is enabled
        fah_cfg = fs_common.get_fah_cfg(config)
        self.fah_enabled = fah_cfg is not None
        adapter_root = fs_common.get_adapter_root(config)
        hetero_ranks_cfg = fs_common.get_hetero_ranks_cfg(config, enabled_only=False)

        # Generate heterogeneous LoRA client configurations if needed
        self.hetero_lora_config = None
        self.fah_client_rank_caps = {}

        # Initialize heterogeneous base model quantization mapping
        # Maps client_id -> nbits (4, 8, or 16)
        self.client_base_quant = {}
        self._setup_base_quant(config)

        if self.fah_enabled:
            # For FAH-QLoRA, adapter.hetero_strategy should define the immutable
            # per-client local capacity / max feasible rank, while warm-up itself
            # remains homogeneous at runtime.
            active_config_local = fs_common.get_active_hetero_config_local(config)
            hetero_strategy = getattr(adapter_root, 'hetero_strategy', 'homo') if adapter_root is not None else 'homo'

            if (
                active_config_local is not None
                and isinstance(active_config_local, dict)
                and any(str(k).startswith('Client_') for k in active_config_local.keys())
            ):
                self.hetero_lora_config = active_config_local
                logger.info(
                    f"[FAH] Using manually provided heterogeneous capacity configs for "
                    f"{config.federate.client_num} clients"
                )
            else:
                from federatedscope.contrib.common.client_config_generator import (
                    get_default_client_types, get_client_lora_config)

                config_types = get_default_client_types()
                default_alpha = getattr(
                    getattr(adapter_root, 'hetero_alpha', None), 'default', 16)
                default_dropout = getattr(
                    hetero_ranks_cfg, 'default_dropout', 0.05) if hetero_ranks_cfg is not None else 0.05

                if hetero_strategy == 'homo':
                    target_modules = fs_common.get_effective_target_modules(config)
                    cap_rank = int(
                        getattr(adapter_root, 'max_rank', None)
                        or getattr(adapter_root, 'rank', 8)
                    )
                    self.hetero_lora_config = {
                        'alpha': default_alpha,
                        'lora_dropout': default_dropout,
                    }
                    for client_id in range(1, config.federate.client_num + 1):
                        client_key = f'Client_{client_id}'
                        self.hetero_lora_config[client_key] = {
                            mod: cap_rank for mod in target_modules
                        }

                    logger.info(
                        f"[FAH] Built homogeneous capacity config at rank={cap_rank} "
                        f"for {config.federate.client_num} clients; warmup remains homogeneous at runtime "
                        f"with init_rank={fah_cfg.init_rank}."
                    )
                else:
                    self.hetero_lora_config = get_client_lora_config(
                        config_types=config_types,
                        num_clients=config.federate.client_num,
                        strategy=hetero_strategy,
                        seed=config.seed,
                        default_alpha=default_alpha,
                        default_dropout=default_dropout,
                    )

                    if fs_common.is_glue_task(config):
                        target_modules = fs_common.get_effective_target_modules(config)
                        if target_modules:
                            remapped = {}

                            for cfg_key, cfg_val in self.hetero_lora_config.items():
                                if not str(cfg_key).startswith('Client_'):
                                    remapped[cfg_key] = cfg_val
                                    continue

                                if not hasattr(cfg_val, 'items'):
                                    remapped[cfg_key] = cfg_val
                                    continue

                                client_rank = next(
                                    (int(v) for v in cfg_val.values()
                                     if isinstance(v, (int, float))),
                                    int(getattr(adapter_root, 'rank', 8))
                                )

                                remapped[cfg_key] = {
                                    mod: client_rank for mod in target_modules
                                }

                            self.hetero_lora_config = remapped
                            logger.info(
                                f"[FAH] Remapped auto-generated capacity configs "
                                f"to target_modules: {target_modules}"
                            )

                    logger.info(
                        f"[FAH] Generated heterogeneous capacity configs for "
                        f"{config.federate.client_num} clients using strategy "
                        f"'{hetero_strategy}'. Warmup remains homogeneous at runtime "
                        f"with init_rank={fah_cfg.init_rank}."
                    )

            from federatedscope.contrib.common.client_config_generator import get_client_rank_caps
            self.fah_client_rank_caps = get_client_rank_caps(
                self.hetero_lora_config, reduction='min'
            )
            logger.info(
                f"[FAH] Derived immutable per-client rank caps from capacity config: "
                f"{self.fah_client_rank_caps}"
            )

        elif (
            adapter_root is not None
            and getattr(adapter_root, 'use', False)
            and getattr(adapter_root, 'hetero_strategy', 'homo') != 'homo'
        ):
            # Check if config_local is already manually set
            active_config_local = fs_common.get_active_hetero_config_local(config)
            if (
                active_config_local is not None
                and isinstance(active_config_local, dict)
                and any(str(k).startswith('Client_') for k in active_config_local.keys())
            ):
                self.hetero_lora_config = active_config_local
                logger.info(
                    f"Using manually provided heterogeneous LoRA configs for "
                    f"{config.federate.client_num} clients"
                )
            else:
                # Auto-generate config
                from federatedscope.contrib.common.client_config_generator import (
                    get_default_client_types, get_client_lora_config)

                config_types = get_default_client_types()
                default_alpha = getattr(
                    getattr(adapter_root, 'hetero_alpha', None), 'default', 16)
                default_dropout = getattr(
                    hetero_ranks_cfg, 'default_dropout', 0.05) if hetero_ranks_cfg is not None else 0.05
                hetero_strategy = getattr(adapter_root, 'hetero_strategy', 'homo')

                self.hetero_lora_config = get_client_lora_config(
                    config_types=config_types,
                    num_clients=config.federate.client_num,
                    strategy=hetero_strategy,
                    seed=config.seed,
                    default_alpha=default_alpha,
                    default_dropout=default_dropout,
                )

                if fs_common.is_glue_task(config):
                    target_modules = fs_common.get_effective_target_modules(config)
                    if target_modules:
                        remapped = {}

                        for cfg_key, cfg_val in self.hetero_lora_config.items():
                            # Preserve top-level metadata such as alpha/lora_dropout
                            if not str(cfg_key).startswith('Client_'):
                                remapped[cfg_key] = cfg_val
                                continue

                            # Only process dict-like client configs
                            if not hasattr(cfg_val, 'items'):
                                remapped[cfg_key] = cfg_val
                                continue

                            client_rank = next(
                                (int(v) for v in cfg_val.values()
                                 if isinstance(v, (int, float))),
                                int(getattr(adapter_root, 'rank', 8))
                            )

                            remapped[cfg_key] = {
                                mod: client_rank for mod in target_modules
                            }

                        self.hetero_lora_config = remapped
                        logger.info(
                            f"[HeteroLoRA] Remapped auto-generated hetero configs "
                            f"to target_modules: {target_modules}"
                        )

                logger.info(
                    f"Generated heterogeneous LoRA configs for "
                    f"{config.federate.client_num} clients using strategy "
                    f"'{hetero_strategy}'"
                )

        # Store final hetero config in the runtime config before server/client setup
        if self.hetero_lora_config is not None:
            config.defrost()

            roots_to_update = []
            for prefer_glue in (True, False):
                root = fs_common.get_adapter_root(config, prefer_glue=prefer_glue)
                if root is not None and hasattr(root, 'hetero_ranks') and all(root is not existing for existing in roots_to_update):
                    roots_to_update.append(root)

            for root in roots_to_update:
                root.hetero_ranks.config_local = self.hetero_lora_config

            config.freeze()

        # Check the completeness of msg_handler.
        self.check()

        # Set up for Runner
        self._set_up()

        
    def _setup_base_quant(self, config):
        """
        Set up *heterogeneous base-model quantization* (per-client nbits mapping).

        Applies to any method where you want heterogeneous base precision, not only
        FAH-QLoRA.

        This method builds `self.client_base_quant: Dict[int, int]` mapping
        `client_id -> nbits`, and `_setup_client()` will translate that into
        `client_specific_config.computation_quantization.{method, nbits}`.

        Where we look for config:
        - Prefer `glue.adapter.base_quant` (GLUE tasks) / `llm.adapter.base_quant`
        - Otherwise use `glue.adapter.fah.base_quant` / `llm.adapter.fah.base_quant`
        """
        self.client_base_quant = {}

        primary_root = fs_common.get_adapter_root(config)
        secondary_root = fs_common.get_adapter_root(
            config, prefer_glue=not fs_common.is_glue_task(config))

        def _get_base_quant_cfg(adapter_root):
            if adapter_root is None:
                return None
            if hasattr(adapter_root, 'base_quant'):
                return adapter_root.base_quant
            if hasattr(adapter_root, 'fah') and hasattr(adapter_root.fah, 'base_quant'):
                return adapter_root.fah.base_quant
            return None

        base_quant_cfg = _get_base_quant_cfg(primary_root)
        if base_quant_cfg is None and secondary_root is not primary_root:
            base_quant_cfg = _get_base_quant_cfg(secondary_root)

        if base_quant_cfg is None:
            # No base-quant subtree present: do nothing
            return

        if not getattr(base_quant_cfg, 'enabled', False):
            logger.info("[BASE_QUANT] Heterogeneous base quantization disabled (base_quant.enabled=False).")
            return

        # Distribution over nbits groups (keys may come as strings from YAML)
        distribution = {}
        if hasattr(base_quant_cfg, 'distribution'):
            dist_cfg = base_quant_cfg.distribution
            for key in ['4', '8', '16', '32', 4, 8, 16, 32]:
                if hasattr(dist_cfg, str(key)):
                    try:
                        distribution[int(key)] = float(getattr(dist_cfg, str(key)))
                    except Exception:
                        pass

        # If no distribution specified, default to 1/3 each over {4,8,16} to match the FAH paper
        if not distribution:
            distribution = {4: 0.333, 8: 0.333, 16: 0.334}
            logger.info("[BASE_QUANT] Using default distribution: 1/3 4-bit, 1/3 8-bit, 1/3 16-bit.")

        # Normalize distribution to sum to 1
        total = sum(distribution.values())
        if total <= 0:
            logger.warning("[BASE_QUANT] Invalid base_quant.distribution (sum<=0). Disabling heterogeneous base quantization.")
            return

        if abs(total - 1.0) > 0.01:
            logger.warning(f"[BASE_QUANT] base_quant.distribution sums to {total:.4f}, normalizing.")
            distribution = {k: v / total for k, v in distribution.items()}

        # Assign quantization levels to clients
        num_clients = config.federate.client_num
        np.random.seed(config.seed)  # For reproducibility

        quant_assignments = []
        for nbits, fraction in sorted(distribution.items()):
            count = int(round(fraction * num_clients))
            quant_assignments.extend([int(nbits)] * count)

        # Adjust for rounding so exactly num_clients levels are assigned
        while len(quant_assignments) < num_clients:
            most_common = max(distribution, key=distribution.get)
            quant_assignments.append(int(most_common))
        while len(quant_assignments) > num_clients:
            most_common = max(distribution, key=distribution.get)
            idx = quant_assignments.index(int(most_common))
            quant_assignments.pop(idx)

        np.random.shuffle(quant_assignments)

        for client_id in range(1, num_clients + 1):
            self.client_base_quant[client_id] = int(quant_assignments[client_id - 1])

        # Log counts
        counts = {}
        for nbits in self.client_base_quant.values():
            counts[nbits] = counts.get(nbits, 0) + 1

        logger.info("[BASE_QUANT] Heterogeneous base quantization enabled:")
        for nbits in sorted(counts.keys()):
            logger.info(f"[BASE_QUANT]   {nbits}-bit clients: {counts[nbits]}")
        logger.info(f"[BASE_QUANT]   Client assignments: {self.client_base_quant}")

    @abc.abstractmethod
    def _set_up(self):
        """
        Set up and instantiate the client/server.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _get_server_args(self, resource_info, client_resource_info):
        """
        Get the args for instantiating the server.

        Args:
            resource_info: information of resource
            client_resource_info: information of client's resource

        Returns:
            (server_data, model, kw): None or data which server holds; model \
            to be aggregated; kwargs dict to instantiate the server.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _get_client_args(self, client_id, resource_info):
        """
        Get the args for instantiating the server.

        Args:
            client_id: ID of client
            resource_info: information of resource

        Returns:
            (client_data, kw): data which client holds; kwargs dict to \
            instantiate the client.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def run(self):
        """
        Launch the FL course

        Returns:
            dict: best results during the FL course
        """
        raise NotImplementedError

    @property
    def ds_rank(self):
        return get_ds_rank()

    def _setup_server(self, resource_info=None, client_resource_info=None):
        """
        Set up and instantiate the server.

        Args:
            resource_info: information of resource
            client_resource_info: information of client's resource

        Returns:
            Instantiate server.
        """
        assert self.server_class is not None, \
            "`server_class` cannot be None."
        self.server_id = 0
        server_data, model, kw = self._get_server_args(resource_info,
                                                       client_resource_info)
        if getattr(self, 'fah_client_rank_caps', None):
            kw['fah_client_rank_caps'] = self.fah_client_rank_caps
        if getattr(self, 'hetero_lora_config', None) is not None and getattr(self, 'fah_enabled', False):
            kw['fah_cap_config_local'] = self.hetero_lora_config
        self._server_device = self.gpu_manager.auto_choice()
        server = self.server_class(
            ID=self.server_id,
            config=self.cfg,
            data=server_data,
            model=model,
            client_num=self.cfg.federate.client_num,
            total_round_num=self.cfg.federate.total_round_num,
            device=self._server_device,
            unseen_clients_id=self.unseen_clients_id,
            **kw)
        if self.cfg.nbafl.use:
            from federatedscope.core.trainers.trainer_nbafl import \
                wrap_nbafl_server
            wrap_nbafl_server(server)
        if self.cfg.vertical.use:
            from federatedscope.vertical_fl.utils import wrap_vertical_server
            server = wrap_vertical_server(server, self.cfg)
        if self.cfg.fedswa.use:
            from federatedscope.core.workers.wrapper import wrap_swa_server
            server = wrap_swa_server(server)
        logger.info('________Server has been set up________')
        return self.feat_engr_wrapper_server(server)

    def _setup_client(self,
                      client_id=-1,
                      client_model=None,
                      resource_info=None):
        """
        Set up and instantiate the client.

        Args:
            client_id: ID of client
            client_model: model of client
            resource_info: information of resource

        Returns:
            Instantiate client.
        """
        assert self.client_class is not None, \
            "`client_class` cannot be None"
        self.server_id = 0
        client_data, kw = self._get_client_args(client_id, resource_info)
        client_specific_config = self.cfg.clone()
        if self.client_cfgs:
            client_specific_config.defrost()
            client_specific_config.merge_from_other_cfg(
                self.client_cfgs.get('client_{}'.format(client_id)))
            client_specific_config.freeze()
        client_device = self._server_device if \
            self.cfg.federate.share_local_model else \
            self.gpu_manager.auto_choice()
        # Apply client-specific base-model quantization if a base-quant mapping is provided
        if client_id in self.client_base_quant:
            client_nbits = int(self.client_base_quant[client_id])
            client_specific_config.defrost()

            # Ensure computation_quantization exists
            if not hasattr(client_specific_config, 'computation_quantization'):
                from federatedscope.core.configs.config import CN
                client_specific_config.computation_quantization = CN()

            if client_nbits in (16, 32):
                # Full precision (no k-bit quantization)
                client_specific_config.computation_quantization.method = 'none'
                client_specific_config.computation_quantization.nbits = client_nbits
            else:
                # k-bit base model via (Q)LoRA
                client_specific_config.computation_quantization.method = 'qlora'
                client_specific_config.computation_quantization.nbits = client_nbits

            client_specific_config.freeze()
            logger.info(
                f"[BASE_QUANT] Client {client_id}: base model nbits={client_nbits} "
                f"(method={client_specific_config.computation_quantization.method})"
            )

        # Get or create client model
        model = client_model or get_model(
            client_specific_config, client_data, backend=self.cfg.backend)
        
        # Apply heterogeneous LoRA rank modifications if needed
        if (self.hetero_lora_config is not None and
                hasattr(model, 'model') and  # AdapterModel wrapper
                hasattr(model.model, 'peft_config')):  # PEFT model
            from federatedscope.contrib.common.heterolora_utils import modify_adapter, is_qlora_client_cfg
            
            client_key = fs_common.resolve_client_key(
                self.hetero_lora_config, client_id)

            if client_key and client_key in self.hetero_lora_config:
                client_rank_config = self.hetero_lora_config[client_key]
                default_alpha = self.hetero_lora_config.get('alpha', 16)
                default_dropout = self.hetero_lora_config.get(
                    'lora_dropout', 0.05)
                
                # Get adapter name (usually 'default' for PEFT)
                adapter_name = 'default'
                if hasattr(model.model, 'peft_config'):
                    adapter_names = list(model.model.peft_config.keys())
                    if adapter_names:
                        adapter_name = adapter_names[0]
                
                # Decide compute dtype for this client from config
                compute_dtype = _client_compute_dtype(client_specific_config)
                is_qlora = is_qlora_client_cfg(client_specific_config)

                # For QLoRA clients (4/8-bit):
                #   - LoRA weights should live in compute_dtype (bf16 or fp16)
                #   - non LoRA trainables (for example classifier head) should stay in fp32
                # For non QLoRA clients:
                #   - all trainables can live in compute_dtype
                non_lora_dtype = torch.float32 if is_qlora else None

                modify_adapter(
                    peft_model=model.model,
                    adapter_name=adapter_name,
                    modify_module_rank=client_rank_config,
                    lora_alpha=default_alpha,
                    lora_dropout=default_dropout,
                    init_lora_weights=True,
                    target_modules=list(client_rank_config.keys()),
                    compute_dtype=compute_dtype,
                    non_lora_trainable_dtype=non_lora_dtype,
                    recast_trainables=True,
                    recast_log_prefix=f"[HeteroLoRA] Client {client_id}:",
                )

                model.model.print_trainable_parameters()

                # Convert to plain dict for logging (in case it's a CfgNode)
                # Filter to only include valid module names (exclude config metadata)
                # Common LoRA target module patterns:
                # - LLM (Llama, etc.): _proj, _attn
                # - NLP (DeBERTa, BERT, etc.): .dense, in_proj
                valid_module_patterns = ['_proj', '_attn', '.dense', 'in_proj']
                
                def is_valid_module_name(name):
                    """Check if a key looks like a LoRA module name."""
                    return (any(pattern in name for pattern in valid_module_patterns) or
                            name in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 
                                    'gate_proj', 'up_proj', 'down_proj',
                                    'c_attn', 'c_proj'])
                
                # Only include keys that look like module names with numeric ranks
                client_rank_dict = {}
                for k, v in client_rank_config.items():
                    k_str = str(k)
                    if is_valid_module_name(k_str):
                        if isinstance(v, (int, float)) and not k_str.startswith('_'):
                            client_rank_dict[k_str] = int(v)
                
                # Serialize the module->rank map as sorted JSON for the log line
                import json
                try:
                    # Sort keys for consistent output
                    dict_str = json.dumps(client_rank_dict, sort_keys=True)
                except Exception as e:
                    # Fallback to repr if json fails
                    dict_str = str(client_rank_dict)
                
                logger.info(
                    f"Applied heterogeneous LoRA ranks to client {client_id}: "
                    f"{dict_str}"
                )
        
        client = self.client_class(
            ID=client_id,
            server_id=self.server_id,
            config=client_specific_config,
            data=client_data,
            model=model,
            device=client_device,
            is_unseen_client=client_id in self.unseen_clients_id,
            **kw)

        if self.cfg.vertical.use:
            from federatedscope.vertical_fl.utils import wrap_vertical_client
            client = wrap_vertical_client(client, config=self.cfg)

        if client_id == -1:
            logger.info('Client (address {}:{}) has been set up ... '.format(
                self.client_address['host'], self.client_address['port']))
        else:
            logger.info(f'________Client {client_id} has been set up________')

        return self.feat_engr_wrapper_client(client)

    def check(self):
        """
        Check the completeness of Server and Client.

        """
        if not self.cfg.check_completeness:
            return
        try:
            import os
            import networkx as nx
            import matplotlib.pyplot as plt
            # Build check graph
            G = nx.DiGraph()
            flags = {0: 'Client', 1: 'Server'}
            msg_handler_dicts = [
                self.client_class.get_msg_handler_dict(),
                self.server_class.get_msg_handler_dict()
            ]
            for flag, msg_handler_dict in zip(flags.keys(), msg_handler_dicts):
                role, oppo = flags[flag], flags[(flag + 1) % 2]
                for msg_in, (handler, msgs_out) in \
                        msg_handler_dict.items():
                    for msg_out in msgs_out:
                        msg_in_key = f'{oppo}_{msg_in}'
                        handler_key = f'{role}_{handler}'
                        msg_out_key = f'{role}_{msg_out}'
                        G.add_node(msg_in_key, subset=1)
                        G.add_node(handler_key, subset=0 if flag else 2)
                        G.add_node(msg_out_key, subset=1)
                        G.add_edge(msg_in_key, handler_key)
                        G.add_edge(handler_key, msg_out_key)
            pos = nx.multipartite_layout(G)
            plt.figure(figsize=(20, 15))
            nx.draw(G,
                    pos,
                    with_labels=True,
                    node_color='white',
                    node_size=800,
                    width=1.0,
                    arrowsize=25,
                    arrowstyle='->')
            fig_path = os.path.join(self.cfg.outdir, 'msg_handler.png')
            plt.savefig(fig_path)
            if nx.has_path(G, 'Client_join_in', 'Server_finish'):
                if nx.is_weakly_connected(G):
                    logger.info(f'Completeness check passes! Save check '
                                f'results in {fig_path}.')
                else:
                    logger.warning(f'Completeness check raises warning for '
                                   f'some handlers not in FL process! Save '
                                   f'check results in {fig_path}.')
            else:
                logger.error(f'Completeness check fails for there is no'
                             f'path from `join_in` to `finish`! Save '
                             f'check results in {fig_path}.')
        except Exception as error:
            logger.warning(f'Completeness check failed for {error}!')
        return


class StandaloneRunner(BaseRunner):
    def _set_up(self):
        """
        To set up server and client for standalone mode.
        """
        self.is_run_online = True if self.cfg.federate.online_aggr else False
        self.shared_comm_queue = deque()

        if self.cfg.backend == 'torch':
            import torch
            torch.set_num_threads(1)

        assert self.cfg.federate.client_num != 0, \
            "In standalone mode, self.cfg.federate.client_num should be " \
            "non-zero. " \
            "This is usually cased by using synthetic data and users not " \
            "specify a non-zero value for client_num"

        if self.cfg.federate.method == "global":
            self.cfg.defrost()
            self.cfg.federate.client_num = 1
            self.cfg.federate.sample_client_num = 1
            self.cfg.freeze()

        # sample resource information
        if self.resource_info is not None:
            if len(self.resource_info) < self.cfg.federate.client_num + 1:
                replace = True
                logger.warning(
                    f"Because the provided the number of resource information "
                    f"{len(self.resource_info)} is less than the number of "
                    f"participants {self.cfg.federate.client_num + 1}, one "
                    f"candidate might be selected multiple times.")
            else:
                replace = False
            sampled_index = np.random.choice(
                list(self.resource_info.keys()),
                size=self.cfg.federate.client_num + 1,
                replace=replace)
            server_resource_info = self.resource_info[sampled_index[0]]
            client_resource_info = [
                self.resource_info[x] for x in sampled_index[1:]
            ]
        else:
            server_resource_info = None
            client_resource_info = None

        self.server = self._setup_server(
            resource_info=server_resource_info,
            client_resource_info=client_resource_info)

        self.client = dict()
        # assume the client-wise data are consistent in their input&output
        # shape
        if self.cfg.federate.online_aggr:
            self._shared_client_model = get_model(
                self.cfg, self.data[1], backend=self.cfg.backend
            ) if self.cfg.federate.share_local_model else None
        else:
            self._shared_client_model = self.server.model \
                if self.cfg.federate.share_local_model else None
        
        # For heterogeneous LoRA, each client must have its own model instance
        # to avoid rank mutations on a shared object
        if self.hetero_lora_config is not None:
            if self._shared_client_model is not None:
                logger.info(
                    "Heterogeneous LoRA detected: disabling shared client model "
                    "to ensure per-client rank independence"
                )
                self._shared_client_model = None
        
        for client_id in range(1, self.cfg.federate.client_num + 1):
            self.client[client_id] = self._setup_client(
                client_id=client_id,
                client_model=self._shared_client_model,
                resource_info=client_resource_info[client_id - 1]
                if client_resource_info is not None else None)

        # in standalone mode, by default, we print the trainer info only
        # once for better logs readability
        trainer_representative = self.client[1].trainer
        if trainer_representative is not None and hasattr(
                trainer_representative, 'print_trainer_meta_info'):
            trainer_representative.print_trainer_meta_info()

    def _get_server_args(self, resource_info=None, client_resource_info=None):
        if self.server_id in self.data:
            server_data = self.data[self.server_id]
            model = get_model(self.cfg, server_data, backend=self.cfg.backend)
        else:
            server_data = None
            data_representative = self.data[1]
            model = get_model(
                self.cfg, data_representative, backend=self.cfg.backend
            )  # get the model according to client's data if the server
            # does not own data
        kw = {
            'shared_comm_queue': self.shared_comm_queue,
            'resource_info': resource_info,
            'client_resource_info': client_resource_info
        }
        return server_data, model, kw

    def _get_client_args(self, client_id=-1, resource_info=None):
        client_data = self.data[client_id]
        kw = {
            'shared_comm_queue': self.shared_comm_queue,
            'resource_info': resource_info
        }
        return client_data, kw

    def run(self):
        for each_client in self.client:
            # Launch each client
            self.client[each_client].join_in()

        if self.is_run_online:
            self._run_simulation_online()
        else:
            self._run_simulation()
        # TODO: avoid using private attr
        self.server._monitor.finish_fed_runner(fl_mode=self.mode)
        return self.server.best_results

    def _handle_msg(self, msg, rcv=-1):
        """
        To simulate the message handling process (used only for the \
        standalone mode)
        """
        if rcv != -1:
            # simulate broadcast one-by-one
            self.client[rcv].msg_handlers[msg.msg_type](msg)
            return

        _, receiver = msg.sender, msg.receiver
        download_bytes, upload_bytes = msg.count_bytes()
        if not isinstance(receiver, list):
            receiver = [receiver]
        for each_receiver in receiver:
            if each_receiver == 0:
                self.server.msg_handlers[msg.msg_type](msg)
                self.server._monitor.track_download_bytes(download_bytes)
            else:
                self.client[each_receiver].msg_handlers[msg.msg_type](msg)
                self.client[each_receiver]._monitor.track_download_bytes(
                    download_bytes)

    def _run_simulation_online(self):
        """
        Run for online aggregation.
        Any broadcast operation would be executed client-by-clien to avoid \
        the existence of #clients messages at the same time. Currently, \
        only consider centralized topology \
        """
        def is_broadcast(msg):
            return len(msg.receiver) >= 1 and msg.sender == 0

        cached_bc_msgs = []
        cur_idx = 0
        while True:
            if len(self.shared_comm_queue) > 0:
                msg = self.shared_comm_queue.popleft()
                if is_broadcast(msg):
                    cached_bc_msgs.append(msg)
                    # assume there is at least one client
                    msg = cached_bc_msgs[0]
                    self._handle_msg(msg, rcv=msg.receiver[cur_idx])
                    cur_idx += 1
                    if cur_idx >= len(msg.receiver):
                        del cached_bc_msgs[0]
                        cur_idx = 0
                else:
                    self._handle_msg(msg)
            elif len(cached_bc_msgs) > 0:
                msg = cached_bc_msgs[0]
                self._handle_msg(msg, rcv=msg.receiver[cur_idx])
                cur_idx += 1
                if cur_idx >= len(msg.receiver):
                    del cached_bc_msgs[0]
                    cur_idx = 0
            else:
                # finished
                break

    def _run_simulation(self):
        """
        Run for standalone simulation (W/O online aggr)
        """
        server_msg_cache = list()
        while True:
            if len(self.shared_comm_queue) > 0:
                msg = self.shared_comm_queue.popleft()
                if not self.cfg.vertical.use and msg.receiver == [
                        self.server_id
                ]:
                    # For the server, move the received message to a
                    # cache for reordering the messages according to
                    # the timestamps
                    msg.serial_num = self.serial_num_for_msg
                    self.serial_num_for_msg += 1
                    heapq.heappush(server_msg_cache, msg)
                else:
                    self._handle_msg(msg)
            elif len(server_msg_cache) > 0:
                msg = heapq.heappop(server_msg_cache)
                if self.cfg.asyn.use and self.cfg.asyn.aggregator \
                        == 'time_up':
                    # When the timestamp of the received message beyond
                    # the deadline for the currency round, trigger the
                    # time up event first and push the message back to
                    # the cache
                    if self.server.trigger_for_time_up(msg.timestamp):
                        heapq.heappush(server_msg_cache, msg)
                    else:
                        self._handle_msg(msg)
                else:
                    self._handle_msg(msg)
            else:
                if self.cfg.asyn.use and self.cfg.asyn.aggregator \
                        == 'time_up':
                    self.server.trigger_for_time_up()
                    if len(self.shared_comm_queue) == 0 and \
                            len(server_msg_cache) == 0:
                        break
                else:
                    # terminate when shared_comm_queue and
                    # server_msg_cache are all empty
                    break


class DistributedRunner(BaseRunner):
    def _set_up(self):
        """
        To set up server or client for distributed mode.
        """
        # sample resource information
        if self.resource_info is not None:
            sampled_index = np.random.choice(list(self.resource_info.keys()))
            sampled_resource = self.resource_info[sampled_index]
        else:
            sampled_resource = None

        self.server_address = {
            'host': self.cfg.distribute.server_host,
            'port': self.cfg.distribute.server_port + self.ds_rank
        }
        if self.cfg.distribute.role == 'server':
            self.server = self._setup_server(resource_info=sampled_resource)
        elif self.cfg.distribute.role == 'client':
            # When we set up the client in the distributed mode, we assume
            # the server has been set up and number with #0
            self.client_address = {
                'host': self.cfg.distribute.client_host,
                'port': self.cfg.distribute.client_port + self.ds_rank
            }
            self.client = self._setup_client(resource_info=sampled_resource)

    def _get_server_args(self, resource_info, client_resource_info):
        server_data = self.data
        model = get_model(self.cfg, server_data, backend=self.cfg.backend)
        kw = self.server_address
        kw.update({'resource_info': resource_info})
        return server_data, model, kw

    def _get_client_args(self, client_id, resource_info):
        client_data = self.data
        kw = self.client_address
        kw['server_host'] = self.server_address['host']
        kw['server_port'] = self.server_address['port']
        kw['resource_info'] = resource_info
        return client_data, kw

    def run(self):
        if self.cfg.distribute.role == 'server':
            self.server.run()
            return self.server.best_results
        elif self.cfg.distribute.role == 'client':
            self.client.join_in()
            self.client.run()


# TODO: remove FedRunner (keep now for forward compatibility)
class FedRunner(object):
    """
    This class is used to construct an FL course, which includes `_set_up`
    and `run`.

    Arguments:
        data: The data used in the FL courses, which are formatted as \
        ``{'ID':data}`` for standalone mode. More details can be found in \
        federatedscope.core.auxiliaries.data_builder .
        server_class: The server class is used for instantiating a ( \
        customized) server.
        client_class: The client class is used for instantiating a ( \
        customized) client.
        config: The configurations of the FL course.
        client_configs: The clients' configurations.

    Warnings:
        ``FedRunner`` will be removed in the future, consider \
        using ``StandaloneRunner`` or ``DistributedRunner`` instead!
    """
    def __init__(self,
                 data,
                 server_class=Server,
                 client_class=Client,
                 config=None,
                 client_configs=None):
        logger.warning('`federate.core.fed_runner.FedRunner` will be '
                       'removed in the future, please use'
                       '`federate.core.fed_runner.get_runner` to get '
                       'Runner.')
        self.data = data
        self.server_class = server_class
        self.client_class = client_class
        assert config is not None, \
            "When using FedRunner, you should specify the `config` para"
        if not config.is_ready_for_run:
            config.ready_for_run()
        self.cfg = config
        self.client_cfgs = client_configs

        self.mode = self.cfg.federate.mode.lower()
        self.gpu_manager = GPUManager(gpu_available=self.cfg.use_gpu,
                                      specified_device=self.cfg.device)

        self.unseen_clients_id = []
        if self.cfg.federate.unseen_clients_rate > 0:
            self.unseen_clients_id = np.random.choice(
                np.arange(1, self.cfg.federate.client_num + 1),
                size=max(
                    1,
                    int(self.cfg.federate.unseen_clients_rate *
                        self.cfg.federate.client_num)),
                replace=False).tolist()
        # get resource information
        self.resource_info = get_resource_info(
            config.federate.resource_info_file)

        # Check the completeness of msg_handler.
        self.check()

    def setup(self):
        if self.mode == 'standalone':
            self.shared_comm_queue = deque()
            self._setup_for_standalone()
            # in standalone mode, by default, we print the trainer info only
            # once for better logs readability
            trainer_representative = self.client[1].trainer
            if trainer_representative is not None:
                trainer_representative.print_trainer_meta_info()
        elif self.mode == 'distributed':
            self._setup_for_distributed()

    def _setup_for_standalone(self):
        """
        To set up server and client for standalone mode.
        """
        if self.cfg.backend == 'torch':
            import torch
            torch.set_num_threads(1)

        assert self.cfg.federate.client_num != 0, \
            "In standalone mode, self.cfg.federate.client_num should be " \
            "non-zero. " \
            "This is usually cased by using synthetic data and users not " \
            "specify a non-zero value for client_num"

        if self.cfg.federate.method == "global":
            self.cfg.defrost()
            self.cfg.federate.client_num = 1
            self.cfg.federate.sample_client_num = 1
            self.cfg.freeze()

        # sample resource information
        if self.resource_info is not None:
            if len(self.resource_info) < self.cfg.federate.client_num + 1:
                replace = True
                logger.warning(
                    f"Because the provided the number of resource information "
                    f"{len(self.resource_info)} is less than the number of "
                    f"participants {self.cfg.federate.client_num+1}, one "
                    f"candidate might be selected multiple times.")
            else:
                replace = False
            sampled_index = np.random.choice(
                list(self.resource_info.keys()),
                size=self.cfg.federate.client_num + 1,
                replace=replace)
            server_resource_info = self.resource_info[sampled_index[0]]
            client_resource_info = [
                self.resource_info[x] for x in sampled_index[1:]
            ]
        else:
            server_resource_info = None
            client_resource_info = None

        self.server = self._setup_server(
            resource_info=server_resource_info,
            client_resource_info=client_resource_info)

        self.client = dict()

        # assume the client-wise data are consistent in their input&output
        # shape
        self._shared_client_model = get_model(
            self.cfg, self.data[1], backend=self.cfg.backend
        ) if self.cfg.federate.share_local_model else None

        for client_id in range(1, self.cfg.federate.client_num + 1):
            self.client[client_id] = self._setup_client(
                client_id=client_id,
                client_model=self._shared_client_model,
                resource_info=client_resource_info[client_id - 1]
                if client_resource_info is not None else None)

    def _setup_for_distributed(self):
        """
        To set up server or client for distributed mode.
        """

        # sample resource information
        if self.resource_info is not None:
            sampled_index = np.random.choice(list(self.resource_info.keys()))
            sampled_resource = self.resource_info[sampled_index]
        else:
            sampled_resource = None

        self.server_address = {
            'host': self.cfg.distribute.server_host,
            'port': self.cfg.distribute.server_port
        }
        if self.cfg.distribute.role == 'server':
            self.server = self._setup_server(resource_info=sampled_resource)
        elif self.cfg.distribute.role == 'client':
            # When we set up the client in the distributed mode, we assume
            # the server has been set up and number with #0
            self.client_address = {
                'host': self.cfg.distribute.client_host,
                'port': self.cfg.distribute.client_port
            }
            self.client = self._setup_client(resource_info=sampled_resource)

    def run(self):
        """
        To run an FL course, which is called after server/client has been
        set up.
        For the standalone mode, a shared message queue will be set up to
        simulate ``receiving message``.
        """
        self.setup()
        if self.mode == 'standalone':
            # trigger the FL course
            for each_client in self.client:
                self.client[each_client].join_in()

            if self.cfg.federate.online_aggr:
                # any broadcast operation would be executed client-by-client
                # to avoid the existence of #clients messages at the same time.
                # currently, only consider centralized topology
                self._run_simulation_online()

            else:
                self._run_simulation()

            self.server._monitor.finish_fed_runner(fl_mode=self.mode)

            return self.server.best_results

        elif self.mode == 'distributed':
            if self.cfg.distribute.role == 'server':
                self.server.run()
                return self.server.best_results
            elif self.cfg.distribute.role == 'client':
                self.client.join_in()
                self.client.run()

    def _run_simulation_online(self):
        def is_broadcast(msg):
            return len(msg.receiver) >= 1 and msg.sender == 0

        cached_bc_msgs = []
        cur_idx = 0
        while True:
            if len(self.shared_comm_queue) > 0:
                msg = self.shared_comm_queue.popleft()
                if is_broadcast(msg):
                    cached_bc_msgs.append(msg)
                    # assume there is at least one client
                    msg = cached_bc_msgs[0]
                    self._handle_msg(msg, rcv=msg.receiver[cur_idx])
                    cur_idx += 1
                    if cur_idx >= len(msg.receiver):
                        del cached_bc_msgs[0]
                        cur_idx = 0
                else:
                    self._handle_msg(msg)
            elif len(cached_bc_msgs) > 0:
                msg = cached_bc_msgs[0]
                self._handle_msg(msg, rcv=msg.receiver[cur_idx])
                cur_idx += 1
                if cur_idx >= len(msg.receiver):
                    del cached_bc_msgs[0]
                    cur_idx = 0
            else:
                # finished
                break

    def _run_simulation(self):
        server_msg_cache = list()
        while True:
            if len(self.shared_comm_queue) > 0:
                msg = self.shared_comm_queue.popleft()
                if msg.receiver == [self.server_id]:
                    # For the server, move the received message to a
                    # cache for reordering the messages according to
                    # the timestamps
                    heapq.heappush(server_msg_cache, msg)
                else:
                    self._handle_msg(msg)
            elif len(server_msg_cache) > 0:
                msg = heapq.heappop(server_msg_cache)
                if self.cfg.asyn.use and self.cfg.asyn.aggregator \
                        == 'time_up':
                    # When the timestamp of the received message beyond
                    # the deadline for the currency round, trigger the
                    # time up event first and push the message back to
                    # the cache
                    if self.server.trigger_for_time_up(msg.timestamp):
                        heapq.heappush(server_msg_cache, msg)
                    else:
                        self._handle_msg(msg)
                else:
                    self._handle_msg(msg)
            else:
                if self.cfg.asyn.use and self.cfg.asyn.aggregator \
                        == 'time_up':
                    self.server.trigger_for_time_up()
                    if len(self.shared_comm_queue) == 0 and \
                            len(server_msg_cache) == 0:
                        break
                else:
                    # terminate when shared_comm_queue and
                    # server_msg_cache are all empty
                    break

    def _setup_server(self, resource_info=None, client_resource_info=None):
        """
        Set up the server
        """
        self.server_id = 0
        if self.mode == 'standalone':
            if self.server_id in self.data:
                server_data = self.data[self.server_id]
                model = get_model(self.cfg,
                                  server_data,
                                  backend=self.cfg.backend)
            else:
                server_data = None
                data_representative = self.data[1]
                model = get_model(
                    self.cfg, data_representative, backend=self.cfg.backend
                )  # get the model according to client's data if the server
                # does not own data
            kw = {
                'shared_comm_queue': self.shared_comm_queue,
                'resource_info': resource_info,
                'client_resource_info': client_resource_info
            }
        elif self.mode == 'distributed':
            server_data = self.data
            model = get_model(self.cfg, server_data, backend=self.cfg.backend)
            kw = self.server_address
            kw.update({'resource_info': resource_info})
        else:
            raise ValueError('Mode {} is not provided'.format(
                self.cfg.mode.type))

        if self.server_class:
            self._server_device = self.gpu_manager.auto_choice()
            server = self.server_class(
                ID=self.server_id,
                config=self.cfg,
                data=server_data,
                model=model,
                client_num=self.cfg.federate.client_num,
                total_round_num=self.cfg.federate.total_round_num,
                device=self._server_device,
                unseen_clients_id=self.unseen_clients_id,
                **kw)

            if self.cfg.nbafl.use:
                from federatedscope.core.trainers.trainer_nbafl import \
                    wrap_nbafl_server
                wrap_nbafl_server(server)

        else:
            raise ValueError

        logger.info('Server has been set up ... ')

        return server

    def _setup_client(self,
                      client_id=-1,
                      client_model=None,
                      resource_info=None):
        """
        Set up the client
        """
        self.server_id = 0
        if self.mode == 'standalone':
            client_data = self.data[client_id]
            kw = {
                'shared_comm_queue': self.shared_comm_queue,
                'resource_info': resource_info
            }
        elif self.mode == 'distributed':
            client_data = self.data
            kw = self.client_address
            kw['server_host'] = self.server_address['host']
            kw['server_port'] = self.server_address['port']
            kw['resource_info'] = resource_info
        else:
            raise ValueError('Mode {} is not provided'.format(
                self.cfg.mode.type))

        if self.client_class:
            client_specific_config = self.cfg.clone()
            if self.client_cfgs and \
                    self.client_cfgs.get('client_{}'.format(client_id)):
                client_specific_config.defrost()
                client_specific_config.merge_from_other_cfg(
                    self.client_cfgs.get('client_{}'.format(client_id)))
                client_specific_config.freeze()
            client_device = self._server_device if \
                self.cfg.federate.share_local_model else \
                self.gpu_manager.auto_choice()
            client = self.client_class(ID=client_id,
                                       server_id=self.server_id,
                                       config=client_specific_config,
                                       data=client_data,
                                       model=client_model
                                       or get_model(client_specific_config,
                                                    client_data,
                                                    backend=self.cfg.backend),
                                       device=client_device,
                                       is_unseen_client=client_id
                                       in self.unseen_clients_id,
                                       **kw)
        else:
            raise ValueError

        if client_id == -1:
            logger.info('Client (address {}:{}) has been set up ... '.format(
                self.client_address['host'], self.client_address['port']))
        else:
            logger.info(f'Client {client_id} has been set up ... ')

        return client

    def _handle_msg(self, msg, rcv=-1):
        """
        To simulate the message handling process (used only for the
        standalone mode)
        """
        if rcv != -1:
            # simulate broadcast one-by-one
            self.client[rcv].msg_handlers[msg.msg_type](msg)
            return

        _, receiver = msg.sender, msg.receiver
        download_bytes, upload_bytes = msg.count_bytes()
        if not isinstance(receiver, list):
            receiver = [receiver]
        for each_receiver in receiver:
            if each_receiver == 0:
                self.server.msg_handlers[msg.msg_type](msg)
                self.server._monitor.track_download_bytes(download_bytes)
            else:
                self.client[each_receiver].msg_handlers[msg.msg_type](msg)
                self.client[each_receiver]._monitor.track_download_bytes(
                    download_bytes)

    def check(self):
        """
        Check the completeness of Server and Client.

        """
        if not self.cfg.check_completeness:
            return
        try:
            import os
            import networkx as nx
            import matplotlib.pyplot as plt
            # Build check graph
            G = nx.DiGraph()
            flags = {0: 'Client', 1: 'Server'}
            msg_handler_dicts = [
                self.client_class.get_msg_handler_dict(),
                self.server_class.get_msg_handler_dict()
            ]
            for flag, msg_handler_dict in zip(flags.keys(), msg_handler_dicts):
                role, oppo = flags[flag], flags[(flag + 1) % 2]
                for msg_in, (handler, msgs_out) in \
                        msg_handler_dict.items():
                    for msg_out in msgs_out:
                        msg_in_key = f'{oppo}_{msg_in}'
                        handler_key = f'{role}_{handler}'
                        msg_out_key = f'{role}_{msg_out}'
                        G.add_node(msg_in_key, subset=1)
                        G.add_node(handler_key, subset=0 if flag else 2)
                        G.add_node(msg_out_key, subset=1)
                        G.add_edge(msg_in_key, handler_key)
                        G.add_edge(handler_key, msg_out_key)
            pos = nx.multipartite_layout(G)
            plt.figure(figsize=(20, 15))
            nx.draw(G,
                    pos,
                    with_labels=True,
                    node_color='white',
                    node_size=800,
                    width=1.0,
                    arrowsize=25,
                    arrowstyle='->')
            fig_path = os.path.join(self.cfg.outdir, 'msg_handler.png')
            plt.savefig(fig_path)
            if nx.has_path(G, 'Client_join_in', 'Server_finish'):
                if nx.is_weakly_connected(G):
                    logger.info(f'Completeness check passes! Save check '
                                f'results in {fig_path}.')
                else:
                    logger.warning(f'Completeness check raises warning for '
                                   f'some handlers not in FL process! Save '
                                   f'check results in {fig_path}.')
            else:
                logger.error(f'Completeness check fails for there is no'
                             f'path from `join_in` to `finish`! Save '
                             f'check results in {fig_path}.')
        except Exception as error:
            logger.warning(f'Completeness check failed for {error}!')
        return