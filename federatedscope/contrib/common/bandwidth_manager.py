"""
Per-round client bandwidth shared by all federated learning methods.

This module provides a shared RoundBandwidthManager that handles:
- Per-client bandwidth state (upload/download in kbit/s)
- Static and dynamic bandwidth modes
- Generated, trace-based and TC JSON bandwidth sources
- Per-round history tracking
- Generic history export

All methods (FAH-QLoRA, AdaS, etc.) read bandwidth_info from this
shared manager.
"""
import csv
import json
import logging
import os
import random
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class RoundBandwidthManager:
    """
    Server-owned bandwidth manager for all federated methods.
    
    Responsibilities:
    - Initialize per-client communication state
    - Assign clients to trace classes when traces are used
    - Create and own per-client trace loaders
    - Sample initial bandwidth values
    - Advance dynamic bandwidth exactly one step per round
    - Expose normalized per-client bandwidth_info
    - Keep round-by-round history for all clients
    - Export generic history for all methods
    
    Supported modes:
    - static: Sample once at init, values fixed for all rounds
    - dynamic: Re-sample (or advance trace) once per round
    
    Supported sources:
    - generated: Random sampling from [uplink_min_mbps, uplink_max_mbps], fixed downlink_mbps
    - trace: Read from network trace CSV files
    - tc_json: Per-round uplink/downlink profiles from JSON files
    """
    
    def __init__(self, cfg: Any):
        """
        Initialize the bandwidth manager from federate.communication config.
        
        Args:
            cfg: Full config object containing federate.communication subtree
        """
        # Extract communication config
        comm_cfg = None
        if hasattr(cfg, 'federate') and hasattr(cfg.federate, 'communication'):
            comm_cfg = cfg.federate.communication
        
        # Core settings with defaults
        self.enabled = getattr(comm_cfg, 'enabled', True) if comm_cfg else True
        self.mode = getattr(comm_cfg, 'mode', 'static') if comm_cfg else 'static'
        self.source = getattr(comm_cfg, 'source', 'generated') if comm_cfg else 'generated'
        
        # Validate mode
        if self.mode not in ('static', 'dynamic'):
            logger.warning(
                f"Invalid bandwidth mode '{self.mode}'. "
                f"Expected 'static' or 'dynamic'. Defaulting to 'static'."
            )
            self.mode = 'static'
        
        # Validate source
        if self.source not in ('generated', 'trace', 'tc_json'):
            logger.warning(
                f"Invalid bandwidth source '{self.source}'. "
                f"Expected 'generated', 'trace', or 'tc_json'. Defaulting to 'generated'."
            )
            self.source = 'generated'

        # TC JSON paths
        self.tc_json_ul_path = getattr(comm_cfg, 'tc_json_ul_path', '') if comm_cfg else ''
        self.tc_json_dl_path = getattr(comm_cfg, 'tc_json_dl_path', '') if comm_cfg else ''

        # Number of clients (needed for per-device DL computation from total server egress)
        self.client_num = getattr(cfg.federate, 'client_num', 12) if hasattr(cfg, 'federate') else 12
        
        # Trace settings
        self.trace_path = getattr(comm_cfg, 'network_trace_path', '') if comm_cfg else ''
        self.trace_uplink_scale = float(getattr(comm_cfg, 'trace_uplink_scale', 10.0)) if comm_cfg else 10.0
        self.trace_distribution = {}
        if comm_cfg and hasattr(comm_cfg, 'network_trace_distribution'):
            dist = comm_cfg.network_trace_distribution
            if hasattr(dist, 'items'):
                # Filter out CN internal attributes
                self.trace_distribution = {
                    k: int(v) for k, v in dist.items()
                    if not str(k).startswith('_') and isinstance(v, (int, float))
                }
        
        # Dedicated communication seed. Falls back to cfg.seed when not set.
        comm_seed = getattr(comm_cfg, 'seed', None) if comm_cfg else None
        if comm_seed is None:
            comm_seed = getattr(cfg, 'seed', None)
        if comm_seed is None:
            comm_seed = 0
        self.communication_seed = int(comm_seed)
        self.rng = random.Random(self.communication_seed)

        # Generated bandwidth settings (in Mbps from config, converted to kbit/s internally)
        uplink_min = getattr(comm_cfg, 'uplink_min_mbps', 5) if comm_cfg else 5
        uplink_max = getattr(comm_cfg, 'uplink_max_mbps', 20) if comm_cfg else 20
        downlink = getattr(comm_cfg, 'downlink_mbps', 50) if comm_cfg else 50
        
        # Convert Mbps to kbit/s for internal canonical unit
        self.uplink_min_kbits = uplink_min * 1000
        self.uplink_max_kbits = uplink_max * 1000
        self.downlink_kbits = downlink * 1000
        
        # Symmetric UL/DL option
        self.force_symmetric_ul_dl = getattr(comm_cfg, 'force_symmetric_ul_dl', False) if comm_cfg else False
        
        # Export settings
        self.export_history = getattr(comm_cfg, 'export_history', True) if comm_cfg else True
        
        # Per-client state
        self.client_bandwidth: Dict[int, Dict[str, Any]] = {}
        self.client_to_class: Dict[int, str] = {}
        self.trace_loaders: Dict[int, Any] = {}
        
        # History tracking: {client_id: [(round, upload_kbits, download_kbits, metadata), ...]}
        self.bandwidth_history: Dict[int, List[Tuple]] = {}
        
        # Track current round
        self.current_round = -1
        self.initialized = False

        # Store total rounds for trace allocation
        self.total_rounds: Optional[int] = None
        
        logger.info(
            f"RoundBandwidthManager created: "
            f"enabled={self.enabled}, mode='{self.mode}', source='{self.source}', "
            f"communication_seed={self.communication_seed}, "
            f"force_symmetric_ul_dl={self.force_symmetric_ul_dl}, "
            f"uplink_range=[{uplink_min}, {uplink_max}] Mbps, "
            f"downlink={downlink} Mbps"
        )
    
    def init(self, client_ids: List[int], total_rounds: Optional[int] = None) -> None:
        """
        Initialize per-client bandwidth state.
        
        Args:
            client_ids: List of client IDs to initialize
            total_rounds: Total number of FL rounds (used for trace allocation)
        """
        client_ids = sorted(client_ids)

        if not self.enabled:
            for client_id in sorted(client_ids):
                self.client_bandwidth[client_id] = {
                    'upload_kbits': float('inf'),
                    'download_kbits': float('inf'),
                }
            logger.info("RoundBandwidthManager disabled, using unlimited bandwidth")
            self.initialized = True
            return
        
        self.total_rounds = total_rounds
        
        if self.source == 'tc_json':
            self._init_tc_json_source(client_ids)
        elif self.source == 'trace':
            self._init_trace_source(client_ids)
        else:
            self._init_generated_source(client_ids)
        
        # Sample initial values
        self._sample_bandwidth_all(client_ids, round_idx=0, is_initial=True)
        
        self.current_round = 0
        self.initialized = True
        
        self._log_bandwidth_summary("initialization")
    
    def _init_trace_source(self, client_ids: List[int]) -> None:
        """Initialize trace-based bandwidth source."""
        if not self.trace_path or not os.path.exists(self.trace_path):
            logger.warning(
                f"Trace path '{self.trace_path}' not found. "
                f"Falling back to generated source."
            )
            self.source = 'generated'
            return
        
        try:
            from federatedscope.contrib.common.network_trace_loader import (
                assign_clients_to_classes,
                create_trace_loaders
            )
            
            # Get distribution or use default
            distribution = self.trace_distribution
            if not distribution:
                # Find first available class
                classes = [
                    d for d in os.listdir(self.trace_path)
                    if os.path.isdir(os.path.join(self.trace_path, d))
                ]
                if classes:
                    distribution = {classes[0]: 100}
                else:
                    logger.warning("No trace classes found. Falling back to generated source.")
                    self.source = 'generated'
                    return
            
            # Assign clients to trace classes
            self.client_to_class = assign_clients_to_classes(client_ids, distribution, rng=self.rng)
            
            # Create trace loaders
            self.trace_loaders = create_trace_loaders(
                self.trace_path,
                self.client_to_class,
                total_rounds=self.total_rounds,
                rng=self.rng,
            )
            
            logger.info(
                f"Trace source initialized: "
                f"{len(self.trace_loaders)} clients, "
                f"{len(set(self.client_to_class.values()))} classes"
            )
            
        except Exception as e:
            logger.warning(f"Failed to initialize trace source: {e}. Falling back to generated.")
            self.source = 'generated'
            self.trace_loaders = {}
            self.client_to_class = {}
    
    def _init_tc_json_source(self, client_ids: List[int]) -> None:
        """Load uplink/downlink bandwidth profiles from TC JSON files."""
        if not self.tc_json_ul_path or not os.path.exists(self.tc_json_ul_path):
            logger.warning(
                f"TC JSON UL path '{self.tc_json_ul_path}' not found. "
                f"Falling back to generated source."
            )
            self.source = 'generated'
            return
        if not self.tc_json_dl_path or not os.path.exists(self.tc_json_dl_path):
            logger.warning(
                f"TC JSON DL path '{self.tc_json_dl_path}' not found. "
                f"Falling back to generated source."
            )
            self.source = 'generated'
            return

        with open(self.tc_json_ul_path) as f:
            self._tc_ul_trace = json.load(f)["bandwidth_limits"]
        with open(self.tc_json_dl_path) as f:
            self._tc_dl_trace = json.load(f)["bandwidth_limits"]

        self._tc_trace_len = len(self._tc_ul_trace)

        # Per-client UL scaling factors (auto-discovered from same directory)
        self._tc_client_factors: Dict[int, float] = {}
        sf_path = os.path.join(
            os.path.dirname(self.tc_json_ul_path),
            "bandwidth_scaling_factors.json",
        )
        if os.path.exists(sf_path):
            with open(sf_path) as f:
                sf_data = json.load(f)
            for cid_str, info in sf_data.get("clients", {}).items():
                self._tc_client_factors[int(cid_str)] = float(info["factor"])
            logger.info(
                "Per-client UL scaling loaded (%d clients): %s",
                len(self._tc_client_factors),
                {k: f"{v:.4f}" for k, v in sorted(self._tc_client_factors.items())},
            )
        else:
            logger.info("No per-client scaling factors found at %s", sf_path)

        logger.info(
            f"TC JSON source initialized: "
            f"{self._tc_trace_len} entries, "
            f"UL range [{min(e['mbps'] for e in self._tc_ul_trace):.2f}, "
            f"{max(e['mbps'] for e in self._tc_ul_trace):.2f}] Mbps, "
            f"DL total range [{min(e['mbps'] for e in self._tc_dl_trace):.2f}, "
            f"{max(e['mbps'] for e in self._tc_dl_trace):.2f}] Mbps, "
            f"DL per-client /{self.client_num}"
        )

    def _sample_tc_json_bandwidth(self, client_id: int, round_idx: int) -> Tuple[float, float]:
        """Sample bandwidth from TC JSON trace using round_idx cycling.

        When per-client scaling factors are loaded, the base UL value is
        multiplied by the client's factor so each device gets a different
        share of the bandwidth budget.
        """
        idx = round_idx % self._tc_trace_len
        ul_mbps = self._tc_ul_trace[idx]["mbps"]
        dl_total_mbps = self._tc_dl_trace[idx]["mbps"]
        dl_per_client_mbps = dl_total_mbps / self.client_num

        if self._tc_client_factors and client_id in self._tc_client_factors:
            ul_mbps *= self._tc_client_factors[client_id]

        upload_kbits = ul_mbps * 1000
        download_kbits = dl_per_client_mbps * 1000
        return upload_kbits, download_kbits

    def _init_generated_source(self, client_ids: List[int]) -> None:
        """Initialize generated bandwidth source (no setup needed)."""
        logger.info(f"Generated source initialized for {len(client_ids)} clients")
    
    def _sample_bandwidth_all(self, client_ids: List[int], round_idx: int, is_initial: bool = False) -> None:
        """
        Sample bandwidth for all clients.
        
        Args:
            client_ids: List of client IDs
            round_idx: Current round index
            is_initial: Whether this is initial sampling (for static mode)
        """
        for client_id in sorted(client_ids):
            # For static mode after initial sampling, skip
            if self.mode == 'static' and not is_initial and client_id in self.client_bandwidth:
                continue
            
            # Sample bandwidth
            upload_kbits, download_kbits, metadata = self._sample_client_bandwidth(client_id)
            
            # Apply symmetric UL/DL if configured
            if self.force_symmetric_ul_dl:
                upload_kbits = download_kbits
            
            # Store current values
            self.client_bandwidth[client_id] = {
                'upload_kbits': upload_kbits,
                'download_kbits': download_kbits,
                'trace_class': metadata.get('trace_class'),
                'trace_file': metadata.get('trace_file'),
                'trace_index': metadata.get('trace_index'),
            }
            
            # Record history
            if client_id not in self.bandwidth_history:
                self.bandwidth_history[client_id] = []
            
            self.bandwidth_history[client_id].append((
                round_idx,
                upload_kbits,
                download_kbits,
                metadata.get('trace_class'),
                metadata.get('trace_file'),
                metadata.get('trace_index'),
            ))
    
    def _sample_client_bandwidth(self, client_id: int) -> Tuple[float, float, Dict]:
        """
        Sample bandwidth for a single client.
        
        Returns:
            Tuple of (upload_kbits, download_kbits, metadata_dict)
        """
        metadata = {
            'trace_class': None,
            'trace_file': None,
            'trace_index': None,
        }
        
        if self.source == 'tc_json':
            upload_kbits, download_kbits = self._sample_tc_json_bandwidth(
                client_id, self.current_round if self.current_round >= 0 else 0
            )
            metadata['trace_index'] = (self.current_round if self.current_round >= 0 else 0) % self._tc_trace_len
            return upload_kbits, download_kbits, metadata

        if self.source == 'trace' and client_id in self.trace_loaders:
            loader = self.trace_loaders[client_id]
            
            # Capture index BEFORE sampling (for accurate metadata)
            # For dynamic mode, sample_dynamic() increments after reading
            sampled_index = loader.get_current_index()
            
            # Sample based on mode
            if self.mode == 'static':
                # Static always returns first row (index 0)
                download_kbits, upload_kbits = loader.sample_static()
                sampled_index = 0
            else:  # dynamic
                download_kbits, upload_kbits = loader.sample_dynamic()
                # sampled_index was captured before the call
            
            # Scale traced uplink rates (federate.communication.trace_uplink_scale)
            upload_kbits *= self.trace_uplink_scale
            
            # Populate metadata
            metadata['trace_class'] = self.client_to_class.get(client_id)
            metadata['trace_file'] = getattr(loader, 'file_name', None)
            metadata['trace_index'] = sampled_index
            
        else:
            # Generated source: random uplink, fixed downlink
            upload_kbits = int(round(self.rng.uniform(self.uplink_min_kbits, self.uplink_max_kbits)))
            download_kbits = int(round(self.downlink_kbits))
        
        return upload_kbits, download_kbits, metadata
    
    def advance_round(self, round_idx: int) -> None:
        """
        Advance bandwidth state for a new round.
        
        For dynamic mode: re-sample or advance trace for all clients.
        For static mode: no-op (values remain fixed).
        
        Args:
            round_idx: The new round index
        """
        if not self.enabled or not self.initialized:
            return
        
        self.current_round = round_idx
        
        if self.mode == 'static':
            # Static mode: values don't change after initialization
            return
        
        # Dynamic mode: re-sample for all clients
        client_ids = sorted(self.client_bandwidth.keys())
        self._sample_bandwidth_all(client_ids, round_idx, is_initial=False)
        
        logger.debug(f"RoundBandwidthManager advanced to round {round_idx}")
    
    def get_bandwidth_info(self, client_id: int, round_idx: Optional[int] = None) -> Dict[str, Any]:
        """
        Get normalized bandwidth_info dict for a client.
        
        Args:
            client_id: Client ID
            round_idx: Optional round index (for history lookup, defaults to current)
        
        Returns:
            Dict with bandwidth info suitable for message payloads
        """
        # Use explicit None check to preserve round_idx=0
        effective_round = self.current_round if round_idx is None else round_idx
        
        if not self.enabled:
            return {
                'round': effective_round,
                'upload_kbits': float('inf'),
                'download_kbits': float('inf'),
                'units': 'kbit/s',
                'mode': self.mode,
                'source': self.source,
                'trace_class': None,
                'trace_file': None,
                'trace_index': None,
            }
        
        if client_id not in self.client_bandwidth:
            # Fallback: sample on-the-fly
            upload_kbits, download_kbits, metadata = self._sample_client_bandwidth(client_id)
            if self.force_symmetric_ul_dl:
                upload_kbits = download_kbits
        else:
            bw = self.client_bandwidth[client_id]
            upload_kbits = bw['upload_kbits']
            download_kbits = bw['download_kbits']
            metadata = {
                'trace_class': bw.get('trace_class'),
                'trace_file': bw.get('trace_file'),
                'trace_index': bw.get('trace_index'),
            }

        return {
            'round': effective_round,
            'upload_kbits': upload_kbits,
            'download_kbits': download_kbits,
            'units': 'kbit/s',
            'mode': self.mode,
            'source': self.source,
            'trace_class': metadata.get('trace_class'),
            'trace_file': metadata.get('trace_file'),
            'trace_index': metadata.get('trace_index'),
        }
    
    def save_history(self, outdir: str, filename: str = 'bandwidth_history.csv') -> None:
        """
        Save bandwidth history to a CSV file.
        
        Args:
            outdir: Output directory
            filename: Output filename
        """
        if not self.export_history or not self.bandwidth_history:
            return
        
        try:
            os.makedirs(outdir, exist_ok=True)
            filepath = os.path.join(outdir, filename)
            
            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                
                # Write header
                writer.writerow([
                    'round',
                    'client_id',
                    'upload_kbits',
                    'download_kbits',
                    'upload_mbps',
                    'download_mbps',
                    'mode',
                    'source',
                    'trace_class',
                    'trace_file',
                    'trace_index',
                ])

                # Write data
                for client_id in sorted(self.bandwidth_history.keys()):
                    for entry in self.bandwidth_history[client_id]:
                        round_idx, ul_kbits, dl_kbits, trace_class, trace_file, trace_index = entry
                        writer.writerow([
                            round_idx,
                            client_id,
                            ul_kbits,
                            dl_kbits,
                            ul_kbits / 1000,
                            dl_kbits / 1000,
                            self.mode,
                            self.source,
                            trace_class or '',
                            trace_file or '',
                            trace_index if trace_index is not None else '',
                        ])
            
            logger.info(f"Bandwidth history saved to {filepath}")
            
        except Exception as e:
            logger.warning(f"Failed to save bandwidth history: {e}")
    
    def _log_bandwidth_summary(self, context: str) -> None:
        """Log bandwidth summary for all clients."""
        if not self.client_bandwidth:
            return
        
        ul_values = [bw['upload_kbits'] for bw in self.client_bandwidth.values()]
        dl_values = [bw['download_kbits'] for bw in self.client_bandwidth.values()]
        
        # Filter out infinite values for stats
        ul_finite = [v for v in ul_values if v != float('inf')]
        dl_finite = [v for v in dl_values if v != float('inf')]
        
        if ul_finite:
            ul_stats = f"{min(ul_finite)/1000:.1f}/{sum(ul_finite)/len(ul_finite)/1000:.1f}/{max(ul_finite)/1000:.1f}"
        else:
            ul_stats = "inf"
        
        if dl_finite:
            dl_stats = f"{min(dl_finite)/1000:.1f}/{sum(dl_finite)/len(dl_finite)/1000:.1f}/{max(dl_finite)/1000:.1f}"
        else:
            dl_stats = "inf"
        
        logger.info(
            f"RoundBandwidthManager {context}: "
            f"n_clients={len(self.client_bandwidth)}, mode={self.mode}, source={self.source}, "
            f"uplink(min/avg/max)={ul_stats} Mbps, "
            f"downlink(min/avg/max)={dl_stats} Mbps"
        )
