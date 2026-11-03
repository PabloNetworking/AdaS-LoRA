"""
Network trace loader for bandwidth simulation.

This module provides utilities to load and sample bandwidth values from network trace CSV files.
All bandwidth values are in kbit/s (the canonical internal unit).

Supported modes (managed by RoundBandwidthManager):
- static: Sample once at initialization, values remain fixed
- dynamic: Advance one sample per round

The force_symmetric_ul_dl behavior is managed by RoundBandwidthManager.
"""
import logging
import os
import random
from typing import Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


class NetworkTraceLoader:
    """
    Loader for network trace CSV files.
    
    Handles reading bandwidth values (DL_bitrate, UL_bitrate) from CSV files
    and sampling them for client bandwidth simulation.
    
    All returned values are in kbit/s (raw values from CSV).
    
    Sampling always starts from the beginning of the trace (deterministic):
    - static mode: returns the first valid sample
    - dynamic mode: starts at index 0 and advances sequentially
    """
    
    def __init__(
        self,
        trace_path: str,
        client_class: str,
        file_name: Optional[str] = None,
        start_index: Optional[int] = None,
        rng: Optional[random.Random] = None,
    ):
        """
        Initialize network trace loader for a specific client class.
        
        Args:
            trace_path: Path to network trace directory
            client_class: Subfolder name (e.g., 'pedestrian', 'bus', 'car')
            file_name: Optional specific file name. If None, selects a file via rng
            start_index: Optional explicit starting index for dynamic sampling.
                        If None, starts from index 0 (deterministic).
            rng: Dedicated RNG for communication-side randomness.
        """
        self.trace_path = trace_path
        self.client_class = client_class
        self.file_name = file_name
        self.rng = rng if rng is not None else random.Random()
        self.data = None
        self.current_index = 0
        self._load_trace_file()
        
        # Set starting index: use provided start_index or default to 0
        if start_index is not None and self.data is not None:
            self.current_index = start_index % len(self.data)
        # Without an explicit start_index, sampling starts from index 0
    
    def _load_trace_file(self):
        """Load the network trace CSV file."""
        class_path = os.path.join(self.trace_path, self.client_class)
        
        if not os.path.exists(class_path):
            raise ValueError(f"Network trace class directory not found: {class_path}")
        
        if self.file_name is None:
            csv_files = [f for f in os.listdir(class_path) if f.endswith('.csv')]
            if not csv_files:
                raise ValueError(f"No CSV files found in {class_path}")
            self.file_name = self.rng.choice(csv_files)
        
        file_path = os.path.join(class_path, self.file_name)
        
        if not os.path.exists(file_path):
            raise ValueError(f"Network trace file not found: {file_path}")
        
        try:
            self.data = pd.read_csv(file_path)
            
            if 'DL_bitrate' not in self.data.columns or 'UL_bitrate' not in self.data.columns:
                raise ValueError(f"CSV file missing required columns: DL_bitrate, UL_bitrate")
            
            # Filter out rows where either DL or UL are 0
            self.data = self.data[(self.data['DL_bitrate'] > 0) & (self.data['UL_bitrate'] > 0)]
            
            if len(self.data) == 0:
                raise ValueError(f"No valid (non-zero) bandwidth entries in {file_path}")
            
            logger.debug(
                f"Loaded {len(self.data)} entries from {self.file_name} "
                f"(class: {self.client_class})"
            )
        except Exception as e:
            raise ValueError(f"Failed to load network trace file {file_path}: {e}")
    
    def sample_static(self) -> Tuple[float, float]:
        """
        Sample the first valid bandwidth pair (static mode).
        
        Returns the first valid (non-zero) sample from the trace file.
        This value remains fixed for static mode.
        
        Returns:
            Tuple of (downlink_kbits, uplink_kbits) in kbit/s
        """
        if self.data is None or len(self.data) == 0:
            raise ValueError("No data loaded")
        
        # Return first valid row (data already filtered for non-zero values)
        row = self.data.iloc[0]
        
        dl_kbits = float(row['DL_bitrate'])
        ul_kbits = float(row['UL_bitrate'])
        
        return dl_kbits, ul_kbits
    
    def sample_dynamic(self) -> Tuple[float, float]:
        """
        Sample the next bandwidth pair in sequence (dynamic mode).
        Advances exactly one step per call.
        
        Returns:
            Tuple of (downlink_kbits, uplink_kbits) in kbit/s
        """
        if self.data is None or len(self.data) == 0:
            raise ValueError("No data loaded")
        
        if self.current_index >= len(self.data):
            self.current_index = 0
        
        row = self.data.iloc[self.current_index]
        
        dl_kbits = float(row['DL_bitrate'])
        ul_kbits = float(row['UL_bitrate'])
        
        self.current_index += 1
        
        if dl_kbits == 0 and ul_kbits == 0:
            while self.current_index < len(self.data):
                row = self.data.iloc[self.current_index]
                dl_kbits = float(row['DL_bitrate'])
                ul_kbits = float(row['UL_bitrate'])
                self.current_index += 1
                if dl_kbits > 0 or ul_kbits > 0:
                    break
            
            if dl_kbits == 0 and ul_kbits == 0:
                self.current_index = 0
                return self.sample_dynamic()
        
        return dl_kbits, ul_kbits
    
    def get_current_index(self) -> int:
        """Get the current sampling index."""
        return self.current_index
    
    def __len__(self) -> int:
        """Return the number of valid entries in the trace file."""
        return len(self.data) if self.data is not None else 0
    
    def reset(self):
        """Reset the dynamic index to start."""
        self.current_index = 0


def assign_clients_to_classes(
    client_ids: List[int],
    distribution: Dict[str, int],
    rng: Optional[random.Random] = None,
) -> Dict[int, str]:
    """
    Assign clients to network trace classes based on distribution percentages.
    
    Args:
        client_ids: List of client IDs
        distribution: Dict mapping class names to percentages
            Example: {'pedestrian': 70, 'bus': 20, 'car': 10}
    
    Returns:
        Dict mapping client_id to class name
    """
    rng = rng if rng is not None else random.Random()
    total = sum(distribution.values())
    if total != 100:
        logger.warning(
            f"Distribution percentages sum to {total}, not 100. Normalizing."
        )
        distribution = {k: int(v * 100 / total) for k, v in distribution.items()}
    
    assignments = []
    for class_name, percentage in distribution.items():
        count = int(len(client_ids) * percentage / 100)
        assignments.extend([class_name] * count)
    
    while len(assignments) < len(client_ids):
        assignments.append(list(distribution.keys())[0])
    
    assignments = assignments[:len(client_ids)]
    
    rng.shuffle(assignments)
    
    client_to_class = {
        client_id: class_name
        for client_id, class_name in zip(client_ids, assignments)
    }
    
    logger.info(
        f"Assigned {len(client_ids)} clients to classes: "
        f"{dict(zip(*zip(*[(k, assignments.count(k)) for k in set(assignments)])))}"
    )
    
    return client_to_class



def create_trace_loaders(
    trace_path: str,
    client_to_class: Dict[int, str],
    total_rounds: Optional[int] = None,
    rng: Optional[random.Random] = None,
) -> Dict[int, NetworkTraceLoader]:
    """
    Create NetworkTraceLoader instances for each client.
    
    Each client is assigned a trace file of its class uniformly at random,
    so clients of the same class may share a file.
    
    All clients start sampling from index 0 (the beginning of their assigned trace).
    If multiple clients share the same file, they will sample the same sequence.
    
    Args:
        trace_path: Path to network trace directory
        client_to_class: Dict mapping client_id to class name
        total_rounds: Total number of FL rounds (unused)
    
    Returns:
        Dict mapping client_id to NetworkTraceLoader instance
    """
    rng = rng if rng is not None else random.Random()
    loaders = {}
    
    class_to_clients: Dict[str, List[int]] = {}
    for client_id, class_name in client_to_class.items():
        if class_name not in class_to_clients:
            class_to_clients[class_name] = []
        class_to_clients[class_name].append(client_id)
    
    for class_name, client_ids in class_to_clients.items():
        class_path = os.path.join(trace_path, class_name)
        
        if not os.path.exists(class_path):
            raise ValueError(f"Network trace class directory not found: {class_path}")
        
        csv_files = [f for f in os.listdir(class_path) if f.endswith('.csv')]
        if not csv_files:
            raise ValueError(f"No CSV files found in {class_path}")
        
        num_files = len(csv_files)
        num_clients = len(client_ids)
        
        logger.info(
            f"Class '{class_name}': {num_files} files available for {num_clients} clients"
        )
        
        # Randomly assign files to clients
        file_to_clients: Dict[int, List[int]] = {i: [] for i in range(num_files)}
        
        for client_id in client_ids:
            file_idx = rng.randint(0, num_files - 1)
            file_to_clients[file_idx].append(client_id)
        
        for file_idx, clients_on_file in file_to_clients.items():
            if not clients_on_file:
                continue
            
            file_name = csv_files[file_idx]
            
            for client_id in clients_on_file:
                try:
                    # All clients start from index 0 (deterministic)
                    loaders[client_id] = NetworkTraceLoader(
                        trace_path,
                        class_name,
                        file_name=file_name,
                        start_index=0,
                        rng=rng,
                    )
                    
                    logger.debug(
                        f"Client {client_id} assigned file '{file_name}' "
                        f"(class: {class_name}, start_index: 0)"
                    )
                except Exception as e:
                    logger.error(f"Failed to create loader for client {client_id}: {e}")
                    raise
        
        if num_clients > num_files:
            logger.info(
                f"Class '{class_name}': {num_clients} clients sharing {num_files} files. "
                f"Clients sharing the same file will sample the same sequence from index 0."
            )
    
    return loaders
