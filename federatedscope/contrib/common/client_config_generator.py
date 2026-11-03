"""
Client configuration generator for HeteroLoRA.

This module provides functions to generate client-specific LoRA configurations
with different ranks, alpha values, and target modules based on various
distribution strategies.
"""
import random
import numpy as np
from scipy.stats import norm
import logging

logger = logging.getLogger(__name__)


def get_client_lora_config(config_types, num_clients, strategy='random',
                          seed=42, default_alpha=16, default_dropout=0.05,
                          **kwargs):
    """
    Generate client-specific LoRA configurations based on distribution strategy.
    
    Args:
        config_types: Dict mapping type names to their LoRA configurations
            Format: {
                'Type_0': {'q_proj': 8, 'v_proj': 8, ...},
                'Type_1': {'q_proj': 16, 'v_proj': 16, ...},
                ...
            }
        num_clients: Total number of clients
        strategy: Distribution strategy
            - 'homo': All clients use the same configuration (returns None)
            - 'random': Random assignment of types
            - 'heavy_tail': Heavy-tailed distribution (80% Type_0, 10% Type_1, etc.)
            - 'heavy_tail_strong': Strong heavy-tailed distribution
            - 'normal': Normal distribution over types
            - 'custom': Fixed type mix for 6 or 12 clients
        seed: Random seed for reproducibility
        default_alpha: Default LoRA alpha value
        default_dropout: Default LoRA dropout rate
    
    Returns:
        Dict mapping client IDs to their configurations:
        {
            'alpha': default_alpha,
            'lora_dropout': default_dropout,
            'Client_0': {'q_proj': 8, 'v_proj': 8, ...},
            'Client_1': {'q_proj': 16, 'v_proj': 16, ...},
            ...
        }
    """
    if strategy == 'homo':
        return None
    
    # Set random seed
    random.seed(seed)
    np.random.seed(seed)
    
    config_local = {
        'alpha': default_alpha,
        'lora_dropout': default_dropout
    }
    
    if strategy == 'random':
        # Random assignment of types
        # Note: FederatedScope uses 1-indexed client IDs, so we generate Client_1, Client_2, etc.
        type_names = list(config_types.keys())
        for i in range(1, num_clients + 1):
            type_name = random.choice(type_names)
            config_local['Client_' + str(i)] = config_types[type_name].copy()
    
    elif strategy == 'heavy_tail':
        # Heavy-tailed distribution: 80% Type_0, 10% Type_1, 5% Type_2, 5% Type_3
        # Note: FederatedScope uses 1-indexed client IDs
        type_names = sorted(config_types.keys())
        for i in range(1, num_clients + 1):
            rand_num = random.random()
            if rand_num < 0.80:
                type_name = type_names[0] if len(type_names) > 0 else 'Type_0'
            elif rand_num < 0.90:
                type_name = type_names[1] if len(type_names) > 1 else type_names[0]
            elif rand_num < 0.95:
                type_name = type_names[2] if len(type_names) > 2 else type_names[0]
            else:
                type_name = type_names[3] if len(type_names) > 3 else type_names[0]
            config_local['Client_' + str(i)] = config_types[type_name].copy()
    
    elif strategy == 'heavy_tail_strong':
        # Strong heavy-tailed distribution toward small ranks:
        # 90% Type_0 (smallest), 7% Type_1, 2% Type_2, 1% Type_3 (largest)
        # Note: FederatedScope uses 1-indexed client IDs
        type_names = sorted(config_types.keys())
        for i in range(1, num_clients + 1):
            rand_num = random.random()
            if rand_num < 0.90:
                type_name = type_names[0] if len(type_names) > 0 else 'Type_0'
            elif rand_num < 0.97:
                type_name = type_names[1] if len(type_names) > 1 else type_names[0]
            elif rand_num < 0.99:
                type_name = type_names[2] if len(type_names) > 2 else type_names[0]
            else:
                type_name = type_names[3] if len(type_names) > 3 else type_names[0]
            config_local['Client_' + str(i)] = config_types[type_name].copy()
    
    elif strategy == 'normal':
        # Normal distribution over types
        # Note: FederatedScope uses 1-indexed client IDs
        type_names = sorted(config_types.keys())
        positions = np.array(list(range(len(type_names))))
        mu = len(type_names) / 2.0
        sigma = len(type_names) / 3.0
        probabilities = norm.pdf(positions, mu, sigma)
        probabilities /= probabilities.sum()
        
        for i in range(1, num_clients + 1):
            selected_idx = np.random.choice(positions, p=probabilities)
            type_name = type_names[int(selected_idx)]
            config_local['Client_' + str(i)] = config_types[type_name].copy()
    
    elif strategy == 'custom':
        # Fixed type mixes for 6 or 12 clients.
        # Uses seed controlled shuffle so which client id gets which type is randomized but reproducible.
        type_names = sorted(config_types.keys())

        if len(type_names) < 4:
            raise ValueError("custom strategy expects at least 4 types: Type_0..Type_3")

        custom_mixes = {
            6:  (2, 2, 1, 1),
            12: (3, 4, 3, 2),
        }
        if num_clients not in custom_mixes:
            raise ValueError(
                f"custom strategy supports num_clients in {sorted(custom_mixes)}, got {num_clients}"
            )
        mix = custom_mixes[num_clients]
        type_list = []
        for t, count in zip(type_names[:4], mix):
            type_list.extend([t] * count)
        random.shuffle(type_list)

        for i in range(1, num_clients + 1):
            type_name = type_list[i - 1]
            config_local['Client_' + str(i)] = config_types[type_name].copy()

    else:
        raise ValueError(f"Unknown distribution strategy: {strategy}")
    
    return config_local


def get_client_rank_caps(config_local, reduction='min'):
    """
    Derive immutable per-client scalar rank caps from a heterogeneous
    client configuration dictionary.

    Args:
        config_local: Client configuration dict from get_client_lora_config
            or a manually supplied config_local.
        reduction: How to reduce module-wise ranks to a single scalar cap.
            - 'min': safe scalar cap for schedulers that use one rank per client
            - 'max': largest module rank per client

    Returns:
        Dict mapping integer client IDs to integer caps, e.g.:
        {1: 64, 2: 120, 3: 200}
    """
    if config_local is None:
        return {}

    if reduction not in {'min', 'max'}:
        raise ValueError("reduction must be either 'min' or 'max'")

    client_rank_caps = {}

    for client_key, client_config in config_local.items():
        client_key_str = str(client_key)
        if not client_key_str.startswith('Client_'):
            continue
        if not hasattr(client_config, 'items'):
            continue

        numeric_ranks = [
            int(v) for _, v in client_config.items()
            if isinstance(v, (int, float))
        ]
        if not numeric_ranks:
            continue

        try:
            client_id = int(client_key_str.split('_')[-1])
        except ValueError:
            continue

        client_rank_caps[client_id] = (
            min(numeric_ranks) if reduction == 'min' else max(numeric_ranks)
        )

    return client_rank_caps

def get_default_client_types():
    """
    Get default client type configurations similar to FlexLoRA.
    
    Returns:
        Dict mapping type names to their LoRA rank configurations
    """
    return {
        'Type_0': {  # small
            'q_proj': 64, 'v_proj': 64, 'k_proj': 64, 'o_proj': 64,
            'gate_proj': 64, 'down_proj': 64, 'up_proj': 64
        },
        'Type_1': {  # medium-low
            'q_proj': 120, 'v_proj': 120, 'k_proj': 120, 'o_proj': 120,
            'gate_proj': 120, 'down_proj': 120, 'up_proj': 120
        },
        'Type_2': {  # medium
            'q_proj': 150, 'v_proj': 150, 'k_proj': 150, 'o_proj': 150,
            'gate_proj': 150, 'down_proj': 150, 'up_proj': 150
        },
        'Type_3': {  # large
            'q_proj': 200, 'v_proj': 200, 'k_proj': 200, 'o_proj': 200,
            'gate_proj': 200, 'down_proj': 200, 'up_proj': 200
        }
    }


