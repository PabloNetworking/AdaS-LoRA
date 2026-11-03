"""Helpers that locate method config blocks and adapter settings in the config tree."""
from __future__ import annotations

from typing import Any, Iterable, Optional


DEFAULT_TARGET_MODULES = [
    'q_proj', 'k_proj', 'v_proj', 'o_proj',
    'gate_proj', 'down_proj', 'up_proj'
]


METHOD_ALIASES = {
    'adas-pruneonly': 'adas_pruneonly',
    'adas_pruneonly': 'adas_pruneonly',
    'adas': 'adas',
    'adas-layeraware': 'adas_layeraware',
    'adas_layeraware': 'adas_layeraware',
    'fah-qlora': 'fah_qlora',
    'fah_qlora': 'fah_qlora',
    'hetlora': 'hetlora',
    'heterolora': 'heterolora',
    'hetero-lora': 'heterolora',
}


def normalize_method_name(method: Any) -> str:
    """Normalize a federated method name and map aliases to the canonical name."""
    if method is None:
        return ''
    name = str(method).strip().lower()
    return METHOD_ALIASES.get(name, name)



def _safe_getattr(obj: Any, name: str, default: Any = None) -> Any:
    return getattr(obj, name, default) if obj is not None else default



def is_glue_task(cfg: Any) -> bool:
    """Return True when the current run is a GLUE-style task.

    Matches data types that contain '@glue' or start with 'glue'.
    """
    data_type = _safe_getattr(_safe_getattr(cfg, 'data', None), 'type', '')
    if not isinstance(data_type, str):
        return False
    lowered = data_type.lower()
    return ('@glue' in lowered) or lowered.startswith('glue')



def get_adapter_root(cfg: Any, prefer_glue: Optional[bool] = None) -> Any:
    """Return the active adapter root, preferring the task-specific one.

    The return value is usually cfg.glue.adapter or cfg.llm.adapter.  When
    neither exists, returns None.
    """
    glue_adapter = None
    if hasattr(cfg, 'glue') and hasattr(cfg.glue, 'adapter'):
        glue_adapter = cfg.glue.adapter

    llm_adapter = None
    if hasattr(cfg, 'llm') and hasattr(cfg.llm, 'adapter'):
        llm_adapter = cfg.llm.adapter

    if prefer_glue is None:
        prefer_glue = is_glue_task(cfg)

    if prefer_glue:
        return glue_adapter if glue_adapter is not None else llm_adapter
    return llm_adapter if llm_adapter is not None else glue_adapter



def get_adapter_args_list(cfg: Any, prefer_glue: Optional[bool] = None) -> list:
    """Return adapter.args as a list, or an empty list when unavailable."""
    root = get_adapter_root(cfg, prefer_glue=prefer_glue)
    args = _safe_getattr(root, 'args', None)
    if args is None:
        return []
    try:
        return list(args)
    except TypeError:
        return []



def _get_named_feature_cfg(adapter_root: Any, feature_name: str, enabled_only: bool = True) -> Any:
    if adapter_root is None or not hasattr(adapter_root, feature_name):
        return None
    feature_cfg = getattr(adapter_root, feature_name)
    if enabled_only and not getattr(feature_cfg, 'enabled', False):
        return None
    return feature_cfg



def _get_method_scoped_cfg(
    cfg: Any,
    *,
    feature_name: str,
    allowed_methods: Optional[Iterable[str]] = None,
    enabled_only: bool = True,
    prefer_glue: Optional[bool] = None,
) -> Any:
    method = normalize_method_name(_safe_getattr(_safe_getattr(cfg, 'federate', None), 'method', ''))
    if allowed_methods is not None:
        normalized_allowed = {normalize_method_name(m) for m in allowed_methods}
        if method not in normalized_allowed:
            return None

    if prefer_glue is None:
        prefer_glue = is_glue_task(cfg)

    roots = []
    primary = get_adapter_root(cfg, prefer_glue=prefer_glue)
    secondary = get_adapter_root(cfg, prefer_glue=not prefer_glue)
    if primary is not None:
        roots.append(primary)
    if secondary is not None and secondary is not primary:
        roots.append(secondary)

    for root in roots:
        feature_cfg = _get_named_feature_cfg(root, feature_name, enabled_only=enabled_only)
        if feature_cfg is not None:
            return feature_cfg
    return None



def get_hetero_ranks_cfg(cfg: Any, enabled_only: bool = False) -> Any:
    """Return hetero_ranks config from GLUE or LLM adapter roots.

    Not gated on federate.method: hetero_ranks holds the client-specific rank
    configs read by several methods (HetLoRA, HeteroLoRA, FAH-QLoRA and AdaS).
    """
    return _get_method_scoped_cfg(
        cfg,
        feature_name='hetero_ranks',
        allowed_methods=None,
        enabled_only=enabled_only,
    )



def get_hetlora_cfg(cfg: Any) -> Any:
    return _get_method_scoped_cfg(
        cfg,
        feature_name='hetlora',
        allowed_methods={'hetlora'},
        enabled_only=True,
    )



def get_adas_pruneonly_cfg(cfg: Any) -> Any:
    return _get_method_scoped_cfg(
        cfg,
        feature_name='adas_pruneonly',
        allowed_methods={'adas_pruneonly', 'adas-pruneonly'},
        enabled_only=True,
    )



def get_adas_cfg(cfg: Any) -> Any:
    return _get_method_scoped_cfg(
        cfg,
        feature_name='adas',
        allowed_methods={'adas'},
        enabled_only=True,
    )



def get_adas_layeraware_cfg(cfg: Any) -> Any:
    """Return AdaS-LayerAware config if method matches and is enabled."""
    return _get_method_scoped_cfg(
        cfg,
        feature_name='adas_layeraware',
        allowed_methods={'adas_layeraware', 'adas-layeraware'},
        enabled_only=True,
    )



def get_fah_cfg(cfg: Any) -> Any:
    """Return FAH config.

    Looks in glue.adapter for GLUE tasks and llm.adapter otherwise, then in
    the other one.
    """
    return _get_method_scoped_cfg(
        cfg,
        feature_name='fah',
        allowed_methods={'fah_qlora', 'fah-qlora'},
        enabled_only=True,
    )



def get_active_hetero_config_local(cfg: Any) -> Any:
    """Resolve the active hetero_ranks.config_local.

    Prefers the task-specific config and otherwise falls back to whichever
    one is defined.
    """
    glue_config_local = None
    if hasattr(cfg, 'glue') and hasattr(cfg.glue, 'adapter') and \
            hasattr(cfg.glue.adapter, 'hetero_ranks') and \
            hasattr(cfg.glue.adapter.hetero_ranks, 'config_local'):
        glue_config_local = cfg.glue.adapter.hetero_ranks.config_local

    llm_config_local = None
    if hasattr(cfg, 'llm') and hasattr(cfg.llm, 'adapter') and \
            hasattr(cfg.llm.adapter, 'hetero_ranks') and \
            hasattr(cfg.llm.adapter.hetero_ranks, 'config_local'):
        llm_config_local = cfg.llm.adapter.hetero_ranks.config_local

    if is_glue_task(cfg):
        return glue_config_local or llm_config_local
    return llm_config_local or glue_config_local



def get_effective_target_modules(cfg: Any, default: Optional[Iterable[str]] = None) -> list:
    """Resolve adapter target_modules from the preferred adapter root."""
    target_modules = list(default) if default is not None else list(DEFAULT_TARGET_MODULES)

    args_list = get_adapter_args_list(cfg)
    if args_list:
        first = args_list[0]
        if isinstance(first, dict):
            tm = first.get('target_modules', None)
            if tm:
                return list(tm)

    secondary_args_list = get_adapter_args_list(cfg, prefer_glue=not is_glue_task(cfg))
    if secondary_args_list:
        first = secondary_args_list[0]
        if isinstance(first, dict):
            tm = first.get('target_modules', None)
            if tm:
                return list(tm)

    return target_modules



def get_effective_max_rank(cfg: Any, default: int = 64) -> int:
    """Resolve max_rank from the preferred adapter root, then fallback root."""
    primary = get_adapter_root(cfg)
    secondary = get_adapter_root(cfg, prefer_glue=not is_glue_task(cfg))

    for root in [primary, secondary]:
        if root is not None and hasattr(root, 'max_rank'):
            return int(root.max_rank)
    return int(default)
