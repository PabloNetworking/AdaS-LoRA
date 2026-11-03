"""Utilities shared by the method implementations."""

from .config_resolver import (
    normalize_method_name,
    is_glue_task,
    get_adapter_root,
    get_adapter_args_list,
    get_hetero_ranks_cfg,
    get_hetlora_cfg,
    get_adas_pruneonly_cfg,
    get_adas_cfg,
    get_adas_layeraware_cfg,
    get_fah_cfg,
    get_active_hetero_config_local,
    get_effective_target_modules,
    get_effective_max_rank,
)
from .payload_utils import (
    is_wrapped_payload,
    extract_model_para,
)
from .rank_utils import (
    resolve_client_key,
    infer_rank_from_client_rank_config,
    indices_from_rank,
    normalize_indices,
)
from .bandwidth_manager import RoundBandwidthManager
from .network_trace_loader import (
    NetworkTraceLoader,
    assign_clients_to_classes,
    create_trace_loaders,
)
from .client_config_generator import (
    get_client_lora_config,
    get_client_rank_caps,
    get_default_client_types,
)
from .heterolora_utils import (
    is_qlora_client_cfg,
    modify_adapter,
    distribute_weight_fast,
    load_weight_local,
    compute_lora_size,
    iter_lora_pairs,
    tail_penalty,
    tail_score,
    truncate_client_lora_to_rank,
    get_current_lora_rank_from_state_dict,
)
from .adas_utils import (
    compute_component_scores,
    compute_lowset_and_score,
    slice_update_by_keep_positions,
    validate_indices,
    get_component_norm,
    compute_per_component_norms_from_state_dict,
    distribute_weights_by_indices,
    compute_model_update_from_snapshot,
    apply_residual_to_update,
    compute_stage2_upload_scores,
    compute_stage2_downlink_scores,
    compute_component_upload_cost,
    compute_component_downlink_cost,
    greedy_select_by_score_cost_ratio,
    slice_model_update_by_indices,
    apply_sparse_update_to_model,
    update_residual_buffers_after_upload,
    prune_residual_buffers,
    validate_upload_subset,
    compute_residual_norm_summary,
)
from .adas_layeraware_utils import (
    # Layer key helpers
    canonicalize_lora_layer_key,
    strip_rank_suffix,
    get_canonical_lora_param_key,
    infer_layer_keys_from_state_dict,
    get_lora_keys_for_layer,
    # ComponentID helpers
    flatten_grouped_indices_by_layer,
    group_component_ids_by_layer,
    normalize_indices_to_grouped,
    # Layer-aware scoring
    compute_component_scores_grouped,
    compute_lowset_grouped,
    # Stage 2 scoring (grouped)
    compute_stage2_upload_scores_grouped,
    compute_stage2_downlink_scores_grouped,
    # Layer-aware cost helpers
    compute_component_upload_cost_grouped,
    compute_component_downlink_cost_grouped,
    # Selection helpers (grouped)
    greedy_select_by_score_cost_ratio_grouped,
    # Distribution/slicing helpers (grouped)
    distribute_weights_by_layer_indices,
    slice_model_update_by_component_ids,
    # Residual buffer helpers (grouped)
    update_residual_buffers_after_upload_grouped,
    apply_residual_to_update_grouped,
    prune_residual_buffers_grouped,
    # Model update computation
    compute_model_update_from_snapshot_grouped,
    # Validation helpers
    compute_residual_norm_summary_grouped,
    validate_layeraware_state,
)