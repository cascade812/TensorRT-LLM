# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Experimental GLM DSA KV-cache offload overhead prototype.

This module intentionally measures transfer overhead without reclaiming HBM:
the normal GPU KV cache remains authoritative while three shared layers after
selected full-indexer layers are mirrored to pinned host memory. During decode,
the full layer gathers the first shared layer's rows, and each of the first two
shared layers gathers the following layer's rows. Each transfer runs on a
stage-specific auxiliary stream while the preceding layer computes.

Enable one group with
TRTLLM_DSA_KV_OFFLOAD_PROTOTYPE_FULL_LAYER=<global layer>, or every complete
local group with TRTLLM_DSA_KV_OFFLOAD_PROTOTYPE_FULL_LAYER=all. The prototype
offloads all three shared positions by default. Set
TRTLLM_DSA_KV_OFFLOAD_PROTOTYPE_GROUP_LAYERS=3,4 to keep the first shared
position resident and offload only the last two. The prototype supports FP8
latent KV, no speculative decoding, and a fresh fixed-batch run. It is not a
user-facing configuration surface.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Dict, NamedTuple, Optional, Tuple, Union

import torch

from tensorrt_llm.bindings import DataType
from tensorrt_llm.logger import logger

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention_backend.interface import AttentionMetadata

    from .backend import DSATrtllmAttention
    from .cache_manager import DSACacheManager
    from .metadata import DSAtrtllmAttentionMetadata


_PROTOTYPE_ENV = "TRTLLM_DSA_KV_OFFLOAD_PROTOTYPE_FULL_LAYER"
_GROUP_LAYERS_ENV = "TRTLLM_DSA_KV_OFFLOAD_PROTOTYPE_GROUP_LAYERS"
_MAX_HOST_GIB_ENV = "TRTLLM_DSA_KV_OFFLOAD_PROTOTYPE_MAX_HOST_GIB"
_NUM_SHARED_LAYERS = 3
_LATENT_BYTES = 576
_HOST_ROW_BYTES = _NUM_SHARED_LAYERS * _LATENT_BYTES
_DEFAULT_MAX_HOST_GIB = 3.5


class _OffloadGroup(NamedTuple):
    shared_layers: Tuple[int, int, int]
    offloaded_layer_indices: Tuple[int, ...]
    host_pool: torch.Tensor


def _get_prototype_selector() -> Optional[Union[int, str]]:
    """Return a full layer, ``all``, or None when the prototype is disabled."""
    value = os.environ.get(_PROTOTYPE_ENV)
    if value is None:
        return None
    if value.lower() == "all":
        return "all"
    try:
        layer_idx = int(value)
    except ValueError as err:
        raise ValueError(
            f"{_PROTOTYPE_ENV} must be a non-negative integer or 'all', got {value!r}"
        ) from err
    if layer_idx < 0:
        raise ValueError(
            f"{_PROTOTYPE_ENV} must be a non-negative integer or 'all', got {layer_idx}"
        )
    return layer_idx


def _get_max_host_bytes() -> int:
    value = os.environ.get(_MAX_HOST_GIB_ENV, str(_DEFAULT_MAX_HOST_GIB))
    try:
        max_host_gib = float(value)
    except ValueError as err:
        raise ValueError(f"{_MAX_HOST_GIB_ENV} must be a positive number, got {value!r}") from err
    if max_host_gib <= 0:
        raise ValueError(f"{_MAX_HOST_GIB_ENV} must be positive, got {max_host_gib}")
    return int(max_host_gib * (1 << 30))


def _get_offloaded_layer_indices() -> Tuple[int, ...]:
    """Return zero-based shared-layer indices selected within each group."""
    value = os.environ.get(_GROUP_LAYERS_ENV, "2,3,4")
    try:
        group_layers = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as err:
        raise ValueError(
            f"{_GROUP_LAYERS_ENV} must be a comma-separated subset of 2,3,4, got {value!r}"
        ) from err
    if (
        not group_layers
        or len(set(group_layers)) != len(group_layers)
        or any(layer not in (2, 3, 4) for layer in group_layers)
    ):
        raise ValueError(
            f"{_GROUP_LAYERS_ENV} must be a comma-separated subset of 2,3,4, got {value!r}"
        )
    return tuple(layer - 2 for layer in sorted(group_layers))


def _shared_layers(full_layer: int) -> Tuple[int, int, int]:
    return (full_layer + 1, full_layer + 2, full_layer + 3)


def _select_full_layers(
    selector: Union[int, str],
    cache_manager: "DSACacheManager",
    sparse_attention_config,
    pretrained_config,
) -> Tuple[int, ...]:
    local_layers = tuple(sorted(cache_manager.layer_offsets))
    local_layer_set = set(local_layers)
    is_full = {
        layer_idx: bool(
            getattr(
                sparse_attention_config.to_sparse_params(
                    pretrained_config=pretrained_config,
                    layer_idx=layer_idx,
                ),
                "is_full_indexer_layer",
                True,
            )
        )
        for layer_idx in local_layers
    }

    def is_complete_group(full_layer: int) -> bool:
        shared_layers = _shared_layers(full_layer)
        return (
            full_layer in local_layer_set
            and all(layer_idx in local_layer_set for layer_idx in shared_layers)
            and is_full[full_layer]
            and all(not is_full[layer_idx] for layer_idx in shared_layers)
        )

    if selector == "all":
        full_layers = tuple(
            layer_idx for layer_idx in local_layers if is_complete_group(layer_idx)
        )
        if not full_layers:
            raise ValueError(f"{_PROTOTYPE_ENV}=all found no complete local IndexShare groups")
        return full_layers

    assert isinstance(selector, int)
    layers = (selector, *_shared_layers(selector))
    missing_layers = [layer_idx for layer_idx in layers if layer_idx not in local_layer_set]
    if missing_layers:
        raise ValueError(
            f"{_PROTOTYPE_ENV} group {layers} must be local to one pipeline stage; "
            f"missing layers: {missing_layers}"
        )
    if not is_full[selector]:
        raise ValueError(f"{_PROTOTYPE_ENV} layer {selector} is not a full-indexer layer")
    invalid_shared = [layer_idx for layer_idx in layers[1:] if is_full[layer_idx]]
    if invalid_shared:
        raise ValueError(
            f"{_PROTOTYPE_ENV} requires three following shared-indexer layers; "
            f"full layers found at {invalid_shared}"
        )
    return (selector,)


def configure_cache_manager(
    cache_manager: "DSACacheManager",
    sparse_attention_config,
    pretrained_config,
) -> None:
    """Allocate pinned host mirrors for the selected overhead experiment."""
    cache_manager.dsa_kv_offload_groups = {}
    cache_manager.dsa_kv_offload_shared_to_group = {}

    selector = _get_prototype_selector()
    if selector is None or cache_manager.is_estimating_kv_cache:
        return
    if pretrained_config is None:
        raise ValueError(f"{_PROTOTYPE_ENV} requires a pretrained model configuration")
    if cache_manager.spec_config is not None:
        raise ValueError(f"{_PROTOTYPE_ENV} does not support speculative decoding")
    if cache_manager.dtype != DataType.FP8:
        raise ValueError(f"{_PROTOTYPE_ENV} requires an FP8 latent KV cache")
    if cache_manager.head_dim != _LATENT_BYTES:
        raise ValueError(
            f"{_PROTOTYPE_ENV} requires {_LATENT_BYTES}-byte latent KV rows, "
            f"got head_dim={cache_manager.head_dim}"
        )

    full_layers = _select_full_layers(
        selector,
        cache_manager,
        sparse_attention_config,
        pretrained_config,
    )
    offloaded_layer_indices = _get_offloaded_layer_indices()
    host_rows = cache_manager.blocks_in_primary_pool * cache_manager.tokens_per_block
    host_bytes_per_group = host_rows * _HOST_ROW_BYTES
    total_host_bytes = len(full_layers) * host_bytes_per_group
    max_host_bytes = _get_max_host_bytes()
    if total_host_bytes > max_host_bytes:
        raise ValueError(
            f"{_PROTOTYPE_ENV} needs {total_host_bytes / (1 << 30):.2f} GiB of pinned "
            f"mirrors for {len(full_layers)} groups, exceeding "
            f"{_MAX_HOST_GIB_ENV}={max_host_bytes / (1 << 30):.2f}. Bound "
            "kv_cache_config.max_tokens for the fixed-batch experiment or raise the limit."
        )

    groups: Dict[int, _OffloadGroup] = {}
    shared_to_group: Dict[int, Tuple[int, int]] = {}
    try:
        for full_layer in full_layers:
            host_pool = torch.empty(
                (host_rows, _HOST_ROW_BYTES),
                dtype=torch.uint8,
                device="cpu",
                pin_memory=True,
            )
            shared_layers = _shared_layers(full_layer)
            groups[full_layer] = _OffloadGroup(
                shared_layers, offloaded_layer_indices, host_pool
            )
            for layer_in_group in offloaded_layer_indices:
                shared_layer = shared_layers[layer_in_group]
                shared_to_group[shared_layer] = (full_layer, layer_in_group)
    except RuntimeError as err:
        groups.clear()
        host_gib = total_host_bytes / (1 << 30)
        raise RuntimeError(
            f"Unable to allocate {host_gib:.2f} GiB of pinned host pools for {_PROTOTYPE_ENV}"
        ) from err

    cache_manager.dsa_kv_offload_groups = groups
    cache_manager.dsa_kv_offload_shared_to_group = shared_to_group
    logger.info(
        "[DSA KV offload prototype] %d group(s), full layers %s, "
        "group layers %s, token-major host mirrors %.2f GiB total "
        "(%.2f GiB/group, %d rows/group)",
        len(groups),
        tuple(groups),
        tuple(layer_in_group + 2 for layer_in_group in offloaded_layer_indices),
        total_host_bytes / (1 << 30),
        host_bytes_per_group / (1 << 30),
        host_rows,
    )


def create_metadata_buffers(
    metadata: "DSAtrtllmAttentionMetadata",
    capture_graph: bool,
) -> None:
    """Create fixed-address per-layer staging and synchronization objects."""
    metadata.dsa_kv_offload_staging = None
    metadata.dsa_kv_offload_streams = ()
    metadata.dsa_kv_offload_start_events = ()
    metadata.dsa_kv_offload_events = ()
    groups = getattr(metadata.kv_cache_manager, "dsa_kv_offload_groups", {})
    if not groups:
        return

    max_rows = (
        metadata.max_num_sequences * (1 + metadata.max_draft_tokens) * metadata.num_sparse_topk
    )
    metadata.dsa_kv_offload_staging = metadata.get_empty(
        metadata.cuda_graph_buffers,
        (_NUM_SHARED_LAYERS, max_rows, _LATENT_BYTES),
        cache_name="dsa_kv_offload_staging",
        dtype=torch.uint8,
        capture_graph=capture_graph,
    )
    metadata.dsa_kv_offload_streams = tuple(
        torch.cuda.Stream() for _ in range(_NUM_SHARED_LAYERS)
    )
    metadata.dsa_kv_offload_start_events = tuple(
        torch.cuda.Event() for _ in range(_NUM_SHARED_LAYERS)
    )
    metadata.dsa_kv_offload_events = tuple(
        torch.cuda.Event() for _ in range(_NUM_SHARED_LAYERS)
    )


def _launch_gather(
    backend: "DSATrtllmAttention",
    global_indices: torch.Tensor,
    metadata: "DSAtrtllmAttentionMetadata",
    group: _OffloadGroup,
    layer_in_group: int,
) -> None:
    """Gather one shared layer's selected rows on its auxiliary stream."""
    staging = metadata.dsa_kv_offload_staging
    streams = metadata.dsa_kv_offload_streams
    start_events = metadata.dsa_kv_offload_start_events
    events = metadata.dsa_kv_offload_events
    assert staging is not None
    assert len(streams) == _NUM_SHARED_LAYERS
    assert len(start_events) == _NUM_SHARED_LAYERS
    assert len(events) == _NUM_SHARED_LAYERS
    assert global_indices.numel() <= staging.shape[1]

    stream = streams[layer_in_group]
    start_event = start_events[layer_in_group]
    event = events[layer_in_group]
    start_event.record()
    with torch.cuda.stream(stream):
        stream.wait_event(start_event)
        torch.ops.trtllm.dsa_kv_cache_offload_gather(
            group.host_pool,
            global_indices,
            staging[layer_in_group, : global_indices.numel()],
            metadata._cached_stride_factor,
            metadata._cached_tokens_per_block,
            backend.get_local_layer_idx(metadata),
            layer_in_group,
        )
        event.record()


def advance_gather_pipeline(
    backend: "DSATrtllmAttention",
    global_indices: torch.Tensor,
    metadata: "DSAtrtllmAttentionMetadata",
) -> None:
    """Wait for this layer's rows, then prefetch the following shared layer."""
    cache_manager = metadata.kv_cache_manager
    groups = getattr(cache_manager, "dsa_kv_offload_groups", {})
    group = groups.get(backend.layer_idx)
    if group is not None:
        _launch_gather(
            backend,
            global_indices,
            metadata,
            group,
            layer_in_group=group.offloaded_layer_indices[0],
        )
        return

    group_info = getattr(cache_manager, "dsa_kv_offload_shared_to_group", {}).get(
        backend.layer_idx
    )
    if group_info is None:
        return

    full_layer, layer_in_group = group_info
    events = metadata.dsa_kv_offload_events
    assert len(events) == _NUM_SHARED_LAYERS
    torch.cuda.current_stream().wait_event(events[layer_in_group])

    group = groups[full_layer]
    selected_position = group.offloaded_layer_indices.index(layer_in_group)
    next_selected_position = selected_position + 1
    if next_selected_position < len(group.offloaded_layer_indices):
        next_layer_in_group = group.offloaded_layer_indices[next_selected_position]
        _launch_gather(
            backend,
            global_indices,
            metadata,
            group,
            next_layer_in_group,
        )


def mirror_appended_kv(
    backend: "DSATrtllmAttention",
    metadata: "AttentionMetadata",
    position_ids: Optional[torch.Tensor],
    is_generation: bool,
) -> None:
    """Dual-write newly appended shared-layer latent KV rows to its host pool."""
    cache_manager = metadata.kv_cache_manager
    group_info = getattr(cache_manager, "dsa_kv_offload_shared_to_group", {}).get(
        backend.layer_idx
    )
    if group_info is None:
        return
    if position_ids is None:
        raise ValueError(f"{_PROTOTYPE_ENV} requires position_ids")

    metadata._ensure_pool_view_cached()
    if is_generation:
        block_table = metadata._cached_block_table_gen
        req_idx = metadata._cached_req_idx_gen
    else:
        block_table = metadata._cached_block_table_ctx
        req_idx = metadata._cached_req_idx_ctx
    token_indices = position_ids.reshape(-1, 1).to(dtype=torch.int32)
    if token_indices.shape[0] != req_idx.shape[0]:
        raise ValueError(
            "DSA KV offload prototype position/request row mismatch: "
            f"{token_indices.shape[0]} != {req_idx.shape[0]}"
        )

    full_layer, layer_in_group = group_info
    group = cache_manager.dsa_kv_offload_groups[full_layer]
    local_layer = backend.get_local_layer_idx(metadata)
    global_indices = torch.ops.trtllm.convert_req_index_to_global(
        req_idx,
        block_table,
        token_indices,
        metadata._cached_tokens_per_block,
        1,
        metadata._cached_stride_factor,
        local_layer,
    )
    torch.ops.trtllm.dsa_kv_cache_offload_mirror(
        metadata._cached_pool_view,
        global_indices,
        group.host_pool,
        metadata._cached_stride_factor,
        metadata._cached_tokens_per_block,
        local_layer,
        layer_in_group,
    )
