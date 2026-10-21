# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3 sparse KV cache offload configuration, without a GPU."""

from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.cache_manager import (
    local_sparse_kv_heads,
    offload_selection_width,
    sparse_layer_ids_from_config,
)
from tensorrt_llm.llmapi.llm_args import (
    DeepSeekSparseAttentionConfig,
    MiniMaxM3SparseAttentionConfig,
    SparseKvCacheOffloadConfig,
)

pytestmark = pytest.mark.cpu_only


def _offload(**kwargs) -> MiniMaxM3SparseAttentionConfig:
    return MiniMaxM3SparseAttentionConfig(
        implementation="msa",
        enable_kv_cache_offload=True,
        kv_cache_offload_config={"gpu_cache_factor": 1.5, "host_history_size": 1 << 30},
        **kwargs,
    )


def test_minimax_m3_offload_lowers_into_sparse_params():
    config = _offload()
    assert config.get_kv_cache_offload_config() == SparseKvCacheOffloadConfig(
        gpu_cache_factor=1.5, host_history_size=1 << 30
    )
    assert config.to_sparse_params().enable_kv_cache_offload
    plain = MiniMaxM3SparseAttentionConfig(implementation="msa")
    assert plain.get_kv_cache_offload_config() is None
    assert not plain.to_sparse_params().enable_kv_cache_offload


def test_minimax_m3_offload_sizing_requires_offload():
    with pytest.raises(ValueError, match="requires enable_kv_cache_offload"):
        MiniMaxM3SparseAttentionConfig(kv_cache_offload_config={"gpu_cache_factor": 3.0})


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"implementation": "triton"}, "'msa' implementation"),
        (
            {"indexer_kv_dtype": "fp8", "fuse_qkv_index_projection": True},
            "fuse_qkv_index_projection",
        ),
        ({"sparse_disable_index_value": False}, "sparse_disable_index_value"),
        ({"sparse_topk_blocks": 1, "sparse_local_blocks": 1}, "exceed sparse_local_blocks"),
    ],
)
def test_minimax_m3_offload_refuses_paths_without_host_history(kwargs, message):
    options = {"implementation": "msa", **kwargs}
    with pytest.raises(ValueError, match=message):
        MiniMaxM3SparseAttentionConfig(enable_kv_cache_offload=True, **options)


def test_dsa_offload_fields_and_checks_are_unchanged():
    config = DeepSeekSparseAttentionConfig(
        index_topk=2048,
        enable_kv_cache_offload=True,
        kv_cache_offload_config={"gpu_cache_factor": 2.0},
    )
    assert config.get_kv_cache_offload_config().gpu_cache_factor == 2.0
    with pytest.raises(ValueError, match="positive index_topk"):
        DeepSeekSparseAttentionConfig(index_topk=0, enable_kv_cache_offload=True)
    with pytest.raises(ValueError, match="requires enable_kv_cache_offload"):
        DeepSeekSparseAttentionConfig(kv_cache_offload_config={"gpu_cache_factor": 3.0})
    # The checkpoint fills an unset top-k later.
    assert DeepSeekSparseAttentionConfig(enable_kv_cache_offload=True).index_topk is None


@pytest.mark.parametrize(
    ("tp_size", "attention_dp", "local_heads"),
    [(1, False, 4), (2, False, 2), (4, False, 1), (8, False, 1), (4, True, 4)],
)
def test_selection_is_the_union_of_the_local_heads_history_picks(
    tp_size, attention_dp, local_heads
):
    mapping = SimpleNamespace(tp_size=tp_size, enable_attention_dp=attention_dp)
    heads = local_sparse_kv_heads(4, mapping, [3, 4])
    assert heads == local_heads
    assert local_sparse_kv_heads([4] * 5, mapping, [3, 4]) == local_heads
    # Each head's 16 picks include the block being written.
    config = MiniMaxM3SparseAttentionConfig(implementation="msa")
    assert offload_selection_width(config, heads) == local_heads * 15


def test_sparse_layers_come_from_the_checkpoint_pattern():
    freq = [0, 0, 0, 1, 1, 0, 1]
    text = SimpleNamespace(sparse_attention_config={"sparse_attention_freq": freq})
    assert sparse_layer_ids_from_config(SimpleNamespace(text_config=text), 7) == [3, 4, 6]
    assert sparse_layer_ids_from_config(text, 7) == [3, 4, 6]
    # Layers past the target count, such as draft layers, are not sparse.
    assert sparse_layer_ids_from_config(text, 5) == [3, 4]
    disabled = SimpleNamespace(
        sparse_attention_config={"use_sparse_attention": False, "sparse_attention_freq": freq}
    )
    assert sparse_layer_ids_from_config(disabled, 7) == []
    assert sparse_layer_ids_from_config(SimpleNamespace(), 7) is None
    assert sparse_layer_ids_from_config(None, 7) is None
