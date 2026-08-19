# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import MethodType, SimpleNamespace

import pytest
import torch
from torch import nn

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modules import attention as attention_module
from tensorrt_llm._torch.modules.attention import Attention
from tensorrt_llm._torch.modules.gated_mlp import GatedMLP
from tensorrt_llm._torch.modules.mlp import MLP

pytestmark = pytest.mark.cpu_only


class _FakeAttentionBackend:
    def __init__(self, layer_idx: int) -> None:
        self.layer_idx = layer_idx

    @staticmethod
    def support_fused_rope() -> bool:
        return False

    @staticmethod
    def support_fused_qkv() -> bool:
        return False


class _RecordingLora(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer_indices: list[int] = []

    def forward(self, x: torch.Tensor, lora_params: dict, layer_idx: int) -> None:
        self.layer_indices.append(layer_idx)
        return None


class _RecordingProjection(nn.Module):
    def __init__(self, out_features: int) -> None:
        super().__init__()
        self.out_features = out_features
        self.has_fp8_qdq = False
        self.has_w4a8_nvfp4_fp8 = False
        self.kwargs: list[dict] = []

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        self.kwargs.append(kwargs)
        return torch.zeros(*x.shape[:-1], self.out_features, dtype=x.dtype)


def _skip_weight_creation_config() -> ModelConfig:
    return ModelConfig(skip_create_weights_in_init=True)


def _make_attention(monkeypatch: pytest.MonkeyPatch, **kwargs) -> Attention:
    monkeypatch.setattr(
        attention_module,
        "get_attention_backend",
        lambda *args, **kwargs: _FakeAttentionBackend,
    )
    monkeypatch.setattr(
        attention_module,
        "create_attention",
        lambda _, layer_idx, *args, **kwargs: _FakeAttentionBackend(layer_idx),
    )
    return Attention(
        hidden_size=4,
        num_attention_heads=1,
        num_key_value_heads=1,
        max_position_embeddings=16,
        bias=False,
        rope_fusion=False,
        dtype=torch.float16,
        config=_skip_weight_creation_config(),
        **kwargs,
    )


def test_lora_layer_idx_defaults_to_layer_idx(monkeypatch: pytest.MonkeyPatch):
    attention = _make_attention(monkeypatch, layer_idx=3)
    mlp = MLP(
        hidden_size=4,
        intermediate_size=8,
        bias=False,
        layer_idx=3,
        config=_skip_weight_creation_config(),
    )
    gated_mlp = GatedMLP(
        hidden_size=4,
        intermediate_size=8,
        bias=False,
        layer_idx=3,
        config=_skip_weight_creation_config(),
    )

    assert attention.lora_layer_idx == attention.layer_idx == 3
    assert mlp.lora_layer_idx == mlp.layer_idx == 3
    assert gated_mlp.lora_layer_idx == gated_mlp.layer_idx == 3


def test_attention_uses_lora_layer_idx_without_changing_backend_layer_idx(
    monkeypatch: pytest.MonkeyPatch,
):
    attention = _make_attention(monkeypatch, layer_idx=2, lora_layer_idx=7)
    attention.qkv_proj = _RecordingProjection(out_features=12)
    attention.splitted_qkv_lora = _RecordingLora()
    attention.fused_qkv_lora = _RecordingLora()
    attention.o_proj = _RecordingProjection(out_features=4)
    attention.forward_impl = MethodType(
        lambda self, *args, **kwargs: torch.zeros(2, 4, dtype=torch.float16),
        attention,
    )

    attention(
        position_ids=None,
        hidden_states=torch.zeros(2, 4, dtype=torch.float16),
        attn_metadata=SimpleNamespace(),
        lora_params={"active": True},
    )

    assert attention.layer_idx == 2
    assert attention.attn.layer_idx == 2
    assert attention.splitted_qkv_lora.layer_indices == [7]
    assert attention.fused_qkv_lora.layer_indices == [7]
    assert attention.o_proj.kwargs[-1]["layer_idx"] == 7


def test_mlp_uses_lora_layer_idx():
    mlp = MLP(
        hidden_size=4,
        intermediate_size=8,
        bias=False,
        activation=lambda x: x,
        layer_idx=2,
        lora_layer_idx=7,
        config=_skip_weight_creation_config(),
    )
    mlp.up_proj = _RecordingProjection(out_features=8)
    mlp.up_lora = _RecordingLora()
    mlp.down_proj = _RecordingProjection(out_features=4)

    mlp(torch.zeros(2, 4), lora_params={"active": True})

    assert mlp.layer_idx == 2
    assert mlp.up_lora.layer_indices == [7]
    assert mlp.down_proj.kwargs[-1]["layer_idx"] == 7


def test_gated_mlp_uses_lora_layer_idx():
    gated_mlp = GatedMLP(
        hidden_size=4,
        intermediate_size=8,
        bias=False,
        activation=lambda x: x[..., : x.shape[-1] // 2],
        layer_idx=2,
        lora_layer_idx=7,
        config=_skip_weight_creation_config(),
    )
    gated_mlp.gate_up_proj = _RecordingProjection(out_features=16)
    gated_mlp.splitted_gate_up_lora = _RecordingLora()
    gated_mlp.fused_gate_up_lora = _RecordingLora()
    gated_mlp.down_proj = _RecordingProjection(out_features=4)

    gated_mlp(torch.zeros(2, 4), lora_params={"active": True})

    assert gated_mlp.layer_idx == 2
    assert gated_mlp.splitted_gate_up_lora.layer_indices == [7]
    assert gated_mlp.fused_gate_up_lora.layer_indices == [7]
    assert gated_mlp.down_proj.kwargs[-1]["layer_idx"] == 7
