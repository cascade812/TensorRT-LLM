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

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from safetensors.torch import save_file

from tensorrt_llm.lora_helper import LoraConfig, get_enc_dec_trtllm_modules_to_hf_modules
from tensorrt_llm.lora_manager import (
    LoraManager,
    LoraModelConfig,
    get_all_hf_lora_weights_enc_dec,
    invert_module_mapping,
    load_torch_lora,
)
from tensorrt_llm.mapping import Mapping


def _projection_weights(prefix: str, output_size: int) -> dict[str, torch.Tensor]:
    return {
        f"{prefix}.lora_A.weight": torch.arange(16, dtype=torch.float32).reshape(2, 8),
        f"{prefix}.lora_B.weight": torch.arange(output_size * 2, dtype=torch.float32).reshape(
            output_size, 2
        ),
    }


def _adapter_weights(model_type: str) -> dict[str, torch.Tensor]:
    if model_type == "t5":
        prefixes = {
            "encoder_q": "base_model.model.encoder.block.0.layer.0.SelfAttention.q",
            "decoder_q": "base_model.model.decoder.block.0.layer.0.SelfAttention.q",
            "cross_q": "base_model.model.decoder.block.0.layer.1.EncDecAttention.q",
            "mlp": "base_model.model.decoder.block.0.layer.2.DenseReluDense.wi_0",
        }
    else:
        prefixes = {
            "encoder_q": "base_model.model.model.encoder.layers.0.self_attn.q_proj",
            "decoder_q": "base_model.model.model.decoder.layers.0.self_attn.q_proj",
            "cross_q": "base_model.model.model.decoder.layers.0.encoder_attn.q_proj",
            "mlp": "base_model.model.model.decoder.layers.0.fc1",
        }

    weights = {}
    for name, prefix in prefixes.items():
        weights.update(_projection_weights(prefix, 16 if name == "mlp" else 8))
    return weights


def _write_adapter(tmp_path, model_type: str, *, modules_to_save=None):
    adapter_dir = tmp_path / f"{model_type}_adapter"
    adapter_dir.mkdir()
    adapter_config = {
        "r": 2,
        "lora_alpha": 2,
        "target_modules": ["q_proj", "fc1"],
        "peft_type": "LORA",
    }
    if modules_to_save is not None:
        adapter_config["modules_to_save"] = modules_to_save
    (adapter_dir / "adapter_config.json").write_text(json.dumps(adapter_config))
    save_file(
        _adapter_weights(model_type),
        str(adapter_dir / "adapter_model.safetensors"),
    )
    return adapter_dir


def _pretrained_config(model_type: str):
    return SimpleNamespace(
        model_type=model_type,
        is_encoder_decoder=True,
        hidden_size=8,
        num_layers=2,
        num_decoder_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        d_kv=4,
    )


def test_t5_parser_uses_block_index_and_offsets_decoder():
    mapping = get_enc_dec_trtllm_modules_to_hf_modules("t5")
    hf_modules = set(invert_module_mapping(mapping))
    weights = {
        "base_model.model.encoder.block.1.layer.0.SelfAttention.q.lora_A.weight": torch.ones(2, 8),
        "base_model.model.decoder.block.1.layer.2.DenseReluDense.wi_0.lora_B.weight": torch.ones(
            16, 2
        ),
    }

    parsed = get_all_hf_lora_weights_enc_dec(
        weights,
        hf_modules,
        model_type="t5",
        num_encoder_layers=3,
        num_decoder_layers=2,
    )

    assert "SelfAttention.q" in parsed[1]
    assert "DenseReluDense.wi" in parsed[4]
    assert 2 not in parsed


@pytest.mark.parametrize("model_type", ["bart", "whisper"])
def test_bart_family_parser_preserves_qualified_attention_names(model_type):
    mapping = get_enc_dec_trtllm_modules_to_hf_modules(model_type)
    hf_modules = set(invert_module_mapping(mapping))
    weights = {
        "base_model.model.model.encoder.layers.1.self_attn.q_proj.lora_A.weight": torch.ones(2, 8),
        "base_model.model.model.decoder.layers.1.encoder_attn.q_proj.lora_B.weight": torch.ones(
            8, 2
        ),
    }

    parsed = get_all_hf_lora_weights_enc_dec(
        weights,
        hf_modules,
        model_type=model_type,
        num_encoder_layers=3,
        num_decoder_layers=2,
    )

    assert "self_attn.q_proj" in parsed[1]
    assert "encoder_attn.q_proj" in parsed[4]


def test_encoder_decoder_parser_rejects_unmapped_weights():
    mapping = get_enc_dec_trtllm_modules_to_hf_modules("bart")
    hf_modules = set(invert_module_mapping(mapping))
    weights = {"base_model.model.model.decoder.layers.0.layer_norm.lora_A.weight": torch.ones(2, 8)}

    with pytest.raises(KeyError, match="Every adapter tensor must map"):
        get_all_hf_lora_weights_enc_dec(
            weights,
            hf_modules,
            model_type="bart",
            num_encoder_layers=2,
            num_decoder_layers=2,
        )


def test_encoder_decoder_parser_rejects_out_of_range_layer():
    mapping = get_enc_dec_trtllm_modules_to_hf_modules("whisper")
    hf_modules = set(invert_module_mapping(mapping))
    weights = {
        "base_model.model.model.decoder.layers.2.self_attn.q_proj.lora_A.weight": torch.ones(2, 8)
    }

    with pytest.raises(ValueError, match="model has 2 decoder layers"):
        get_all_hf_lora_weights_enc_dec(
            weights,
            hf_modules,
            model_type="whisper",
            num_encoder_layers=2,
            num_decoder_layers=2,
        )


@pytest.mark.parametrize("model_type", ["t5", "bart", "whisper"])
def test_lora_manager_packs_encoder_and_decoder_global_layer_ids(tmp_path, monkeypatch, model_type):
    adapter_dir = _write_adapter(tmp_path, model_type)
    pretrained_config = _pretrained_config(model_type)
    lora_config = LoraConfig(lora_dir=[str(adapter_dir)])
    load_torch_lora(lora_config, pretrained_config)

    assert "attn_q" in lora_config.lora_target_modules
    assert "cross_attn_q" in lora_config.lora_target_modules
    assert "mlp_h_to_4h" in lora_config.lora_target_modules

    model_config = LoraModelConfig.from_pretrained_config(
        lora_target_modules=lora_config.lora_target_modules,
        trtllm_modules_to_hf_modules=(lora_config.trtllm_modules_to_hf_modules),
        hidden_size=8,
        dtype="float32",
        swap_gate_up_proj_lora_b_weight=True,
        pretrained_config=pretrained_config,
    )
    manager = LoraManager(
        mapping=Mapping(world_size=1, rank=0, tp_size=1),
        model_config=SimpleNamespace(head_size=4, num_heads=2, num_kv_heads=1),
        cpp_peft_cache_manager=MagicMock(),
    )

    zero_shapes = []
    torch_zeros = torch.zeros

    def record_zeros(*size, **kwargs):
        shape = size[0] if len(size) == 1 and isinstance(size[0], tuple) else size
        zero_shapes.append(tuple(shape))
        return torch_zeros(*size, **kwargs)

    monkeypatch.setattr(torch, "zeros", record_zeros)
    monkeypatch.setattr(torch.Tensor, "cuda", lambda tensor, *args, **kwargs: tensor)
    manager.load_from_hf([str(adapter_dir)], model_config=model_config, uids=["adapter"])

    packed_keys = {
        (int(module_id), int(layer_id))
        for module_id, layer_id, _, _ in manager.cpp_lora_config["adapter"]
    }
    assert (LoraManager.LORA_MODULE_IDS["attn_q"], 0) in packed_keys
    assert (LoraManager.LORA_MODULE_IDS["attn_q"], 2) in packed_keys
    assert (LoraManager.LORA_MODULE_IDS["cross_attn_q"], 2) in packed_keys
    assert (LoraManager.LORA_MODULE_IDS["mlp_h_to_4h"], 2) in packed_keys
    assert not any(
        module_id == LoraManager.LORA_MODULE_IDS["cross_attn_q"] and layer_id < 2
        for module_id, layer_id in packed_keys
    )

    # Missing K/V projections use KV-head geometry, not hidden_size, for B.
    assert (4, 2) in zero_shapes
    assert model_config.attention_output_size("attn_q") == 8
    assert model_config.attention_output_size("attn_k") == 4


def test_encoder_decoder_loader_rejects_modules_to_save(tmp_path):
    adapter_dir = _write_adapter(tmp_path, "t5", modules_to_save=["lm_head"])
    lora_config = LoraConfig(lora_dir=[str(adapter_dir)])

    with pytest.raises(ValueError, match="does not support modules_to_save"):
        load_torch_lora(lora_config, _pretrained_config("t5"))
