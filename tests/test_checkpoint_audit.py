from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import pytest
import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONVERTER_PATH = PROJECT_ROOT / "tools" / "convert_surgmllm_to_hf.py"
HF_ROOT = PROJECT_ROOT / "projects" / "surgmllm" / "hf"


@pytest.fixture(scope="module")
def converter_module():
    """Load audit helpers without importing the heavyweight SAM2 remote graph."""
    hf_stub = ModuleType("projects.surgmllm.hf")
    hf_stub.SurgMLLMConfig = type("SurgMLLMConfig", (), {})
    hf_stub.SurgMLLMForConditionalGeneration = type(
        "SurgMLLMForConditionalGeneration", (), {}
    )
    module_name = "_surgmllm_converter_checkpoint_test"
    spec = importlib.util.spec_from_file_location(module_name, CONVERTER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"projects.surgmllm.hf": hf_stub}):
        spec.loader.exec_module(module)
    return module


class _AuditModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mllm = nn.Module()
        self.mllm.register_parameter("lora_A", nn.Parameter(torch.zeros(2, 3)))
        self.mllm.structure = nn.Module()
        self.mllm.structure.register_parameter(
            "trainable_rows", nn.Parameter(torch.zeros(7, 3))
        )
        self.seg_projector = nn.Module()
        self.seg_projector.register_parameter("weight", nn.Parameter(torch.zeros(3, 3)))
        self.temporal_fusion = nn.Module()
        self.temporal_fusion.register_parameter("weight", nn.Parameter(torch.zeros(3, 3)))
        self.grounding_encoder = nn.Module()
        self.grounding_encoder.sam_mask_decoder = nn.Module()
        self.grounding_encoder.sam_mask_decoder.register_parameter(
            "weight", nn.Parameter(torch.zeros(1, 3))
        )
        self.register_parameter(
            "frozen_backbone", nn.Parameter(torch.zeros(4), requires_grad=False)
        )

    def expected_trainable_state_keys(self) -> list[str]:
        return sorted(name for name, value in self.named_parameters() if value.requires_grad)


EXPECTED_GROUPS = {
    "lora": ["mllm.lora_A"],
    "structure_tokens": ["mllm.structure.trainable_rows"],
    "projector": ["seg_projector.weight"],
    "temporal_fusion": ["temporal_fusion.weight"],
    "sam2_mask_decoder": ["grounding_encoder.sam_mask_decoder.weight"],
}


def test_checkpoint_audit_covers_every_trainable_group(converter_module):
    model = _AuditModel()

    audit = converter_module._audit_checkpoint(model, model.state_dict())

    assert audit["trainable_key_count"] == 5
    assert audit["checkpoint_key_count"] == 6
    assert audit["reference_base_key_count"] == 6
    assert audit["frozen_key_count_from_reference_base"] == 1
    assert audit["ignored_checkpoint_frozen_key_count"] == 1
    assert audit["groups"] == EXPECTED_GROUPS


def test_export_state_never_accepts_frozen_checkpoint_overrides(converter_module):
    model = _AuditModel()
    reference = {key: value.detach().clone() for key, value in model.state_dict().items()}
    checkpoint = {key: value.detach().clone() for key, value in reference.items()}
    checkpoint["frozen_backbone"] = torch.full_like(
        checkpoint["frozen_backbone"], 99
    )
    checkpoint["mllm.lora_A"] = torch.full_like(checkpoint["mllm.lora_A"], 7)

    audit, resolved = converter_module._resolve_export_state(
        model, checkpoint, reference
    )

    assert torch.equal(resolved["frozen_backbone"], reference["frozen_backbone"])
    assert torch.equal(resolved["mllm.lora_A"], checkpoint["mllm.lora_A"])
    assert audit["ignored_checkpoint_frozen_key_count"] == 1


@pytest.mark.parametrize("missing_key", [key for keys in EXPECTED_GROUPS.values() for key in keys])
def test_checkpoint_audit_fails_for_each_missing_trainable_key(
    converter_module, missing_key
):
    model = _AuditModel()
    state = dict(model.state_dict())
    del state[missing_key]

    with pytest.raises(
        RuntimeError,
        match="missing_trainable=.*" + missing_key.replace(".", r"\."),
    ):
        converter_module._audit_checkpoint(model, state)


@pytest.mark.parametrize("bad_key", [key for keys in EXPECTED_GROUPS.values() for key in keys])
def test_checkpoint_audit_fails_for_each_trainable_shape_mismatch(converter_module, bad_key):
    model = _AuditModel()
    state = dict(model.state_dict())
    original = state[bad_key]
    state[bad_key] = torch.zeros(original.shape[0] + 1, *original.shape[1:])

    with pytest.raises(
        RuntimeError, match="shape_mismatch=.*" + bad_key.replace(".", r"\.")
    ):
        converter_module._audit_checkpoint(model, state)


@pytest.mark.parametrize(
    ("group", "parameter_name"),
    [(group, keys[0]) for group, keys in EXPECTED_GROUPS.items()],
)
def test_checkpoint_audit_rejects_a_model_that_omits_a_required_group(
    converter_module, group, parameter_name
):
    model = _AuditModel()
    model.get_parameter(parameter_name).requires_grad_(False)

    with pytest.raises(RuntimeError, match=r"no expected trainable keys.*" + group):
        converter_module._audit_checkpoint(model, model.state_dict())


def test_converter_never_uses_non_strict_state_loading():
    tree = ast.parse(CONVERTER_PATH.read_text(encoding="utf-8"))
    load_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "load_state_dict"
    ]

    assert len(load_calls) >= 2
    for call in load_calls:
        strict_values = [keyword.value for keyword in call.keywords if keyword.arg == "strict"]
        assert len(strict_values) == 1
        assert isinstance(strict_values[0], ast.Constant)
        assert strict_values[0].value is True


def test_hf_remote_bundle_contains_legal_closure(converter_module, tmp_path):
    converter_module._copy_remote_bundle(tmp_path)

    expected = set(converter_module.REMOTE_CODE_FILES) | set(
        converter_module.LEGAL_FILES
    )
    assert {path.name for path in tmp_path.iterdir()} == expected
    for name in converter_module.LEGAL_FILES:
        assert (tmp_path / name).read_bytes() == (PROJECT_ROOT / name).read_bytes()


def test_conversion_manifest_source_names_are_portable(converter_module):
    assert converter_module._portable_source_name(
        Path("/external/root/configs/fold1.py")
    ) == "fold1.py"
    assert converter_module._portable_source_name(
        Path("/external/root/checkpoints/epoch_5.pth")
    ) == "epoch_5.pth"


def _base_config_assignments(tree: ast.AST) -> dict[str, object]:
    assignments: dict[str, object] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not (
            isinstance(target, ast.Subscript)
            and isinstance(target.value, ast.Name)
            and target.value.id == "base_config"
            and isinstance(target.slice, ast.Constant)
            and isinstance(target.slice.value, str)
        ):
            continue
        try:
            assignments[target.slice.value] = ast.literal_eval(node.value)
        except (ValueError, TypeError):
            continue
    return assignments


def test_hf_config_and_remote_entrypoints_use_only_surgmllm_identity():
    converter_tree = ast.parse(CONVERTER_PATH.read_text(encoding="utf-8"))
    assignments = _base_config_assignments(converter_tree)
    assert assignments["architectures"] == ["SurgMLLMForConditionalGeneration"]
    assert assignments["auto_map"] == {
        "AutoConfig": "configuration_surgmllm.SurgMLLMConfig",
        "AutoModel": "modeling_surgmllm.SurgMLLMForConditionalGeneration",
        "AutoModelForCausalLM": "modeling_surgmllm.SurgMLLMForConditionalGeneration",
    }

    config_source = (HF_ROOT / "configuration_surgmllm.py").read_text(encoding="utf-8")
    modeling_source = (HF_ROOT / "modeling_surgmllm.py").read_text(encoding="utf-8")
    init_source = (HF_ROOT / "__init__.py").read_text(encoding="utf-8")
    config_tree = ast.parse(config_source)
    modeling_tree = ast.parse(modeling_source)
    config_classes = {node.name: node for node in config_tree.body if isinstance(node, ast.ClassDef)}
    modeling_classes = {
        node.name: node for node in modeling_tree.body if isinstance(node, ast.ClassDef)
    }

    assert set(config_classes) == {"SurgMLLMConfig"}
    assert "SurgMLLMForConditionalGeneration" in modeling_classes
    model_type = [
        node.value.value
        for node in config_classes["SurgMLLMConfig"].body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "model_type" for target in node.targets)
        and isinstance(node.value, ast.Constant)
    ]
    assert model_type == ["surgmllm"]
    assert "SurgMLLMConfig" in init_source
    assert "SurgMLLMForConditionalGeneration" in init_source
