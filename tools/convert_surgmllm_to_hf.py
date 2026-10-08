"""Strictly convert one MMEngine checkpoint into a reloadable HF model."""

from __future__ import annotations

import argparse
import atexit
import gc
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import torch
from mmengine.config import Config
from transformers import AutoModel
from xtuner.registry import BUILDER

from projects.surgmllm.hf import SurgMLLMConfig, SurgMLLMForConditionalGeneration


TRAINABLE_GROUPS = {
    "lora": lambda key: "lora_" in key,
    "structure_tokens": lambda key: "trainable_rows" in key,
    "projector": lambda key: key.startswith("seg_projector."),
    "temporal_fusion": lambda key: key.startswith("temporal_fusion."),
    "sam2_mask_decoder": lambda key: ".sam_mask_decoder." in key,
}

REMOTE_CODE_FILES = (
    "configuration_surgmllm.py",
    "modeling_surgmllm.py",
    "configuration_intern_vit.py",
    "configuration_internvl_chat.py",
    "modeling_intern_vit.py",
    "modeling_internvl_chat.py",
    "conversation.py",
    "sam2_runtime.py",
)
LEGAL_FILES = ("LICENSE", "LICENSE_InternVL", "NOTICE", "THIRD_PARTY.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("checkpoint")
    parser.add_argument("--save-path", required=True)
    return parser.parse_args()


def _tensor_digest(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(value).hexdigest()


def _copy_remote_bundle(destination: Path) -> None:
    repository_root = Path(__file__).resolve().parents[1]
    code_root = repository_root / "projects" / "surgmllm" / "hf"
    for name in REMOTE_CODE_FILES:
        shutil.copy2(code_root / name, destination / name)
    for name in LEGAL_FILES:
        shutil.copy2(repository_root / name, destination / name)


def _portable_source_name(path: Path) -> str:
    """Return a portable name for an input artifact."""

    return path.name


def _audit_checkpoint(
    model,
    state_dict: dict[str, torch.Tensor],
    reference: dict[str, torch.Tensor] | None = None,
) -> dict:
    reference = model.state_dict() if reference is None else reference
    expected = model.expected_trainable_state_keys()
    missing = [key for key in expected if key not in state_dict]
    unexpected = sorted(set(state_dict) - set(reference))
    shape_mismatch = [
        key
        for key, value in state_dict.items()
        if key in reference and tuple(value.shape) != tuple(reference[key].shape)
    ]
    if missing or unexpected or shape_mismatch:
        raise RuntimeError(
            "Checkpoint audit failed; "
            f"missing_trainable={missing}, unexpected={unexpected}, "
            f"shape_mismatch={shape_mismatch}"
        )
    groups = {
        name: sorted(key for key in expected if predicate(key))
        for name, predicate in TRAINABLE_GROUPS.items()
    }
    empty_groups = [name for name, keys in groups.items() if not keys]
    if empty_groups:
        raise RuntimeError(f"Checkpoint has no expected trainable keys for: {empty_groups}")
    classified = {key for keys in groups.values() for key in keys}
    unclassified = sorted(set(expected) - classified)
    if unclassified:
        raise RuntimeError(
            "Model exposes parameters outside the configured trainable-module set: "
            f"{unclassified}"
        )
    return {
        "trainable_key_count": len(expected),
        "checkpoint_key_count": len(state_dict),
        "reference_base_key_count": len(reference),
        "frozen_key_count_from_reference_base": len(set(reference) - set(expected)),
        "ignored_checkpoint_frozen_key_count": len(set(state_dict) - set(expected)),
        "groups": groups,
    }


def _resolve_export_state(model, state_dict, reference):
    """Overlay trainable checkpoint state while preserving reference frozen keys."""

    audit = _audit_checkpoint(model, state_dict, reference)
    expected = set(model.expected_trainable_state_keys())
    resolved = dict(reference)
    resolved.update({key: value for key, value in state_dict.items() if key in expected})
    return audit, resolved


def _export_digests(model) -> dict[str, str]:
    state = model.state_dict()
    selected = {
        key: value
        for key, value in state.items()
        if key.startswith("seg_projector.")
        or key.startswith("temporal_fusion.")
        or ".sam_mask_decoder." in key
    }
    input_rows = model.language_model.get_input_embeddings().weight[
        model.config.structure_token_ids
    ]
    output_rows = model.language_model.get_output_embeddings().weight[
        model.config.structure_token_ids
    ]
    selected["__structure_input_rows__"] = input_rows
    selected["__structure_output_rows__"] = output_rows
    return {key: _tensor_digest(value) for key, value in selected.items()}


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    final_save_path = Path(args.save_path).resolve()
    if not config_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing config/checkpoint: {config_path}, {checkpoint_path}")
    if final_save_path.exists():
        raise FileExistsError(f"Refusing to overwrite an existing HF path: {final_save_path}")
    final_save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path = Path(
        tempfile.mkdtemp(
            prefix=f".{final_save_path.name}.tmp-", dir=final_save_path.parent
        )
    )

    def cleanup_staging() -> None:
        shutil.rmtree(save_path, ignore_errors=True)

    atexit.register(cleanup_staging)

    cfg = Config.fromfile(config_path)
    training_model = BUILDER.build(cfg.model)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict") if isinstance(checkpoint, dict) else None
    if not isinstance(state_dict, dict):
        raise TypeError("Expected an MMEngine checkpoint containing state_dict")
    # ZeRO-2/MMEngine may intentionally omit frozen tensors. Reconstruct those
    # only from the freshly loaded reference base, overlay only the validated
    # trainable checkpoint keys, and then perform one strict full-graph load.
    # No trainable key may fall back to the base model, and no frozen key may
    # override the reference base.
    reference_state = training_model.state_dict()
    audit, resolved_state = _resolve_export_state(
        training_model, state_dict, reference_state
    )
    incompatible = training_model.load_state_dict(resolved_state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Strict checkpoint load failed: {incompatible}")
    del resolved_state, reference_state
    training_model.merge_adapters_for_export()

    base_config = training_model.mllm.model.config.to_dict()
    base_config["architectures"] = ["SurgMLLMForConditionalGeneration"]
    base_config["auto_map"] = {
        "AutoConfig": "configuration_surgmllm.SurgMLLMConfig",
        "AutoModel": "modeling_surgmllm.SurgMLLMForConditionalGeneration",
        "AutoModelForCausalLM": "modeling_surgmllm.SurgMLLMForConditionalGeneration",
    }
    base_config["llm_config"]["vocab_size"] = len(training_model.tokenizer)
    base_config["dynamic_image_size"] = True
    base_config["min_dynamic_patch"] = 1
    base_config["max_dynamic_patch"] = 5
    base_config["use_thumbnail"] = True
    base_config["window_size"] = 5
    base_config["image_tokens_per_tile"] = 256
    base_config["sam2_image_size"] = 1024
    base_config["mask_threshold"] = 0.0
    base_config["structure_tokens"] = list(training_model.tokenizer.additional_special_tokens)
    base_config["structure_token_ids"] = [
        training_model.tokenizer.convert_tokens_to_ids(token)
        for token in training_model.tokenizer.additional_special_tokens
        if token in {"[SEG]", "<p>", "</p>", "<think>", "</think>", "<answer>", "</answer>"}
    ]
    required_order = ["[SEG]", "<p>", "</p>", "<think>", "</think>", "<answer>", "</answer>"]
    base_config["structure_tokens"] = required_order
    base_config["structure_token_ids"] = [
        training_model.tokenizer.convert_tokens_to_ids(token) for token in required_order
    ]
    hf_config = SurgMLLMConfig(**base_config)
    hf_config.structure_token_ids = base_config["structure_token_ids"]

    hf_model = SurgMLLMForConditionalGeneration(
        hf_config,
        vision_model=training_model.mllm.model.vision_model,
        language_model=training_model.mllm.model.language_model,
    )
    merged_state = training_model.state_dict()
    hf_state = {}
    for key, value in merged_state.items():
        mapped = key[len("mllm.model.") :] if key.startswith("mllm.model.") else key
        hf_state[mapped] = value
    incompatible = hf_model.load_state_dict(hf_state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Strict HF graph load failed: {incompatible}")

    hf_model.save_pretrained(save_path, safe_serialization=True, max_shard_size="4GB")
    training_model.tokenizer.save_pretrained(save_path)
    _copy_remote_bundle(save_path)
    before_reload = _export_digests(hf_model)

    manifest = {
        "format": "surgmllm-hf-v1",
        "source_config": _portable_source_name(config_path),
        "source_checkpoint": _portable_source_name(checkpoint_path),
        "checkpoint_audit": audit,
        "reload_digests": before_reload,
    }
    (save_path / "conversion_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    del training_model, hf_model, merged_state, hf_state, state_dict, checkpoint
    gc.collect()
    reloaded, loading_info = AutoModel.from_pretrained(
        save_path,
        trust_remote_code=True,
        force_download=True,
        low_cpu_mem_usage=True,
        output_loading_info=True,
    )
    if loading_info.get("missing_keys") or loading_info.get("unexpected_keys"):
        raise RuntimeError(f"HF reload was not strict: {loading_info}")
    after_reload = _export_digests(reloaded)
    if before_reload != after_reload:
        changed = sorted(key for key in before_reload if before_reload[key] != after_reload.get(key))
        raise RuntimeError(f"HF reload changed trainable-module tensors: {changed}")
    del reloaded
    gc.collect()
    os.replace(save_path, final_save_path)
    atexit.unregister(cleanup_staging)
    summary = {
        key: value for key, value in audit.items() if key != "groups"
    }
    summary["group_key_counts"] = {
        name: len(keys) for name, keys in audit["groups"].items()
    }
    print(
        json.dumps(
            {"save_path": str(final_save_path), "checkpoint_audit": summary},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
