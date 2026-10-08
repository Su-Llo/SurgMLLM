"""Unified structured reasoning and role-aware visual grounding model."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Iterable

import torch
import torch.nn.functional as F
from mmengine.model import BaseModel
from torch import nn
from xtuner.registry import BUILDER


STRUCTURE_TOKENS = (
    "[SEG]",
    "<p>",
    "</p>",
    "<think>",
    "</think>",
    "<answer>",
    "</answer>",
)
ENTITY_WEIGHTS = {"instrument": 5.0, "verb": 2.0, "target": 5.0, "phase": 5.0}


def normalize_role_key(key: str) -> str:
    if ":" not in key:
        raise ValueError(f"Role-aware key must contain ':': {key!r}")
    role, label = key.split(":", 1)
    role = role.strip().lower()
    if role not in {"instrument", "target"}:
        raise ValueError(f"Only instrument/target may own masks: {key!r}")
    label = re.sub(r"[^a-z0-9]+", "_", label.strip().lower()).strip("_")
    if not label:
        raise ValueError(f"Empty normalized entity label: {key!r}")
    return f"{role}:{label}"


class SurgMLLMModel(BaseModel):
    """InternVL reasoning with `[SEG]`-prompted SAM2 binary masks."""

    def __init__(
        self,
        mllm,
        tokenizer,
        grounding_encoder,
        special_tokens: Iterable[str] = STRUCTURE_TOKENS,
        bce_weight: float = 2.0,
        dice_weight: float = 0.5,
        entity_weight: float = 1.0,
        entity_token_weights: dict | None = None,
        torch_dtype=torch.bfloat16,
        data_preprocessor=None,
        init_cfg=None,
    ) -> None:
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        tokens = tuple(special_tokens)
        if tokens != STRUCTURE_TOKENS:
            raise ValueError(
                "Exactly the seven configured structure tokens are allowed: "
                f"{STRUCTURE_TOKENS}"
            )
        self.mllm = BUILDER.build(mllm) if isinstance(mllm, dict) else mllm
        tokenizer = BUILDER.build(tokenizer) if isinstance(tokenizer, dict) else tokenizer
        token_ids = self.mllm.add_structure_tokens(tokenizer, list(tokens))
        self.tokenizer = tokenizer
        self.seg_token_id = token_ids[0]
        self.mllm.prepare_adapters()

        self.grounding_encoder = (
            BUILDER.build(grounding_encoder)
            if isinstance(grounding_encoder, dict)
            else grounding_encoder
        )
        input_dim = int(self.mllm.get_embedding_size())
        prompt_dim = int(self.grounding_encoder.hidden_dim)
        self.seg_projector = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim, prompt_dim),
        )
        self.temporal_fusion = nn.Sequential(
            nn.Linear(prompt_dim, prompt_dim),
            nn.ReLU(inplace=True),
            nn.Linear(prompt_dim, prompt_dim),
        )
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.entity_weight = float(entity_weight)
        self.entity_token_weights = dict(ENTITY_WEIGHTS)
        if entity_token_weights is not None:
            if set(entity_token_weights) != set(ENTITY_WEIGHTS):
                raise ValueError(f"Entity weights must have keys {sorted(ENTITY_WEIGHTS)}")
            self.entity_token_weights.update(
                {key: float(value) for key, value in entity_token_weights.items()}
            )
        self.torch_dtype = torch_dtype

    def merge_adapters_for_export(self) -> None:
        self.mllm.merge_adapters_for_export()

    def expected_trainable_state_keys(self) -> list[str]:
        return sorted(name for name, parameter in self.named_parameters() if parameter.requires_grad)

    def fuse_role_aware_embeddings(
        self,
        embeddings: torch.Tensor,
        role_keys_per_frame: list[list[str]],
        counts_per_frame: list[int] | None = None,
    ) -> torch.Tensor:
        if counts_per_frame is None:
            counts_per_frame = [len(keys) for keys in role_keys_per_frame]
        if counts_per_frame != [len(keys) for keys in role_keys_per_frame]:
            raise ValueError("counts_per_frame does not match role_keys_per_frame")
        if sum(counts_per_frame) != embeddings.shape[0]:
            raise ValueError(
                f"Embedding/entity mismatch: {embeddings.shape[0]} vs {sum(counts_per_frame)}"
            )
        flat_keys = [normalize_role_key(key) for keys in role_keys_per_frame for key in keys]
        groups: dict[str, list[int]] = defaultdict(list)
        for index, key in enumerate(flat_keys):
            groups[key].append(index)
        fused = embeddings.clone()
        for indices in groups.values():
            index_tensor = torch.tensor(indices, device=embeddings.device, dtype=torch.long)
            context = embeddings.index_select(0, index_tensor).mean(dim=0, keepdim=True)
            context = self.temporal_fusion(context)
            fused.index_copy_(0, index_tensor, embeddings.index_select(0, index_tensor) + context)
        return fused

    def _entity_position_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        ranges_batch: list[dict] | None,
    ) -> torch.Tensor:
        if not ranges_batch:
            return logits.sum() * 0.0
        weighted_loss = logits.sum() * 0.0
        denominator = 0.0
        for batch_index, ranges in enumerate(ranges_batch):
            if not ranges:
                continue
            for category, weight in self.entity_token_weights.items():
                for span in ranges.get(category, []):
                    start, end = int(span[0]), int(span[1])
                    if not (0 <= start < end <= labels.shape[1]):
                        raise ValueError(f"Invalid {category} token span: {span}")
                    for label_position in range(start, end):
                        prediction_position = label_position - 1
                        if prediction_position < 0 or labels[batch_index, label_position] == -100:
                            continue
                        token_loss = F.cross_entropy(
                            logits[batch_index, prediction_position].float().unsqueeze(0),
                            labels[batch_index, label_position].unsqueeze(0),
                        )
                        weighted_loss = weighted_loss + token_loss * weight
                        denominator += weight
        if denominator == 0:
            return logits.sum() * 0.0
        return weighted_loss / denominator * self.entity_weight

    @staticmethod
    def _dice_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probabilities = logits.sigmoid().flatten(1)
        targets = targets.float().flatten(1)
        numerator = 2.0 * (probabilities * targets).sum(dim=1) + 1.0
        denominator = probabilities.sum(dim=1) + targets.sum(dim=1) + 1.0
        return (1.0 - numerator / denominator).mean()

    def _decode_training_masks(
        self,
        projected_embeddings: list[torch.Tensor],
        grounding_pixels,
        frames_per_batch: list[int],
        masks_batch,
        role_keys_batch: list[list[list[str]]],
        counts_batch: list[list[int]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(projected_embeddings) != len(frames_per_batch):
            raise ValueError("Batch split mismatch for projected embeddings")
        if isinstance(grounding_pixels, torch.Tensor):
            grounding_pixels = list(grounding_pixels)
        cursor = 0
        all_predictions: list[torch.Tensor] = []
        all_targets: list[torch.Tensor] = []
        for batch_index, frame_count in enumerate(frames_per_batch):
            frame_pixels = grounding_pixels[cursor : cursor + frame_count]
            cursor += frame_count
            counts = [int(value) for value in counts_batch[batch_index]]
            role_keys = role_keys_batch[batch_index]
            if len(counts) != frame_count or len(role_keys) != frame_count:
                raise ValueError("Every clip must provide role keys for every frame")
            embeddings = projected_embeddings[batch_index]
            targets = masks_batch[batch_index]
            expected = sum(counts)
            if embeddings.shape[0] != expected or len(targets) != expected:
                raise ValueError(
                    f"[SEG]/role/mask mismatch in sample {batch_index}: "
                    f"{embeddings.shape[0]}/{expected}/{len(targets)}"
                )
            if expected == 0:
                continue
            fused = self.fuse_role_aware_embeddings(embeddings, role_keys, counts)
            per_frame = torch.split(fused, counts, dim=0)
            max_objects = max(counts)
            padded = []
            valid = []
            for frame_embeddings, count in zip(per_frame, counts):
                padding = frame_embeddings.new_zeros(max_objects - count, frame_embeddings.shape[-1])
                padded.append(torch.cat([frame_embeddings, padding], dim=0))
                validity = torch.zeros(max_objects, dtype=torch.bool, device=fused.device)
                validity[:count] = True
                valid.append(validity)
            language = torch.stack(padded).reshape(-1, 1, fused.shape[-1])
            images = torch.stack(
                [self.grounding_encoder.preprocess_image(image) for image in frame_pixels]
            ).to(fused.device)
            states = self.grounding_encoder.get_sam2_embeddings(images, expand_size=max_objects)
            predictions = self.grounding_encoder.inject_language_embeddings(
                states, language, (frame_count, max_objects)
            )
            predictions = predictions.flatten(0, 1)[torch.stack(valid).flatten()]
            targets = targets.to(predictions.device)
            targets = F.interpolate(
                targets.float().unsqueeze(1),
                size=predictions.shape[-2:],
                mode="nearest",
            ).squeeze(1)
            all_predictions.append(predictions)
            all_targets.append(targets)
        if cursor != len(grounding_pixels):
            raise ValueError("Unused grounding frames remain after batch split")
        if not all_predictions:
            reference = projected_embeddings[0]
            empty = reference.new_empty((0, 1, 1))
            return empty, empty
        return torch.cat(all_predictions), torch.cat(all_targets)

    def forward(self, data, data_samples=None, mode: str = "loss"):
        if mode != "loss":
            raise ValueError("Training model supports mode='loss'; use the HF model for generation")
        batch = dict(data)
        grounding_pixels = batch.pop("g_pixel_values")
        masks_batch = batch.pop("masks")
        frames_per_batch = [int(value) for value in batch.pop("frames_per_batch")]
        role_keys_batch = batch.pop("role_keys_per_frame")
        counts_batch = batch.pop("num_segs_per_frame")
        ranges_batch = batch.pop("entity_token_ranges", None)

        outputs = self.mllm(batch, data_samples, mode)
        hidden_states = outputs.hidden_states[-1]
        seg_mask = batch["input_ids"].eq(self.seg_token_id)
        projected = self.seg_projector(hidden_states[seg_mask])
        counts_per_sample = seg_mask.sum(dim=1).tolist()
        projected_batch = list(torch.split(projected, counts_per_sample, dim=0))
        predicted_masks, target_masks = self._decode_training_masks(
            projected_batch,
            grounding_pixels,
            frames_per_batch,
            masks_batch,
            role_keys_batch,
            counts_batch,
        )
        if predicted_masks.numel() == 0:
            loss_bce = projected.sum() * 0.0
            loss_dice = projected.sum() * 0.0
        else:
            loss_bce = (
                F.binary_cross_entropy_with_logits(
                    predicted_masks.float(), target_masks.float()
                )
                * self.bce_weight
            )
            loss_dice = self._dice_loss(predicted_masks, target_masks) * self.dice_weight
        loss_entity = self._entity_position_loss(
            outputs.logits, batch["labels"], ranges_batch
        )
        return {
            "loss_llm": outputs.loss,
            "loss_bce": loss_bce,
            "loss_dice": loss_dice,
            "loss_entity": loss_entity,
        }
