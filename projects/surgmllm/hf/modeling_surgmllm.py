"""Hugging Face remote code for structured generation and mask decoding."""

from __future__ import annotations

import re
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

# See configuration_surgmllm.py: direct imports make the complete remote-code
# closure available to a fresh Hugging Face dynamic-module cache.
from .configuration_intern_vit import InternVisionConfig as _InternVisionConfig
from .configuration_internvl_chat import InternVLChatConfig as _InternVLChatConfig
from .conversation import get_conv_template as _get_conv_template
from .configuration_surgmllm import SurgMLLMConfig
from .modeling_intern_vit import InternVisionModel as _InternVisionModel
from .modeling_internvl_chat import InternVLChatModel
from .sam2_runtime import SAM2


def _normalize_role_key(role: str, label: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", label.strip().lower()).strip("_")
    return f"{role.lower()}:{normalized}"


_PHASE_CLASSES = frozenset(
    {
        "preparation",
        "calot triangle dissection",
        "clipping and cutting",
        "gallbladder dissection",
        "gallbladder packaging",
        "cleaning and coagulation",
        "gallbladder extraction",
    }
)
_INSTRUMENT_CLASSES = frozenset(
    {"grasper", "bipolar", "hook", "scissors", "clipper", "irrigator"}
)
_VERB_CLASSES = frozenset(
    {
        "grasp",
        "retract",
        "dissect",
        "coagulate",
        "clip",
        "cut",
        "aspirate",
        "irrigate",
        "pack",
        "null verb",
    }
)
_TARGET_CLASSES = frozenset(
    {
        "gallbladder",
        "cystic plate",
        "cystic duct",
        "cystic artery",
        "cystic pedicle",
        "blood vessel",
        "fluid",
        "abdominal wall cavity",
        "liver",
        "adhesion",
        "omentum",
        "peritoneum",
        "gut",
        "specimen bag",
        "null target",
    }
)
_ANSWER_RE = re.compile(
    r"\ADuring the (?P<phase>[a-z][a-z ]*) phase, "
    r"(?P<count>\d+) surgical action triplet(?P<plural>s?) "
    r"(?P<copula>is|are) identified:(?P<body>.*)\Z",
    re.DOTALL,
)
_TRIPLET_RE = re.compile(
    r"\s*\((?P<number>\d+)\) the instrument is "
    r"(?P<instrument>[a-z][a-z ]*?), the target is "
    r"(?P<target>[a-z][a-z ]*?), "
    r"based on the two components, the action is "
    r"(?P<verb>[a-z][a-z ]*?)(?P<terminator>[.;])"
)
_GROUNDING_MARKER_RE = re.compile(
    r"<p>\s*(?P<label>[a-z][a-z ]*?)\s*</p>\s*\[SEG\]"
)


def _canonical_label(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower().replace("_", " ").replace("-", " "))


@dataclass(frozen=True)
class _MarkerSpan:
    label: str
    semantic_start: int
    semantic_end: int


def _strip_grounding_markers(answer: str) -> tuple[str, list[_MarkerSpan]]:
    parts: list[str] = []
    markers: list[_MarkerSpan] = []
    cursor = 0
    semantic_length = 0
    for match in _GROUNDING_MARKER_RE.finditer(answer):
        prefix = answer[cursor : match.start()]
        parts.append(prefix)
        semantic_length += len(prefix)
        label = match.group("label").strip()
        if label != _canonical_label(label):
            raise ValueError(f"Noncanonical grounded label: {label!r}")
        start = semantic_length
        parts.append(label)
        semantic_length += len(label)
        markers.append(_MarkerSpan(label, start, semantic_length))
        cursor = match.end()
    parts.append(answer[cursor:])
    token_counts = (
        answer.count("<p>"),
        answer.count("</p>"),
        answer.count("[SEG]"),
    )
    if any(count != len(markers) for count in token_counts):
        raise ValueError(
            "Every generated grounding must be one complete "
            f"<p> entity </p> [SEG] marker; complete={len(markers)}, "
            f"tokens={token_counts}"
        )
    return "".join(parts), markers


def _parse_answer_grounding_assignments(answer: str) -> list[dict[str, object]]:
    """Parse marker roles and exact triplet slots without project imports.

    This implementation intentionally lives in the HF remote-code closure.  A
    converted checkpoint therefore keeps identical mask-order semantics when
    loaded from a clean Transformers dynamic-module cache.
    """

    semantic_answer, markers = _strip_grounding_markers(answer)
    answer_match = _ANSWER_RE.fullmatch(semantic_answer)
    if answer_match is None:
        raise ValueError("Generated answer does not match the strict phase/count grammar")
    phase = answer_match.group("phase")
    if phase not in _PHASE_CLASSES:
        raise ValueError(f"Unknown generated phase label: {phase!r}")
    count = int(answer_match.group("count"))
    expected_plural = "" if count == 1 else "s"
    expected_copula = "is" if count == 1 else "are"
    if (
        answer_match.group("plural") != expected_plural
        or answer_match.group("copula") != expected_copula
    ):
        raise ValueError(f"Generated triplet count agreement is invalid for {count}")

    body = answer_match.group("body")
    triplets: list[tuple[str, str, str]] = []
    field_spans: list[tuple[int, int, int, str, str]] = []
    if count == 0:
        if body != " none.":
            raise ValueError("Zero generated triplets must use exactly 'none.'")
    else:
        cursor = 0
        for expected_number in range(1, count + 1):
            match = _TRIPLET_RE.match(body, cursor)
            if match is None:
                raise ValueError(
                    f"Could not parse generated triplet {expected_number} at offset {cursor}"
                )
            cursor = match.end()
            if int(match.group("number")) != expected_number:
                raise ValueError(
                    f"Expected generated triplet {expected_number}, got "
                    f"{match.group('number')}"
                )
            expected_terminator = "." if expected_number == count else ";"
            if match.group("terminator") != expected_terminator:
                raise ValueError(
                    f"Generated triplet {expected_number} requires "
                    f"{expected_terminator!r}"
                )
            instrument = _canonical_label(match.group("instrument"))
            target = _canonical_label(match.group("target"))
            verb = _canonical_label(match.group("verb"))
            if (
                instrument != match.group("instrument")
                or target != match.group("target")
                or verb != match.group("verb")
            ):
                raise ValueError("Generated triplet labels must be canonical lowercase text")
            triplet_index = len(triplets)
            triplets.append((instrument, verb, target))
            body_offset = answer_match.start("body")
            for role, label in (("instrument", instrument), ("target", target)):
                start, end = match.span(role)
                field_spans.append(
                    (body_offset + start, body_offset + end, triplet_index, role, label)
                )
        if cursor != len(body):
            raise ValueError(f"Unparsed generated answer suffix at offset {cursor}")

    if len(triplets) != count:
        raise ValueError(f"Declared {count} generated triplets, parsed {len(triplets)}")

    assignments: list[dict[str, object] | None] = [None] * len(markers)
    occupied: set[tuple[int, str]] = set()
    deferred: list[int] = []
    for marker_index, marker in enumerate(markers):
        contexts = [
            span
            for span in field_spans
            if span[0] <= marker.semantic_start and marker.semantic_end <= span[1]
        ]
        if not contexts:
            deferred.append(marker_index)
            continue
        if len(contexts) != 1:
            raise ValueError(f"Ambiguous generated marker context for {marker.label!r}")
        _, _, triplet_index, role, field_label = contexts[0]
        if marker.label != field_label:
            raise ValueError(
                f"Generated {role} field {field_label!r} cannot be grounded by "
                f"{marker.label!r}"
            )
        vocabulary = _INSTRUMENT_CLASSES if role == "instrument" else _TARGET_CLASSES
        if marker.label not in vocabulary:
            raise ValueError(f"Unknown generated {role} grounding label: {marker.label!r}")
        slot = (triplet_index, role)
        if slot in occupied:
            raise ValueError(
                f"Generated triplet {triplet_index + 1} {role} has duplicate markers"
            )
        occupied.add(slot)
        assignments[marker_index] = {
            "role": role,
            "label": marker.label,
            "triplet_index": triplet_index,
        }

    for marker_index in deferred:
        marker = markers[marker_index]
        candidate_roles: list[str] = []
        if any(instrument == marker.label for instrument, _, _ in triplets):
            candidate_roles.append("instrument")
        if any(target == marker.label for _, _, target in triplets):
            candidate_roles.append("target")
        if not candidate_roles:
            raise ValueError(
                f"Unknown generated grounding label {marker.label!r}: no parsed field matches"
            )
        if len(candidate_roles) != 1:
            raise ValueError(
                f"Ambiguous generated grounding role for {marker.label!r}: "
                "both instrument and target match"
            )
        role = candidate_roles[0]
        vocabulary = _INSTRUMENT_CLASSES if role == "instrument" else _TARGET_CLASSES
        if marker.label not in vocabulary:
            raise ValueError(f"Unknown generated {role} grounding label: {marker.label!r}")
        matching_indices = [
            index
            for index, (instrument, _, target) in enumerate(triplets)
            if (instrument if role == "instrument" else target) == marker.label
            and (index, role) not in occupied
        ]
        if not matching_indices:
            raise ValueError(
                f"Excess generated marker {_normalize_role_key(role, marker.label)!r}: "
                "no unassigned triplet field remains"
            )
        triplet_index = matching_indices[0]
        occupied.add((triplet_index, role))
        assignments[marker_index] = {
            "role": role,
            "label": marker.label,
            "triplet_index": triplet_index,
        }

    for instrument, verb, target in triplets:
        if instrument not in _INSTRUMENT_CLASSES:
            raise ValueError(f"Unknown generated instrument label: {instrument!r}")
        if verb not in _VERB_CLASSES:
            raise ValueError(f"Unknown generated action label: {verb!r}")
        if target not in _TARGET_CLASSES:
            raise ValueError(f"Unknown generated target label: {target!r}")
    for assignment in assignments:
        if assignment is not None and assignment["label"] == "null target":
            raise ValueError("The non-spatial null target cannot own a generated mask")
    if any(assignment is None for assignment in assignments):
        raise ValueError("At least one generated marker could not be assigned")
    return [assignment for assignment in assignments if assignment is not None]


def _cuda_autocast(tensor: torch.Tensor):
    if tensor.device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


class SurgMLLMForConditionalGeneration(InternVLChatModel):
    config_class = SurgMLLMConfig
    _no_split_modules = InternVLChatModel._no_split_modules + ["SAM2"]

    def __init__(
        self,
        config: SurgMLLMConfig,
        vision_model=None,
        language_model=None,
        use_flash_attn: bool = True,
    ) -> None:
        super().__init__(
            config,
            vision_model=vision_model,
            language_model=language_model,
            use_flash_attn=use_flash_attn,
        )
        self.grounding_encoder = SAM2()
        hidden_size = int(config.llm_config.hidden_size)
        prompt_size = int(self.grounding_encoder.hidden_dim)
        self.seg_projector = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, prompt_size),
        )
        self.temporal_fusion = nn.Sequential(
            nn.Linear(prompt_size, prompt_size),
            nn.ReLU(inplace=True),
            nn.Linear(prompt_size, prompt_size),
        )
        self.seg_token_id = None

    def prepare_for_inference(self, tokenizer) -> None:
        self.seg_token_id = tokenizer.convert_tokens_to_ids("[SEG]")
        self.img_context_token_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
        for token in self.config.structure_tokens:
            token_id = tokenizer.convert_tokens_to_ids(token)
            if tokenizer.encode(token, add_special_tokens=False) != [token_id]:
                raise ValueError(f"Structure token is not atomic after reload: {token!r}")

    @staticmethod
    def parse_generated_grounding_assignments(
        text: str, expected_frames: int
    ) -> list[list[dict[str, object]]]:
        frame_pattern = re.compile(
            r"<think>(.*?)</think>\s*<answer>(.*?)</answer>", re.DOTALL
        )
        frames = frame_pattern.findall(text)
        if len(frames) != expected_frames:
            raise ValueError(f"Expected {expected_frames} generated frames, found {len(frames)}")
        for tag in ("<think>", "</think>", "<answer>", "</answer>"):
            if text.count(tag) != expected_frames:
                raise ValueError(
                    f"Expected {expected_frames} generated {tag} tags, found {text.count(tag)}"
                )
        output: list[list[dict[str, object]]] = []
        for think, answer in frames:
            if any(token in think for token in ("<p>", "</p>", "[SEG]")):
                raise ValueError("Generated grounding markers are forbidden in <think>")
            output.append(_parse_answer_grounding_assignments(answer.strip()))
        complete_markers = sum(len(frame) for frame in output)
        token_counts = (text.count("<p>"), text.count("</p>"), text.count("[SEG]"))
        if any(count != complete_markers for count in token_counts):
            raise ValueError(
                "Every generated structural grounding token must be inside one "
                f"answer marker; complete={complete_markers}, tokens={token_counts}"
            )
        return output

    @staticmethod
    def parse_generated_role_keys(text: str, expected_frames: int) -> list[list[str]]:
        assignments = (
            SurgMLLMForConditionalGeneration.parse_generated_grounding_assignments(
                text, expected_frames
            )
        )
        return [
            [
                _normalize_role_key(str(item["role"]), str(item["label"]))
                for item in frame
            ]
            for frame in assignments
        ]

    def fuse_role_aware_embeddings(
        self, embeddings: torch.Tensor, role_keys_per_frame: list[list[str]]
    ) -> torch.Tensor:
        keys = [key for frame in role_keys_per_frame for key in frame]
        if len(keys) != embeddings.shape[0]:
            raise ValueError(f"Role/[SEG] mismatch: {len(keys)} vs {embeddings.shape[0]}")
        groups = defaultdict(list)
        for index, key in enumerate(keys):
            groups[key].append(index)
        fused = embeddings.clone()
        for indices in groups.values():
            idx = torch.tensor(indices, device=embeddings.device, dtype=torch.long)
            context = self.temporal_fusion(
                embeddings.index_select(0, idx).mean(dim=0, keepdim=True)
            )
            fused.index_copy_(0, idx, embeddings.index_select(0, idx) + context)
        return fused

    def _decode_masks(
        self,
        grounding_pixels: torch.Tensor,
        embeddings: torch.Tensor,
        role_keys_per_frame: list[list[str]],
    ) -> list[torch.Tensor]:
        counts = [len(keys) for keys in role_keys_per_frame]
        if not counts:
            raise ValueError("At least one generated frame is required")
        if max(counts) == 0:
            height, width = grounding_pixels.shape[-2:]
            return [
                embeddings.new_empty((0, height, width)) for _ in role_keys_per_frame
            ]
        embeddings = self.fuse_role_aware_embeddings(embeddings, role_keys_per_frame)
        per_frame = torch.split(embeddings, counts, dim=0)
        max_objects = max(counts)
        padded, valid = [], []
        for frame_embeddings, count in zip(per_frame, counts):
            padding = frame_embeddings.new_zeros(max_objects - count, embeddings.shape[-1])
            padded.append(torch.cat([frame_embeddings, padding], dim=0))
            row = torch.zeros(max_objects, dtype=torch.bool, device=embeddings.device)
            row[:count] = True
            valid.append(row)
        language = torch.stack(padded).reshape(-1, 1, embeddings.shape[-1])

        grounding_pixels = grounding_pixels.float().div(255.0)
        mean = grounding_pixels.new_tensor((0.485, 0.456, 0.406))[None, :, None, None]
        std = grounding_pixels.new_tensor((0.229, 0.224, 0.225))[None, :, None, None]
        grounding_pixels = (grounding_pixels - mean) / std
        with torch.no_grad(), _cuda_autocast(grounding_pixels):
            features = self.grounding_encoder.sam2_model.forward_image(grounding_pixels)
            for index, feature in enumerate(features["backbone_fpn"]):
                features["backbone_fpn"][index] = (
                    feature[:, None]
                    .expand(-1, max_objects, -1, -1, -1)
                    .flatten(0, 1)
                )
            for index, position in enumerate(features["vision_pos_enc"]):
                features["vision_pos_enc"][index] = (
                    position[:, None]
                    .expand(-1, max_objects, -1, -1, -1)
                    .flatten(0, 1)
                )
            _, vision, _, sizes = self.grounding_encoder.sam2_model._prepare_backbone_features(
                features
            )
        high_resolution = [
            feature.permute(1, 2, 0).view(feature.size(1), feature.size(2), *size)
            for feature, size in zip(vision[:-1], sizes[:-1])
        ]
        final = vision[-1]
        height, width = sizes[-1]
        image_features = final + self.grounding_encoder.sam2_model.no_mem_embed
        image_features = image_features.permute(1, 2, 0).view(
            final.size(1), self.grounding_encoder.hidden_dim, height, width
        )
        with _cuda_autocast(image_features):
            masks = self.grounding_encoder.sam2_model._forward_sam_heads(
                backbone_features=image_features,
                point_inputs=None,
                mask_inputs=None,
                high_res_features=high_resolution,
                multimask_output=False,
                language_embd=language,
            )[4].squeeze(1)
        masks = masks.unflatten(0, (len(counts), max_objects))
        return [frame_masks[:count] for frame_masks, count in zip(masks, counts)]

    @torch.inference_mode()
    def generate_with_grounding(
        self,
        tokenizer,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        grounding_pixels: torch.Tensor,
        generation_config=None,
        **generation_kwargs,
    ) -> dict:
        if self.seg_token_id is None:
            self.prepare_for_inference(tokenizer)
        image_flags = torch.ones(pixel_values.shape[0], 1, dtype=torch.long, device=pixel_values.device)
        with _cuda_autocast(pixel_values):
            generated = self.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                **generation_kwargs,
            )
        step_count = len(generated.scores)
        generated_ids = generated.sequences[:, -step_count:] if step_count else generated.sequences[:, :0]
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=False)
        transition_scores = self.language_model.compute_transition_scores(
            generated.sequences,
            generated.scores,
            beam_indices=getattr(generated, "beam_indices", None),
            normalize_logits=True,
        )
        token_probabilities = (
            transition_scores[0, -step_count:].float().exp().cpu().tolist()
            if step_count
            else []
        )

        full_ids = torch.cat([input_ids, generated_ids], dim=1)
        full_attention = torch.ones_like(full_ids, dtype=attention_mask.dtype)
        with _cuda_autocast(pixel_values):
            teacher_output = self.forward(
                pixel_values=pixel_values,
                input_ids=full_ids,
                attention_mask=full_attention,
                position_ids=None,
                image_flags=image_flags,
                labels=None,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
        generated_hidden = teacher_output.hidden_states[-1][:, input_ids.shape[1] :]
        seg_hidden = generated_hidden[generated_ids.eq(self.seg_token_id)]
        grounding_assignments = self.parse_generated_grounding_assignments(
            generated_text, grounding_pixels.shape[0]
        )
        role_keys = [
            [
                _normalize_role_key(str(item["role"]), str(item["label"]))
                for item in frame
            ]
            for frame in grounding_assignments
        ]
        if seg_hidden.shape[0] != sum(map(len, role_keys)):
            raise ValueError("Generated hidden-state/[SEG]/role counts are not aligned")
        projected = self.seg_projector(seg_hidden)
        mask_logits = self._decode_masks(grounding_pixels, projected, role_keys)
        binary_masks = [
            (frame > self.config.mask_threshold).to(torch.uint8).cpu()
            for frame in mask_logits
        ]
        return {
            "generated_text": generated_text,
            "generated_ids": generated_ids.cpu(),
            "token_probabilities": token_probabilities,
            "role_keys_per_frame": role_keys,
            "grounding_assignments_per_frame": grounding_assignments,
            "mask_logits": [frame.float().cpu() for frame in mask_logits],
            "binary_masks": binary_masks,
        }
