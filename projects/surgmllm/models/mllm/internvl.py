"""Thin XTuner adapter for an InternVL2.5 checkpoint."""

from __future__ import annotations

from typing import Iterable, Optional

import torch
import torch.nn.functional as F
from mmengine.logging import print_log
from peft import PeftModel
from torch import nn
from xtuner.model import InternVL_V1_5


class SelectiveTokenEmbedding(nn.Module):
    """Train only selected vocabulary rows while keeping the base table frozen."""

    def __init__(self, base: nn.Embedding, token_ids: Iterable[int]):
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        ids = torch.tensor(sorted(set(int(i) for i in token_ids)), dtype=torch.long)
        self.register_buffer("token_ids", ids, persistent=True)
        self.trainable_rows = nn.Parameter(base.weight.detach()[ids].clone())
        self.num_embeddings = base.num_embeddings
        self.embedding_dim = base.embedding_dim
        self.padding_idx = base.padding_idx

    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight.index_copy(
            0, self.token_ids.to(self.base.weight.device), self.trainable_rows.to(self.base.weight)
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return F.embedding(
            input_ids,
            self.weight,
            self.padding_idx,
            self.base.max_norm,
            self.base.norm_type,
            self.base.scale_grad_by_freq,
            self.base.sparse,
        )

    def merge(self) -> nn.Embedding:
        with torch.no_grad():
            self.base.weight.index_copy_(
                0, self.token_ids.to(self.base.weight.device), self.trainable_rows.to(self.base.weight)
            )
        return self.base


class SelectiveTokenLinear(nn.Module):
    """Vocabulary projection with trainable rows only for new structure tokens."""

    def __init__(self, base: nn.Linear, token_ids: Iterable[int]):
        super().__init__()
        if base.bias is not None:
            raise ValueError("SelectiveTokenLinear requires a bias-free language head")
        self.base = base
        self.base.requires_grad_(False)
        ids = torch.tensor(sorted(set(int(i) for i in token_ids)), dtype=torch.long)
        self.register_buffer("token_ids", ids, persistent=True)
        self.trainable_rows = nn.Parameter(base.weight.detach()[ids].clone())
        self.in_features = base.in_features
        self.out_features = base.out_features

    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight.index_copy(
            0, self.token_ids.to(self.base.weight.device), self.trainable_rows.to(self.base.weight)
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.linear(hidden_states, self.weight)

    def merge(self) -> nn.Linear:
        with torch.no_grad():
            self.base.weight.index_copy_(
                0, self.token_ids.to(self.base.weight.device), self.trainable_rows.to(self.base.weight)
            )
        return self.base


class SurgMLLMInternVL(InternVL_V1_5):
    """InternVL2.5 model with LoRA and sparse token-row tuning."""

    def __init__(
        self,
        model_path: str,
        freeze_llm: bool = True,
        freeze_visual_encoder: bool = True,
        llm_lora: Optional[dict] = None,
        visual_encoder_lora: Optional[dict] = None,
        quantization_vit: bool = False,
        quantization_llm: bool = False,
        pretrained_pth: Optional[str] = None,
    ) -> None:
        if pretrained_pth is not None:
            raise ValueError(
                "pretrained_pth is unsupported; initialize the MLLM with model_path"
            )
        super().__init__(
            model_path=model_path,
            freeze_llm=freeze_llm,
            freeze_visual_encoder=freeze_visual_encoder,
            llm_lora=None,
            visual_encoder_lora=None,
            quantization_vit=quantization_vit,
            quantization_llm=quantization_llm,
            pretrained_pth=None,
        )
        self.llm_lora_config = llm_lora
        self.visual_encoder_lora_config = visual_encoder_lora
        # The configured recipe adapts the language model through LoRA only;
        # the InternVL vision-to-language projector remains part of the
        # frozen backbone just like InternViT and the dense LLM weights.
        self.model.mlp1.requires_grad_(False)
        self.tokenizer = None
        self.structure_token_ids: list[int] = []

    def add_structure_tokens(self, tokenizer, tokens: list[str]) -> list[int]:
        tokenizer.add_special_tokens({"additional_special_tokens": list(tokens)})
        self.model.language_model.resize_token_embeddings(len(tokenizer))
        ids = [tokenizer.convert_tokens_to_ids(token) for token in tokens]
        if len(set(ids)) != len(tokens):
            raise ValueError(f"Structure tokens do not have unique ids: {dict(zip(tokens, ids))}")
        for token, token_id in zip(tokens, ids):
            encoded = tokenizer.encode(token, add_special_tokens=False)
            if encoded != [token_id]:
                raise ValueError(f"{token!r} is not exactly one token: {encoded}")
        self.tokenizer = tokenizer
        self.structure_token_ids = ids
        self.model.img_context_token_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
        print_log(
            f"Registered {len(ids)} SurgMLLM structure tokens; vocab={len(tokenizer)}",
            logger="current",
        )
        return ids

    def prepare_adapters(self) -> None:
        if self.llm_lora_config is None:
            raise ValueError("llm_lora must be configured")
        if getattr(self.llm_lora_config, "modules_to_save", None):
            raise ValueError("Use sparse structure-token rows, not full modules_to_save")
        self._prepare_llm_for_lora(self.llm_lora_config)
        language_model = self.model.language_model
        input_embedding = language_model.get_input_embeddings()
        output_projection = language_model.get_output_embeddings()
        if not isinstance(input_embedding, nn.Embedding):
            raise TypeError(f"Unexpected input embedding type: {type(input_embedding)!r}")
        if not isinstance(output_projection, nn.Linear):
            raise TypeError(f"Unexpected output projection type: {type(output_projection)!r}")
        language_model.set_input_embeddings(
            SelectiveTokenEmbedding(input_embedding, self.structure_token_ids)
        )
        language_model.set_output_embeddings(
            SelectiveTokenLinear(output_projection, self.structure_token_ids)
        )

    def get_embedding_size(self) -> int:
        return int(self.model.config.llm_config.hidden_size)

    def merge_adapters_for_export(self) -> None:
        language_model = self.model.language_model
        input_embedding = language_model.get_input_embeddings()
        output_projection = language_model.get_output_embeddings()
        if isinstance(input_embedding, SelectiveTokenEmbedding):
            language_model.set_input_embeddings(input_embedding.merge())
        if isinstance(output_projection, SelectiveTokenLinear):
            language_model.set_output_embeddings(output_projection.merge())
        if isinstance(language_model, PeftModel):
            self.model.language_model = language_model.merge_and_unload(safe_merge=True)

    def forward(self, data: dict, data_samples=None, mode: str = "loss"):
        del data_samples, mode
        pixel_values = data["pixel_values"]
        if isinstance(pixel_values, list):
            frames = [item.unsqueeze(0) if item.ndim == 3 else item for item in pixel_values]
            concat_images = torch.cat(frames, dim=0)
        elif pixel_values.ndim == 5:
            concat_images = pixel_values.flatten(0, 1)
        else:
            raise ValueError("pixel_values must be a list of frame batches or a 5-D tensor")
        concat_images = concat_images.to(self.model.vision_model.dtype)
        image_flags = (concat_images.abs().sum(dim=(1, 2, 3)) != 0).long().unsqueeze(-1)
        return self.model(
            pixel_values=concat_images,
            input_ids=data["input_ids"],
            attention_mask=data["attention_mask"],
            position_ids=data.get("position_ids"),
            image_flags=image_flags,
            labels=data.get("labels"),
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
