"""Hugging Face configuration for the converted SurgMLLM model."""

# Keep transitive remote-code dependencies as direct relative imports.  The
# Transformers dynamic-module cache copies direct siblings before it inspects
# nested imports.
from .configuration_intern_vit import InternVisionConfig as _InternVisionConfig
from .configuration_internvl_chat import InternVLChatConfig


class SurgMLLMConfig(InternVLChatConfig):
    model_type = "surgmllm"

    def __init__(
        self,
        structure_tokens=None,
        sam2_image_size: int = 1024,
        mask_threshold: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.structure_tokens = structure_tokens or [
            "[SEG]",
            "<p>",
            "</p>",
            "<think>",
            "</think>",
            "<answer>",
            "</answer>",
        ]
        if len(self.structure_tokens) != 7:
            raise ValueError("SurgMLLM requires exactly seven structure tokens")
        self.sam2_image_size = int(sam2_image_size)
        self.mask_threshold = float(mask_threshold)
