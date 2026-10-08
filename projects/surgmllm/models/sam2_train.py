"""SAM2-H runner used by the training model."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import torch
from hydra import compose
from hydra.utils import instantiate
from mmengine.model import BaseModule
from omegaconf import OmegaConf


def _autocast_for(tensor: torch.Tensor):
    if tensor.device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


class SAM2TrainRunner(BaseModule):
    """Build the Hiera-Large graph and expose its language mask head."""

    def __init__(
        self,
        ckpt_path: str,
        cfg_path: str = "sam2_hiera_l.yaml",
    ) -> None:
        super().__init__(init_cfg=None)
        checkpoint = Path(ckpt_path).expanduser()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"SAM2-H checkpoint not found: {checkpoint}")

        import third_parts.sam2  # noqa: F401

        cfg = compose(
            config_name=cfg_path,
            overrides=[
                "++model._target_=projects.surgmllm.models.extension.SurgMLLMSAM2Base"
            ],
        )
        OmegaConf.resolve(cfg)
        self.sam2_model = instantiate(cfg.model, _recursive_=True)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if isinstance(payload, dict) and "model" in payload:
            payload = payload["model"]
        elif isinstance(payload, dict) and "state_dict" in payload:
            payload = payload["state_dict"]
        if not isinstance(payload, dict):
            raise TypeError(f"Unsupported SAM2 checkpoint payload: {type(payload)!r}")
        incompatible = self.sam2_model.load_state_dict(payload, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(f"SAM2 strict load failed: {incompatible}")

        self.sam2_model.requires_grad_(False)
        self.sam2_model.sam_mask_decoder.requires_grad_(True)
        self.hidden_dim = int(self.sam2_model.hidden_dim)
        self.img_mean = (0.485, 0.456, 0.406)
        self.img_std = (0.229, 0.224, 0.225)

    def train(self, mode: bool = True):
        super().train(mode)
        self.sam2_model.eval()
        self.sam2_model.sam_mask_decoder.train(mode)
        return self

    def preprocess_image(self, image: torch.Tensor) -> torch.Tensor:
        image = image.float().div(255.0)
        mean = image.new_tensor(self.img_mean)[:, None, None]
        std = image.new_tensor(self.img_std)[:, None, None]
        return (image - mean) / std

    def get_sam2_embeddings(self, images: torch.Tensor, expand_size: int = 1) -> dict:
        if images.ndim != 4 or images.shape[-2:] != (1024, 1024):
            raise ValueError(f"SAM2 images must be [B,3,1024,1024], got {tuple(images.shape)}")
        with torch.no_grad(), _autocast_for(images):
            # Freeze only the Hiera image encoder.  SAM2's high-resolution
            # conv_s0/conv_s1 projections live inside the mask decoder and must
            # remain on the gradient path when the decoder is fine-tuned.
            features = self.sam2_model.image_encoder(images)
        if self.sam2_model.use_high_res_features_in_sam:
            with _autocast_for(images):
                features["backbone_fpn"][0] = self.sam2_model.sam_mask_decoder.conv_s0(
                    features["backbone_fpn"][0]
                )
                features["backbone_fpn"][1] = self.sam2_model.sam_mask_decoder.conv_s1(
                    features["backbone_fpn"][1]
                )
        with _autocast_for(images):
            if expand_size > 1:
                for index, feature in enumerate(features["backbone_fpn"]):
                    features["backbone_fpn"][index] = (
                        feature[:, None]
                        .expand(-1, expand_size, -1, -1, -1)
                        .flatten(0, 1)
                    )
                for index, position in enumerate(features["vision_pos_enc"]):
                    features["vision_pos_enc"][index] = (
                        position[:, None]
                        .expand(-1, expand_size, -1, -1, -1)
                        .flatten(0, 1)
                    )
            _, vision, positions, sizes = self.sam2_model._prepare_backbone_features(features)
        return {
            "current_vision_feats": vision,
            "current_vision_pos_embeds": positions,
            "feat_sizes": sizes,
        }

    def inject_language_embeddings(
        self,
        sam_states: dict,
        language_embeddings: torch.Tensor,
        frame_object_shape: tuple[int, int],
    ) -> torch.Tensor:
        high_resolution = [
            feature.permute(1, 2, 0).view(feature.size(1), feature.size(2), *size)
            for feature, size in zip(
                sam_states["current_vision_feats"][:-1], sam_states["feat_sizes"][:-1]
            )
        ]
        final_feature = sam_states["current_vision_feats"][-1]
        batch_size = final_feature.size(1)
        channels = self.hidden_dim
        height, width = sam_states["feat_sizes"][-1]
        if not self.sam2_model.directly_add_no_mem_embed:
            raise RuntimeError(
                "This configuration requires directly_add_no_mem_embed"
            )
        image_features = final_feature + self.sam2_model.no_mem_embed
        image_features = image_features.permute(1, 2, 0).view(
            batch_size, channels, height, width
        )
        with _autocast_for(image_features):
            outputs = self.sam2_model._forward_sam_heads(
                backbone_features=image_features,
                point_inputs=None,
                mask_inputs=None,
                high_res_features=high_resolution,
                multimask_output=False,
                language_embd=language_embeddings,
            )
        low_resolution_masks = outputs[3].squeeze(1)
        return low_resolution_masks.unflatten(0, frame_object_shape)

    def forward(self, batch):
        raise NotImplementedError("Use get_sam2_embeddings and inject_language_embeddings")
