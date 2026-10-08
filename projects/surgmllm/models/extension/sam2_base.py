"""SAM2 mask-head extension accepting a language embedding as a sparse prompt."""

import torch
import torch.nn.functional as F

from third_parts.sam2.modeling.sam2_base import SAM2Base


class SurgMLLMSAM2Base(SAM2Base):
    def _forward_sam_heads(
        self,
        backbone_features,
        point_inputs=None,
        mask_inputs=None,
        high_res_features=None,
        multimask_output=False,
        language_embd=None,
    ):
        batch_size = backbone_features.size(0)
        device = backbone_features.device
        if backbone_features.shape[1:] != (
            self.sam_prompt_embed_dim,
            self.sam_image_embedding_size,
            self.sam_image_embedding_size,
        ):
            raise ValueError(f"Unexpected SAM2 feature shape: {tuple(backbone_features.shape)}")

        if point_inputs is None:
            point_coords = torch.zeros(batch_size, 1, 2, device=device)
            point_labels = -torch.ones(batch_size, 1, dtype=torch.int32, device=device)
        else:
            point_coords = point_inputs["point_coords"]
            point_labels = point_inputs["point_labels"]

        if mask_inputs is not None and mask_inputs.shape[-2:] != self.sam_prompt_encoder.mask_input_size:
            mask_inputs = F.interpolate(
                mask_inputs.float(),
                size=self.sam_prompt_encoder.mask_input_size,
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        sparse_embeddings, dense_embeddings = self.sam_prompt_encoder(
            points=(point_coords, point_labels), boxes=None, masks=mask_inputs
        )
        if language_embd is not None:
            if language_embd.ndim != 3 or language_embd.shape[0] != batch_size:
                raise ValueError(
                    f"language_embd must be [B,N,C], got {tuple(language_embd.shape)}"
                )
            sparse_embeddings = torch.cat(
                [sparse_embeddings, language_embd.to(sparse_embeddings)], dim=1
            )

        low_res_multimasks, ious, output_tokens, object_score_logits = self.sam_mask_decoder(
            image_embeddings=backbone_features,
            image_pe=self.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            repeat_image=False,
            high_res_features=high_res_features,
        )
        low_res_multimasks = low_res_multimasks.float()
        high_res_multimasks = F.interpolate(
            low_res_multimasks,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        output_token = output_tokens[:, 0]
        if multimask_output:
            best = torch.argmax(ious, dim=-1)
            batch = torch.arange(batch_size, device=device)
            low_res_masks = low_res_multimasks[batch, best].unsqueeze(1)
            high_res_masks = high_res_multimasks[batch, best].unsqueeze(1)
            output_token = output_tokens[batch, best]
        else:
            low_res_masks = low_res_multimasks
            high_res_masks = high_res_multimasks
        object_pointer = self.obj_ptr_proj(output_token)
        return (
            low_res_multimasks,
            high_res_multimasks,
            ious,
            low_res_masks,
            high_res_masks,
            object_pointer,
            object_score_logits,
        )
