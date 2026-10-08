from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from projects.surgmllm.models.mllm.internvl import (
    SelectiveTokenEmbedding,
    SelectiveTokenLinear,
)
from projects.surgmllm.models.surgmllm import STRUCTURE_TOKENS, SurgMLLMModel
from third_parts.sam2.modeling.position_encoding import (
    PositionEmbeddingSine,
    apply_rotary_enc,
    compute_axial_cis,
)


class _DummyTokenizer:
    def __init__(self, token_ids: list[int]) -> None:
        self._available_ids = iter(token_ids)
        self._token_to_id: dict[str, int] = {}

    def add_special_tokens(self, payload: dict[str, list[str]]) -> int:
        tokens = payload["additional_special_tokens"]
        for token in tokens:
            self._token_to_id.setdefault(token, next(self._available_ids))
        return len(tokens)

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._token_to_id[token]

    def encode(self, token: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [self.convert_tokens_to_ids(token)]


class _DummyMLLM(nn.Module):
    def __init__(self, vocab_size: int = 40, hidden_size: int = 6) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.input_embedding: nn.Module = nn.Embedding(vocab_size, hidden_size)
        self.output_projection: nn.Module = nn.Linear(hidden_size, vocab_size, bias=False)
        self.lora_adapter: nn.Module = nn.Identity()
        self.structure_token_ids: list[int] = []

    def add_structure_tokens(self, tokenizer, tokens: list[str]) -> list[int]:
        tokenizer.add_special_tokens({"additional_special_tokens": tokens})
        self.structure_token_ids = [tokenizer.convert_tokens_to_ids(token) for token in tokens]
        return self.structure_token_ids

    def prepare_adapters(self) -> None:
        self.lora_adapter = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.input_embedding = SelectiveTokenEmbedding(
            self.input_embedding, self.structure_token_ids
        )
        self.output_projection = SelectiveTokenLinear(
            self.output_projection, self.structure_token_ids
        )

    def get_embedding_size(self) -> int:
        return self.hidden_size

    def forward(self, data: dict, data_samples=None, mode: str = "loss"):
        del data_samples, mode
        hidden = self.lora_adapter(self.input_embedding(data["input_ids"]))
        logits = self.output_projection(hidden)
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, self.vocab_size),
            data["labels"][:, 1:].reshape(-1),
        )
        return SimpleNamespace(loss=loss, logits=logits, hidden_states=(hidden,))


class _DummySAM2Model(nn.Module):
    def __init__(self, prompt_size: int) -> None:
        super().__init__()
        self.sam_mask_decoder = nn.Linear(prompt_size, 1)


class _DummyGroundingEncoder(nn.Module):
    def __init__(self, hidden_dim: int = 4) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.sam2_model = _DummySAM2Model(hidden_dim)

    @staticmethod
    def preprocess_image(image: torch.Tensor) -> torch.Tensor:
        return image.float()

    @staticmethod
    def get_sam2_embeddings(images: torch.Tensor, expand_size: int = 1) -> dict:
        return {"images": images, "expand_size": expand_size}

    def inject_language_embeddings(
        self,
        sam_states: dict,
        language_embeddings: torch.Tensor,
        frame_object_shape: tuple[int, int],
    ) -> torch.Tensor:
        frame_count, object_count = frame_object_shape
        assert sam_states["images"].shape[0] == frame_count
        assert sam_states["expand_size"] == object_count
        values = self.sam2_model.sam_mask_decoder(language_embeddings.squeeze(1)).squeeze(-1)
        return values.view(frame_count, object_count, 1, 1).expand(-1, -1, 2, 2)


def _build_dummy_model() -> SurgMLLMModel:
    token_ids = list(range(33, 40))
    return SurgMLLMModel(
        mllm=_DummyMLLM(),
        tokenizer=_DummyTokenizer(token_ids),
        grounding_encoder=_DummyGroundingEncoder(),
        torch_dtype=torch.float32,
    )


def test_selective_token_embedding_only_trains_seven_rows_and_merges_exactly():
    torch.manual_seed(0)
    base = nn.Embedding(19, 5)
    original = base.weight.detach().clone()
    token_ids = [18, 12, 17, 13, 16, 14, 15]
    embedding = SelectiveTokenEmbedding(base, token_ids)

    embedding(torch.tensor(token_ids)).sum().backward()

    assert base.weight.grad is None
    assert [name for name, parameter in embedding.named_parameters() if parameter.requires_grad] == [
        "trainable_rows"
    ]
    assert embedding.trainable_rows.grad is not None
    assert embedding.trainable_rows.grad.shape == (len(STRUCTURE_TOKENS), 5)
    assert torch.count_nonzero(embedding.trainable_rows.grad.sum(dim=1)) == len(STRUCTURE_TOKENS)

    with torch.no_grad():
        embedding.trainable_rows.add_(torch.arange(7).unsqueeze(1) + 0.25)
    expected = embedding.weight.detach().clone()
    merged = embedding.merge()

    assert merged is base
    assert torch.equal(merged.weight, expected)
    untouched = sorted(set(range(base.num_embeddings)) - set(token_ids))
    assert torch.equal(merged.weight[untouched], original[untouched])
    assert not merged.weight.requires_grad


def test_sam2_position_cache_and_repeated_rope_are_cpu_safe():
    position = PositionEmbeddingSine(num_pos_feats=8)
    image_features = torch.zeros(2, 8, 3, 4)
    first = position(image_features)
    cached = position(image_features)
    torch.testing.assert_close(cached, first, rtol=0, atol=0)
    assert cached.device == image_features.device

    query = torch.randn(1, 1, 4, 8)
    key = torch.randn(1, 1, 8, 8)
    frequencies = compute_axial_cis(dim=8, end_x=2, end_y=2)
    rotated_query, rotated_key = apply_rotary_enc(
        query, key, frequencies, repeat_freqs_k=True
    )
    assert rotated_query.shape == query.shape
    assert rotated_key.shape == key.shape


def test_selective_token_linear_only_trains_seven_rows_and_merges_exactly():
    torch.manual_seed(1)
    base = nn.Linear(5, 19, bias=False)
    original = base.weight.detach().clone()
    token_ids = [18, 12, 17, 13, 16, 14, 15]
    projection = SelectiveTokenLinear(base, token_ids)

    projection(torch.ones(3, 5)).sum().backward()

    assert base.weight.grad is None
    assert [name for name, parameter in projection.named_parameters() if parameter.requires_grad] == [
        "trainable_rows"
    ]
    assert projection.trainable_rows.grad is not None
    assert projection.trainable_rows.grad.shape == (len(STRUCTURE_TOKENS), 5)
    assert torch.count_nonzero(projection.trainable_rows.grad.sum(dim=1)) == len(STRUCTURE_TOKENS)

    with torch.no_grad():
        projection.trainable_rows.sub_(torch.arange(7).unsqueeze(1) + 0.5)
    expected = projection.weight.detach().clone()
    merged = projection.merge()

    assert merged is base
    assert torch.equal(merged.weight, expected)
    untouched = sorted(set(range(base.out_features)) - set(token_ids))
    assert torch.equal(merged.weight[untouched], original[untouched])
    assert not merged.weight.requires_grad


def test_selective_token_linear_rejects_a_biased_language_head():
    with pytest.raises(ValueError, match="bias-free"):
        SelectiveTokenLinear(nn.Linear(3, 10, bias=True), range(3, 10))


def test_role_aware_temporal_fusion_is_normalized_mean_mlp_residual_and_deterministic():
    model = _build_dummy_model()
    first = nn.Linear(2, 2, bias=False)
    second = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        first.weight.copy_(torch.eye(2))
        second.weight.copy_(torch.eye(2))
    model.temporal_fusion = nn.Sequential(first, nn.ReLU(), second)
    embeddings = torch.tensor(
        [
            [1.0, 2.0],
            [100.0, 200.0],
            [3.0, 4.0],
            [300.0, 400.0],
            [5.0, 6.0],
        ]
    )
    keys = [
        ["Instrument: Maryland Bipolar", "target:Maryland-Bipolar"],
        ["instrument:maryland_bipolar", "TARGET: maryland bipolar"],
        ["instrument: hook"],
    ]
    expected = torch.tensor(
        [
            [3.0, 5.0],
            [300.0, 500.0],
            [5.0, 7.0],
            [500.0, 700.0],
            [10.0, 12.0],
        ]
    )

    first_result = model.fuse_role_aware_embeddings(embeddings, keys, [2, 2, 1])
    second_result = model.fuse_role_aware_embeddings(embeddings, keys, [2, 2, 1])

    assert first_result.shape == embeddings.shape
    torch.testing.assert_close(first_result, expected, rtol=0, atol=0)
    torch.testing.assert_close(second_result, expected, rtol=0, atol=0)
    # The target rows would be orders of magnitude larger if same-label roles were mixed.
    assert first_result[0, 0] == 3.0
    assert first_result[1, 0] == 300.0


@pytest.mark.parametrize(
    ("embeddings", "keys", "counts", "error", "match"),
    [
        (torch.zeros(2, 4), [["instrument:hook"]], [2], ValueError, "counts_per_frame"),
        (
            torch.zeros(2, 4),
            [["instrument:hook"]],
            [1],
            ValueError,
            "Embedding/entity mismatch",
        ),
        (torch.zeros(1, 4), [["verb:cut"]], [1], ValueError, "instrument/target"),
        (torch.zeros(1, 4), [["instrument"]], [1], ValueError, "contain ':'"),
        (torch.zeros(1, 3), [["instrument:hook"]], [1], RuntimeError, "mat1 and mat2"),
    ],
)
def test_role_aware_temporal_fusion_rejects_misaligned_inputs(
    embeddings, keys, counts, error, match
):
    model = _build_dummy_model()
    with pytest.raises(error, match=match):
        model.fuse_role_aware_embeddings(embeddings, keys, counts)


def test_synthetic_five_frame_forward_has_four_named_losses_and_backpropagates():
    torch.manual_seed(2)
    model = _build_dummy_model()
    seg_id = model.seg_token_id
    other_structure_ids = model.mllm.structure_token_ids[1:]
    sequence = [seg_id, 1] * 10 + other_structure_ids
    input_ids = torch.tensor([sequence], dtype=torch.long)
    labels = input_ids.clone()
    masks = torch.tensor(
        [
            [[(index + row + col) % 2 for col in range(2)] for row in range(2)]
            for index in range(10)
        ],
        dtype=torch.float32,
    )
    role_keys = [
        ["instrument:grasper", "target:gallbladder"],
        ["instrument: grasper", "target: gallbladder"],
        ["instrument:hook", "target: gallbladder"],
        ["instrument:grasper", "target:liver"],
        ["instrument: hook", "target:liver"],
    ]
    data = {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "labels": labels,
        "pixel_values": torch.zeros(1, 5, 3, 2, 2),
        "g_pixel_values": [torch.full((3, 2, 2), index) for index in range(5)],
        "masks": [masks],
        "frames_per_batch": [5],
        "role_keys_per_frame": [role_keys],
        "num_segs_per_frame": [[2, 2, 2, 2, 2]],
        "entity_token_ranges": [
            {
                "instrument": [[1, 2]],
                "verb": [[3, 4]],
                "target": [[5, 6]],
                "phase": [[7, 8]],
            }
        ],
    }

    losses = model(data, mode="loss")

    assert set(losses) == {"loss_llm", "loss_bce", "loss_dice", "loss_entity"}
    for loss in losses.values():
        assert loss.ndim == 0
        assert torch.isfinite(loss)
    sum(losses.values()).backward()

    assert model.mllm.input_embedding.trainable_rows.grad is not None
    assert model.mllm.output_projection.trainable_rows.grad is not None
    assert model.mllm.input_embedding.base.weight.grad is None
    assert model.mllm.output_projection.base.weight.grad is None
    assert model.mllm.lora_adapter.weight.grad is not None
    assert model.seg_projector[0].weight.grad is not None
    assert model.temporal_fusion[0].weight.grad is not None
    assert model.grounding_encoder.sam2_model.sam_mask_decoder.weight.grad is not None


def test_all_zero_entity_clip_returns_graph_connected_zero_mask_losses():
    model = _build_dummy_model()
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    data = {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "labels": input_ids.clone(),
        "pixel_values": torch.zeros(1, 5, 3, 2, 2),
        "g_pixel_values": [torch.zeros(3, 2, 2) for _ in range(5)],
        "masks": [torch.empty(0, 2, 2)],
        "frames_per_batch": [5],
        "role_keys_per_frame": [[[], [], [], [], []]],
        "num_segs_per_frame": [[0, 0, 0, 0, 0]],
        "entity_token_ranges": [
            {"instrument": [], "verb": [], "target": [], "phase": []}
        ],
    }

    losses = model(data, mode="loss")

    assert losses["loss_bce"].item() == 0.0
    assert losses["loss_dice"].item() == 0.0
    sum(losses.values()).backward()
    assert model.seg_projector[0].weight.grad is not None
    assert torch.count_nonzero(model.seg_projector[0].weight.grad) == 0


def test_five_frame_forward_rejects_seg_role_mask_mismatch():
    model = _build_dummy_model()
    input_ids = torch.tensor([[model.seg_token_id] * 10], dtype=torch.long)
    data = {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "labels": input_ids.clone(),
        "pixel_values": torch.zeros(1, 5, 3, 2, 2),
        "g_pixel_values": [torch.zeros(3, 2, 2) for _ in range(5)],
        "masks": [torch.zeros(9, 2, 2)],
        "frames_per_batch": [5],
        "role_keys_per_frame": [
            [["instrument:hook", "target:tissue"] for _ in range(5)]
        ],
        "num_segs_per_frame": [[2, 2, 2, 2, 2]],
    }

    with pytest.raises(ValueError, match=r"\[SEG\]/role/mask mismatch"):
        model(data, mode="loss")
