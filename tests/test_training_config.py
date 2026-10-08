from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch
from mmengine.config import Config
from mmengine.registry import LOOPS
from mmengine.runner import FlexibleRunner, Runner
from torch.utils.data import DataLoader
from xtuner.registry import BUILDER

from projects.surgmllm.datasets import FOLD1_TRAIN_VIDEO_IDS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FOLD1_CONFIG = PROJECT_ROOT / "projects/surgmllm/configs/surgmllm_llm_fold1.py"


def _set_runtime_paths(monkeypatch, tmp_path):
    data_root = tmp_path / "CholecT45-Scene"
    internvl_root = tmp_path / "InternVL2_5-4B"
    internvl_root.mkdir()
    (internvl_root / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["InternVLChatModel"],
                "model_type": "internvl_chat",
                "downsample_ratio": 0.5,
                "select_layer": -1,
                "dynamic_image_size": True,
                "use_thumbnail": True,
                "template": "internvl2_5",
                "llm_config": {
                    "architectures": ["Qwen2ForCausalLM"],
                    "model_type": "qwen2",
                    "hidden_size": 2048,
                    "intermediate_size": 11008,
                    "num_hidden_layers": 36,
                    "num_attention_heads": 16,
                    "num_key_value_heads": 2,
                    "vocab_size": 151674,
                },
                "vision_config": {
                    "architectures": ["InternVisionModel"],
                    "hidden_size": 1024,
                    "intermediate_size": 4096,
                    "num_hidden_layers": 24,
                    "num_attention_heads": 16,
                    "image_size": 448,
                    "patch_size": 14,
                },
            }
        ),
        encoding="utf-8",
    )
    sam2_checkpoint = tmp_path / "sam2_hiera_large.pt"
    sam2_checkpoint.touch()
    monkeypatch.setenv("SURGMLLM_DATA_ROOT", str(data_root))
    monkeypatch.setenv("INTERNVL_MODEL_PATH", str(internvl_root))
    monkeypatch.setenv("SAM2_CHECKPOINT", str(sam2_checkpoint))
    return data_root


def test_training_config_rejects_wrong_internvl_variant(monkeypatch, tmp_path):
    _set_runtime_paths(monkeypatch, tmp_path)
    config_path = Path(os.environ["INTERNVL_MODEL_PATH"]) / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["llm_config"]["hidden_size"] = 4096
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(RuntimeError, match="InternVL2.5-4B.*llm_config.hidden_size"):
        Config.fromfile(FOLD1_CONFIG)


def test_training_config_is_printable_and_registry_resolvable(monkeypatch, tmp_path):
    data_root = _set_runtime_paths(monkeypatch, tmp_path)

    fold1 = Config.fromfile(FOLD1_CONFIG)
    assert "model = dict(" in fold1.pretty_text
    assert fold1.max_epochs == 5
    assert fold1.batch_size == 4
    assert fold1.accumulative_counts == 8
    expected_loop = "mmengine.runner.loops.EpochBasedTrainLoop"
    assert fold1.train_cfg.type == expected_loop
    assert LOOPS.get(expected_loop).__name__ == "EpochBasedTrainLoop"
    assert fold1.train_dataset.datasets[0].annotation_file == str(
        data_root / "annotations/VID80_GCG.json"
    )
    assert fold1.train_dataset.datasets[0].image_root == str(data_root / "videos")
    assert len(fold1.train_dataset.datasets) == 36
    assert tuple(
        int(Path(dataset.annotation_file).stem.removeprefix("VID").removesuffix("_GCG"))
        for dataset in fold1.train_dataset.datasets
    ) == FOLD1_TRAIN_VIDEO_IDS
    assert BUILDER.get(fold1.model.type).__name__ == "SurgMLLMModel"
    assert BUILDER.get(fold1.train_dataset.type).__name__ == "ConcatDataset"
    assert BUILDER.get(fold1.optim_wrapper.optimizer.type).__name__ == "AdamW"
    expected_collate = "projects.surgmllm.datasets.surgmllm_collate_fn"
    assert fold1.train_dataloader.collate_fn.type == expected_collate


class _TinyTrainModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def train_step(self, data, optim_wrapper):
        loss = self.weight.square()
        optim_wrapper.update_params(loss)
        return {"loss": loss.detach()}


def test_epoch_loop_saves_one_checkpoint_per_epoch(tmp_path, monkeypatch):
    """FlexibleRunner must preserve real epochs, including scheduler dispatch."""

    monkeypatch.setattr(
        "mmengine._strategy.single_device.get_device", lambda: "cpu"
    )
    runner = FlexibleRunner(
        model=_TinyTrainModel(),
        work_dir=str(tmp_path),
        train_dataloader=DataLoader([0, 1, 2], batch_size=1),
        optim_wrapper=dict(optimizer=dict(type="SGD", lr=0.01)),
        param_scheduler=[
            dict(
                type="LinearLR",
                start_factor=0.1,
                by_epoch=True,
                begin=0,
                end=1,
                convert_to_iter_based=True,
            ),
            dict(
                type="CosineAnnealingLR",
                eta_min=0.0,
                by_epoch=True,
                begin=1,
                end=2,
                convert_to_iter_based=True,
            ),
        ],
        train_cfg=dict(
            type="mmengine.runner.loops.EpochBasedTrainLoop", max_epochs=2
        ),
        default_hooks=dict(
            checkpoint=dict(
                type="CheckpointHook",
                by_epoch=True,
                interval=1,
                max_keep_ckpts=-1,
                save_optimizer=False,
                save_last=True,
            )
        ),
        visualizer=None,
        randomness=dict(seed=0),
    )

    runner.train()

    assert runner.max_epochs == 2
    assert runner.max_iters == 6
    assert runner.epoch == 2
    assert runner.iter == 6
    assert all(not scheduler.by_epoch for scheduler in runner.param_schedulers)
    assert (tmp_path / "epoch_1.pth").is_file()
    assert (tmp_path / "epoch_2.pth").is_file()
    assert (tmp_path / "last_checkpoint").read_text() == str(
        tmp_path / "epoch_2.pth"
    )


def _minimal_five_frame_sample(input_ids):
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(input_ids, dtype=torch.long),
        "pixel_values": [torch.zeros(1, 3, 2, 2) for _ in range(5)],
        "g_pixel_values": [torch.zeros(3, 2, 2) for _ in range(5)],
        "frames_per_sample": 5,
        "role_keys_per_frame": [[] for _ in range(5)],
        "num_segs_per_frame": [0] * 5,
        "masks": torch.zeros(0, 2, 2, dtype=torch.bool),
    }


def test_mmengine_dataloader_invokes_configured_collate_function(monkeypatch, tmp_path):
    """MMEngine partial-binds collate kwargs, so config must name a function."""

    _set_runtime_paths(monkeypatch, tmp_path)
    config = Config.fromfile(FOLD1_CONFIG)
    assert config.train_dataloader.collate_fn.type.endswith("surgmllm_collate_fn")

    dataloader_config = dict(config.train_dataloader)
    dataloader_config.update(
        dataset=[
            _minimal_five_frame_sample([1, 2, 3]),
            _minimal_five_frame_sample([4, 5]),
        ],
        sampler=dict(type="DefaultSampler", shuffle=False),
        batch_size=2,
        num_workers=0,
        persistent_workers=False,
    )
    dataloader = Runner.build_dataloader(dataloader_config, seed=0)
    batch = next(iter(dataloader))

    assert batch["data"]["input_ids"].tolist() == [[1, 2, 3], [4, 5, 0]]
    assert batch["data"]["frames_per_batch"] == [5, 5]
