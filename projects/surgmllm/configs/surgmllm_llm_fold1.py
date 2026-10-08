"""Fold-1 training configuration."""

_base_ = []

import json
import os

from projects.surgmllm.datasets.splits import FOLD1_TRAIN_VIDEO_IDS


def require_environment(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Required environment variable is not set: {name}")
    return os.path.abspath(os.path.expanduser(value))


DATA_ROOT = require_environment("SURGMLLM_DATA_ROOT")
path = require_environment("INTERNVL_MODEL_PATH")
sam2_checkpoint = require_environment("SAM2_CHECKPOINT")
if "converted" in path.lower() or os.path.splitext(path)[1] in {".pth", ".pt", ".ckpt"}:
    raise RuntimeError(
        "INTERNVL_MODEL_PATH must point to an OpenGVLab InternVL2.5-4B snapshot directory"
    )
__config_json = os.path.join(path, "config.json")
if not os.path.isfile(__config_json):
    raise FileNotFoundError(f"InternVL config is missing: {__config_json}")
with open(__config_json, "r", encoding="utf-8") as __handle:
    __reference_config = json.load(__handle)
__internvl25_4b_fingerprint = {
    ("architectures",): ["InternVLChatModel"],
    ("model_type",): "internvl_chat",
    ("downsample_ratio",): 0.5,
    ("select_layer",): -1,
    ("dynamic_image_size",): True,
    ("use_thumbnail",): True,
    ("template",): "internvl2_5",
    ("llm_config", "architectures"): ["Qwen2ForCausalLM"],
    ("llm_config", "model_type"): "qwen2",
    ("llm_config", "hidden_size"): 2048,
    ("llm_config", "intermediate_size"): 11008,
    ("llm_config", "num_hidden_layers"): 36,
    ("llm_config", "num_attention_heads"): 16,
    ("llm_config", "num_key_value_heads"): 2,
    ("llm_config", "vocab_size"): 151674,
    ("vision_config", "architectures"): ["InternVisionModel"],
    ("vision_config", "hidden_size"): 1024,
    ("vision_config", "intermediate_size"): 4096,
    ("vision_config", "num_hidden_layers"): 24,
    ("vision_config", "num_attention_heads"): 16,
    ("vision_config", "image_size"): 448,
    ("vision_config", "patch_size"): 14,
}
for __field_path, __expected_value in __internvl25_4b_fingerprint.items():
    __actual_value = __reference_config
    for __field_name in __field_path:
        __actual_value = (
            __actual_value.get(__field_name)
            if isinstance(__actual_value, dict)
            else None
        )
    if __actual_value != __expected_value:
        raise RuntimeError(
            "INTERNVL_MODEL_PATH does not match the InternVL2.5-4B config "
            f"fingerprint at {'.'.join(__field_path)}: expected "
            f"{__expected_value!r}, got {__actual_value!r}"
        )
if not os.path.isfile(sam2_checkpoint):
    raise FileNotFoundError(f"SAM2-H checkpoint is missing: {sam2_checkpoint}")
SPECIAL_TOKENS = ("[SEG]", "<p>", "</p>", "<think>", "</think>", "<answer>", "</answer>")

seed = 42
max_length = 8192
batch_size = 4
accumulative_counts = 8
dataloader_num_workers = 16
max_epochs = 5
learning_rate = 6e-5
warmup_ratio = 0.05

tokenizer = dict(
    type="transformers.AutoTokenizer.from_pretrained",
    pretrained_model_name_or_path=path,
    trust_remote_code=True,
    padding_side="right",
    use_fast=True,
)

model = dict(
    type="projects.surgmllm.models.SurgMLLMModel",
    special_tokens=SPECIAL_TOKENS,
    bce_weight=2.0,
    dice_weight=0.5,
    entity_weight=1.0,
    entity_token_weights=dict(instrument=5.0, verb=2.0, target=5.0, phase=5.0),
    mllm=dict(
        type="projects.surgmllm.models.SurgMLLMInternVL",
        model_path=path,
        freeze_llm=True,
        freeze_visual_encoder=True,
        llm_lora=dict(
            type="peft.LoraConfig",
            r=128,
            lora_alpha=256,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    ),
    tokenizer=tokenizer,
    grounding_encoder=dict(
        type="projects.surgmllm.models.SAM2TrainRunner",
        ckpt_path=sam2_checkpoint,
        cfg_path="sam2_hiera_l.yaml",
    ),
)

dataset_defaults = dict(
    type="projects.surgmllm.datasets.SurgMLLMGCGVideoDataset",
    image_root=f"{DATA_ROOT}/videos",
    tokenizer=tokenizer,
    num_frames=5,
    stride=1,
    max_length=max_length,
    special_tokens=SPECIAL_TOKENS,
    image_size=448,
    min_dynamic_patch=1,
    max_dynamic_patch=5,
    use_thumbnail=True,
    extra_image_processor=dict(
        type="projects.surgmllm.models.DirectResize", target_length=1024
    ),
)
train_dataset = dict(
    type="xtuner.dataset.ConcatDataset",
    datasets=[
        dict(
            **dataset_defaults,
            name=f"SurgMLLM_GCG_VID{video_id:02d}",
            annotation_file=f"{DATA_ROOT}/annotations/VID{video_id:02d}_GCG.json",
            video_ids=[video_id],
        )
        for video_id in FOLD1_TRAIN_VIDEO_IDS
    ],
)
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    persistent_workers=True,
    dataset=train_dataset,
    sampler=dict(type="mmengine.dataset.DefaultSampler", shuffle=True, seed=seed),
    collate_fn=dict(
        type="projects.surgmllm.datasets.surgmllm_collate_fn", pad_index=0
    ),
)

optim_wrapper = dict(
    type="mmengine.optim.AmpOptimWrapper",
    optimizer=dict(
        type="torch.optim.AdamW",
        lr=learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.05,
    ),
    clip_grad=dict(max_norm=1.0, error_if_nonfinite=True),
    accumulative_counts=accumulative_counts,
    loss_scale="dynamic",
    dtype="bfloat16",
)
param_scheduler = [
    dict(
        type="mmengine.optim.LinearLR",
        start_factor=1e-5,
        by_epoch=True,
        begin=0,
        end=warmup_ratio * max_epochs,
        convert_to_iter_based=True,
    ),
    dict(
        type="mmengine.optim.CosineAnnealingLR",
        eta_min=0.0,
        by_epoch=True,
        begin=warmup_ratio * max_epochs,
        end=max_epochs,
        convert_to_iter_based=True,
    ),
]
train_cfg = dict(
    type="mmengine.runner.loops.EpochBasedTrainLoop", max_epochs=max_epochs
)

default_hooks = dict(
    timer=dict(type="mmengine.hooks.IterTimerHook"),
    logger=dict(type="mmengine.hooks.LoggerHook", log_metric_by_epoch=False, interval=10),
    param_scheduler=dict(type="mmengine.hooks.ParamSchedulerHook"),
    checkpoint=dict(
        type="mmengine.hooks.CheckpointHook",
        by_epoch=True,
        interval=1,
        max_keep_ckpts=-1,
        save_optimizer=False,
        save_last=True,
    ),
    sampler_seed=dict(type="mmengine.hooks.DistSamplerSeedHook"),
)
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    dist_cfg=dict(backend="nccl"),
)
randomness = dict(seed=seed, deterministic=True, diff_rank_seed=False)
log_level = "INFO"
resume = False
load_from = None
custom_hooks = []
visualizer = None
