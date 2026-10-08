"""Run a bounded real-weight training smoke and save an MMEngine-style checkpoint."""

from __future__ import annotations

import argparse
import json
import time
from functools import partial
from pathlib import Path

import torch
from mmengine.config import Config
from xtuner.registry import BUILDER


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--save-trainable-only",
        action="store_true",
        help="mirror ZeRO-2 checkpoints that omit frozen reference-base tensors",
    )
    return parser.parse_args()


def _to_device(data: dict, device: torch.device) -> dict:
    moved = {}
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        elif isinstance(value, list) and value and all(
            isinstance(item, torch.Tensor) for item in value
        ):
            moved[key] = [item.to(device) for item in value]
        else:
            moved[key] = value
    return moved


def main() -> None:
    args = parse_args()
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    checkpoint_path = Path(args.checkpoint).resolve()
    if checkpoint_path.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint: {checkpoint_path}")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    cfg = Config.fromfile(args.config)
    dataset = BUILDER.build(cfg.train_dataset)
    collator_cfg = dict(cfg.train_dataloader.collate_fn)
    collator_type = collator_cfg.pop("type")
    collator = partial(BUILDER.get(collator_type), **collator_cfg)
    sample = dataset[args.sample_index]
    packed = collator([sample])

    device = torch.device(args.device)
    model = BUILDER.build(cfg.model).to(device)
    model.train()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(cfg.optim_wrapper.optimizer.lr),
        betas=tuple(cfg.optim_wrapper.optimizer.betas),
        weight_decay=float(cfg.optim_wrapper.optimizer.weight_decay),
    )
    data = _to_device(packed["data"], device)
    losses_history = []
    started = time.time()
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            losses = model(data, packed.get("data_samples"), mode="loss")
            total = sum(losses.values())
        if not torch.isfinite(total):
            raise FloatingPointError(f"Non-finite total loss at step {step + 1}: {total}")
        total.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError(
                f"Non-finite gradient norm at step {step + 1}: {gradient_norm}"
            )
        optimizer.step()
        record = {
            key: float(value.detach().float().cpu()) for key, value in losses.items()
        }
        record.update(
            step=step + 1,
            total=float(total.detach().float().cpu()),
            gradient_norm=float(gradient_norm.detach().float().cpu()),
        )
        losses_history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)

    full_state_dict = model.state_dict()
    expected = model.expected_trainable_state_keys()
    missing = sorted(set(expected) - set(full_state_dict))
    if missing:
        raise RuntimeError(f"Trainable state keys missing before save: {missing}")
    state_dict = (
        {key: full_state_dict[key] for key in expected}
        if args.save_trainable_only
        else full_state_dict
    )
    torch.save(
        {
            "meta": {
                "format": "MMEngine",
                "smoke": True,
                "config": Path(args.config).name,
                "sample_index": args.sample_index,
                "video_id": sample["video_id"],
                "frame_ids": sample["frame_ids"],
                "steps": args.steps,
                "trainable_only": args.save_trainable_only,
                "elapsed_seconds": time.time() - started,
                "loss_history": losses_history,
                "expected_trainable_keys": expected,
            },
            "state_dict": state_dict,
        },
        checkpoint_path,
    )
    print(
        json.dumps(
            {
                "checkpoint": checkpoint_path.name,
                "state_key_count": len(state_dict),
                "trainable_key_count": len(expected),
                "loss_history": losses_history,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
