"""Canonical CholecT45 Fold-1 split."""

from __future__ import annotations

from typing import Literal


FOLD1_TEST_VIDEO_IDS = (79, 2, 51, 6, 25, 14, 66, 23, 50)

# Fold 2, 3, 4, then 5.  Keeping the established fold order makes generated
# configs stable while still being exactly the CholecT45 complement of Fold 1.
FOLD1_TRAIN_VIDEO_IDS = (
    80,
    32,
    5,
    15,
    40,
    47,
    26,
    48,
    70,
    31,
    57,
    36,
    18,
    52,
    68,
    10,
    8,
    73,
    42,
    29,
    60,
    27,
    65,
    75,
    22,
    49,
    12,
    78,
    43,
    62,
    35,
    74,
    1,
    56,
    4,
    13,
)

EXCLUDED_EXTENSION_VIDEO_IDS = (92, 96, 103, 110, 111)
CHOLECT45_VIDEO_IDS = FOLD1_TEST_VIDEO_IDS + FOLD1_TRAIN_VIDEO_IDS

# Readable aliases for config files.
FOLD1_TEST_VIDEOS = FOLD1_TEST_VIDEO_IDS
FOLD1_TRAIN_VIDEOS = FOLD1_TRAIN_VIDEO_IDS
FOLD1_EXCLUDED_VIDEOS = EXCLUDED_EXTENSION_VIDEO_IDS


def get_fold1_video_ids(
    split: Literal["train", "test"] | str,
) -> tuple[int, ...]:
    """Return one immutable Fold-1 split for config construction."""

    normalized = split.strip().lower()
    if normalized == "train":
        return FOLD1_TRAIN_VIDEO_IDS
    if normalized in {"test", "val", "validation"}:
        return FOLD1_TEST_VIDEO_IDS
    raise ValueError(
        f"Unknown Fold-1 split {split!r}; expected train or test."
    )


def format_video_id(video_id: int | str) -> str:
    """Normalize an integer or string video identifier to ``VIDxx`` form."""

    if isinstance(video_id, int):
        return f"VID{video_id:02d}"
    value = str(video_id).strip()
    if value.upper().startswith("VID"):
        suffix = value[3:]
        return f"VID{int(suffix):02d}" if suffix.isdigit() else value.upper()
    return f"VID{int(value):02d}" if value.isdigit() else value


def _validate_constants() -> None:
    test = set(FOLD1_TEST_VIDEO_IDS)
    train = set(FOLD1_TRAIN_VIDEO_IDS)
    excluded = set(EXCLUDED_EXTENSION_VIDEO_IDS)
    assert len(FOLD1_TEST_VIDEO_IDS) == len(test) == 9
    assert len(FOLD1_TRAIN_VIDEO_IDS) == len(train) == 36
    assert not test & train
    assert not excluded & (test | train)
    assert len(set(CHOLECT45_VIDEO_IDS)) == 45


_validate_constants()
