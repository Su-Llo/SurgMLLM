import pytest

from projects.surgmllm.datasets import (
    CHOLECT45_VIDEO_IDS,
    EXCLUDED_EXTENSION_VIDEO_IDS,
    FOLD1_TEST_VIDEO_IDS,
    FOLD1_TRAIN_VIDEO_IDS,
    format_video_id,
    get_fold1_video_ids,
)
from projects.surgmllm.evaluation.io import FOLD1_VIDEO_IDS


EXPECTED_TEST = (79, 2, 51, 6, 25, 14, 66, 23, 50)
EXPECTED_TRAIN = (
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


def test_fold1_is_exact_and_has_no_cross_fold_overlap():
    assert FOLD1_TEST_VIDEO_IDS == EXPECTED_TEST
    assert FOLD1_TRAIN_VIDEO_IDS == EXPECTED_TRAIN
    assert len(FOLD1_TEST_VIDEO_IDS) == 9
    assert len(FOLD1_TRAIN_VIDEO_IDS) == 36
    assert set(FOLD1_TEST_VIDEO_IDS).isdisjoint(FOLD1_TRAIN_VIDEO_IDS)
    assert len(set(CHOLECT45_VIDEO_IDS)) == 45
    assert set(CHOLECT45_VIDEO_IDS) == set(EXPECTED_TEST) | set(EXPECTED_TRAIN)
    assert set(EXCLUDED_EXTENSION_VIDEO_IDS) == {92, 96, 103, 110, 111}
    assert set(EXCLUDED_EXTENSION_VIDEO_IDS).isdisjoint(CHOLECT45_VIDEO_IDS)


def test_split_resolver_and_video_formatting():
    assert get_fold1_video_ids("train") == EXPECTED_TRAIN
    assert get_fold1_video_ids("test") == EXPECTED_TEST
    assert get_fold1_video_ids("validation") == EXPECTED_TEST
    assert format_video_id(2) == "VID02"
    assert format_video_id("VID2") == "VID02"
    assert format_video_id("79") == "VID79"
    with pytest.raises(ValueError, match="Unknown Fold-1 split"):
        get_fold1_video_ids("fold2")


def test_evaluation_fold1_names_derive_from_canonical_split():
    assert FOLD1_VIDEO_IDS == tuple(format_video_id(value) for value in EXPECTED_TEST)
