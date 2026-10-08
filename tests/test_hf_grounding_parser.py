from __future__ import annotations

import pytest

from projects.surgmllm.hf.modeling_surgmllm import (
    SurgMLLMForConditionalGeneration,
)


HEADER_GROUNDED_TEXT = (
    "<think>The hook dissects the gallbladder.</think>"
    "<answer>During the <p> gallbladder </p> [SEG] dissection phase, "
    "1 surgical action triplet is identified: (1) the instrument is "
    "<p> hook </p> [SEG], the target is gallbladder, based on the two "
    "components, the action is dissect.</answer>"
)

REPEATED_TARGET_HEADER_TEXT = (
    "<think>The grasper acts twice.</think>"
    "<answer>During the <p> gallbladder </p> [SEG] dissection phase, "
    "2 surgical action triplets are identified: (1) the instrument is "
    "grasper, the target is gallbladder, based on the two components, the "
    "action is retract; (2) the instrument is grasper, the target is "
    "gallbladder, based on the two components, the action is dissect.</answer>"
)


def test_hf_role_keys_follow_seg_order_across_header_and_field_contexts():
    assignments = (
        SurgMLLMForConditionalGeneration.parse_generated_grounding_assignments(
            HEADER_GROUNDED_TEXT, 1
        )
    )
    assert assignments == [
        [
            {"role": "target", "label": "gallbladder", "triplet_index": 0},
            {"role": "instrument", "label": "hook", "triplet_index": 0},
        ]
    ]
    assert SurgMLLMForConditionalGeneration.parse_generated_role_keys(
        HEADER_GROUNDED_TEXT, 1
    ) == [["target:gallbladder", "instrument:hook"]]


@pytest.mark.parametrize(
    ("text", "message"),
    [
        (
            HEADER_GROUNDED_TEXT.replace(
                "the instrument is <p> hook </p> [SEG]",
                "the instrument is gallbladder",
            ),
            "Ambiguous generated grounding role",
        ),
        (
            HEADER_GROUNDED_TEXT.replace(
                "the target is gallbladder", "the target is liver"
            ),
            "Unknown generated grounding label",
        ),
        (
            HEADER_GROUNDED_TEXT.replace(
                "the target is gallbladder",
                "the target is <p> gallbladder </p> [SEG]",
            ),
            "Excess generated marker",
        ),
        (
            HEADER_GROUNDED_TEXT.replace("the action is dissect", "the action is dissect[SEG]"),
            "one complete",
        ),
    ],
)
def test_hf_grounding_parser_rejects_ambiguous_unknown_excess_and_stray(text, message):
    with pytest.raises(ValueError, match=message):
        SurgMLLMForConditionalGeneration.parse_generated_role_keys(text, 1)


def test_hf_deferred_marker_uses_first_unassigned_matching_occurrence():
    assignments = SurgMLLMForConditionalGeneration.parse_generated_grounding_assignments(
        REPEATED_TARGET_HEADER_TEXT, 1
    )
    assert assignments == [
        [{"role": "target", "label": "gallbladder", "triplet_index": 0}]
    ]

    second_field = REPEATED_TARGET_HEADER_TEXT.replace(
        "During the <p> gallbladder </p> [SEG] dissection phase",
        "During the gallbladder dissection phase",
    ).replace(
        "(2) the instrument is grasper, the target is gallbladder",
        "(2) the instrument is grasper, the target is <p> gallbladder </p> [SEG]",
    )
    assignments = SurgMLLMForConditionalGeneration.parse_generated_grounding_assignments(
        second_field, 1
    )
    assert assignments == [
        [{"role": "target", "label": "gallbladder", "triplet_index": 1}]
    ]
