"""Strict text/grounding protocol for five-frame grounded captions."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .tokens import (
    ANSWER_END_TOKEN,
    ANSWER_START_TOKEN,
    PHRASE_END_TOKEN,
    PHRASE_START_TOKEN,
    SEG_TOKEN,
    THINK_END_TOKEN,
    THINK_START_TOKEN,
)


ENTITY_ROLES = ("instrument", "verb", "target", "phase")
MASK_ROLES = ("instrument", "target")

INSTRUMENT_LABELS = (
    "grasper",
    "bipolar",
    "hook",
    "scissors",
    "clipper",
    "irrigator",
)
VERB_LABELS = (
    "grasp",
    "retract",
    "dissect",
    "coagulate",
    "clip",
    "cut",
    "aspirate",
    "irrigate",
    "pack",
    "null_verb",
)
TARGET_LABELS = (
    "blood_vessel",
    "fluid",
    "abdominal_wall_cavity",
    "liver",
    "adhesion",
    "omentum",
    "peritoneum",
    "gut",
    "specimen_bag",
    "null_target",
    "gallbladder",
    "cystic_plate",
    "cystic_duct",
    "cystic_artery",
    "cystic_pedicle",
)
PHASE_LABELS = (
    "preparation",
    "calot_triangle_dissection",
    "clipping_and_cutting",
    "gallbladder_dissection",
    "gallbladder_packaging",
    "cleaning_and_coagulation",
    "gallbladder_extraction",
)


class ProtocolError(ValueError):
    """Raised when text, labels, segments, and masks cannot be aligned safely."""


@dataclass(frozen=True)
class GroundingSpec:
    role: str
    label: str
    start: int
    end: int
    payload: Mapping[str, Any]

    @property
    def role_key(self) -> str:
        return make_role_key(self.role, self.label)


@dataclass(frozen=True)
class EntityCharSpan:
    start: int
    end: int
    label: str


@dataclass(frozen=True)
class CanonicalFrame:
    text: str
    answer_start: int
    answer_end: int
    groundings: tuple[GroundingSpec, ...]
    role_keys: tuple[str, ...]
    labels: Mapping[str, tuple[str, ...]]
    entity_char_spans: Mapping[str, tuple[EntityCharSpan, ...]]


def normalize_label(label: Any) -> str:
    """Return the stable label spelling used in role keys and supervision."""

    value = str(label).strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def normalize_role(role: Any) -> str:
    value = normalize_label(role)
    aliases = {
        "i": "instrument",
        "inst": "instrument",
        "tool": "instrument",
        "action": "verb",
        "v": "verb",
        "t": "target",
        "anatomy": "target",
        "stage": "phase",
        "p": "phase",
    }
    value = aliases.get(value, value)
    if value not in ENTITY_ROLES:
        raise ProtocolError(f"Unsupported semantic role {role!r}.")
    return value


def make_role_key(role: Any, label: Any) -> str:
    normalized_role = normalize_role(role)
    normalized_label = normalize_label(label)
    if not normalized_label:
        raise ProtocolError("A role key cannot contain an empty label.")
    return f"{normalized_role}:{normalized_label}"


def _deduplicate(values: Sequence[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = normalize_label(value)
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


def _as_label_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        if "label" in value:
            return [value["label"]]
        return list(value.values())
    if isinstance(value, Sequence):
        return list(value)
    return [value]


def parse_frame_labels(raw_labels: Any) -> dict[str, list[str]]:
    """Accept role dictionaries, role records, or IVT triplet dictionaries."""

    result: dict[str, list[Any]] = {role: [] for role in ENTITY_ROLES}
    if raw_labels is None:
        return {role: [] for role in ENTITY_ROLES}

    if isinstance(raw_labels, Mapping):
        for role in ENTITY_ROLES:
            result[role].extend(_as_label_list(raw_labels.get(role)))
        raw_triplets = raw_labels.get("triplets")
        if isinstance(raw_triplets, Mapping):
            triplets = [raw_triplets]
        else:
            triplets = _as_label_list(raw_triplets)
        for triplet in triplets:
            if isinstance(triplet, Mapping):
                for role in ("instrument", "verb", "target"):
                    result[role].extend(_as_label_list(triplet.get(role)))
            elif isinstance(triplet, Sequence) and not isinstance(triplet, str):
                values = list(triplet)
                if len(values) != 3:
                    raise ProtocolError(
                        f"A positional triplet must have three labels, got {values!r}."
                    )
                for role, value in zip(("instrument", "verb", "target"), values):
                    result[role].append(value)
    elif isinstance(raw_labels, Sequence) and not isinstance(raw_labels, str):
        for record in raw_labels:
            if not isinstance(record, Mapping) or "role" not in record:
                raise ProtocolError(
                    "A label list must contain mappings with role and label fields."
                )
            role = normalize_role(record["role"])
            result[role].extend(_as_label_list(record.get("label")))
    else:
        raise ProtocolError(f"Unsupported frame-label structure: {raw_labels!r}.")

    return {role: _deduplicate(values) for role, values in result.items()}


def _one_token_positive(value: Any) -> tuple[int, int]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = list(value)
        if len(values) == 2 and all(isinstance(item, int) for item in values):
            start, end = values
        elif values:
            return _one_token_positive(values[0])
        else:
            raise ProtocolError("token_positives cannot be empty.")
    else:
        raise ProtocolError(f"Invalid token_positives value: {value!r}.")
    if start < 0 or end <= start:
        raise ProtocolError(f"Invalid character span [{start}, {end}].")
    return start, end


def _role_from_key(key: str) -> str | None:
    if ":" not in key:
        return None
    prefix = key.split(":", 1)[0]
    try:
        return normalize_role(prefix)
    except ProtocolError:
        return None


def _infer_grounding_role(label: str, labels: Mapping[str, Sequence[str]]) -> str:
    normalized = normalize_label(label)
    candidates = [
        role
        for role in MASK_ROLES
        if normalized in {normalize_label(item) for item in labels.get(role, ())}
    ]
    if len(candidates) == 1:
        return candidates[0]
    if normalized in INSTRUMENT_LABELS:
        return "instrument"
    if normalized in TARGET_LABELS:
        return "target"
    raise ProtocolError(
        f"Cannot infer whether grounded label {label!r} is an instrument or target; "
        "provide an explicit role field."
    )


def normalize_groundings(
    raw_groundings: Any,
    labels: Mapping[str, Sequence[str]] | None = None,
) -> list[GroundingSpec]:
    """Normalize grounding records and order them by original caption offset."""

    labels = labels or {role: () for role in ENTITY_ROLES}
    records: list[tuple[str, Mapping[str, Any]]] = []
    if isinstance(raw_groundings, Mapping):
        for key, value in raw_groundings.items():
            values = value if isinstance(value, list) else [value]
            for item in values:
                if not isinstance(item, Mapping):
                    raise ProtocolError(f"Grounding {key!r} must be a mapping.")
                records.append((str(key), item))
    elif isinstance(raw_groundings, Sequence) and not isinstance(
        raw_groundings, (str, bytes)
    ):
        for index, item in enumerate(raw_groundings):
            if not isinstance(item, Mapping):
                raise ProtocolError(f"Grounding #{index} must be a mapping.")
            key = str(item.get("label", item.get("role_key", f"grounding_{index}")))
            records.append((key, item))
    else:
        raise ProtocolError("groundings must be a mapping or list of mappings.")

    normalized: list[GroundingSpec] = []
    for key, payload in records:
        key_label = key.split(":", 1)[1] if _role_from_key(key) else key
        label = str(payload.get("label", key_label))
        role_value = payload.get("role") or payload.get("semantic_role")
        role = normalize_role(role_value) if role_value else _role_from_key(key)
        if role is None:
            role = _infer_grounding_role(label, labels)
        if role not in MASK_ROLES:
            raise ProtocolError(
                f"Grounded role {role!r} is invalid: verbs and phases never own masks."
            )
        positive = payload.get("token_positives", payload.get("token_positive"))
        start, end = _one_token_positive(positive)
        normalized.append(
            GroundingSpec(
                role=role,
                label=normalize_label(label),
                start=start,
                end=end,
                payload=payload,
            )
        )

    normalized.sort(key=lambda item: (item.start, item.end, item.role_key))
    previous_end = -1
    for item in normalized:
        if item.start < previous_end:
            raise ProtocolError("Grounded character spans may not overlap in caption text.")
        previous_end = item.end
    return normalized


def _wrapper_matches(caption: str, start_token: str, end_token: str) -> list[re.Match[str]]:
    pattern = re.compile(
        re.escape(start_token) + r"(?P<content>.*?)" + re.escape(end_token),
        re.DOTALL | re.IGNORECASE,
    )
    return list(pattern.finditer(caption))


def _raw_answer_intervals(caption: str) -> list[tuple[int, int]]:
    answer_matches = _wrapper_matches(caption, ANSWER_START_TOKEN, ANSWER_END_TOKEN)
    think_matches = _wrapper_matches(caption, THINK_START_TOKEN, THINK_END_TOKEN)
    if len(answer_matches) > 1 or len(think_matches) > 1:
        raise ProtocolError("Each frame may contain at most one think and one answer block.")
    if answer_matches:
        match = answer_matches[0]
        return [match.span("content")]
    if not think_matches:
        return [(0, len(caption))]
    match = think_matches[0]
    intervals = []
    if match.start() > 0:
        intervals.append((0, match.start()))
    if match.end() < len(caption):
        intervals.append((match.end(), len(caption)))
    return intervals


def _validate_grounding_offsets(caption: str, specs: Sequence[GroundingSpec]) -> None:
    answer_intervals = _raw_answer_intervals(caption)
    for spec in specs:
        if spec.end > len(caption):
            raise ProtocolError(
                f"Grounding {spec.role_key} ends at {spec.end}, beyond caption length "
                f"{len(caption)}."
            )
        if not any(left <= spec.start and spec.end <= right for left, right in answer_intervals):
            raise ProtocolError(
                f"Grounding {spec.role_key} at [{spec.start}, {spec.end}] is not wholly "
                "inside the frame answer."
            )
        entity = caption[spec.start : spec.end]
        if normalize_label(entity).replace("_", "") != spec.label.replace("_", ""):
            raise ProtocolError(
                f"Stale offset for {spec.role_key}: caption span is {entity!r}."
            )


def insert_grounding_tags(
    caption: str,
    groundings: Sequence[GroundingSpec],
) -> str:
    """Insert phrase/segment tags using offsets from the untouched caption."""

    _validate_grounding_offsets(caption, groundings)
    tagged = caption
    for spec in reversed(groundings):
        entity = caption[spec.start : spec.end]
        replacement = (
            f"{PHRASE_START_TOKEN} {entity} {PHRASE_END_TOKEN} {SEG_TOKEN}"
        )
        tagged = tagged[: spec.start] + replacement + tagged[spec.end :]
    return tagged


def _strip_existing_wrappers(
    tagged_caption: str,
    explicit_think: str | None,
) -> tuple[str, str]:
    think_matches = _wrapper_matches(tagged_caption, THINK_START_TOKEN, THINK_END_TOKEN)
    answer_matches = _wrapper_matches(tagged_caption, ANSWER_START_TOKEN, ANSWER_END_TOKEN)
    if len(think_matches) > 1 or len(answer_matches) > 1:
        raise ProtocolError("Each frame may contain at most one think and one answer block.")

    think = explicit_think or ""
    if think_matches:
        embedded = think_matches[0].group("content").strip()
        if explicit_think is not None and embedded != explicit_think.strip():
            raise ProtocolError("Explicit think text conflicts with the caption think block.")
        think = embedded

    if answer_matches:
        answer = answer_matches[0].group("content").strip()
    else:
        answer = tagged_caption
        if think_matches:
            match = think_matches[0]
            answer = tagged_caption[: match.start()] + tagged_caption[match.end() :]
        answer = answer.strip()

    for token in (THINK_START_TOKEN, THINK_END_TOKEN, ANSWER_START_TOKEN, ANSWER_END_TOKEN):
        if token.lower() in answer.lower():
            raise ProtocolError("Malformed or nested frame wrappers in answer text.")
    return think.strip(), answer.strip()


def _label_pattern(label: str) -> re.Pattern[str]:
    words = [re.escape(piece) for piece in normalize_label(label).split("_") if piece]
    body = r"[\s_-]+".join(words)
    return re.compile(rf"(?<![A-Za-z0-9]){body}(?![A-Za-z0-9])", re.IGNORECASE)


def _tag_free_text_with_source_offsets(text: str) -> tuple[str, list[int | None]]:
    """Remove grounding markup while retaining a map to visible source text.

    A mask-owning entity may also be a word in a non-mask label.  For example,
    the source phrase ``gallbladder dissection`` becomes ``<p> gallbladder
    </p> [SEG] dissection`` after the annotation's exact character offset is
    tagged.  The structural tokens are not semantic phase words, so each token
    is represented by one unmapped space in the searchable view.
    """

    token_pattern = re.compile(
        "|".join(
            re.escape(token)
            for token in (PHRASE_START_TOKEN, PHRASE_END_TOKEN, SEG_TOKEN)
        ),
        re.IGNORECASE,
    )
    visible: list[str] = []
    source_offsets: list[int | None] = []
    cursor = 0
    for match in token_pattern.finditer(text):
        for index in range(cursor, match.start()):
            visible.append(text[index])
            source_offsets.append(index)
        # Keep adjacent words separated even if an input omitted spaces around
        # its structural markers.  This placeholder can never become a span.
        visible.append(" ")
        source_offsets.append(None)
        cursor = match.end()
    for index in range(cursor, len(text)):
        visible.append(text[index])
        source_offsets.append(index)
    return "".join(visible), source_offsets


def _tag_aware_label_spans(
    answer: str,
    label: str,
    search_start: int = 0,
) -> list[tuple[int, int]]:
    """Find one canonical label without treating grounding tags as its words.

    The ordinary path returns a single contiguous span.  The fallback searches
    a tag-free semantic view, then maps every canonical word back separately.
    Consequently ``<p>``, ``</p>``, and ``[SEG]`` are never included in a
    verb/phase loss range, while the visible words must still normalize to the
    complete canonical label.
    """

    canonical = normalize_label(label)
    direct = _label_pattern(canonical).search(answer, search_start)
    if direct is not None:
        return [direct.span()]

    searchable, source_offsets = _tag_free_text_with_source_offsets(answer)
    semantic = _label_pattern(canonical).search(searchable, search_start)
    if semantic is None:
        return []

    word_matches = list(
        re.finditer(r"[A-Za-z0-9]+", searchable[semantic.start() : semantic.end()])
    )
    expected_words = canonical.split("_")
    observed_words = [normalize_label(match.group()) for match in word_matches]
    if observed_words != expected_words:
        raise ProtocolError(
            f"Tag-aware label normalization mismatch for {canonical!r}: "
            f"found {observed_words!r}."
        )

    spans: list[tuple[int, int]] = []
    for match in word_matches:
        left = semantic.start() + match.start()
        right = semantic.start() + match.end()
        mapped = [
            source_offsets[index]
            for index in range(left, right)
            if source_offsets[index] is not None
        ]
        if not mapped or any(
            current != previous + 1 for previous, current in zip(mapped, mapped[1:])
        ):
            raise ProtocolError(
                f"A structural token splits a word in canonical label {canonical!r}."
            )
        spans.append((mapped[0], mapped[-1] + 1))

    reconstructed = "_".join(normalize_label(answer[start:end]) for start, end in spans)
    if reconstructed != canonical:
        raise ProtocolError(
            f"Tag-aware spans reconstruct {reconstructed!r}, expected {canonical!r}."
        )
    return spans


def _infer_labels_from_answer(
    caption: str,
    groundings: Sequence[GroundingSpec],
) -> dict[str, list[str]]:
    labels = {role: [] for role in ENTITY_ROLES}
    for spec in groundings:
        labels[spec.role].append(spec.label)

    intervals = _raw_answer_intervals(caption)
    answer = " ".join(caption[left:right] for left, right in intervals)
    for role, vocabulary in (("verb", VERB_LABELS), ("phase", PHASE_LABELS)):
        matches: list[tuple[int, str]] = []
        for label in vocabulary:
            if label.startswith("null_"):
                continue
            match = _label_pattern(label).search(answer)
            if match:
                matches.append((match.start(), label))
        labels[role].extend(label for _, label in sorted(matches))
    return {role: _deduplicate(values) for role, values in labels.items()}


def _validate_frame_labels(
    labels: Mapping[str, Sequence[str]],
    groundings: Sequence[GroundingSpec],
) -> None:
    grounded = {(spec.role, spec.label) for spec in groundings}
    expected = {
        (role, normalize_label(label))
        for role in MASK_ROLES
        for label in labels.get(role, ())
    }
    if grounded != expected:
        raise ProtocolError(
            "Every instrument/target label must own at least one grounding mask in the "
            f"same frame; labels={sorted(expected)}, groundings={sorted(grounded)}."
        )


def _marker_entity_spans(
    text: str,
    groundings: Sequence[GroundingSpec],
) -> tuple[dict[str, list[EntityCharSpan]], list[str]]:
    spans = {role: [] for role in ENTITY_ROLES}
    role_keys: list[str] = []
    marker_pattern = re.compile(
        re.escape(PHRASE_START_TOKEN)
        + r"\s(?P<entity>.*?)\s"
        + re.escape(PHRASE_END_TOKEN)
        + r"\s+"
        + re.escape(SEG_TOKEN),
        re.DOTALL,
    )
    matches = list(marker_pattern.finditer(text))
    if len(matches) != len(groundings):
        raise ProtocolError(
            f"Expected {len(groundings)} segment markers, found {len(matches)}."
        )
    for match, spec in zip(matches, groundings):
        entity = match.group("entity")
        if normalize_label(entity).replace("_", "") != spec.label.replace("_", ""):
            raise ProtocolError(
                f"Segment order mismatch: expected {spec.role_key}, found {entity!r}."
            )
        start, end = match.span("entity")
        spans[spec.role].append(EntityCharSpan(start, end, spec.label))
        role_keys.append(spec.role_key)
    return spans, role_keys


def _non_mask_entity_spans(
    text: str,
    answer_start: int,
    answer_end: int,
    labels: Mapping[str, Sequence[str]],
    spans: dict[str, list[EntityCharSpan]],
) -> None:
    answer = text[answer_start:answer_end]
    for role in ("verb", "phase"):
        search_from: dict[str, int] = {}
        for raw_label in labels.get(role, ()):
            label = normalize_label(raw_label)
            matches = _tag_aware_label_spans(
                answer,
                label,
                search_start=search_from.get(label, 0),
            )
            if not matches:
                raise ProtocolError(
                    f"Frame {role} label {label!r} is not present in its answer text."
                )
            search_from[label] = matches[-1][1]
            spans[role].extend(
                EntityCharSpan(
                    answer_start + start,
                    answer_start + end,
                    label,
                )
                for start, end in matches
            )


def build_canonical_frame(
    caption: str,
    raw_groundings: Any,
    raw_labels: Any = None,
    explicit_think: str | None = None,
) -> CanonicalFrame:
    """Tag the raw caption first, then create one think/answer pair for a frame."""

    if not isinstance(caption, str) or not caption.strip():
        raise ProtocolError("Each frame requires a non-empty caption.")
    labels = parse_frame_labels(raw_labels)
    groundings = normalize_groundings(raw_groundings, labels)
    if not groundings:
        raise ProtocolError("Each selected frame requires at least one grounding.")
    if not any(labels.values()):
        labels = _infer_labels_from_answer(caption, groundings)
    else:
        for spec in groundings:
            if spec.label not in labels[spec.role]:
                labels[spec.role].append(spec.label)
        labels = {role: _deduplicate(values) for role, values in labels.items()}
    _validate_frame_labels(labels, groundings)

    tagged_caption = insert_grounding_tags(caption, groundings)
    think, answer = _strip_existing_wrappers(tagged_caption, explicit_think)
    text = (
        f"{THINK_START_TOKEN}\n{think}\n{THINK_END_TOKEN}\n"
        f"{ANSWER_START_TOKEN}\n{answer}\n{ANSWER_END_TOKEN}"
    )
    answer_start = text.index(ANSWER_START_TOKEN) + len(ANSWER_START_TOKEN) + 1
    answer_end = text.rindex("\n" + ANSWER_END_TOKEN)

    spans, role_keys = _marker_entity_spans(text, groundings)
    if any(
        not (answer_start <= span.start < span.end <= answer_end)
        for role in MASK_ROLES
        for span in spans[role]
    ):
        raise ProtocolError("All grounded entity markers must be inside the answer block.")
    _non_mask_entity_spans(text, answer_start, answer_end, labels, spans)

    frozen_labels = {role: tuple(labels[role]) for role in ENTITY_ROLES}
    frozen_spans = {role: tuple(spans[role]) for role in ENTITY_ROLES}
    return CanonicalFrame(
        text=text,
        answer_start=answer_start,
        answer_end=answer_end,
        groundings=tuple(groundings),
        role_keys=tuple(role_keys),
        labels=frozen_labels,
        entity_char_spans=frozen_spans,
    )


def _plain_ids(value: Any) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list) and value and isinstance(value[0], list):
        if len(value) != 1:
            raise ProtocolError("Only unbatched tokenizer output is supported.")
        value = value[0]
    if not isinstance(value, list):
        raise ProtocolError(f"Tokenizer input_ids must be a list, got {type(value)!r}.")
    return [int(item) for item in value]


def _plain_offsets(value: Any) -> list[tuple[int, int]]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list) and value and len(value) == 1 and isinstance(value[0], list):
        first = value[0]
        if not first or isinstance(first[0], (list, tuple)):
            value = first
    return [(int(start), int(end)) for start, end in value]


def tokenize_text_with_offsets(
    tokenizer: Any,
    text: str,
) -> tuple[list[int], list[tuple[int, int]] | None]:
    """Tokenize without model-added tokens and request character offsets when available."""

    try:
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        input_ids = _plain_ids(encoded["input_ids"])
        offsets = _plain_offsets(encoded["offset_mapping"])
        if len(input_ids) != len(offsets):
            raise ProtocolError("Tokenizer IDs and offset mapping have different lengths.")
        return input_ids, offsets
    except (KeyError, TypeError, NotImplementedError, AttributeError):
        input_ids = _plain_ids(tokenizer.encode(text, add_special_tokens=False))
        return input_ids, None


def _decode_matches(tokenizer: Any, token_ids: Sequence[int], expected: str) -> bool:
    if not hasattr(tokenizer, "decode"):
        return True
    decoded = tokenizer.decode(list(token_ids), skip_special_tokens=False)
    return normalize_label(decoded).replace("_", "") == normalize_label(expected).replace(
        "_", ""
    )


def map_entity_spans_to_tokens(
    text: str,
    entity_char_spans: Mapping[str, Sequence[EntityCharSpan]],
    tokenizer: Any,
    token_base: int = 0,
    strict_decode: bool = True,
) -> tuple[
    list[int],
    dict[str, list[tuple[int, int]]],
    dict[str, list[str]],
]:
    """Map answer-relative character spans into the complete sequence indices."""

    token_ids, offsets = tokenize_text_with_offsets(tokenizer, text)
    ranges: dict[str, list[tuple[int, int]]] = {role: [] for role in ENTITY_ROLES}
    range_labels: dict[str, list[str]] = {role: [] for role in ENTITY_ROLES}

    ordered = sorted(
        (
            (span.start, span.end, role, span.label)
            for role in ENTITY_ROLES
            for span in entity_char_spans.get(role, ())
        ),
        key=lambda item: (item[0], item[1], ENTITY_ROLES.index(item[2])),
    )
    for char_start, char_end, role, label in ordered:
        if offsets is not None:
            covered = [
                index
                for index, (start, end) in enumerate(offsets)
                if end > char_start and start < char_end
            ]
            if not covered:
                raise ProtocolError(
                    f"Tokenizer produced no tokens for {role}:{label} at "
                    f"[{char_start}, {char_end}]."
                )
            token_start, token_end = covered[0], covered[-1] + 1
        else:
            prefix_ids = _plain_ids(
                tokenizer.encode(text[:char_start], add_special_tokens=False)
            )
            through_ids = _plain_ids(
                tokenizer.encode(text[:char_end], add_special_tokens=False)
            )
            token_start, token_end = len(prefix_ids), len(through_ids)
            if token_end <= token_start:
                raise ProtocolError(f"Could not tokenize entity {role}:{label}.")

        if strict_decode and not _decode_matches(
            tokenizer, token_ids[token_start:token_end], text[char_start:char_end]
        ):
            decoded = tokenizer.decode(
                token_ids[token_start:token_end], skip_special_tokens=False
            )
            raise ProtocolError(
                f"Tokenizer span mismatch for {role}:{label}: decoded {decoded!r}, "
                f"expected {text[char_start:char_end]!r}."
            )
        ranges[role].append((token_base + token_start, token_base + token_end))
        range_labels[role].append(label)

    return token_ids, ranges, range_labels
