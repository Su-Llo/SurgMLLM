"""Strict parser for one generated frame.

Prediction parsing deliberately has no permissive fallback.  Legacy caption
parsing is exposed separately and is used only to adapt ground-truth files.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from .taxonomy import (
    INSTRUMENT_CLASSES,
    PHASE_CLASSES,
    TARGET_CLASSES,
    TRIPLET_TO_ID,
    VERB_CLASSES,
    canonical_triplet,
    normalize_label,
)


class GenerationParseError(ValueError):
    """Raised when ``raise_on_error=True`` and a frame violates the grammar."""


@dataclass(frozen=True)
class ParseIssue:
    code: str
    message: str

    def as_dict(self) -> dict[str, str]:
        return {"code": self.code, "message": self.message}


@dataclass(frozen=True)
class ParsedTriplet:
    instrument: str
    verb: str
    target: str
    taxonomy_id: int | None
    instrument_grounded: bool = False
    target_grounded: bool = False

    @property
    def valid(self) -> bool:
        return self.taxonomy_id is not None

    def as_tuple(self) -> tuple[str, str, str]:
        return self.instrument, self.verb, self.target

    def as_dict(self) -> dict[str, Any]:
        return {
            "instrument": self.instrument,
            "verb": self.verb,
            "target": self.target,
            "taxonomy_id": self.taxonomy_id,
            "valid": self.valid,
            "instrument_grounded": self.instrument_grounded,
            "target_grounded": self.target_grounded,
        }


@dataclass(frozen=True)
class ParsedGrounding:
    """One complete ``<p> entity </p> [SEG]`` marker in source order.

    ``triplet_index`` is zero based and records the prediction-schema slot that
    owns the corresponding decoded mask.  Keeping this association explicit is
    important when the marker occurs outside the triplet itself (for example,
    inside ``gallbladder dissection`` in the phase header) or when the same
    role/label is present in more than one triplet.
    """

    role: str
    label: str
    triplet_index: int
    semantic_offset: int

    @property
    def role_key(self) -> str:
        return f"{self.role}:{self.label.replace(' ', '_')}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "label": self.label,
            "role_key": self.role_key,
            "triplet_index": self.triplet_index,
            "semantic_offset": self.semantic_offset,
        }


@dataclass(frozen=True)
class FrameParseResult:
    think: str | None
    answer: str | None
    phase: str | None
    declared_triplets: int | None
    triplets: tuple[ParsedTriplet, ...]
    issues: tuple[ParseIssue, ...]
    groundings: tuple[ParsedGrounding, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.issues

    @property
    def parse_error(self) -> str | None:
        if self.ok:
            return None
        return "; ".join(f"{issue.code}: {issue.message}" for issue in self.issues)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "think": self.think,
            "answer": self.answer,
            "phase": self.phase,
            "declared_triplets": self.declared_triplets,
            "triplets": [triplet.as_dict() for triplet in self.triplets],
            "groundings": [grounding.as_dict() for grounding in self.groundings],
            "grounding_keys": [grounding.role_key for grounding in self.groundings],
            "issues": [issue.as_dict() for issue in self.issues],
            "parse_error": self.parse_error,
        }


@dataclass(frozen=True)
class WindowParseResult:
    """Strict parse result for one numbered multi-frame generation."""

    expected_frames: int
    frame_numbers: tuple[int, ...]
    frame_texts: tuple[str, ...]
    frames: tuple[FrameParseResult, ...]
    issues: tuple[ParseIssue, ...]

    @property
    def ok(self) -> bool:
        return (
            not self.issues
            and len(self.frames) == self.expected_frames
            and all(frame.ok for frame in self.frames)
        )

    @property
    def parse_error(self) -> str | None:
        messages = [f"{issue.code}: {issue.message}" for issue in self.issues]
        messages.extend(
            f"frame_{index}: {frame.parse_error}"
            for index, frame in zip(self.frame_numbers, self.frames)
            if not frame.ok
        )
        return "; ".join(messages) if messages else None

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "expected_frames": self.expected_frames,
            "frame_numbers": list(self.frame_numbers),
            "frames": [frame.as_dict() for frame in self.frames],
            "issues": [issue.as_dict() for issue in self.issues],
            "parse_error": self.parse_error,
        }


_FRAME_RE = re.compile(
    r"\A\s*<think>(?P<think>.*?)</think>\s*"
    r"<answer>(?P<answer>.*?)</answer>\s*\Z",
    re.DOTALL,
)
_ANSWER_RE = re.compile(
    r"\ADuring the (?P<phase>[a-z][a-z ]*) phase, "
    r"(?P<count>\d+) surgical action triplet(?P<plural>s?) "
    r"(?P<copula>is|are) identified:(?P<body>.*)\Z",
    re.DOTALL,
)
_TRIPLET_RE = re.compile(
    r"\s*\((?P<number>\d+)\) the instrument is "
    r"(?P<instrument>[a-z][a-z ]*?), the target is "
    r"(?P<target>[a-z][a-z ]*?), "
    r"based on the two components, the action is "
    r"(?P<verb>[a-z][a-z ]*?)(?P<terminator>[.;])"
)
_GROUNDING_MARKER_RE = re.compile(
    r"<p>\s*(?P<label>[a-z][a-z ]*?)\s*</p>\s*\[SEG\]"
)
_WINDOW_FRAME_RE = re.compile(
    r"\s*Frame\s+(?P<number>\d+)\s*:\s*"
    r"(?P<frame><think>.*?</think>\s*<answer>.*?</answer>)",
    re.DOTALL,
)
_KNOWN_GENERATION_SUFFIXES = ("<|im_end|>", "<|endoftext|>", "</s>")


def _strip_generation_suffix(text: str) -> str:
    stripped = text.rstrip()
    for suffix in _KNOWN_GENERATION_SUFFIXES:
        if stripped.endswith(suffix):
            return stripped[: -len(suffix)].rstrip()
    return stripped


def _empty_result(issue: ParseIssue) -> FrameParseResult:
    return FrameParseResult(None, None, None, None, (), (issue,))


@dataclass(frozen=True)
class _MarkerSpan:
    label: str
    semantic_start: int
    semantic_end: int


def _strip_grounding_markers(
    answer: str,
) -> tuple[str, tuple[_MarkerSpan, ...], tuple[ParseIssue, ...]]:
    """Remove complete markers while retaining their semantic answer spans.

    The replacement is the visible entity itself, so the ordinary strict
    phase/triplet grammar sees exactly the canonical sentence that the markup
    annotates.  Partial/nested/orphan marker tokens remain in the semantic text
    and therefore cannot accidentally become valid grammar.
    """

    semantic_parts: list[str] = []
    markers: list[_MarkerSpan] = []
    cursor = 0
    semantic_length = 0
    for match in _GROUNDING_MARKER_RE.finditer(answer):
        prefix = answer[cursor : match.start()]
        semantic_parts.append(prefix)
        semantic_length += len(prefix)
        raw_label = match.group("label").strip()
        start = semantic_length
        semantic_parts.append(raw_label)
        semantic_length += len(raw_label)
        markers.append(_MarkerSpan(raw_label, start, semantic_length))
        cursor = match.end()
    semantic_parts.append(answer[cursor:])
    semantic = "".join(semantic_parts)

    issues: list[ParseIssue] = []
    token_counts = {
        "<p>": answer.count("<p>"),
        "</p>": answer.count("</p>"),
        "[SEG]": answer.count("[SEG]"),
    }
    if any(count != len(markers) for count in token_counts.values()):
        detail = ", ".join(f"{token}={count}" for token, count in token_counts.items())
        issues.append(
            ParseIssue(
                "invalid_grounding_marker",
                f"every grounding must be one complete <p> entity </p> [SEG]; "
                f"complete={len(markers)}, {detail}",
            )
        )
    for marker in markers:
        canonical = normalize_label(marker.label)
        if marker.label != canonical:
            issues.append(
                ParseIssue(
                    "noncanonical_grounding_label",
                    "grounded labels must be canonical lowercase text: "
                    f"{marker.label!r}",
                )
            )
    return semantic, tuple(markers), tuple(issues)


def parse_generated_frame(
    text: object, *, raise_on_error: bool = False
) -> FrameParseResult:
    """Parse exactly one ``think``/``answer`` frame and validate its taxonomy.

    The accepted answer body is intentionally literal.  For ``N > 0`` every
    numbered item must be ``instrument -> target -> action``.  Instrument and
    target fields may be plain canonical labels or a grounded
    ``<p>entity</p>[SEG]`` occurrence.  A complete marker at another answer
    offset is associated by a unique role/label match to the parsed triplets.
    The non-spatial ``null target`` class is always plain.  ``N == 0`` is
    represented by ``none.`` after the colon.
    """

    if not isinstance(text, str):
        result = _empty_result(ParseIssue("not_string", "generated text is not a string"))
        if raise_on_error:
            raise GenerationParseError(result.parse_error)
        return result

    text = _strip_generation_suffix(text)
    tags = ("<think>", "</think>", "<answer>", "</answer>")
    bad_counts = {tag: text.count(tag) for tag in tags if text.count(tag) != 1}
    if bad_counts:
        detail = ", ".join(f"{tag}={count}" for tag, count in bad_counts.items())
        result = _empty_result(
            ParseIssue("tag_count", f"expected each frame tag exactly once; got {detail}")
        )
        if raise_on_error:
            raise GenerationParseError(result.parse_error)
        return result

    frame_match = _FRAME_RE.fullmatch(text)
    if frame_match is None:
        result = _empty_result(
            ParseIssue(
                "frame_grammar",
                "expected <think>...</think><answer>...</answer> with no extra text",
            )
        )
        if raise_on_error:
            raise GenerationParseError(result.parse_error)
        return result

    # Newlines immediately inside a wrapper and spaces around a phrase marker
    # are part of the canonical training protocol.  Whitespace is normalized
    # only at those boundaries; tag count, nesting, field order, and all
    # remaining answer text are still checked literally below.
    think = frame_match.group("think").strip()
    answer = frame_match.group("answer").strip()
    issues: list[ParseIssue] = []
    if not think:
        issues.append(ParseIssue("empty_think", "think segment must not be empty"))
    if any(marker in think for marker in ("<p>", "</p>", "[SEG]")):
        issues.append(
            ParseIssue(
                "grounding_marker_in_think",
                "grounding markers are only valid inside the answer block",
            )
        )

    semantic_answer, markers, marker_issues = _strip_grounding_markers(answer)
    issues.extend(marker_issues)
    answer_match = _ANSWER_RE.fullmatch(semantic_answer)
    if answer_match is None:
        issues.append(
            ParseIssue("answer_grammar", "answer does not match the required phase/count header")
        )
        result = FrameParseResult(think, answer, None, None, (), tuple(issues))
        if raise_on_error:
            raise GenerationParseError(result.parse_error)
        return result

    phase = answer_match.group("phase")
    count = int(answer_match.group("count"))
    plural = answer_match.group("plural")
    copula = answer_match.group("copula")
    body = answer_match.group("body")

    expected_plural = "" if count == 1 else "s"
    expected_copula = "is" if count == 1 else "are"
    if plural != expected_plural or copula != expected_copula:
        issues.append(
            ParseIssue(
                "count_agreement",
                f"count {count} requires 'triplet{expected_plural} {expected_copula}'",
            )
        )
    if phase not in PHASE_CLASSES:
        issues.append(ParseIssue("invalid_phase", f"unknown phase label: {phase!r}"))

    parsed: list[ParsedTriplet] = []
    # Spans use semantic-answer offsets (after complete markers have been
    # replaced by their visible label).  Each tuple is
    # ``(start, end, triplet_index, role, canonical_field_label)``.
    field_spans: list[tuple[int, int, int, str, str]] = []
    if count == 0:
        if body != " none.":
            issues.append(
                ParseIssue("zero_body", "zero triplets must be represented exactly as 'none.'")
            )
    else:
        cursor = 0
        for expected_number in range(1, count + 1):
            match = _TRIPLET_RE.match(body, cursor)
            if match is None:
                issues.append(
                    ParseIssue(
                        "triplet_grammar",
                        f"could not parse triplet {expected_number} at answer offset {cursor}",
                    )
                )
                break
            cursor = match.end()
            number = int(match.group("number"))
            if number != expected_number:
                issues.append(
                    ParseIssue(
                        "triplet_number",
                        f"expected item ({expected_number}), got ({number})",
                    )
                )

            raw_instrument = match.group("instrument").strip()
            raw_target = match.group("target").strip()
            raw_verb = match.group("verb").strip()
            values = (raw_instrument, raw_verb, raw_target)
            field_names = ("instrument", "action", "target")
            for field_name, value in zip(field_names, values):
                if value != normalize_label(value):
                    issues.append(
                        ParseIssue(
                            "noncanonical_label",
                            f"{field_name} must be canonical lowercase text: {value!r}",
                        )
                    )

            triplet = canonical_triplet(raw_instrument, raw_verb, raw_target)
            taxonomy_id = TRIPLET_TO_ID.get(triplet)
            parsed.append(
                ParsedTriplet(
                    *triplet,
                    taxonomy_id=taxonomy_id,
                )
            )
            triplet_index = len(parsed) - 1
            body_offset = answer_match.start("body")
            instrument_start, instrument_end = match.span("instrument")
            target_start, target_end = match.span("target")
            field_spans.extend(
                (
                    body_offset + start,
                    body_offset + end,
                    triplet_index,
                    role,
                    label,
                )
                for start, end, role, label in (
                    (
                        instrument_start,
                        instrument_end,
                        "instrument",
                        triplet[0],
                    ),
                    (target_start, target_end, "target", triplet[2]),
                )
            )

            if triplet[0] not in INSTRUMENT_CLASSES:
                issues.append(
                    ParseIssue("invalid_instrument", f"unknown instrument: {triplet[0]!r}")
                )
            if triplet[1] not in VERB_CLASSES:
                issues.append(ParseIssue("invalid_action", f"unknown action: {triplet[1]!r}"))
            if triplet[2] not in TARGET_CLASSES:
                issues.append(ParseIssue("invalid_target", f"unknown target: {triplet[2]!r}"))
            if taxonomy_id is None:
                issues.append(
                    ParseIssue(
                        "invalid_triplet",
                        f"combination is outside CholecT100: {triplet!r}",
                    )
                )
            expected_terminator = "." if expected_number == count else ";"
            if match.group("terminator") != expected_terminator:
                issues.append(
                    ParseIssue(
                        "triplet_terminator",
                        f"item {expected_number} requires {expected_terminator!r}",
                    )
                )

        if cursor != len(body):
            issues.append(
                ParseIssue(
                    "answer_trailing_text",
                    f"unparsed answer suffix at offset {cursor}: {body[cursor:]!r}",
                )
            )

    if len(parsed) != count:
        issues.append(
            ParseIssue(
                "triplet_count",
                f"declared {count} triplets but parsed {len(parsed)}",
            )
        )

    # Resolve all complete markers only after the full semantic sentence has
    # been parsed.  Explicit field context wins.  A marker elsewhere in the
    # answer is legal only when its label has exactly one instrument/target
    # role among the parsed triplets.  Direct assignments are reserved first,
    # so an inferred duplicate can never steal a field marker's schema slot.
    marker_assignments: list[tuple[str, str, int] | None] = [None] * len(markers)
    occupied: set[tuple[int, str]] = set()
    deferred: list[int] = []
    for marker_index, marker in enumerate(markers):
        contexts = [
            span
            for span in field_spans
            if span[0] <= marker.semantic_start
            and marker.semantic_end <= span[1]
        ]
        if not contexts:
            deferred.append(marker_index)
            continue
        if len(contexts) != 1:
            issues.append(
                ParseIssue(
                    "ambiguous_grounding_context",
                    f"marker {marker.label!r} overlaps multiple triplet fields",
                )
            )
            continue
        _, _, triplet_index, role, field_label = contexts[0]
        marker_label = normalize_label(marker.label)
        if marker_label != field_label:
            issues.append(
                ParseIssue(
                    "grounding_field_mismatch",
                    f"{role} field {field_label!r} cannot be grounded by "
                    f"{marker_label!r}",
                )
            )
            continue
        slot = (triplet_index, role)
        if slot in occupied:
            issues.append(
                ParseIssue(
                    "duplicate_grounding",
                    f"triplet {triplet_index + 1} {role} owns more than one marker",
                )
            )
            continue
        occupied.add(slot)
        marker_assignments[marker_index] = (role, marker_label, triplet_index)

    for marker_index in deferred:
        marker = markers[marker_index]
        marker_label = normalize_label(marker.label)
        candidate_roles: list[str] = []
        if any(triplet.instrument == marker_label for triplet in parsed):
            candidate_roles.append("instrument")
        if any(triplet.target == marker_label for triplet in parsed):
            candidate_roles.append("target")
        if not candidate_roles:
            issues.append(
                ParseIssue(
                    "unknown_grounding_label",
                    f"marker {marker_label!r} matches no parsed instrument or target",
                )
            )
            continue
        if len(candidate_roles) != 1:
            issues.append(
                ParseIssue(
                    "ambiguous_grounding_role",
                    f"marker {marker_label!r} matches both instrument and target roles",
                )
            )
            continue
        role = candidate_roles[0]
        matches = [
            index
            for index, triplet in enumerate(parsed)
            if (triplet.instrument if role == "instrument" else triplet.target)
            == marker_label
            and (index, role) not in occupied
        ]
        if not matches:
            issues.append(
                ParseIssue(
                    "excess_grounding",
                    f"marker {role}:{marker_label.replace(' ', '_')} has no "
                    "unassigned matching triplet field",
                )
            )
            continue
        triplet_index = matches[0]
        occupied.add((triplet_index, role))
        marker_assignments[marker_index] = (role, marker_label, triplet_index)

    groundings = tuple(
        ParsedGrounding(
            role=assignment[0],
            label=assignment[1],
            triplet_index=assignment[2],
            semantic_offset=marker.semantic_start,
        )
        for marker, assignment in zip(markers, marker_assignments)
        if assignment is not None
    )
    grounded_slots = {
        (grounding.triplet_index, grounding.role) for grounding in groundings
    }
    parsed = [
        ParsedTriplet(
            instrument=triplet.instrument,
            verb=triplet.verb,
            target=triplet.target,
            taxonomy_id=triplet.taxonomy_id,
            instrument_grounded=(index, "instrument") in grounded_slots,
            target_grounded=(index, "target") in grounded_slots,
        )
        for index, triplet in enumerate(parsed)
    ]
    for index, triplet in enumerate(parsed):
        if triplet.target == "null target" and triplet.target_grounded:
            issues.append(
                ParseIssue(
                    "grounded_null_target",
                    f"triplet {index + 1} null target is non-spatial and must not "
                    "own a [SEG] marker",
                )
            )
    result = FrameParseResult(
        think=think,
        answer=answer,
        phase=phase,
        declared_triplets=count,
        triplets=tuple(parsed),
        issues=tuple(issues),
        groundings=groundings,
    )
    if raise_on_error and not result.ok:
        raise GenerationParseError(result.parse_error)
    return result


def parse_generated_window(
    text: object,
    *,
    expected_frames: int = 5,
    raise_on_error: bool = False,
) -> WindowParseResult:
    """Parse exactly ``expected_frames`` ordered ``Frame N`` blocks.

    Each block is delegated to :func:`parse_generated_frame`; the window layer
    additionally rejects missing, duplicated, reordered, or trailing blocks.
    """

    if not isinstance(expected_frames, int) or expected_frames <= 0:
        raise ValueError("expected_frames must be a positive integer")
    if not isinstance(text, str):
        result = WindowParseResult(
            expected_frames,
            (),
            (),
            (),
            (ParseIssue("not_string", "generated window is not a string"),),
        )
        if raise_on_error:
            raise GenerationParseError(result.parse_error)
        return result

    normalized = _strip_generation_suffix(text)
    cursor = 0
    numbers: list[int] = []
    frame_texts: list[str] = []
    frames: list[FrameParseResult] = []
    issues: list[ParseIssue] = []
    while cursor < len(normalized):
        match = _WINDOW_FRAME_RE.match(normalized, cursor)
        if match is None:
            if normalized[cursor:].strip():
                issues.append(
                    ParseIssue(
                        "window_trailing_text",
                        f"unparsed window text at offset {cursor}: "
                        f"{normalized[cursor:cursor + 80]!r}",
                    )
                )
            break
        number = int(match.group("number"))
        expected_number = len(numbers) + 1
        if number != expected_number:
            issues.append(
                ParseIssue(
                    "frame_number",
                    f"expected Frame {expected_number}, got Frame {number}",
                )
            )
        frame_text = match.group("frame")
        numbers.append(number)
        frame_texts.append(frame_text)
        frames.append(parse_generated_frame(frame_text))
        cursor = match.end()

    if len(frames) != expected_frames:
        issues.append(
            ParseIssue(
                "frame_count",
                f"expected {expected_frames} frame blocks, found {len(frames)}",
            )
        )
    result = WindowParseResult(
        expected_frames=expected_frames,
        frame_numbers=tuple(numbers),
        frame_texts=tuple(frame_texts),
        frames=tuple(frames),
        issues=tuple(issues),
    )
    if raise_on_error and not result.ok:
        raise GenerationParseError(result.parse_error)
    return result


def parse_generated_text(text: object, *, raise_on_error: bool = False) -> FrameParseResult:
    """Compatibility alias for :func:`parse_generated_frame`."""

    return parse_generated_frame(text, raise_on_error=raise_on_error)


class StrictFrameParser:
    """Small callable wrapper convenient for inference pipelines."""

    def parse(self, text: object, *, raise_on_error: bool = False) -> FrameParseResult:
        return parse_generated_frame(text, raise_on_error=raise_on_error)

    def __call__(self, text: object, *, raise_on_error: bool = False) -> FrameParseResult:
        return self.parse(text, raise_on_error=raise_on_error)


_LEGACY_PHASE_RE = re.compile(r"During the\s+(.+?)\s+phase,", re.IGNORECASE)
_LEGACY_TRIPLET_RE = re.compile(
    r"the instrument is\s*(?P<instrument>.+?)\s*,\s*"
    r"the target is\s*(?P<target>.+?)\s*,\s*"
    r"(?:based on the two components,\s*)?the action is\s*"
    r"(?P<verb>[^.;\r\n]+)",
    re.IGNORECASE,
)
_MARKUP_RE = re.compile(r"</?p>|\[SEG\]", re.IGNORECASE)


def parse_legacy_annotation_caption(
    text: object,
) -> tuple[str | None, tuple[tuple[str, str, str], ...]]:
    """Adapt the historical annotation caption format.

    This helper must not be used for predictions: it intentionally accepts the
    older files that predate the strict generation tags and grounding markers.
    """

    if not isinstance(text, str):
        return None, ()
    without_think = re.sub(
        r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE
    ).strip()
    phase_match = _LEGACY_PHASE_RE.search(without_think)
    phase = normalize_label(phase_match.group(1)) if phase_match else None
    triplets: list[tuple[str, str, str]] = []
    for match in _LEGACY_TRIPLET_RE.finditer(without_think):
        instrument = _MARKUP_RE.sub("", match.group("instrument"))
        target = _MARKUP_RE.sub("", match.group("target"))
        verb = _MARKUP_RE.sub("", match.group("verb"))
        triplets.append(canonical_triplet(instrument, verb, target))
    return phase, tuple(triplets)
