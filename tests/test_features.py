"""Feature functions are pure, so they get plain input/output tests."""

from __future__ import annotations

import pytest

from src.features.clean import (
    apply_clean,
    clean_aggressive,
    clean_conservative,
    heuristic_label,
    normalize_whitespace,
)
from src.features.conversation import (
    conversation_image_path,
    conversation_reference,
    inference_messages,
    to_conversation,
)


def test_normalize_whitespace_collapses_runs():
    assert normalize_whitespace("  a \n\t b  ") == "a b"


def test_conservative_cleaning_keeps_digits_and_punctuation():
    text = "Hubble imaged M31, the Andromeda Galaxy, in 2019."
    assert clean_conservative(text) == text


def test_aggressive_cleaning_destroys_designations():
    """Documented data loss - this is why conservative is the default."""
    assert "31" not in clean_aggressive("Hubble imaged M31.")
    assert clean_aggressive("Apollo 11 landed.") == "apollo landed"


def test_apply_clean_none_is_identity():
    assert apply_clean("  Raw  Text ", "none") == "  Raw  Text "


def test_apply_clean_rejects_unknown_mode():
    with pytest.raises(ValueError, match="clean_mode"):
        apply_clean("x", "medium")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("The Curiosity rover on Mars", "Mars Rover"),
        ("Mars Science Laboratory descent", "Mars Rover"),
        ("A Hubble deep field", "Hubble"),
        ("The Milky Way over the desert", "Milky Way"),
        ("Dust storms on Mars", "Mars"),
        ("Earth seen from the ISS", "Earth"),
        ("A distant quasar", "Unknown"),
    ],
)
def test_heuristic_label(text, expected):
    assert heuristic_label(text) == expected


def test_rover_beats_mars_in_rule_order():
    """Ordering matters: a rover-on-Mars caption must not be filed under Mars."""
    assert heuristic_label("rover on mars") == "Mars Rover"


def test_to_conversation_shape():
    record = {"image_id": "007", "image": "images/007.jpg", "text": "A nebula."}
    conversation = to_conversation(record, "/data/raw", "Describe it.")

    assert conversation["image_id"] == "007"
    assert [m["role"] for m in conversation["messages"]] == ["user", "assistant"]
    assert conversation_reference(conversation) == "A nebula."
    assert conversation_image_path(conversation).replace("\\", "/").endswith(
        "/data/raw/images/007.jpg"
    )


def test_inference_messages_carry_no_image_path():
    messages = inference_messages("Describe it.")
    assert len(messages) == 1
    image_parts = [p for p in messages[0]["content"] if p["type"] == "image"]
    assert image_parts == [{"type": "image"}]


def test_build_records_dedups_and_labels(cfg, dataset_root):
    from src.data.build import build_records
    from src.data.download import load_raw

    raw = load_raw(cfg)
    duplicated = raw + [dict(raw[0])]
    records = build_records(cfg, duplicated)

    assert len(records) == len(raw)
    assert all("label" in r for r in records)
    assert len({r["image_id"] for r in records}) == len(records)
