"""Caption cleaning and label heuristics. Pure functions, no I/O, unit testable."""

from __future__ import annotations

import re

CLEAN_MODES = ("none", "conservative", "aggressive")

_WHITESPACE = re.compile(r"\s+")
_NON_ALPHA = re.compile(r"[^a-z\s]")

# First rule that matches wins. Order matters: "curiosity rover on mars" must
# land in Mars Rover rather than being intercepted by the Mars rule.
_LABEL_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Mars Rover", ("rover", "curiosity", "perseverance", "laboratory")),
    ("Hubble", ("hubble",)),
    ("Milky Way", ("milky way", "milkyway")),
    ("Mars", ("mars", "martian")),
    ("Earth", ("earth",)),
)


def normalize_whitespace(text: str) -> str:
    """Collapse whitespace runs to single spaces and strip the ends."""
    return _WHITESPACE.sub(" ", text).strip()


def clean_conservative(text: str) -> str:
    """Keep case, punctuation and digits; only normalise whitespace. The default."""
    return normalize_whitespace(text)


def clean_aggressive(text: str) -> str:
    """The original project's approach: lowercase and strip every non-letter.

    Lossy: M31, Apollo 11, 2019 disappear for good, the model learns to emit the
    same mutilated text, and BLEU/ROUGE can no longer measure any of it. Kept
    only so the original results can be reproduced; not a good default.
    """
    lowered = text.lower()
    stripped = _NON_ALPHA.sub("", lowered)
    return normalize_whitespace(stripped)


def apply_clean(text: str, mode: str) -> str:
    if mode not in CLEAN_MODES:
        raise ValueError(f"clean_mode must be one of {CLEAN_MODES}, got {mode!r}")
    if mode == "none":
        return text
    if mode == "conservative":
        return clean_conservative(text)
    return clean_aggressive(text)


def heuristic_label(text: str) -> str:
    """Guess a topic from the caption. Used for stratified splitting and the
    distribution plot only - never a training target."""
    lowered = text.lower()
    for label, keywords in _LABEL_RULES:
        if any(keyword in lowered for keyword in keywords):
            return label
    return "Unknown"
