"""Decoding-path regressions: the prompt echo and the inert temperature."""

from __future__ import annotations

import pytest

from src.models.infer import build_gen_kwargs, strip_prompt


class FakeTokenizer:
    """Decodes ids to letters so a decode result is easy to assert on."""

    def decode(self, ids, skip_special_tokens=True):  # noqa: ARG002
        return "".join(chr(ord("a") + int(i)) for i in ids)


def test_strip_prompt_removes_the_echoed_prompt():
    sequence = [0, 1, 2, 3, 4, 5]  # first three are the prompt
    assert list(strip_prompt(sequence, 3)) == [3, 4, 5]


def test_strip_prompt_with_zero_prompt_is_identity():
    assert list(strip_prompt([7, 8], 0)) == [7, 8]


def test_strip_prompt_rejects_negative_length():
    with pytest.raises(ValueError):
        strip_prompt([1, 2], -1)


def test_decoding_the_full_sequence_is_what_the_notebook_got_wrong():
    """Contrast the buggy and fixed decode on the same generate() output."""
    tokenizer = FakeTokenizer()
    prompt_tokens = 4
    sequence = [0, 1, 2, 3, 17, 4, 13]

    buggy = tokenizer.decode(sequence)
    fixed = tokenizer.decode(strip_prompt(sequence, prompt_tokens))

    assert buggy == "abcdren"
    assert fixed == "ren"
    assert buggy.startswith("abcd"), "the prompt leaks into every prediction"
    assert not fixed.startswith("abcd")


def test_greedy_kwargs_carry_no_sampling_parameters():
    """transformers silently ignores temperature when do_sample=False."""
    kwargs = build_gen_kwargs(128, do_sample=False, temperature=1.0, top_p=0.9)
    assert kwargs["do_sample"] is False
    assert "temperature" not in kwargs
    assert "top_p" not in kwargs
    assert kwargs["max_new_tokens"] == 128


def test_sampling_kwargs_keep_the_parameters_that_were_supplied():
    kwargs = build_gen_kwargs(64, do_sample=True, temperature=1.2, top_p=0.9, min_p=0.1)
    assert kwargs["do_sample"] is True
    assert kwargs["temperature"] == pytest.approx(1.2)
    assert kwargs["top_p"] == pytest.approx(0.9)
    assert kwargs["min_p"] == pytest.approx(0.1)


def test_sampling_omits_parameters_left_as_none():
    kwargs = build_gen_kwargs(64, do_sample=True, temperature=None, top_p=0.9)
    assert "temperature" not in kwargs
    assert kwargs["top_p"] == pytest.approx(0.9)


def test_use_cache_is_always_on():
    assert build_gen_kwargs(8, do_sample=False)["use_cache"] is True
