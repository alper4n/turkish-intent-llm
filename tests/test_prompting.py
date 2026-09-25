"""The parser decides what counts as a model's answer, so it decides the score.

Every loosening here would move accuracy upward without the model improving, which is
why the rejections matter more than the acceptances.
"""

import pytest

from src.prompting import build_messages, build_system_prompt, parse_label, prepare_tokenizer


@pytest.mark.parametrize(
    "raw",
    [
        "alarm_set",
        "  alarm_set  ",
        '"alarm_set"',
        "ALARM_SET",
        "**alarm_set**",
        "alarm_set.",
        "alarm_set — sets an alarm",
    ],
)
def test_tolerates_packaging_around_a_real_label(raw, labels):
    assert parse_label(raw, labels) == "alarm_set"


@pytest.mark.parametrize(
    "raw",
    [
        "The intent is alarm_set",     # a label buried mid-sentence is not extracted
        "kitchen_timer",               # plausible, but not in the inventory
        "",                            # an empty answer is an answer, and it is wrong
        "I am not sure",
    ],
)
def test_refuses_to_guess(raw, labels):
    """Anything unmatched comes back unmatched, for `compute_metrics` to count."""
    assert parse_label(raw, labels) not in set(labels)


def test_longer_labels_win_over_shorter_prefixes(labels):
    """`audio_volume_up` must not be shadowed by a shorter label that prefixes it."""
    extended = labels + ["audio_volume"]
    assert parse_label("audio_volume_up", extended) == "audio_volume_up"


def test_messages_put_the_inventory_in_the_system_turn(labels):
    messages = build_messages("ışıkları kıs", labels)

    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[0]["content"] == build_system_prompt(labels)
    assert "ışıkları kıs" in messages[1]["content"]


def test_generation_needs_left_padding_and_a_pad_token():
    """Decoder-only batching pads on the left; training pads on the right elsewhere."""

    class Bare:
        padding_side = "right"
        pad_token = None
        eos_token = "</s>"

    tokenizer = prepare_tokenizer(Bare())

    assert tokenizer.padding_side == "left"
    assert tokenizer.pad_token == "</s>"
