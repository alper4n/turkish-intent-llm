"""Supervised fine-tuning turns on a mask that is easy to get silently wrong.

If the mask slips by one token the model learns to predict the wrong thing and nothing
crashes, so these tests check the alignment directly rather than trusting the loss curve.
"""

import pytest

from src.prompting import build_system_prompt
from src.sft import (IGNORE_INDEX, CompletionOnlyCollator, build_example,
                     cast_trainable_to_fp32)


def test_only_the_intent_tokens_are_supervised(labels, tokenizer):
    example = build_example("yarın sabah yedide alarm kur", "alarm_set", labels, tokenizer)

    supervised = [
        token
        for token, label in zip(example["input_ids"], example["labels"])
        if label != IGNORE_INDEX
    ]

    assert len(example["input_ids"]) == len(example["labels"])
    assert tokenizer.decode(supervised) == "alarm_set <eos>"
    # The prompt is the overwhelming majority of the sequence and none of it is scored.
    assert len(supervised) == 2
    assert len(example["input_ids"]) > 10


def test_the_mask_covers_a_prefix_and_the_answer_is_the_suffix(labels, tokenizer):
    """Masked positions must be exactly the leading prompt, with no gaps."""
    example = build_example("dur", "audio_volume_mute", labels, tokenizer)
    flags = [label == IGNORE_INDEX for label in example["labels"]]

    first_supervised = flags.index(False)
    assert all(flags[:first_supervised])          # everything before is masked
    assert not any(flags[first_supervised:])      # everything after is scored

    # The supervised tail is the completion, and it echoes the input ids beneath it.
    assert example["labels"][first_supervised:] == example["input_ids"][first_supervised:]


def test_the_supervised_span_tracks_the_intent_it_was_given(labels, tokenizer):
    for intent in ("alarm_set", "play_music", "calendar_query"):
        example = build_example("bir şey", intent, labels, tokenizer)
        supervised = [
            token
            for token, label in zip(example["input_ids"], example["labels"])
            if label != IGNORE_INDEX
        ]
        assert tokenizer.decode(supervised).startswith(intent)


def test_the_prompt_carries_the_whole_inventory(labels):
    prompt = build_system_prompt(labels)
    for intent in labels:
        assert intent in prompt


def test_collator_pads_right_and_keeps_padding_out_of_the_loss(tokenizer):
    torch = pytest.importorskip("torch")

    short = {"input_ids": [5, 6], "labels": [IGNORE_INDEX, 6]}
    long = {"input_ids": [5, 6, 7, 8], "labels": [IGNORE_INDEX, IGNORE_INDEX, 7, 8]}
    batch = CompletionOnlyCollator(pad_token_id=0)([short, long])

    assert batch["input_ids"].shape == (2, 4)
    # Right padding: the short row keeps its content at the front.
    assert batch["input_ids"][0].tolist() == [5, 6, 0, 0]
    assert batch["attention_mask"][0].tolist() == [1, 1, 0, 0]
    # Padding must never contribute to the loss.
    assert (batch["labels"][batch["attention_mask"] == 0] == IGNORE_INDEX).all()
    assert batch["labels"][0].tolist() == [IGNORE_INDEX, 6, IGNORE_INDEX, IGNORE_INDEX]


def test_only_the_trainable_adapters_are_promoted_to_fp32():
    """The frozen base must stay in fp16; promoting it would defeat the point."""
    torch = pytest.importorskip("torch")

    model = torch.nn.Linear(4, 4).half()
    model.weight.requires_grad_(True)
    model.bias.requires_grad_(False)

    promoted = cast_trainable_to_fp32(model)

    assert promoted == model.weight.numel()
    assert model.weight.dtype == torch.float32
    assert model.bias.dtype == torch.float16
