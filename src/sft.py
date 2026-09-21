"""Supervised fine-tuning data pipeline for the LoRA experiment (notebook 04).

The model is trained to continue the *same* prompt that notebook 03 evaluates zero-shot
(`src.prompting.build_messages`) with the intent name. Loss is computed on the intent
tokens only: the prompt is 300+ tokens of fixed instructions repeated in every example,
and letting the model spend its capacity learning to reproduce that boilerplate would
both waste the run and inflate the training loss into meaninglessness.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from .prompting import build_messages

IGNORE_INDEX = -100


def build_example(
    utterance: str,
    intent: str,
    labels: Sequence[str],
    tokenizer: Any,
) -> dict[str, list[int]]:
    """Tokenize one (utterance, intent) pair with the prompt masked out of the loss.

    The prompt and the completion are tokenized separately and concatenated, rather than
    tokenizing the joined string and slicing it. Slicing would be wrong wherever the
    tokenizer merges characters across the boundary, which silently shifts the mask by a
    token and teaches the model to predict the wrong thing.
    """
    prompt = tokenizer.apply_chat_template(
        build_messages(utterance, labels), tokenize=False, add_generation_prompt=True
    )
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(intent, add_special_tokens=False)["input_ids"]
    completion_ids = completion_ids + [tokenizer.eos_token_id]

    return {
        "input_ids": prompt_ids + completion_ids,
        "labels": [IGNORE_INDEX] * len(prompt_ids) + completion_ids,
    }


@dataclass
class CompletionOnlyCollator:
    """Pad a batch on the right, masking pad positions out of the loss.

    This pads explicitly instead of calling `tokenizer.pad`, because the tokenizer is
    configured for left padding so that batched *generation* works. Training wants right
    padding, and the two must not fight over one shared setting.
    """

    pad_token_id: int

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, Any]:
        import torch

        longest = max(len(item["input_ids"]) for item in features)
        input_ids, labels, attention_mask = [], [], []

        for item in features:
            padding = longest - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [self.pad_token_id] * padding)
            labels.append(item["labels"] + [IGNORE_INDEX] * padding)
            attention_mask.append([1] * len(item["input_ids"]) + [0] * padding)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


def build_dataset(frame, labels: Sequence[str], tokenizer: Any):
    """Turn a split's DataFrame into a `datasets.Dataset` ready for the Trainer."""
    from datasets import Dataset

    records = [
        build_example(row.utt, row.intent, labels, tokenizer)
        for row in frame.itertuples(index=False)
    ]
    return Dataset.from_list(records)


def cast_trainable_to_fp32(model: Any) -> int:
    """Keep the frozen base in fp16 but train the LoRA weights in fp32.

    Pure fp16 optimizer states lose small updates to rounding; the adapters are a tiny
    fraction of the parameters, so promoting just those costs almost no memory and
    removes the usual source of instability in fp16 LoRA runs.
    """
    import torch

    promoted = 0
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.to(torch.float32)
            promoted += parameter.numel()
    return promoted
