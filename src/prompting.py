"""Prompting and label parsing for the generative experiments (notebooks 03 and 04).

Both the zero-shot baseline and the LoRA fine-tune use the prompt built here, so the
only thing that differs between them is the weights. If the prompts diverged, the
comparison would measure prompt engineering instead of fine-tuning.
"""

from __future__ import annotations

import re
from typing import Any, Sequence

SYSTEM_TEMPLATE = (
    "You are an intent classifier for a Turkish voice assistant.\n"
    "Classify the user's utterance into exactly one of these intents:\n"
    "{labels}\n\n"
    "Answer with the intent name only, nothing else."
)

_STRIP = re.compile(r"^[\s\"'`*\-.:]+|[\s\"'`*\-.:]+$")


def build_system_prompt(labels: Sequence[str]) -> str:
    """The instruction block listing the full intent inventory."""
    return SYSTEM_TEMPLATE.format(labels=", ".join(labels))


def build_messages(utterance: str, labels: Sequence[str]) -> list[dict[str, str]]:
    """Chat messages for one utterance, ready for `apply_chat_template`."""
    return [
        {"role": "system", "content": build_system_prompt(labels)},
        {"role": "user", "content": f'Utterance: "{utterance}"'},
    ]


def parse_label(raw: str, labels: Sequence[str]) -> str:
    """Map raw model output onto an intent name.

    Matching is deliberately strict: surrounding whitespace and punctuation are
    stripped, case is ignored, and an answer that opens with a valid label followed by
    commentary is accepted. Anything else is returned cleaned but unmatched, so that
    `compute_metrics` counts it as an invalid prediction rather than quietly guessing.
    """
    cleaned = _STRIP.sub("", raw).strip()
    label_set = set(labels)
    if cleaned in label_set:
        return cleaned

    lowered = cleaned.lower()
    by_lower = {label.lower(): label for label in labels}
    if lowered in by_lower:
        return by_lower[lowered]

    # "alarm_set — sets an alarm" and similar: accept the leading label, longest first
    # so that e.g. audio_volume_up is not shadowed by a shorter prefix.
    for label in sorted(labels, key=len, reverse=True):
        if lowered.startswith(label.lower()):
            return label

    return cleaned


def generate_labels(
    model: Any,
    tokenizer: Any,
    utterances: Sequence[str],
    labels: Sequence[str],
    *,
    batch_size: int = 16,
    max_new_tokens: int = 12,
    device: str = "cpu",
    progress_every: int = 20,
) -> list[str]:
    """Greedy-decode one intent name per utterance. Returns the raw model outputs."""
    import time

    import torch

    prompts = [
        tokenizer.apply_chat_template(
            build_messages(utterance, labels), tokenize=False, add_generation_prompt=True
        )
        for utterance in utterances
    ]

    outputs: list[str] = []
    n_batches = (len(prompts) + batch_size - 1) // batch_size
    start = time.time()

    for index in range(0, len(prompts), batch_size):
        batch = prompts[index:index + batch_size]
        encoded = tokenizer(batch, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        prompt_length = encoded["input_ids"].shape[1]
        for row in range(len(batch)):
            outputs.append(
                tokenizer.decode(generated[row][prompt_length:], skip_special_tokens=True)
            )

        batch_number = index // batch_size + 1
        if progress_every and batch_number % progress_every == 0:
            done = len(outputs)
            elapsed = time.time() - start
            rate = elapsed / done
            print(f"  {done}/{len(prompts)} utterances | {elapsed/60:.1f} min elapsed | "
                  f"~{rate*(len(prompts)-done)/60:.1f} min left")

    print(f"  done: {len(outputs)} utterances in {(time.time()-start)/60:.1f} min "
          f"({n_batches} batches)")
    return outputs


def prepare_tokenizer(tokenizer: Any) -> Any:
    """Decoder-only generation needs left padding and a pad token."""
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
