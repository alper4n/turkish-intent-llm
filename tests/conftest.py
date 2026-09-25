"""Shared fixtures.

These tests deliberately download nothing. They check the logic this repository wrote —
the label masking, the metric convention, the parser's strictness — not Hugging Face's.
A stub tokenizer stands in wherever one is needed, so the suite runs offline in about a
second and can gate every push.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# A small stand-in inventory. `cooking_query` is here for the same reason it matters in
# the real data: it is the label that can be absent from a split.
LABELS = [
    "alarm_query",
    "alarm_set",
    "audio_volume_mute",
    "audio_volume_up",
    "calendar_query",
    "calendar_set",
    "cooking_query",
    "play_music",
]


@pytest.fixture
def labels() -> list[str]:
    return list(LABELS)


class StubTokenizer:
    """Whitespace tokenizer with stable ids, enough for `src.sft` to exercise.

    Real tokenizers merge characters across a prompt/completion boundary, which is the
    bug `build_example` avoids by tokenizing the two halves separately. A stub cannot
    reproduce that merge, but it can prove the mask lines up with the completion — which
    is the property that has to hold either way.
    """

    eos_token_id = 99

    def __init__(self) -> None:
        self._ids: dict[str, int] = {}

    def _id(self, token: str) -> int:
        return self._ids.setdefault(token, len(self._ids) + 1)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "<s> " + " ".join(m["content"] for m in messages) + " <assistant>"

    def __call__(self, text, add_special_tokens=False, **kwargs):
        return {"input_ids": [self._id(token) for token in text.split()]}

    def decode(self, ids, **kwargs) -> str:
        names = {value: key for key, value in self._ids.items()}
        names[self.eos_token_id] = "<eos>"
        return " ".join(names[i] for i in ids)


@pytest.fixture
def tokenizer() -> StubTokenizer:
    return StubTokenizer()
