"""Orthographic perturbations for probing Turkish robustness.

MASSIVE's Turkish was produced by paid translators, so it is spelled consistently: every
diacritic is where it belongs (notebook 01). Real typed Turkish is not like that — `yarin`
for `yarın` is ordinary on a phone, and ASR output is not always consistent either. A test
score on this corpus therefore says nothing about how a model handles input it will meet in
production, which is why we perturb the test set deliberately instead of assuming coverage.
"""

from __future__ import annotations

from typing import Iterable

# Turkish-specific letters and their bare-ASCII stand-ins. Note the two `i`s: dotless ı
# maps to i, and dotted İ maps to I — the pair that ASCII has no room for.
FOLD_PAIRS: tuple[tuple[str, str], ...] = (
    ("ç", "c"), ("ğ", "g"), ("ı", "i"), ("ö", "o"), ("ş", "s"), ("ü", "u"),
    ("Ç", "C"), ("Ğ", "G"), ("İ", "I"), ("Ö", "O"), ("Ş", "S"), ("Ü", "U"),
)

_FULL_TABLE = str.maketrans(
    "".join(src for src, _ in FOLD_PAIRS),
    "".join(dst for _, dst in FOLD_PAIRS),
)


def ascii_fold(text: str) -> str:
    """Strip every Turkish diacritic: `yarın sabah` -> `yarin sabah`."""
    return text.translate(_FULL_TABLE)


def fold_one(text: str, letter: str) -> str:
    """Fold a single Turkish letter, leaving the others intact.

    Folding one letter at a time is what separates "diacritics matter" from *which*
    diacritic matters — the letters are not interchangeable in how much work they do.
    """
    pairs = [(src, dst) for src, dst in FOLD_PAIRS if src.lower() == letter.lower()]
    if not pairs:
        raise ValueError(f"{letter!r} is not a Turkish-specific letter.")
    table = str.maketrans("".join(s for s, _ in pairs), "".join(d for _, d in pairs))
    return text.translate(table)


def lowercase(text: str) -> str:
    """Turkish-aware lowercasing: I -> ı, İ -> i, unlike `str.lower()`."""
    return text.replace("I", "ı").replace("İ", "i").lower()


def single_letter_probes() -> Iterable[str]:
    """The six lowercase Turkish letters, ordered by how often they occur in the corpus."""
    return ("ı", "ş", "ç", "ü", "ö", "ğ")
