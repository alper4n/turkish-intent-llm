"""Turkish orthography has one trap that generic string handling gets wrong."""

import pytest

from src.robustness import ascii_fold, fold_one, lowercase, single_letter_probes


def test_folding_strips_every_turkish_diacritic():
    assert ascii_fold("çğıöşü ÇĞİÖŞÜ") == "cgiosu CGIOSU"


def test_folding_one_letter_leaves_the_others_alone():
    text = "yarın sabah çığlık şarkısı"

    assert fold_one(text, "ı") == "yarin sabah çiğlik şarkisi"
    assert fold_one(text, "ş") == "yarın sabah çığlık sarkısı"
    assert fold_one(text, "ç") == "yarın sabah cığlık şarkısı"


def test_only_turkish_specific_letters_can_be_folded():
    with pytest.raises(ValueError):
        fold_one("herhangi bir metin", "x")


def test_lowercasing_respects_the_dotted_and_dotless_i():
    """`str.lower()` is wrong for Turkish: it turns I into i and İ into i-plus-a-dot."""
    assert lowercase("IĞDIR") == "ığdır"
    assert lowercase("İSTANBUL") == "istanbul"
    assert lowercase("IĞDIR") != "IĞDIR".lower()


def test_every_probe_letter_is_foldable():
    for letter in single_letter_probes():
        assert fold_one(letter, letter) != letter
