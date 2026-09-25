"""The macro-F1 convention is the load-bearing decision in this comparison.

Every headline number depends on averaging over the labels present in the split rather
than all 60, so the arithmetic that relationship implies is pinned here.
"""

import pytest

from src.metrics import compute_metrics, confusion_pairs, per_class_scores


def test_macro_f1_averages_over_present_labels_only(labels):
    """`macro_f1_all_labels` must be `macro_f1` scaled by present/total.

    This is exactly the gap the README explains: an intent with no test utterances
    contributes a hard 0.0 to the all-labels average, so the two numbers differ by a
    factor of (present / total) and by nothing else.
    """
    present = [item for item in labels if item != "cooking_query"]
    y_true = present * 3
    y_pred = list(y_true)
    y_pred[0] = "alarm_set"       # one deliberate mistake, so F1 is not a flat 1.0

    scored = compute_metrics(y_true, y_pred, labels)

    assert scored["label_space"]["n_labels"] == len(labels)
    assert scored["label_space"]["n_labels_present"] == len(present)
    assert scored["label_space"]["absent_from_split"] == ["cooking_query"]
    assert scored["macro_f1_all_labels"] == pytest.approx(
        scored["macro_f1"] * len(present) / len(labels), abs=1e-4
    )


def test_perfect_predictions_score_one_over_present_labels(labels):
    present = [item for item in labels if item != "cooking_query"]
    scored = compute_metrics(present, list(present), labels)

    assert scored["accuracy"] == 1.0
    assert scored["macro_f1"] == 1.0
    # Still short of 1.0 overall, because the absent label is averaged in as zero.
    assert scored["macro_f1_all_labels"] < 1.0


def test_predictions_outside_the_inventory_are_counted_not_absorbed(labels):
    """A generated label that does not exist must be visible, never quietly dropped.

    Lenient handling here would turn a model failure into a parsing convenience and
    flatter the zero-shot baseline, which emitted 101 such strings on the real test set.
    """
    y_true = ["alarm_set", "play_music", "calendar_set"]
    y_pred = ["alarm_set", "transport_radio", "qa_math"]

    space = compute_metrics(y_true, y_pred, labels)["label_space"]

    assert space["n_invalid_predictions"] == 2
    assert {item["prediction"] for item in space["invalid_predictions"]} == {
        "transport_radio",
        "qa_math",
    }


def test_mismatched_lengths_are_rejected(labels):
    with pytest.raises(ValueError):
        compute_metrics(["alarm_set", "play_music"], ["alarm_set"], labels)


def test_confusion_pairs_rank_by_frequency_and_ignore_correct_answers():
    y_true = ["alarm_set"] * 3 + ["play_music"] * 2 + ["calendar_set"]
    y_pred = ["calendar_set"] * 3 + ["calendar_query"] * 2 + ["calendar_set"]

    pairs = confusion_pairs(y_true, y_pred, top_k=5)

    assert pairs[0] == {"gold": "alarm_set", "predicted": "calendar_set", "count": 3}
    assert pairs[1]["count"] == 2
    assert all(pair["gold"] != pair["predicted"] for pair in pairs)


def test_per_class_scores_lead_with_the_worst_intent(labels):
    y_true = ["alarm_set", "alarm_set", "play_music", "play_music"]
    y_pred = ["alarm_set", "alarm_set", "alarm_set", "alarm_set"]

    rows = per_class_scores(y_true, y_pred, ["alarm_set", "play_music"])

    assert rows[0]["intent"] == "play_music"
    assert rows[0]["f1"] == 0.0
    assert rows[-1]["intent"] == "alarm_set"
