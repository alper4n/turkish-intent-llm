"""Evaluation metrics shared by every experiment in this repository.

All three experiments are scored the same way so their numbers are directly
comparable, and each result file records which convention produced them.

**Macro-F1 convention.** The label space is always the full inventory of 60 intents —
models may predict any of them. Macro-F1, however, is averaged only over the intents
that actually occur in the evaluation split. Turkish MASSIVE has no `cooking_query`
utterances in test, so including it would contribute a hard 0.0 to the average and
depress the score by ~1/60 for reasons unrelated to the model. Both numbers are
reported: `macro_f1` (present labels) and `macro_f1_all_labels` (all 60).
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Sequence

from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support


def confusion_pairs(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    top_k: int = 15,
) -> list[dict[str, Any]]:
    """Most frequent (gold, predicted) mistakes, ordered by count."""
    mistakes = Counter(
        (gold, pred) for gold, pred in zip(y_true, y_pred) if gold != pred
    )
    return [
        {"gold": gold, "predicted": pred, "count": int(count)}
        for (gold, pred), count in mistakes.most_common(top_k)
    ]


def per_class_scores(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str],
) -> list[dict[str, Any]]:
    """Precision / recall / F1 / support for each label, worst F1 first."""
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(labels), zero_division=0
    )
    rows = [
        {
            "intent": label,
            "precision": round(float(p), 4),
            "recall": round(float(r), 4),
            "f1": round(float(f), 4),
            "support": int(s),
        }
        for label, p, r, f, s in zip(labels, precision, recall, f1, support)
    ]
    return sorted(rows, key=lambda row: (row["f1"], -row["support"]))


def compute_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str],
    *,
    top_k_confusions: int = 15,
) -> dict[str, Any]:
    """Score one set of predictions.

    `labels` is the full 60-intent inventory. Returns headline metrics, the
    per-class breakdown and the most common confusions.
    """
    y_true = list(y_true)
    y_pred = list(y_pred)
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Prediction count {len(y_pred)} does not match gold count {len(y_true)}."
        )

    labels = list(labels)
    present = [label for label in labels if label in set(y_true)]
    absent = [label for label in labels if label not in set(y_true)]

    # Predictions outside the known inventory only happen for generative models that
    # ignore the instructions; they are always wrong, and we count them explicitly.
    invalid = [pred for pred in y_pred if pred not in set(labels)]

    return {
        "n_examples": len(y_true),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(
            float(f1_score(y_true, y_pred, labels=present, average="macro",
                           zero_division=0)), 4
        ),
        "macro_f1_all_labels": round(
            float(f1_score(y_true, y_pred, labels=labels, average="macro",
                           zero_division=0)), 4
        ),
        "weighted_f1": round(
            float(f1_score(y_true, y_pred, labels=present, average="weighted",
                           zero_division=0)), 4
        ),
        "label_space": {
            "n_labels": len(labels),
            "n_labels_present": len(present),
            "absent_from_split": absent,
            "n_invalid_predictions": len(invalid),
            "invalid_predictions": [
                {"prediction": pred, "count": int(count)}
                for pred, count in Counter(invalid).most_common(10)
            ],
        },
        "per_class": per_class_scores(y_true, y_pred, present),
        "top_confusions": confusion_pairs(y_true, y_pred, top_k_confusions),
    }


def hf_metrics_fn(labels: Sequence[str]):
    """Build a `compute_metrics` callback for `transformers.Trainer`.

    The Trainer hands us integer ids, so we map them back to intent names and reuse
    the same scoring code as every other experiment.
    """
    labels = list(labels)

    def _fn(eval_pred):
        import numpy as np

        logits, gold_ids = eval_pred
        pred_ids = np.asarray(logits).argmax(axis=-1)
        scored = compute_metrics(
            [labels[i] for i in gold_ids],
            [labels[i] for i in pred_ids],
            labels,
        )
        # Trainer only wants flat scalars during training.
        return {
            "accuracy": scored["accuracy"],
            "macro_f1": scored["macro_f1"],
            "macro_f1_all_labels": scored["macro_f1_all_labels"],
        }

    return _fn
