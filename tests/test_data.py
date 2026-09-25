"""Label ordering is shared by three experiments, so it has to be stable."""

import pandas as pd

from src.data import PARTITION_TO_SPLIT, SPLIT_ORDER, intent_labels, label_mappings


def test_label_ids_round_trip(labels):
    label2id, id2label = label_mappings(labels)

    assert len(label2id) == len(labels)
    assert all(id2label[label2id[name]] == name for name in labels)
    assert sorted(label2id.values()) == list(range(len(labels)))


def test_labels_come_back_sorted_so_ids_never_shift():
    """Ids are derived from sorted names; a different order would silently remap heads."""
    frame = pd.DataFrame({"intent": ["play_music", "alarm_set", "play_music", "calendar_set"]})

    assert intent_labels(frame) == ["alarm_set", "calendar_set", "play_music"]


def test_massive_dev_partition_is_exposed_as_validation():
    assert PARTITION_TO_SPLIT == {"train": "train", "dev": "validation", "test": "test"}
    assert SPLIT_ORDER == ("train", "validation", "test")
