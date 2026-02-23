import numpy as np 
from pathlib import Path 
from sklearn.model_selection import StratifiedShuffleSplit

SAMPLING_RATE = 173.61 # Hz

DATA_DIR = "../data/raw/"

TEST_SIZE = 0.2  # proportion of segments to use as test set

SET_LABELS = {
    "F": 0,  # healthy, eyes open
    "N": 0,  # healthy, eyes closed
    "O": 0,  # interictal (between seizures)
    "S": 1,  # interictal (between seizures)
    "Z": 0,  # ictal (seizure)
}

def load_segment(filepath):
    """Load a single .txt EEG segment"""
    samples = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            try:
                samples.append(float(line))
            except ValueError:
                continue
    return np.array(samples)

def split_segments(dataset, test_size=TEST_SIZE):
    """Split segments into train/test, preserving set proportions"""
    segments = np.array(dataset)
    set_names = np.array([s["set_name"] for s in dataset])

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size)
    train_idx, test_idx = next(splitter.split(segments, set_names))
    return segments[train_idx].tolist(), segments[test_idx].tolist()


def load_dataset(data_dir=DATA_DIR):
    """
    Load all segments from all sets, then split into train/test sets.
    Returns a list of dicts: {signal, label, set_name, segment_id}
    """
    data_dir = Path(data_dir)
    dataset = []

    for set_name, label in SET_LABELS.items():
        set_dir = data_dir / set_name
        if not set_dir.exists():
            print(f"Warning: {set_dir} not found, skipping.")
            continue
        for filepath in sorted(set_dir.glob("*.txt")) + sorted(set_dir.glob("*.TXT")):
            signal = load_segment(filepath)
            dataset.append({
                "signal": signal,
                "label": label,
                "set_name": set_name,
                "segment_id": filepath.stem,
            })

    print(f"Loaded {len(dataset)} segments.")

    print("Splitting segments...")
    train_segs, test_segs = split_segments(dataset)
    print(f"  {len(train_segs)} train segments, {len(test_segs)} test segments")

    return train_segs, test_segs
