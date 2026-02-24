import numpy as np 
from pathlib import Path 
from sklearn.model_selection import StratifiedShuffleSplit

from variables import DATA_DIR, TEST_SIZE, SET_LABELS

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

def split_segments(dataset):
    """Split segments into train/test, preserving set proportions"""
    segments = np.array(dataset)
    set_names = np.array([s["set_name"] for s in dataset])

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=2)
    train_idx, test_idx = next(splitter.split(segments, set_names))
    return segments[train_idx].tolist(), segments[test_idx].tolist()


def load_dataset():
    """
    Load all segments from all sets, then split into train/test sets.
    Returns a list of dicts: {signal, label, set_name, segment_id}
    """
    data_dir = Path(DATA_DIR)
    dataset = []

    for set_name, label in SET_LABELS.items():
        set_dir = data_dir / set_name
        if not set_dir.exists():
            print(f"Warning: {set_dir} not found, skipping.")
            continue
        for filepath in sorted(set_dir.glob("*.txt")) + sorted(set_dir.glob("*.TXT")): # For some reason just the files in the N folder are TXT?
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
