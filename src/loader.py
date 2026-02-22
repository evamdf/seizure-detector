import numpy as np 
from pathlib import Path 

SAMPLING_RATE = 173.61 # Hz

DATA_DIR = "../data/raw/"

SET_LABELS = {
    "F": 0,  # healthy, eyes open
    "N": 0,  # healthy, eyes closed
    "O": 0,  # interictal (between seizures)
    "S": 0,  # interictal (between seizures)
    "Z": 1,  # ictal (seizure)
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

def load_dataset(data_dir=DATA_DIR):
    """
    Load all segments from all sets.
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
    return dataset

