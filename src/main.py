"""
main.py
Entry point for the EEG seizure detection + MIDI generation project.

Usage:
    python main.py                   # run everything
    python main.py --part classifier # train model
    python main.py --part midi       # generate the MIDI files only
    python main.py --part stream     # stream demo only (this will create and train model as well if it isn't found)
"""

import argparse
import random
from pathlib import Path

from loader import load_dataset
from classifier import train, load_model
from streaming import stream_segment
from midi.midi import midi

from variables import MODEL_PATH

def run_classifier(train_segs):
    print("\n" + "=" * 50)
    print("PART 1 — Seizure Classifier")
    print("=" * 50)
    model = train(train_segs)
    return model


def run_streaming(test_segs, model):
    print("\n" + "=" * 50)
    print("PART 1 — Simulated Real-Time Streaming Demo")
    print("=" * 50)
    seizure_segs = [s for s in test_segs if s["label"] == 1]
    normal_segs  = [s for s in test_segs if s["label"] == 0]

    print("\n--- Normal segment demo ---")
    stream_segment(random.choice(normal_segs), model)

    print("\n--- Seizure segment demo ---")
    stream_segment(random.choice(seizure_segs), model)


def run_midi(train_segs):
    print("\n" + "=" * 50)
    print("PART 2 — EEG-to-MIDI Generation")
    print("=" * 50)
    midi(train_segs)


def main():
    parser = argparse.ArgumentParser(description="EEG seizure detection + MIDI generation")
    parser.add_argument(
        "--part",
        choices=["classifier", "stream", "midi", "all"],
        default="all",
    )

    args = parser.parse_args()

    train_segs, test_segs = load_dataset()
    
    if not train_segs:
        print("No training data found. Check DATA_DIR in variables.py.")
        return

    if args.part in ("classifier", "stream", "all"):
        if Path(MODEL_PATH).exists():
            print(f"Loading existing model from {MODEL_PATH}...")
            model = load_model()
        else:
            model = run_classifier(train_segs)

        if args.part in ("stream", "all"):
            run_streaming(test_segs, model)

    if args.part in ("midi", "all"):
        run_midi(train_segs)


if __name__ == "__main__":
    main()