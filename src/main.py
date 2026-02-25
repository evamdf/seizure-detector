"""
Entry point for the EEG seizure detection + MIDI generation project.

Usage:
    python main.py                      # run everything
    python main.py --part model         # train/load model
    python main.py --part midi          # generate the MIDI files only
    python main.py --part stream-demo   # stream demo only (this will create and train model as well if it isn't found)
"""

import argparse

from loader import loader
from classifier import classifier
from streamer import streamer_demo
from midi.midi import midi

def main():
    parser = argparse.ArgumentParser(description="EEG seizure detection + MIDI generation")
    parser.add_argument(
        "--part",
        choices=["model", "stream-demo", "midi", "all"],
        default="all",
    )

    args = parser.parse_args()

    print("Loading dataset...")
    train_segs, test_segs = loader()
    
    if not train_segs:
        print("No training data found. Check DATA_DIR in variables.py.")
        return

    if args.part in ("model", "stream-demo", "all"):
        
        model = classifier(train_segs)

        if args.part in ("stream-demo", "all"):
            streamer_demo(test_segs, model)

    if args.part in ("midi", "all"):
        midi(train_segs)


if __name__ == "__main__":
    main()