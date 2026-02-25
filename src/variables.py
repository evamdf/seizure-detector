SIMULATED_SPEED = 5.0  # How many times faster than real-time to simulate
WINDOW_SIZE = 173   # 1 second of EEG data at 173.61 Hz
STEP_SIZE   = 87    # 50% overlap - window moves 0.5 seconds at a time 
SAMPLING_RATE = 173.61 # Hz
DATA_DIR = "../data/raw/"
MODEL_PATH = "../model.pkl"
MIDI_OUTPUT_DIR = "../midi-output" # Where to store generated MIDI files 
ALERT_THRESHOLD = 6 # Alert seizure after this many consecutive windows predicted as seizure (6 is 3 seconds)
TEST_SIZE = 0.2  # Proportion of segments to use as test set

SET_LABELS = {
    "F": 0,  
    "N": 0,  
    "O": 0,  
    "S": 1,  
    "Z": 0,  
}

BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta":  (13, 30),
    "gamma": (30, 60),
}



