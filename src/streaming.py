"""
Simulates real-time EEG streaming for a single segment. (Doesn't actually stream data).
Processes windows at the real-world rate and prints a seizure alert
when ALERT_THRESHOLD consecutive seizure predictions occur in a row.
"""

import time

from features import extract_features
from windows import segment_windows

from variables import SIMULATED_SPEED, SAMPLING_RATE, STEP_SIZE, ALERT_THRESHOLD

def stream_segment(segment, model):
    """
    Stream a single EEG segment window-by-window, predicting and alerting.
    """
    
    sleep_time = (STEP_SIZE / SAMPLING_RATE) / SIMULATED_SPEED

    true_label = segment["label"]
    label_str  = "SEIZURE" if true_label == 1 else "normal"
    print(f"\n{'='*50}")
    print(f"Streaming segment: {segment['segment_id']} (set={segment['set_name']}, true={label_str})")
    print(f"{'='*50}")

    consecutive = 0
    windows = list(segment_windows(segment))

    for i, (window, _) in enumerate(windows):
        features = extract_features(window).reshape(1, -1)
        prob       = model.predict_proba(features)[0][1]   # P(seizure)
        prediction = int(prob >= 0.5)

        if prediction == 1:
            consecutive += 1
        else:
            consecutive = 0

        status = "⚡ SEIZURE" if prediction == 1 else "  normal "
        print(f"  Window {i+1:3d}/{len(windows)} | {status} | P={prob:.2f} |"
              f" consecutive={consecutive}")

        if consecutive >= ALERT_THRESHOLD:
            print(f"\n  🚨  SEIZURE ALERT — {consecutive} consecutive detections  🚨\n")
            consecutive = 0  # reset counter after alert

        time.sleep(sleep_time)

    print(f"\nDone. True label: {label_str}")