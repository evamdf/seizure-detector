"""
Functions for breaking EEG segments into sliding windows and pairing each window with the segment's label.
"""

from variables import WINDOW_SIZE, STEP_SIZE

def get_windows(signal):
    """
    Generates numpy array of sliding windows (shape (window_size,)) from 1D signal 
    """
    n = len(signal)
    start = 0
    while start + WINDOW_SIZE <= n:
        yield signal[start:start + WINDOW_SIZE]
        start += STEP_SIZE


def windows(segment):
    """
    Breaks a segment into windows and pairs each window with the segment's label.
    Returns list of (window, label) tuples.
    """
    windows = []
    for window in get_windows(segment["signal"]):
        windows.append((window, segment["label"]))
    return windows