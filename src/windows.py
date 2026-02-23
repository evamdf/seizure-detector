import numpy as np

def get_windows(signal, window_size, step_size):
    """
    Generates numpy array of sliding windows (shape (window_size,)) from 1D signal 
    """
    n = len(signal)
    start = 0
    while start + window_size <= n:
        yield signal[start:start + window_size]
        start += step_size


def segment_windows(segment, window_size, step_size):
    """
    Breaks a segment into windows and pairs each window with the segment's label.
    Returns list of (window, label) tuples.
    """
    windows = []
    for window in get_windows(segment["signal"], window_size, step_size):
        windows.append((window, segment["label"]))
    return windows