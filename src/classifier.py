import numpy as np
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pickle

from features import extract_features
from windows import segment_windows

from variables import MODEL_PATH

def window_features(segments):
    """Extract features from every window across a list of segments."""
    X, y = [], []
    for segment in segments:
        for window, label in segment_windows(segment):
            X.append(extract_features(window))
            y.append(label)
    return np.array(X), np.array(y)

def train(train_segs):
    
    print("Extracting features...")
    X_train, y_train = window_features(train_segs)
 
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    RandomForestClassifier(n_estimators=100, n_jobs=-1)),
    ])

    pipeline.fit(X_train, y_train)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"Model saved to {MODEL_PATH}")
    return pipeline


def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)
