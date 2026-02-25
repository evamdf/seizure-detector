"""
Trains a simple Random Forest classifier.
- Extracts features from windows of training segments. 
- Scales features and trains a Random Forest.
- Saves the trained model for later use.
- Provides a function to load the model from disk.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from pathlib import Path
import pickle

from features import features
from windows import windows

from variables import MODEL_PATH

def window_features(segments):
    """
    Extract features from every window across a list of segments. This is just for training
    """
    X, y = [], []
    for segment in segments:
        for window, label in windows(segment):
            X.append(features(window))
            y.append(label)
    return np.array(X), np.array(y)


def train(train_segs):
    """
    Train a Random Forest classifier on the training segments. Saves and returns the model
    """
    print("Training classifier...")
    X_train, y_train = window_features(train_segs)
 
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    RandomForestClassifier(n_estimators=100, n_jobs=-1)),
    ])

    pipeline.fit(X_train, y_train)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"Classifier model saved to {MODEL_PATH}")
    return pipeline


def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

def classifier(train_segs):
    if Path(MODEL_PATH).exists():
        print(f"Loading existing model from {MODEL_PATH}...")
        model = load_model()
    else:
        model = train(train_segs)
    return model