"""
Trains a simple Random Forest classifier.
- Extracts features from windows of training segments. 
- Scales features and trains a Random Forest.
- Saves the trained model for later use.
- Provides a function to load the model from disk.
"""

from pathlib import Path
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix

from features import features
from windows import windows

from variables import MODEL_PATH

def window_features(segments):
    """
    Extract features from every window across a list of segments. This is just for training
    """
    x, y = [], []
    for segment in segments:
        for window, label in windows(segment):
            x.append(features(window))
            y.append(label)
    return np.array(x), np.array(y)


def train(train_segs):
    """
    Train a Random Forest classifier on the training segments. Saves and returns the model
    """
    print("Training classifier...")
    x_train, y_train = window_features(train_segs)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    RandomForestClassifier(n_estimators=100, n_jobs=-1)),
    ])

    pipeline.fit(x_train, y_train)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"Classifier model saved to {MODEL_PATH}")
    return pipeline


def load_model():
    """
    Loads an existing model from pkl file 
    """
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def evaluate(model, test_segs):
    """
    Evaluate the model on the test segments and report a confusion matrix
    Evaluation is at window level so might be optimistic!!
    """

    x_test, y_test = window_features(test_segs)

    y_pred = model.predict(x_test)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    print("\nConfusion Matrix:")
    print(f"  True Negatives  (correct non-seizure)  : {tn}")
    print(f"  False Positives (false alarm)          : {fp}")
    print(f"  False Negatives (missed seizure)       : {fn}")
    print(f"  True Positives  (correct seizure)      : {tp}")


def classifier(train_segs, test_segs):
    """
    Loads the model if it already exists. If not, trains a new one
    """
    if Path(MODEL_PATH).exists():
        print(f"Loading existing model from {MODEL_PATH}...")
        model = load_model()
    else:
        model = train(train_segs)

    evaluate(model, test_segs)

    return model
