import numpy as np
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pickle

from shared.features import extract_features
from shared.loader import load_dataset
from shared.windows import segment_windows

WINDOW_SIZE = 173   # ~1 second at 173.61 Hz
STEP_SIZE   = 87    # 50% overlap


def window_features(segments):
    """Extract features from every window across a list of segments."""
    X, y = [], []
    for segment in segments:
        for window, label in segment_windows(segment, WINDOW_SIZE, STEP_SIZE):
            X.append(extract_features(window))
            y.append(label)
    return np.array(X), np.array(y)

def train(train_segs, model_path="model.pkl"):
    
    print("Extracting features...")
    X_train, y_train = window_features(train_segs)
 
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    RandomForestClassifier(n_estimators=100, n_jobs=-1)),
    ])

    pipeline.fit(X_train, y_train)

    with open(model_path, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"Model saved to {model_path}")
    return pipeline


def load_model(model_path="model.pkl"):
    with open(model_path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    dataset = load_dataset()
    train(dataset)