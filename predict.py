import json
import sys
from pathlib import Path

import joblib
import pandas as pd

BASE_DIR = Path(__file__).parent
model = joblib.load(BASE_DIR / "banknote_model.joblib")
FEATURES = json.loads((BASE_DIR / "features.json").read_text())
LABELS = {0: "genuine", 1: "forged"}


def predict(values: dict) -> dict:
    missing = [f for f in FEATURES if f not in values]
    if missing:
        raise ValueError(f"Missing features: {missing}")
    row = pd.DataFrame([values])[FEATURES]
    p_forged = float(model.predict_proba(row)[0][1])
    return {
        "label": LABELS[int(p_forged >= 0.5)],
        "probability_forged": round(p_forged, 3),
    }


if __name__ == "__main__":
    numbers = [float(x) for x in sys.argv[1:]]
    print(predict(dict(zip(FEATURES, numbers))))