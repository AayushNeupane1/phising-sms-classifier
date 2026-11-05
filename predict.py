"""
Phishing / Spam SMS Classifier - prediction tool.

Loads the trained model and classifies a message as spam or legitimate,
using the tuned decision threshold saved during training.

Usage:
    python predict.py "Congratulations! You've won a $1000 gift card. Click here to claim"
    python predict.py
"""

import sys
from pathlib import Path

import joblib

MODEL_FILE = Path(__file__).resolve().parent / "model.joblib"


def load_model():
    """Return (pipeline, threshold, spam_index).

    Supports both the current bundle format (a dict with the tuned
    threshold) and an older plain-pipeline model, which falls back to 0.5.
    """
    if not MODEL_FILE.exists():
        sys.exit("Model not found. Run 'python train.py' first.")

    obj = joblib.load(MODEL_FILE)
    if isinstance(obj, dict):
        return obj["pipeline"], obj["threshold"], obj.get("spam_index", 1)
    return obj, 0.5, 1


def classify(pipeline, threshold, spam_index, message: str) -> None:
    spam_score = pipeline.predict_proba([message])[0][spam_index]
    label = "SPAM" if spam_score >= threshold else "legitimate"
    print(f"  [{label}]  spam score: {spam_score:.1%}  "
          f"(flagged at >= {threshold:.0%})  {message}")


def main() -> None:
    pipeline, threshold, spam_index = load_model()

    if len(sys.argv) > 1:
        classify(pipeline, threshold, spam_index, " ".join(sys.argv[1:]))
        return

    print("Type a message to classify (blank line to quit):")
    while True:
        try:
            message = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not message:
            break
        classify(pipeline, threshold, spam_index, message)


if __name__ == "__main__":
    main()
