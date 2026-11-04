"""
Phishing / Spam SMS Classifier — prediction tool.

Loads the trained model and classifies a message as spam or legitimate,
with a confidence score.

Usage:
    python predict.py "Congratulations! You've won a $1000 gift card. Click here to claim"
    python predict.py            # then type messages interactively
"""

import sys
from pathlib import Path

import joblib

MODEL_FILE = Path(__file__).resolve().parent / "model.joblib"


def load_model():
    if not MODEL_FILE.exists():
        sys.exit("Model not found. Run 'python train.py' first.")
    return joblib.load(MODEL_FILE)


def classify(model, message: str) -> None:
    label_idx = model.predict([message])[0]
    confidence = model.predict_proba([message])[0][label_idx]
    label = "SPAM" if label_idx == 1 else "legitimate"
    print(f"  [{label}]  (confidence: {confidence:.1%})  {message}")


def main() -> None:
    model = load_model()

    if len(sys.argv) > 1:
        # Classify the message passed on the command line.
        classify(model, " ".join(sys.argv[1:]))
        return

    # Interactive mode.
    print("Type a message to classify (blank line to quit):")
    while True:
        try:
            message = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not message:
            break
        classify(model, message)


if __name__ == "__main__":
    main()
