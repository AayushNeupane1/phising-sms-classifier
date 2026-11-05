from pathlib import Path

import joblib
from flask import Flask, render_template, request

ROOT = Path(__file__).resolve().parent
MODEL_FILE = ROOT / "model.joblib"
MAX_MESSAGE_LENGTH = 1000

app = Flask(__name__)


def load_model():
    """Return (pipeline, threshold, spam_index).

    Supports both the current bundle format (a dict with the tuned
    threshold) and an older plain-pipeline model, which falls back to 0.5.
    """
    if not MODEL_FILE.exists():
        raise RuntimeError("Model not found. Run 'python train.py' first.")

    obj = joblib.load(MODEL_FILE)
    if isinstance(obj, dict):
        return obj["pipeline"], obj["threshold"], obj.get("spam_index", 1)
    return obj, 0.5, 1


pipeline, THRESHOLD, SPAM_INDEX = load_model()


@app.route("/", methods=["GET", "POST"])
def index():
    message = ""
    result = None
    error = None

    if request.method == "POST":
        message = request.form.get("message", "").strip()

        if not message:
            error = "Enter a message to classify."
        elif len(message) > MAX_MESSAGE_LENGTH:
            error = f"Message must be {MAX_MESSAGE_LENGTH} characters or fewer."
        else:
            spam_score = float(pipeline.predict_proba([message])[0][SPAM_INDEX])
            is_spam = spam_score >= THRESHOLD
            result = {
                "label": "SPAM" if is_spam else "Legitimate",
                "spam_score": f"{spam_score:.1%}",
                "is_spam": is_spam,
            }

    return render_template(
        "index.html",
        message=message,
        result=result,
        error=error,
        threshold=f"{THRESHOLD:.0%}",
    )


if __name__ == "__main__":
    app.run(debug=True)
