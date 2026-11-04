from pathlib import Path

import joblib
from flask import Flask, render_template, request

ROOT = Path(__file__).resolve().parent
MODEL_FILE = ROOT / "model.joblib"
MAX_MESSAGE_LENGTH = 1000

app = Flask(__name__)


def load_model():
    if not MODEL_FILE.exists():
        raise RuntimeError("Model not found. Run 'python train.py' first.")
    return joblib.load(MODEL_FILE)


model = load_model()


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
            label_index = int(model.predict([message])[0])
            confidence = float(model.predict_proba([message])[0][label_index])
            result = {
                "label": "SPAM" if label_index == 1 else "Legitimate",
                "confidence": f"{confidence:.1%}",
                "is_spam": label_index == 1,
            }

    return render_template(
        "index.html",
        message=message,
        result=result,
        error=error,
    )


if __name__ == "__main__":
    app.run(debug=True)
