

from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data" / "sms_spam.tsv"
RESULTS_DIR = ROOT / "results"
MODEL_FILE = ROOT / "model.joblib"


def load_data(path: Path) -> pd.DataFrame:
    """Load the tab-separated dataset (columns: label, message)."""
    df = pd.read_csv(path, sep="\t", header=None, names=["label", "message"])

    df["target"] = (df["label"] == "spam").astype(int)
    return df


def build_model() -> Pipeline:
    """TF-IDF features -> Multinomial Naive Bayes, as a single pipeline."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=2,
                ),
            ),
            ("nb", MultinomialNB()),
        ]
    )


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    df = load_data(DATA_FILE)
    print(f"Loaded {len(df)} messages "
          f"({df['target'].sum()} spam, {(df['target'] == 0).sum()} legitimate)")

    X_train, X_test, y_train, y_test = train_test_split(
        df["message"],
        df["target"],
        test_size=0.20,
        random_state=42,
        stratify=df["target"],
    )

    model = build_model()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    report = classification_report(
        y_test, y_pred, target_names=["legitimate", "spam"]
    )
    print("\n" + report)

    report_path = RESULTS_DIR / "report.txt"
    report_path.write_text(
        "Phishing / Spam SMS Classifier — evaluation on held-out test set\n"
        f"Test set size: {len(y_test)} messages\n\n" + report
    )

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["legitimate", "spam"]
    )
    disp.plot(cmap="Blues", colorbar=False)
    plt.title("Confusion matrix — spam detection")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=130)
    print(f"Saved metrics to {report_path}")
    print(f"Saved confusion matrix to {RESULTS_DIR / 'confusion_matrix.png'}")

    joblib.dump(model, MODEL_FILE)
    print(f"Saved trained model to {MODEL_FILE}")


if __name__ == "__main__":
    main()
