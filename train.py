import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
)
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data" / "sms_spam.tsv"
RESULTS_DIR = ROOT / "results"
MODEL_FILE = ROOT / "model.joblib"

SPAM_INDEX = 1


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


def tune_threshold(model: Pipeline, X_train, y_train):
    """Choose the spam probability cutoff that maximises F1.

    The cutoff is selected on 5-fold out-of-fold predictions over the
    training set, so it is never fitted on the held-out test data.
    Returns the chosen threshold and the precision/recall curve for plotting.
    """
    oof_proba = cross_val_predict(
        model, X_train, y_train, cv=5, method="predict_proba"
    )[:, SPAM_INDEX]

    precision, recall, thresholds = precision_recall_curve(y_train, oof_proba)
    # precision/recall are one longer than thresholds; drop the trailing point.
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    best = int(np.nanargmax(f1[:-1]))
    return float(thresholds[best]), (precision, recall, thresholds, best)


def spam_metrics(y_true, proba, threshold: float) -> dict:
    """Precision, recall and F1 for the spam class at a given threshold."""
    from sklearn.metrics import f1_score, precision_score, recall_score

    y_pred = (proba >= threshold).astype(int)
    return {
        "threshold": threshold,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def plot_pr_curve(precision, recall, best, path: Path) -> None:
    chosen_p, chosen_r = precision[best], recall[best]
    plt.figure()
    plt.plot(recall, precision, label="Naive Bayes")
    plt.scatter(
        [chosen_r], [chosen_p], color="red", zorder=5,
        label=f"chosen threshold (recall={chosen_r:.2f}, precision={chosen_p:.2f})",
    )
    plt.xlabel("Recall (spam)")
    plt.ylabel("Precision (spam)")
    plt.title("Precision-recall curve (spam class)")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


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

    # Pick the decision threshold on training data (out-of-fold), then fit
    # the final model on all of the training data.
    threshold, (precision, recall, _thr, best) = tune_threshold(
        model, X_train, y_train
    )
    model.fit(X_train, y_train)

    print(f"\nChosen spam threshold: {threshold:.3f} "
          f"(default is 0.500)")

    test_proba = model.predict_proba(X_test)[:, SPAM_INDEX]

    default = spam_metrics(y_test, test_proba, 0.50)
    tuned = spam_metrics(y_test, test_proba, threshold)

    print("\nSpam-class metrics on the held-out test set:")
    print(f"              precision  recall    f1")
    print(f"  default      {default['precision']:.2f}      "
          f"{default['recall']:.2f}    {default['f1']:.2f}")
    print(f"  tuned        {tuned['precision']:.2f}      "
          f"{tuned['recall']:.2f}    {tuned['f1']:.2f}")

    # Full per-class report at the tuned threshold.
    y_pred_tuned = (test_proba >= threshold).astype(int)
    report = classification_report(
        y_test, y_pred_tuned, target_names=["legitimate", "spam"]
    )
    print("\n" + report)

    # Threshold sweep for the report file.
    sweep_lines = ["thresh  precision  recall    f1"]
    for t in [round(x, 2) for x in np.arange(0.10, 0.95, 0.05)]:
        m = spam_metrics(y_test, test_proba, t)
        sweep_lines.append(
            f"{t:.2f}     {m['precision']:.2f}      {m['recall']:.2f}    {m['f1']:.2f}"
        )

    report_path = RESULTS_DIR / "report.txt"
    report_path.write_text(
        "Phishing / Spam SMS Classifier - evaluation on held-out test set\n"
        f"Test set size: {len(y_test)} messages\n"
        f"Chosen spam threshold: {threshold:.3f} "
        "(maximises F1 on 5-fold out-of-fold training predictions)\n\n"
        "Spam-class metrics (default 0.50 vs tuned):\n"
        f"              precision  recall    f1\n"
        f"  default      {default['precision']:.2f}      "
        f"{default['recall']:.2f}    {default['f1']:.2f}\n"
        f"  tuned        {tuned['precision']:.2f}      "
        f"{tuned['recall']:.2f}    {tuned['f1']:.2f}\n\n"
        "Full per-class report at the tuned threshold:\n" + report + "\n"
        "Threshold sweep on the test set (spam class):\n"
        + "\n".join(sweep_lines) + "\n",
        encoding="utf-8",
    )

    plot_pr_curve(precision, recall, best, RESULTS_DIR / "pr_curve.png")

    cm = confusion_matrix(y_test, y_pred_tuned)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["legitimate", "spam"]
    )
    disp.plot(cmap="Blues", colorbar=False)
    plt.title(f"Confusion matrix (spam detection, threshold={threshold:.2f})")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=130)
    plt.close()

    # Save the pipeline together with its tuned threshold so the serving
    # code applies the same cutoff used at evaluation time.
    joblib.dump(
        {"pipeline": model, "threshold": threshold, "spam_index": SPAM_INDEX},
        MODEL_FILE,
    )

    print(f"\nSaved metrics to {report_path}")
    print(f"Saved precision-recall curve to {RESULTS_DIR / 'pr_curve.png'}")
    print(f"Saved confusion matrix to {RESULTS_DIR / 'confusion_matrix.png'}")
    print(f"Saved trained model to {MODEL_FILE}")


if __name__ == "__main__":
    main()
