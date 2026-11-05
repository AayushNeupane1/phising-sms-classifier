
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from train import DATA_FILE, RESULTS_DIR, load_data


def tfidf() -> TfidfVectorizer:
    """Shared TF-IDF settings so models differ only in the classifier."""
    return TfidfVectorizer(
        lowercase=True, stop_words="english", ngram_range=(1, 2), min_df=2
    )


def candidates() -> dict:
    return {
        "Multinomial NB": Pipeline(
            [("tfidf", tfidf()), ("clf", MultinomialNB())]
        ),
        "Logistic Regression": Pipeline(
            [
                ("tfidf", tfidf()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=1000, class_weight="balanced"
                    ),
                ),
            ]
        ),
    }


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    df = load_data(DATA_FILE)
    X_train, _X_test, y_train, _y_test = train_test_split(
        df["message"],
        df["target"],
        test_size=0.20,
        random_state=42,
        stratify=df["target"],
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # precision/recall/f1 default to the positive class (spam == 1).
    scoring = ["accuracy", "precision", "recall", "f1"]

    header = (
        f"{'Model':<22}{'accuracy':>16}{'precision':>16}"
        f"{'recall':>16}{'f1':>16}"
    )
    lines = [
        "Model comparison - stratified 5-fold CV on the training split",
        "Spam is the positive class; values are mean +/- std across folds.",
        "",
        header,
        "-" * len(header),
    ]

    for name, model in candidates().items():
        res = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring)
        cells = [f"{name:<22}"]
        for metric in ["accuracy", "precision", "recall", "f1"]:
            scores = res[f"test_{metric}"]
            cells.append(f"{scores.mean():.3f}+/-{scores.std():.3f}".rjust(16))
        lines.append("".join(cells))

    table = "\n".join(lines)
    print(table)

    out = RESULTS_DIR / "model_comparison.txt"
    out.write_text(table + "\n", encoding="utf-8")
    print(f"\nSaved comparison to {out}")


if __name__ == "__main__":
    main()
