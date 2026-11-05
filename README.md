# Phishing / Spam SMS Classifier

Detects spam and phishing-style ("smishing") SMS messages. It trains a
TF-IDF + Multinomial Naive Bayes pipeline on labelled messages, tunes the
decision threshold, evaluates on a held-out test set, and exposes the saved
model through a command-line tool and a small Flask web page.

## Results

Trained and evaluated on the public UCI SMS Spam Collection (5,572 labelled
messages, 80/20 stratified split). Because missing a phishing message is worse
than occasionally over-flagging, the decision threshold is tuned for recall
rather than left at the default 0.5.

Spam-class metrics on the held-out test set:

| Threshold      | Precision | Recall | F1   |
| -------------- | --------- | ------ | ---- |
| Default (0.50) | 1.00      | 0.78   | 0.88 |
| Tuned (0.20)   | 0.97      | 0.90   | 0.93 |

Tuning lifts spam recall from 0.78 to 0.90 (about one in five missed spam down
to one in ten) for a 0.03 drop in precision. Overall accuracy is 98%. See
`results/report.txt` for the full per-class report and threshold sweep, and
`results/pr_curve.png` for the precision-recall curve.

### How the threshold is chosen

The cutoff is selected on 5-fold out-of-fold predictions over the *training*
set (the value that maximises spam F1), then applied unchanged to the held-out
test set. Tuning on cross-validated training folds rather than on the test set
keeps the reported metrics honest.

## Known limitations

The model is trained on short SMS text, so it generalises poorly to other
formats. A promotional *email* ("you missed yesterday's session, join us
live...") scores only ~3% spam and is passed as legitimate: its wording
overlaps with the legitimate SMS it learned from, and none of the SMS spam
cues are present. Catching email-style spam would require retraining on an
email corpus, not just a threshold change.

## Project structure

```text
.
├── data/
│   └── sms_spam.tsv          # Training dataset
├── results/
│   ├── report.txt            # Metrics + threshold sweep
│   ├── pr_curve.png          # Precision-recall curve (spam class)
│   └── confusion_matrix.png  # Errors at the tuned threshold
├── train.py                  # Train, tune threshold, and evaluate
├── predict.py                # Classify new SMS messages (CLI)
├── app.py                    # Flask web application
├── templates/
│   └── index.html            # Web interface
├── static/
│   └── style.css             # Web interface styles
├── requirements.txt          # Python dependencies
└── README.md
```

## Getting started

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Train the model, tune the threshold, and generate evaluation results:

```bash
python train.py
```

Classify a message from the command line:

```bash
python predict.py "Congratulations! You won a prize. Click here to claim it."
```

Or start interactive mode:

```bash
python predict.py
```

The trained model is saved as `model.joblib` (the pipeline plus its tuned
threshold). It is git-ignored and can be recreated at any time by running
`train.py`.

## Web interface

After training, start the Flask app:

```bash
python app.py
```

Open `http://127.0.0.1:5000`, enter an SMS message, and select **Classify
message** to see the prediction and spam score.

## Possible next steps

- Compare Naive Bayes with linear models (e.g. logistic regression, linear SVM)
  under cross-validation
- Add a sentence-embedding baseline and compare accuracy, latency, and size
- Retrain on an email corpus to handle email-style spam
- Show which words pushed a given message toward spam or legitimate
- Add automated tests and a CI workflow
