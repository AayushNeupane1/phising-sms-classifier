# Phishing / Spam SMS Classifier

Detects spam and phishing-style ("smishing") SMS messages. It trains a
TF-IDF + Multinomial Naive Bayes pipeline on labelled messages, evaluates it
on a held-out test set, and exposes the saved model through a command-line
tool and a small Flask web page.

## Results

Trained and evaluated on the public UCI SMS Spam Collection (5,572 labelled
messages, 80/20 stratified split):

- Accuracy: 97%
- Spam precision: 1.00
- Spam recall: 0.78
- Spam F1-score: 0.88

Precision on spam is high (no legitimate messages flagged as spam), but recall
of 0.78 means roughly one spam message in five is missed. See `results/` for
the full report and confusion matrix.

## Project structure

```text
.
├── data/
│   └── sms_spam.tsv          # Training dataset
├── results/
│   ├── report.txt            # Evaluation metrics
│   └── confusion_matrix.png  # Model error visualization
├── train.py                  # Train and evaluate the model
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

Train the model and generate evaluation results:

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

The trained model is saved as `model.joblib`, which is git-ignored and can be
recreated at any time by running `train.py`.

## Web interface

After training, start the Flask app:

```bash
python app.py
```

Open `http://127.0.0.1:5000`, enter an SMS message, and select **Classify
message** to see the prediction and confidence score.

## Possible next steps

- Tune the decision threshold to raise spam recall
- Compare Naive Bayes with linear models (e.g. logistic regression, linear SVM)
- Add cross-validation and automated tests
- Show which words pushed a given message toward spam or legitimate
