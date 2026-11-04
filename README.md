# Phishing / Spam SMS Classifier

An end-to-end machine-learning portfolio project that detects spam and
phishing-style ("smishing") SMS messages. The project demonstrates how to
transform raw text into machine-learning features, train a classification
pipeline, evaluate its performance, and use the saved model for new
predictions.

## Why this project matters

SMS phishing is a practical cybersecurity problem: a convincing message can
lead users to reveal information, send money, or open a malicious link. This
project applies natural language processing to build a simple, explainable
first layer of defense.

It showcases:

- Text preprocessing and TF-IDF feature extraction
- A Multinomial Naive Bayes classification model
- Stratified train/test evaluation
- Precision, recall, F1-score, and confusion-matrix reporting
- Model persistence with `joblib`
- A command-line prediction tool for real-world testing

## Results

The model was trained and evaluated on the public UCI SMS Spam Collection,
containing 5,572 labelled messages:

- **Accuracy:** 97%
- **Spam precision:** 1.00
- **Spam recall:** 0.78
- **Spam F1-score:** 0.88

These metrics are included for transparency. In a production system, recall
would be especially important because missed phishing messages can be costly.

## Project structure

```text
.
├── data/
│   └── sms_spam.tsv          # Training dataset
├── results/
│   ├── report.txt            # Evaluation metrics
│   └── confusion_matrix.png  # Model error visualization
├── predict.py                # Classify new SMS messages
├── app.py                    # Flask web application
├── templates/
│   └── index.html            # Web interface
├── static/
│   └── style.css             # Web interface styles
├── train.py                  # Train and evaluate the model
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

The trained model is saved as `model.joblib`. This generated artifact is
ignored by Git and can be recreated at any time by running `train.py`.

## Web interface

After training the model, start the Flask app:

```bash
python app.py
```

Open `http://127.0.0.1:5000` in a browser, enter an SMS message, and select
**Classify message** to view the prediction and confidence score.

## Future improvements

- Compare Naive Bayes with linear models and transformer-based approaches
- Tune the decision threshold to improve spam recall
- Add cross-validation and automated tests
- Build a small web interface or REST API
- Add explainability for words that influence each prediction

## Portfolio value

This project is designed to demonstrate practical machine-learning workflow
skills rather than only model training: reproducible setup, data preparation,
evaluation, artifact generation, and a usable prediction interface. It can be
extended into a larger cybersecurity or NLP application as new features are
added.
