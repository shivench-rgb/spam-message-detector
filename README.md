# Spam Message Detector

A machine learning project that classifies SMS messages as **Spam** or **Not Spam** using Natural Language Processing.

## Features

* Takes an SMS/message as input
* Converts text into numerical features using TF-IDF
* Uses Multinomial Naive Bayes for classification
* Predicts whether the message is Spam or Not Spam
* Displays model accuracy

## Technologies Used

* Python
* Pandas
* Scikit-learn
* NLP
* TF-IDF
* Multinomial Naive Bayes

## How It Works

```text
SMS Message
     ↓
Text Data
     ↓
TF-IDF Vectorization
     ↓
Multinomial Naive Bayes
     ↓
Prediction
     ↓
Spam / Not Spam
```

## Installation

```bash
pip install -r requirements.txt
```

## Run the Project

```bash
python spam_detector.py
```

## Example

```text
Enter a message: Congratulations! You won a free prize
Prediction: SPAM
```

```text
Enter a message: Are you coming to college tomorrow?
Prediction: HAM
```

## Future Improvements

* Use a larger real-world SMS dataset
* Add text preprocessing
* Compare multiple machine learning algorithms
* Add a Streamlit web interface
* Improve model accuracy

## Author

Shiven Chaware

BCA Student | Python | AI/ML
