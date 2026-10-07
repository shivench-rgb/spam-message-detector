import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

data = {
    "message": [
        "Congratulations! You won a lottery of 10000 rupees",
        "Win a free iPhone now, click this link",
        "Claim your prize immediately",
        "You have won a cash reward",
        "Get free recharge by clicking this link",
        "Urgent! You won a free gift",
        "Congratulations, you are selected for a prize",
        "Free entry in a lucky draw",
        "Win money now by entering your details",
        "You have received a free voucher",
        "Hey, are you coming to college tomorrow?",
        "Please send me the notes",
        "What time is the class today?",
        "I will call you after college",
        "Can you share the assignment?",
        "Let's meet tomorrow morning",
        "Your exam is scheduled for Monday",
        "Please bring your practical file",
        "I am going home now",
        "See you tomorrow"
    ],
    "label": [
        "spam", "spam", "spam", "spam", "spam",
        "spam", "spam", "spam", "spam", "spam",
        "ham", "ham", "ham", "ham", "ham",
        "ham", "ham", "ham", "ham", "ham"
    ]
}

df = pd.DataFrame(data)

X_train, X_test, y_train, y_test = train_test_split(
    df["message"], df["label"], test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)

print("Accuracy:", accuracy_score(y_test, y_pred))

message = input("Enter a message: ")

message_tfidf = vectorizer.transform([message])
prediction = model.predict(message_tfidf)

print("Prediction:", prediction[0].upper())