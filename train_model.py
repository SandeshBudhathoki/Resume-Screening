import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from model_utils import LogisticRegressionSoftmaxScratch


#  Load & Preprocess Dataset

df = pd.read_csv("resumes_dataset.csv")
df = df.dropna(subset=["Category", "Text"])
df = df.drop_duplicates(subset=["Text"])

X = df["Text"]
y = df["Category"]

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# TF-IDF
vectorizer = TfidfVectorizer(stop_words="english", max_features=2000)
X_train_tfidf = vectorizer.fit_transform(X_train).toarray()
X_test_tfidf = vectorizer.transform(X_test).toarray()


#  Custom Logistic Regression

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # stability
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

class LogisticRegressionSoftmax:
    def __init__(self, learning_rate=0.1, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        y_one_hot = np.zeros((n_samples, n_classes))
        y_one_hot[np.arange(n_samples), y] = 1

        self.weights = np.zeros((n_features, n_classes))
        self.bias = np.zeros((1, n_classes))

        for _ in range(self.n_iters):
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = softmax(linear_model)

            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y_one_hot))
            db = (1 / n_samples) * np.sum(y_pred - y_one_hot, axis=0, keepdims=True)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_pred = softmax(linear_model)
        return np.argmax(y_pred, axis=1)

    def predict_proba(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        return softmax(linear_model)

# Train model
model = LogisticRegressionSoftmax(learning_rate=0.1, n_iters=1000)
model.fit(X_train_tfidf, y_train)

# Evaluate
y_pred = model.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Save Model 

with open("resume_model.pkl", "wb") as f:
    pickle.dump(model, f)
# Vectorizer 
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
#Encoder
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print("\n Model, vectorizer, and encoder saved!")
