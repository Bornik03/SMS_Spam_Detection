import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
# Loading the custom Kaggle dataset
df = pd.read_csv("C:\\Users\\borni\\Downloads\\Compressed\\archive_2\\spam.csv", encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'message']
# Encoding labels
df['label_num'] = df.label.map({'ham': 0, 'spam': 1})
# Spliting into features and labels
X = df['message']
y = df['label_num']
# Vectorizing the text
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)
# Training Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train, y_train)
# Predicting and evaluating
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
# Function to classify new messages, not in use but can be used for testing
def predict_sms(message):
    msg_vector = vectorizer.transform([message])
    prediction = model.predict(msg_vector)[0]
    return "Spam" if prediction == 1 else "Not Spam"
# Saving the model
joblib.dump(model, 'spam_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')