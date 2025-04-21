import joblib

# Loading model and vectorizer
loaded_model = joblib.load('spam_model.pkl')
loaded_vectorizer = joblib.load('vectorizer.pkl')
# Function to classify new messages using loaded model
def predict_sms_loaded(message):
    msg_vector = loaded_vectorizer.transform([message])
    prediction = loaded_model.predict(msg_vector)[0]
    return "Spam" if prediction == 1 else "Not Spam"
# Take input from user
print("Enter message")
text=input()
print(predict_sms_loaded(text)) # Printing Spam or not