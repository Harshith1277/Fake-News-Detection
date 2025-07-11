# test_fake_news.py

import joblib

# Load model and vectorizer
model = joblib.load("models/fake_news_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# User input
print("📰 Enter a news article to check if it's FAKE or REAL:\n")
user_input = input()

# Transform and predict
input_vector = vectorizer.transform([user_input])
prediction = model.predict(input_vector)

# Output result
if prediction[0] == 'FAKE':
    print("\n🚨 This news article is FAKE.")
else:
    print("\n✅ This news article is REAL.")
