
import streamlit as st
import joblib
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

model = joblib.load('logistic_regression_model (1).pkl')
tfidf = joblib.load('tfidf_vectorizer (1).pkl')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(word)
             for word in words if word not in stop_words]
    return ' '.join(words)

st.title("Smart Fake News Detection System")
st.write("Enter a news article to check its predicted category.")

news = st.text_area("Enter News Article")

if st.button("Detect News"):
    if news.strip():
        cleaned = clean_text(news)
        vectorized = tfidf.transform([cleaned])

        prediction = model.predict(vectorized)[0]
        probability = model.predict_proba(vectorized)[0]

        if prediction == 0:
            st.error("Prediction: FAKE NEWS")
        else:
            st.success("Prediction: REAL NEWS")

        st.write("Model confidence:",
                 round(max(probability) * 100, 2), "%")
    else:
        st.warning("Please enter a news article.")
