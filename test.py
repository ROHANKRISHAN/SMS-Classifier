import streamlit as st
import sklearn
import pickle
import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
nltk.download('stopwords')
nltk.download('punkt')
from nltk.stem import PorterStemmer

port_stemmer = PorterStemmer()

# Load pre-trained TF-IDF vectorizer and model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Create a function to generate cleaned data from raw text
def clean_text(text):
    text = word_tokenize(text)  # Tokenize the text
    text = " ".join(text)  # Join tokens
    text = [char for char in text if char not in string.punctuation]  # Remove punctuations
    text = ''.join(text)  # Join the characters
    text = [char for char in text if not char.isdigit()]  # Remove numbers
    text = ''.join(text)  # Join the characters
    text = [word.lower() for word in text.split() if word.lower() not in stopwords.words('english')]  # Remove stopwords
    text = ' '.join(text)  # Join the words
    text = [port_stemmer.stem(word) for word in text.split()]  # Stemming
    return " ".join(text)

st.title('SMS Spam Classifier')

input_sms = st.text_input("Enter the Message")

if st.button('Predict'):
    if input_sms == "":
        st.header('Please Enter Your Message !!!')
    else:
        # 1. Preprocess
        transform_text = clean_text(input_sms)

        # 2. Vectorize
        vector_input = tfidf.transform([transform_text])

        # 3. Prediction
        result = model.predict(vector_input)

        # 4. Display
        if result == 1:
            st.header("Spam")
        else:
            st.header("Not Spam")
