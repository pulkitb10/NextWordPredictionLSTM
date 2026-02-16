import streamlit as st
import pickle
import numpy as np
import pandas as pd
import nltk
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, LSTM,Embedding
from tensorflow.keras.models import Sequential
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# Load the trained model
model = load_model('next_word_predictor_model.h5')

#Load the tokenizer
tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))

def predict_values(model,tokenizer,text):
  token_text = tokenizer.texts_to_sequences([text])[0]
  padded_token_text = pad_sequences([token_text], maxlen= 183, padding='pre')
  pos = np.argmax(model.predict(padded_token_text))
  
  for word,index in tokenizer.word_index.items():
    if index == pos:
      text = text + " " + word
  return text    


st.title("Next Word Predictor Using LSTM")

input_text = st.text_input("Enter a sentence to predict its next word:")

if st.button("Predict"):
    if input_text:
        predicted_text = predict_values(model, tokenizer, input_text)
        st.write("Predicted Text: ", predicted_text)
    else:
        st.write("Please enter a sentence to predict its next word.")


