import numpy as np
import pandas as pd
import tensorflow as tf

import string
import streamlit as st
from tensorflow.keras.layers import SimpleRNN,Dense,Embedding
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence

#load IMDB
word_index=imdb.get_word_index()
reverse_word_index={value:key for key,value in word_index.items()}

import h5py

# Load the model file
model_path = 'simpleRNN.h5'

try:
    model = load_model('simpleRNN.h5')
    print("Model loaded successfully")
except OSError as e:
    st.error("Error: Could not find or load the model file 'simpleRNN.h5'. Please ensure the model file exists in the correct location.")
    st.stop()

# Remove the h5py file handling block as it's not necessary if we're using load_model
# The following section can be removed:
# with h5py.File(model_path, 'r+') as f:
#     # Navigate to the SimpleRNN layer configuration
#     if 'model_weights' in f and 'simple_rnn_1' in f['model_weights']:
#         simple_rnn_group = f['model_weights']['simple_rnn_1']
#         
#         # Check if 'time_major' exists in the configuration
#         if 'time_major' in simple_rnn_group.attrs:
#             # Remove the 'time_major' attribute
#             del simple_rnn_group.attrs['time_major']
#             print("Removed 'time_major' from SimpleRNN layer configuration.")
#         else:
#             print("'time_major' not found in SimpleRNN layer configuration.")
#     else:
#         print("SimpleRNN layer not found in the model.")
# 
# print("Model file has been updated.")
# model=load_model('simpleRNN.h5')
# 
# function to decode the review
def decode_review(encoded_review):
    return " ".join(
    [reverse_word_index.get(i-3,'?') 
     for i in encoded_review])
#function to preprocess the data
def preprocess_text(text):
    max_features=10000
    text = text.translate(str.maketrans('', '', string.punctuation))
    words=text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 if word_index.get(word, 2) < max_features else 2 for word in words]
    padded_review=sequence.pad_sequences([encoded_review],maxlen=500)
    return padded_review

#prediction function
def predict_sentiment(review):
    preprocessed_ip=preprocess_text(review)
    prediction=model.predict(preprocessed_ip)
    sentiment='Positive' if prediction[0][0]>0.5 else 'Negative'
    return sentiment,prediction[0][0]


#streamlit app
st.title('IMDB Movie review rating Analysis')
st.write('Enter the moview review to classify it as positive or negative')

#user input
user_input=st.text_area("Movie Review")

if st.button('Classify'):
    if user_input:
        #make prediction
        sentiment,score=predict_sentiment(user_input)
        #Display result with better formatting
        st.markdown("### Results:")
        st.markdown(f"**Sentiment:** {sentiment}")
        st.markdown(f"**Confidence Score:** {score:.2f}")
    else:
        st.warning('Please enter a movie review before classifying')

else:
    st.write('Please enter the movie review')