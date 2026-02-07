import streamlit as st
import pickle
import numpy as np

# Load saved files
lda = pickle.load(open("lda_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
topic_labels = pickle.load(open("topic_labels.pkl", "rb"))

st.title("🛍️ Customer Review Topic Predictor")

user_input = st.text_area("Enter Customer Review")

if st.button("Predict Topic"):
    
    # Vectorize input
    vect_text = vectorizer.transform([user_input])
    
    # Get topic probabilities
    topic_prob = lda.transform(vect_text)
    
    # Get dominant topic
    topic_num = np.argmax(topic_prob)
    
    # Get label
    topic_name = topic_labels[topic_num]
    
    st.success(f"Predicted Topic: {topic_name}")
