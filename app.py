import streamlit as st
import pickle
import numpy as np

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="Topic Predictor", layout="centered")

# ------------------ CUSTOM CSS ------------------
st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    font-family: 'Segoe UI';
}

/* Glass Container */
.glass {
    background: rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(10px);
    padding: 25px;
    border-radius: 20px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
}

/* Title Styling */
.title {
    font-size: 42px;
    font-weight: bold;
    color: white;
    text-align: center;
}

/* Prediction Box */
.prediction-box {
    background: linear-gradient(90deg, #00c853, #64dd17);
    padding: 15px;
    border-radius: 15px;
    font-size: 20px;
    font-weight: bold;
    color: black;
    text-align: center;
}

/* Button Styling */
div.stButton > button {
    background: linear-gradient(90deg, #ff7e5f, #feb47b);
    border-radius: 12px;
    height: 45px;
    font-weight: bold;
    color: white;
    border: none;
    transition: 0.3s;
}

div.stButton > button:hover {
    transform: scale(1.05);
    background: linear-gradient(90deg, #ff9966, #ff5e62);
}

</style>
""", unsafe_allow_html=True)

# ------------------ LOAD MODEL ------------------
lda = pickle.load(open("lda_model.pkl","rb"))
vectorizer = pickle.load(open("vectorizer.pkl","rb"))
topic_labels = pickle.load(open("topic_labels.pkl","rb"))

# ------------------ HEADER ------------------
st.markdown('<div class="title">🛍 Customer Review Topic Predictor</div>', unsafe_allow_html=True)
st.write("")

# ------------------ GLASS CONTAINER ------------------
with st.container():
    st.markdown('<div class="glass">', unsafe_allow_html=True)

    user_input = st.text_area("Enter Customer Review")

    if st.button("Predict Topic"):

        vect = vectorizer.transform([user_input])
        prob = lda.transform(vect)

        topic_num = np.argmax(prob)
        topic_name = topic_labels[topic_num]

        st.markdown(
            f'<div class="prediction-box">Predicted Topic: {topic_name}</div>',
            unsafe_allow_html=True
        )

        # Confidence chart
        st.subheader("Topic Confidence")
        st.bar_chart(prob[0])

    st.markdown('</div>', unsafe_allow_html=True)

