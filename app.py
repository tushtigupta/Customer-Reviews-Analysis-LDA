import streamlit as st
import pickle
import numpy as np
import os

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="Customer Review Topic Predictor", layout="centered")

# ------------------ CUSTOM CSS ------------------
st.markdown("""
<style>

/* Background */
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    font-family: 'Segoe UI';
}

/* Title Styling */
.title {
    font-size: 42px;
    font-weight: 800;
    color: white;
    text-align: center;
    margin-bottom: 25px;
}

/* Premium Input Container */
.premium-input {
    background: rgba(255,255,255,0.07);
    padding: 20px;
    border-radius: 18px;
    backdrop-filter: blur(12px);
    box-shadow: 0 8px 30px rgba(0,0,0,0.35);
    margin-bottom: 20px;
}

/* Input Label */
.input-label {
    font-size: 20px;
    font-weight: 800;
    color: white;
    margin-bottom: 10px;
}

/* Text Area Styling */
textarea {
    background: rgba(255,255,255,0.05) !important;
    border-radius: 12px !important;
    border: 2px solid transparent !important;
    color: white !important;
    font-size: 16px !important;
    padding: 12px !important;
}

/* Glow on focus */
textarea:focus {
    border: 2px solid #00e5ff !important;
    box-shadow: 0 0 12px #00e5ff !important;
    outline: none !important;
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
    width: 100%;
}

div.stButton > button:hover {
    transform: scale(1.03);
    background: linear-gradient(90deg, #ff9966, #ff5e62);
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
    margin-top: 15px;
}

</style>
""", unsafe_allow_html=True)

# ------------------ LOAD MODEL FILES ------------------
BASE_DIR = os.path.dirname(__file__)

lda = pickle.load(open(os.path.join(BASE_DIR, "lda_model.pkl"), "rb"))
vectorizer = pickle.load(open(os.path.join(BASE_DIR, "vectorizer.pkl"), "rb"))
topic_labels = pickle.load(open(os.path.join(BASE_DIR, "topic_labels.pkl"), "rb"))

# ------------------ HEADER ------------------
st.markdown('<div class="title">🛍 Customer Review Topic Predictor</div>', unsafe_allow_html=True)

# ------------------ PREMIUM INPUT BOX ------------------
st.markdown('<div class="premium-input">', unsafe_allow_html=True)

st.markdown(
    '<div class="input-label">📝 Enter Customer Review</div>',
    unsafe_allow_html=True
)

user_input = st.text_area(
    "",
    height=160,
    placeholder="Example: The product has smooth texture and balanced sweetness..."
)

st.markdown('</div>', unsafe_allow_html=True)

# ------------------ PREDICTION ------------------
if st.button("Predict Topic"):

    if user_input.strip() == "":
        st.warning("Please enter a review first.")
    else:
        vect = vectorizer.transform([user_input])
        prob = lda.transform(vect)

        topic_num = np.argmax(prob)
        topic_name = topic_labels[topic_num]

        # Prediction Card
        st.markdown(
            f'<div class="prediction-box">Predicted Topic: {topic_name}</div>',
            unsafe_allow_html=True
        )

        # Confidence Chart
        st.subheader("📊 Topic Confidence")
        st.bar_chart(prob[0])

# ------------------ FOOTER ------------------
st.markdown("---")
st.caption("Built with ❤️ using NLP & LDA Topic Modeling")

