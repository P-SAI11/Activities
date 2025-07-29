# app.py

import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load models and resources
tokenizer = pickle.load(open("tokenizer.pkl", "rb"))
label_map = pickle.load(open("label_encoder.pkl", "rb"))

rnn_model = load_model("rnn_model.h5")
gru_model = load_model("gru_model.h5")
lstm_model = load_model("lstm_model.h5")

MAX_LEN = 100
# 🔮 Prediction function
def predict_emotion(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LEN)

    rnn_pred = rnn_model.predict(padded)[0]
    gru_pred = gru_model.predict(padded)[0]
    lstm_pred = lstm_model.predict(padded)[0]

    ensemble_probs = (rnn_pred + gru_pred + lstm_pred) / 3
    final_index = np.argmax(ensemble_probs)
    final_emotion = label_map[final_index]

    return final_emotion, ensemble_probs

# 🎨 Streamlit UI
st.set_page_config(page_title="Emotion Detector 🤖", layout="centered")

st.title("Emotion Detector 💬")
st.subheader("Type a sentence and I'll guess the emotion!")

user_input = st.text_area("Enter your sentence:", height=150)

if st.button("Detect Emotion"):
    if user_input.strip() == "":
        st.warning("Please enter something.")
    else:
        predicted_emotion, probs = predict_emotion(user_input)
        st.success(f"🔮 Predicted Emotion: **{predicted_emotion.upper()}**")
