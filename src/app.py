import streamlit as st
import pickle
import pandas as pd
import os

# 1. Path to your saved models
# This gets the directory where app.py is located (the 'src' folder)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# This goes UP one level from 'src', then into 'models'
MODEL_DIR = os.path.join(BASE_DIR, '..', 'models')

# 2. Load the components
model = pickle.load(open(os.path.join(MODEL_DIR, 'rf_model.pkl'), 'rb'))
vectorizer = pickle.load(open(os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl'), 'rb'))
label_encoder = pickle.load(open(os.path.join(MODEL_DIR, 'label_encoder.pkl'), 'rb'))

# 3. Streamlit UI
st.set_page_config(page_title="Fintech Categorizer", page_icon="💳")

st.title("💳 Smart Expense Categorizer")
st.markdown("---")
st.write("Enter a transaction description below to see how the AI classifies it.")

# User Input
user_input = st.text_input("Transaction Description", placeholder="e.g., Netflix.com or Shell Gas Station")

if st.button("Classify Transaction"):
    if user_input:
        # Preprocessing (Keep it simple for the demo)
        cleaned_text = user_input.lower().strip()
        
        # Transform using the SAVED vectorizer
        vectorized_text = vectorizer.transform([cleaned_text])
        
        # Predict
        prediction = model.predict(vectorized_text)
        category = label_encoder.inverse_transform(prediction)[0]
        
        # Get Confidence Score
        probs = model.predict_proba(vectorized_text)
        confidence = max(probs[0]) * 100

        # Display results
        st.success(f"**Predicted Category:** {category}")
        st.info(f"**AI Confidence Score:** {confidence:.2f}%")
        
        # Visual feedback based on confidence
        if confidence < 60:
            st.warning("⚠️ Low confidence. This might need manual review.")
    else:
        st.error("Please enter a description first!")