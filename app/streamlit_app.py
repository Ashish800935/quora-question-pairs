"""
Streamlit demo for duplicate-question detection.

Run:
    streamlit run app/streamlit_app.py
"""
import streamlit as st

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.predict import get_predictor

st.set_page_config(page_title="Quora Duplicate Question Detector", page_icon="🔎")
st.title("🔎 Quora Duplicate Question Detector")
st.caption("Type two questions and check whether they carry the same intent.")

q1 = st.text_area("Question 1", placeholder="e.g. How do I learn Python?")
q2 = st.text_area("Question 2", placeholder="e.g. What is the best way to learn Python?")

if st.button("Check", type="primary"):
    if not q1.strip() or not q2.strip():
        st.warning("Please enter both questions.")
    else:
        try:
            predictor = get_predictor()
        except FileNotFoundError:
            st.error("Model not trained yet. Run `python -m src.train` first.")
        else:
            result = predictor.predict(q1, q2)
            label = "✅ Duplicate" if result["is_duplicate"] else "❌ Not Duplicate"
            st.subheader(label)
            st.progress(result["duplicate_probability"])
            st.write(f"**Duplicate probability:** {result['duplicate_probability']:.2%}")
            st.write(f"**Confidence in prediction:** {result['confidence']:.2%}")
            st.caption(f"Model: {result['model_used']}")
