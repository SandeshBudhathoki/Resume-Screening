import streamlit as st
import pickle
import numpy as np
import pandas as pd
import docx
import PyPDF2
from io import StringIO
from model_utils import LogisticRegressionSoftmax


def read_txt(file):
    return file.read().decode("utf-8")

def read_docx(file):
    doc = docx.Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

def read_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        if page.extract_text():
            text += page.extract_text() + "\n"
    return text

with open("resume_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)


st.title(" Resume Screening with Logistic Regression")
st.write("Upload a resume file or paste text below to get the predicted job category.")

# File uploader
uploaded_file = st.file_uploader("Upload Resume", type=["txt", "docx", "pdf"])

resume_input = ""

if uploaded_file is not None:
    if uploaded_file.name.endswith(".txt"):
        resume_input = read_txt(uploaded_file)
    elif uploaded_file.name.endswith(".docx"):
        resume_input = read_docx(uploaded_file)
    elif uploaded_file.name.endswith(".pdf"):
        resume_input = read_pdf(uploaded_file)

# Fallback: manual text input
if not resume_input:
    resume_input = st.text_area("Or Paste Resume Text", height=200)

# Prediction
if st.button("Predict"):
    if resume_input.strip():
        resume_tfidf = vectorizer.transform([resume_input]).toarray()
        probs = model.predict_proba(resume_tfidf)[0]

        pred_class = np.argmax(probs)
        pred_label = label_encoder.inverse_transform([pred_class])[0]

        st.success(f" Predicted Category: **{pred_label}**")

        st.subheader(" Category Probabilities:")
        prob_df = pd.DataFrame({
            "Category": label_encoder.classes_,
            "Probability": np.round(probs, 4)
        }).sort_values(by="Probability", ascending=False)

        st.table(prob_df)
    else:
        st.warning(" Please upload or paste a resume before predicting.")
