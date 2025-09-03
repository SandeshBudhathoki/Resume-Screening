# 📄 Resume Screening with Custom Logistic Regression

This project is a **machine learning application** that classifies resumes into job categories using a **Logistic Regression (Softmax) model implemented from scratch**.  
It includes:  
- A training pipeline (data preprocessing, TF-IDF vectorization, model training, evaluation).  
- A **Streamlit web app** where users can upload resumes (`.txt`, `.docx`, `.pdf`) or paste text for category prediction.  

---

## 🚀 Features
- Preprocesses raw resume text using **TF-IDF**.
- Implements **Logistic Regression with Softmax** from scratch (no sklearn classifier).
- Trains and evaluates the model on labeled resume data.
- Saves the trained model, vectorizer, and encoder using `pickle`.
- Provides a **Streamlit UI** to upload resumes or paste text for prediction.
- Supports input from `.txt`, `.docx`, `.pdf`.

---

## 📂 Project Structure

```
resume_screening/
│── resumes_dataset.csv           # Dataset of resumes
│── train_model.py                # Training pipeline
│── model_utils.py                # Custom Logistic Regression class
│── app.py                        # Streamlit application
│── resume_model.pkl              # Trained model (saved)
│── vectorizer.pkl                # TF-IDF vectorizer (saved)
│── label_encoder.pkl             # Label encoder (saved)
│── requirements.txt              # Python dependencies
│── README.md                     # Project documentation
```

---

## ⚙️ Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/SandeshBudhathoki/resume-screening.git
   cd resume-screening
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate   # Mac/Linux
   venv\Scripts\activate      # Windows
   ```

3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🧑‍🏫 Training the Model

1. Make sure `resumes_dataset.csv` is in the project folder.
2. Run the training script:
   ```bash
   python train_model.py
   ```
3. This will:
   - Load and preprocess the dataset.
   - Train the **LogisticRegressionSoftmax** model.
   - Print evaluation results (accuracy + classification report).
   - Save:
     - `resume_model.pkl`
     - `vectorizer.pkl`
     - `label_encoder.pkl`

---

## 🌐 Running the Streamlit App

1. Start the app:
   ```bash
   streamlit run app.py
   ```

2. Open in your browser (default: `http://localhost:8501`).

3. Options:
   - **Upload a file** (`.txt`, `.docx`, `.pdf`)  
   - **Paste resume text** manually

4. The app will output:
   - **Predicted job category**  
   - **Category probabilities (table view)**  

---

## 🖥️ Example Workflow
- Train the model:
  ```bash
  python train_model.py
  ```
- Run the app:
  ```bash
  streamlit run app.py
  ```
- Upload `resume.pdf` or paste resume text:
  - Example prediction:  
    ```
    Predicted Category: Data Scientist
    ```

---

## 📝 Future Improvements
- Add support for extracting resumes from **images (OCR)**.
- Improve preprocessing (remove stopwords, stemming, etc.).
- Experiment with **deep learning models (BERT, RoBERTa)**.
- Deploy the Streamlit app to **Streamlit Cloud / Heroku**.

---
