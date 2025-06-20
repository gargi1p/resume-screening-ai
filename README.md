# 🤖 Resume Screening using BERT

A smart AI-powered tool to screen resumes and suggest the most suitable job roles based on the content of the uploaded PDF. This project uses **BERT (Bidirectional Encoder Representations from Transformers)** to analyze resumes and classify them into predefined job categories.

---

## 🧠 Project Overview

This system automates resume screening for recruiters or educational institutions by:
- Extracting text from uploaded PDFs
- Processing it with a fine-tuned BERT model
- Predicting the most appropriate job role
- Providing insightful suggestions for the candidate

---

## 🎯 Objectives
- Develop a **binary classifier** and a **multi-label classifier** to classify resumes
- Build an intuitive **Streamlit UI** for file upload, text extraction, and prediction display
- Help candidates or HR professionals get immediate feedback and insights

---

## 📂 Project Structure

resume_screening_project/<br>
│<br>
├── app.py # Streamlit frontend for resume upload and prediction<br>
├── main.py # Model training script<br>
├── model.py # BERT-based classifier architecture<br>
├── utils.py # PDF parsing and resume loading utilities<br>
├── requirements.txt # Project dependencies<br>
└── README.md # You're here!<br>



---

## 🖼️ UI Output Snapshots

### 📄 Resume Upload Interface
![Upload UI](assets/upload_ui.png)

### 🔍 Prediction & Confidence Score
![Prediction UI](assets/prediction_ui.png)

### 💡 Insights & Suggestions
![Insights UI](assets/insights_ui.png)

> 📌 *(You can add these screenshots by saving the UI output from Streamlit and uploading them to an `assets/` folder in the repo.)*

---

## 🛠️ Technologies Used

- **BERT** (`bert-base-uncased`)
- **Transformers** by Hugging Face
- **PyTorch** for model training
- **Streamlit** for frontend UI
- **pdfplumber** for PDF text extraction
- **scikit-learn**, **pandas**, **tqdm** for processing

---

## ⚙️ How to Run

 🧪 Step 1: Install Dependencies

pip install -r requirements.txt


🚀 Step 2: Run the Streamlit App

streamlit run app.py


 🧠 Step 3: Train the Model (Optional)
You can train the model using:


python main.py
