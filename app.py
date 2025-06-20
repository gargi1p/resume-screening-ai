import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pdfplumber
import matplotlib.pyplot as plt
import seaborn as sns


# Load tokenizer and model from saved directory
# model_dir = "/content/resume_screening_project"  # or wherever you saved
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased")

# UI Layout
st.title("🤖 Resume Screening using BERT")
st.write("Upload a resume in PDF format to evaluate its fit for various job roles based on its content.")

uploaded_file = st.file_uploader("📄 Upload Resume (PDF)", type="pdf")

# PDF Text Extraction
def extract_text_from_pdf(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    return text

# Prediction
def classify_resume(text):
    inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1)
    return pred.item(), probs.squeeze().tolist()

# Role labels
labels = [
    "Data Scientist", "Software Engineer", "DevOps Engineer", "Machine Learning Engineer",
    "Frontend Developer", "Backend Developer", "Full Stack Developer", "Cloud Engineer",
    "AI Researcher", "Business Analyst", "Product Manager", "UX/UI Designer",
    "Network Engineer", "QA Tester", "IT Support Specialist", "Other"
]

# Insights dictionary
insights = {
    "Data Scientist": "Strong in data analysis, Python, and statistics. Highlight ML and analytics projects.",
    "Software Engineer": "Solid coding skills. Showcase large-scale systems or backend architecture knowledge.",
    "DevOps Engineer": "Emphasize CI/CD, Docker, Kubernetes, and automation tools experience.",
    "Machine Learning Engineer": "Demonstrate ML pipeline, model training/deployment, and TensorFlow/PyTorch expertise.",
    "Frontend Developer": "Focus on HTML, CSS, JS, React, and responsive design. UI/UX sense is a bonus.",
    "Backend Developer": "Show server-side knowledge—Node.js, Django, databases, APIs, and RESTful services.",
    "Full Stack Developer": "Full project delivery from front to back. Showcase real-world web applications.",
    "Cloud Engineer": "AWS, GCP, or Azure expertise. Mention deployments, serverless, and cloud monitoring.",
    "AI Researcher": "Highlight publications, advanced projects, and deep learning/NLP/CV work.",
    "Business Analyst": "Focus on SQL, dashboards, stakeholder communication, and business metrics.",
    "Product Manager": "Show leadership, roadmap planning, stakeholder alignment, and tech fluency.",
    "UX/UI Designer": "Portfolio is key. Mention Figma, Adobe XD, user testing, and wireframes.",
    "Network Engineer": "Highlight routing, switching, firewalls, and network troubleshooting certifications.",
    "QA Tester": "Manual + automation testing knowledge. Show test cases, tools like Selenium, and bug tracking.",
    "IT Support Specialist": "Focus on hardware/software troubleshooting, ticketing tools, and communication.",
    "Other": "Resume may not clearly align with any specific role. Consider tailoring it for clarity."
}

# Handle Uploaded File
if uploaded_file is not None:
    text = extract_text_from_pdf(uploaded_file)
    st.subheader("📃 Extracted Resume Text:")
    st.text_area("", text[:2000], height=200)

    st.subheader("🧩 Role Suitability Analysis")
    label_index, probabilities = classify_resume(text)
    predicted_role = labels[label_index]
    confidence = probabilities[label_index]

    # Best Matched Role
    st.markdown(f"💼 **Best Matched Role:** `{predicted_role}`")
    st.markdown(f"📈 **Confidence Score:** `{confidence:.2%}`")

    # Insightful suggestions
    st.markdown("💡 **Insights & Suggestions:**")
    st.info(insights.get(predicted_role, "No specific insight available."))
