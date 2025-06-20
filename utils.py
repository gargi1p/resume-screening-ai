import pdfplumber
import os

def extract_text_from_pdf(file_path):
    with pdfplumber.open(file_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text.strip()

def load_resumes_from_folder(folder_path):
    resumes = []
    filenames = os.listdir(folder_path)
    for file in filenames:
        if file.endswith(".pdf"):
            text = extract_text_from_pdf(os.path.join(folder_path, file))
            resumes.append((file, text))
    return resumes
