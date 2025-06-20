import torch
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader
from model import BERTClassifier
from utils import load_resumes_from_folder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
from tqdm import tqdm

class ResumeDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(self.texts[idx], padding="max_length", truncation=True, max_length=self.max_len, return_tensors="pt")
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx])
        }

resumes = load_resumes_from_folder("/content/resume_screening_project/data/resumes")
texts = [text for _, text in resumes]
labels = [1 if "python" in text.lower() else 0 for text in texts]

X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

train_ds = ResumeDataset(X_train, y_train, tokenizer)
test_ds = ResumeDataset(X_test, y_test, tokenizer)
train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BERTClassifier().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
loss_fn = torch.nn.CrossEntropyLoss()

model.train()
for epoch in range(3):
    print(f"Epoch {epoch+1}")
    for batch in tqdm(train_loader):
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        outputs = model(input_ids, attention_mask)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        outputs = model(input_ids, attention_mask)
        preds = torch.argmax(outputs, dim=1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

print(classification_report(y_true, y_pred))

