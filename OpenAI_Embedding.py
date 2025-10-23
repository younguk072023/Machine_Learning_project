import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sentence_transformers import SentenceTransformer

dataset = load_dataset("ag_news")
texts = dataset["train"]["text"]
labels = dataset["train"]["label"]

x_train, x_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

model = SentenceTransformer("all-MiniLM-L6-v2")

train_embeddings = model.encode(x_train, show_progress_bar=True)
test_embeddings  = model.encode(x_test, show_progress_bar=True)

svm_model = SVC(kernel="linear")
svm_model.fit(train_embeddings, y_train)
pred = svm_model.predict(test_embeddings)

print("SentenceTransformer Embedding + SVM 정확도:", round(accuracy_score(y_test, pred), 3))
print(classification_report(y_test, pred))
