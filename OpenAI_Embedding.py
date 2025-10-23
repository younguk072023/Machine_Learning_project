import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sentence_transformers import SentenceTransformer

# 1️⃣ 데이터 불러오기
dataset = load_dataset("ag_news")
texts = dataset["train"]["text"]
labels = dataset["train"]["label"]

x_train, x_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

# 2️⃣ 무료 임베딩 모델 로드
# 영어용: all-MiniLM-L6-v2 (작고 빠름, 384차원)
model = SentenceTransformer("all-MiniLM-L6-v2")

# 3️⃣ 텍스트 → 임베딩 벡터로 변환
train_embeddings = model.encode(x_train, show_progress_bar=True)
test_embeddings  = model.encode(x_test, show_progress_bar=True)

# 4️⃣ SVM 학습
svm_model = SVC(kernel="linear")
svm_model.fit(train_embeddings, y_train)
pred = svm_model.predict(test_embeddings)

# 5️⃣ 성능 평가
print("SentenceTransformer Embedding + SVM 정확도:", round(accuracy_score(y_test, pred), 3))
print(classification_report(y_test, pred))
