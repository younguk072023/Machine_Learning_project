import pandas as pd
import pathlib
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text  import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import  accuracy_score, classification_report


# Tf는 하나의 문서에서 특정 단어가 등장하는 횟수
# Idf 역수를 취해서 적은 문장에 등장할수록 큰 숫자가 되게하고 반대로 많은 문서에 등장할수록 숫자를 작아지게 함으로써 문장에서 의미없이 사용되는 단어의 가중치를 줄임.

df = pd.read_csv("data/mtsamples.csv")

x_train, x_test, y_train, y_test = train_test_split(
    df["transcription"], df["medical_specialty"],test_size=0.2,random_state=42)


vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
Xtr = vectorizer.fit_transform(x_train)
Xte = vectorizer.transform(x_test)

svm_model = SVC(kernel = "linear")
svm_model.fit(x_train, y_train)
pred = svm_model.predict(Xte)

print("TF-IDF + SVM 정확도:", round(accuracy_score(y_test, pred), 3))
print(classification_report(y_test, pred))
                                                  
    

