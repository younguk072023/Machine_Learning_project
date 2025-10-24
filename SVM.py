from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

dataset = load_dataset("ag_news")
texts = dataset["train"]["text"]
labels = dataset["train"]["label"]

x_train, x_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
Xtr = vectorizer.fit_transform(x_train)
Xte = vectorizer.transform(x_test)

svm_model = SVC(kernel="linear")
svm_model.fit(Xtr, y_train)
pred = svm_model.predict(Xte)


print("TF-IDF + SVM 정확도:", round(accuracy_score(y_test, pred), 3))
print(classification_report(y_test, pred))

#정확도 : 0.91