import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


df = pd.read_csv('clean_event_classification_dataset.csv')
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)

tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2), stop_words='english')
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(X_train_tfidf, y_train)

sample_texts = [
    "Photoshoot for a grand wedding ceremony in the city gardens",
    "Team building event for a multinational corporation",
    "My son's 10th birthday party with magicians and games",
    "Rock band concert happening downtown this Saturday",
    "Football match finals at the national stadium"
]
sample_features = tfidf.transform(sample_texts)
sample_preds = clf.predict(sample_features)

for text, label in zip(sample_texts, sample_preds):
    print(f"Description: {text}")
    print(f"Predicted Category: {label}")
    print('-'*40)
