import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(y_true, y_pred, class_labels=None):
    """
    Evaluate classification model performance:
    - Prints accuracy
    - Prints full classification report
    - Prints macro and weighted F1 scores
    - Visualizes confusion matrix
    
    Parameters:
    - y_true: array-like, true labels
    - y_pred: array-like, predicted labels
    - class_labels: list of class labels in consistent order (optional)
    """
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1 Score: {macro_f1:.4f}")
    print(f"Weighted F1 Score: {weighted_f1:.4f}\n")

    print("Classification Report:")
    print(classification_report(y_true, y_pred))

    if class_labels is None:
       
        class_labels = sorted(list(set(y_true)))

    cm = confusion_matrix(y_true, y_pred, labels=class_labels)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels,
                yticklabels=class_labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()

def main():
    
    df = pd.read_csv('clean_event_classification_dataset.csv')

    print(f"Dataset loaded with {len(df)} samples.")
    print(f"Label distribution:\n{df['label'].value_counts()}")


    X_train, X_test, y_train, y_test = train_test_split(
        df['text'],
        df['label'],
        test_size=0.2,
        random_state=42,
        stratify=df['label']
    )

    tfidf = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words='english'
    )
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    print(f"TF-IDF matrix shapes: Train {X_train_tfidf.shape}, Test {X_test_tfidf.shape}")

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train_tfidf, y_train)

    y_pred = clf.predict(X_test_tfidf)


    evaluate_model(y_test, y_pred, class_labels=clf.classes_)


    print("\nSample predictions on test samples:")
    samples = X_test.sample(5, random_state=42)
    for text in samples:
        pred = clf.predict(tfidf.transform([text]))[0]
        print(f"Text: {text}\nPredicted Category: {pred}\n")

if __name__ == '__main__':
    main()
