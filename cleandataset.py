import pandas as pd
from sqlalchemy import create_engine
from urllib.parse import quote_plus
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def get_sqlalchemy_engine():
    user = 'root'
    password = 'srinath@25k'  
    host = 'localhost'
    database = 'Photography_Project'

    password_encoded = quote_plus(password)
    connection_string = f"mysql+mysqlconnector://{user}:{password_encoded}@{host}/{database}"
    engine = create_engine(connection_string)
    return engine


def clean_text(text):
    if not isinstance(text, str):
        return ''
    text = text.lower().strip()
    text = re.sub(r'[^a-z\s]', ' ', text)  
    text = re.sub(r'\s+', ' ', text) 
    return text


def extract_clean_dataset(engine):
  
    query = """
    SELECT event_category AS label,
           event_category AS event_description
    FROM bookings
    WHERE event_category IS NOT NULL
    """

    df = pd.read_sql(query, con=engine)

    
    df['cleaned_text'] = df['event_description'].apply(clean_text)

    # Filter out empty labels or texts if any
    df = df[(df['cleaned_text'] != '') & (df['label'] != '')]

    # Rename columns
    clean_df = df[['cleaned_text', 'label']].rename(columns={'cleaned_text': 'text'})
    return clean_df

def train_evaluate_model(df):
    X = df['text']
    y = df['label']

    # Split train-test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Extract TF-IDF features
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    # Train Logistic Regression classifier
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_tfidf, y_train)

    # Predict on test
    y_pred = clf.predict(X_test_tfidf)

    # Evaluation metrics
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")

    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Sample predictions
    print("\nSample Predictions:")
    samples = X_test.sample(min(5, len(X_test)), random_state=42)
    for sample_text in samples:
        sample_vec = tfidf.transform([sample_text])
        pred_label = clf.predict(sample_vec)[0]
        print(f"Text: {sample_text}\nPredicted Label: {pred_label}\n")

    return clf, tfidf


if __name__ == '__main__':
    engine = get_sqlalchemy_engine()
    dataset = extract_clean_dataset(engine)

    print(f"Loaded {len(dataset)} samples from database.")

    if len(dataset) == 0:
        print("No data available to train. Make sure your bookings table has event_category values.")
    else:
        model, vectorizer = train_evaluate_model(dataset)
import pandas as pd
import re
from sqlalchemy import create_engine
from urllib.parse import quote_plus

def get_sqlalchemy_engine():
    user = 'root'
    password = 'srinath@25k'  # change if needed
    host = 'localhost'
    database = 'Photography_Project'

    password_encoded = quote_plus(password)
    connection_string = f"mysql+mysqlconnector://{user}:{password_encoded}@{host}/{database}"
    engine = create_engine(connection_string)
    return engine

def clean_text(text):
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = re.sub(r'<.*?>', ' ', text)         
    text = re.sub(r'[^a-z\s]', ' ', text)       
    text = re.sub(r'\s+', ' ', text).strip()   
    return text

def extract_clean_dataset(engine):
    """
    Extracts a labeled dataset with:
    - label: event_category from bookings
    - text: Combine multiple possible textual sources:
        - event_category itself (use as fallback)
        - portfolio titles and descriptions related to photographers booked (optional)
        - user search terms for events matching the category (optional)
    """

   
    bookings_query = """
    SELECT booking_id, user_id, photographer_id, event_category
    FROM bookings
    WHERE event_category IS NOT NULL
    """
    bookings_df = pd.read_sql(bookings_query, con=engine)

    if bookings_df.empty:
        print("No bookings data found with event_category!")
        return None

   
    portfolios_query = """
    SELECT p.portfolio_id, p.photographer_id, p.title, p.description
    FROM portfolios p
    """
    portfolios_df = pd.read_sql(portfolios_query, con=engine)

   
    search_logs_query = """
    SELECT user_id, search_term
    FROM search_logs
    WHERE search_term IS NOT NULL
    """
    search_logs_df = pd.read_sql(search_logs_query, con=engine)


    combined_texts = []

    for idx, row in bookings_df.iterrows():
        user_id = row['user_id']
        photographer_id = row['photographer_id']
        event_category = row['event_category']

       
        pf_texts = ''
        if not portfolios_df.empty:
            pf_portfolios = portfolios_df[portfolios_df['photographer_id'] == photographer_id]
            titles = pf_portfolios['title'].fillna('').tolist()
            descriptions = pf_portfolios['description'].fillna('').tolist()
            pf_texts = ' '.join(titles + descriptions)

        searches = search_logs_df[search_logs_df['user_id'] == user_id]
        user_search_terms = ' '.join(searches['search_term'].dropna().tail(5).tolist())

        combined_text = f"{event_category} {pf_texts} {user_search_terms}"

        cleaned = clean_text(combined_text)

        combined_texts.append({'text': cleaned, 'label': event_category})

    dataset_df = pd.DataFrame(combined_texts)

 
    dataset_df = dataset_df[(dataset_df['text'].str.strip() != '') & (dataset_df['label'].str.strip() != '')]

    print(f"Extracted dataset with {len(dataset_df)} samples.")

    return dataset_df

if __name__ == '__main__':
    engine = get_sqlalchemy_engine()
    dataset = extract_clean_dataset(engine)

    if dataset is not None:
        print(dataset.head(5))
        dataset.to_csv('clean_event_classification_dataset.csv', index=False)
        print("Clean dataset saved to clean_event_classification_dataset.csv")
