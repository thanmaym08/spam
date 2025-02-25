import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report


nltk.download('stopwords')


df = pd.read_csv('spam.csv', encoding='latin-1')
df = df[['v1', 'v2']]
df.columns = ['label', 'message']
df['label'] = df['label'].map({'ham': 0, 'spam': 1})


def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)  
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]  
    return " ".join(words)

df['cleaned_message'] = df['message'].apply(clean_text)


X_train, X_test, y_train, y_test = train_test_split(
    df['cleaned_message'], df['label'], test_size=0.2, random_state=42
)

spam_pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('classifier', MultinomialNB())
])


spam_pipeline.fit(X_train, y_train)


y_pred = spam_pipeline.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


def predict_spam(text):
    cleaned_text = clean_text(text)
    prediction = spam_pipeline.predict([cleaned_text])
    return "Spam" if prediction[0] == 1 else "valid"


sample_text = "Let's catch up soon, it's been a while!"
print(f"Sample Prediction: {predict_spam(sample_text)}")

