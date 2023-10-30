import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re

def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
download_nltk_data()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A)
    tokens = word_tokenize(text)
    stopword_list = stopwords.words('english')
    tokens = [token for token in tokens if token not in stopword_list]
    tokens = [WordNetLemmatizer().lemmatize(token) for token in tokens]
    return ' '.join(tokens)

df = pd.read_csv('DatasetAdTranscript.csv')

df['processed_text'] = df['text'].apply(preprocess_text)

X_train, X_test, y_train, y_test = train_test_split(df['processed_text'], df['ad'], test_size=0.2, random_state=42)

pipeline = make_pipeline(TfidfVectorizer(), MultinomialNB())

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))
