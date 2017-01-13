import pandas as pd
from clean_data import make_fraud_col
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from sklearn.naive_bayes import MultinomialNB



def tokenize(doc):
    snowball = SnowballStemmer('english')
    return [snowball.stem(word) for word in word_tokenize(doc.lower())]

def make_train_test(path='data/data.json'):
    df = pd.read_json(path)
    df = make_fraud_col(df)
    y = df['fraud'].values
    X = df['name'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    # return X_train, X_test, y_train, y_test

# X_train, X_test, y_train, y_test = make_train_test()


    vectorizer = TfidfVectorizer(stop_words='english', tokenizer=tokenize)
    tfidf_vectorized = vectorizer.fit_transform(X_train)
    model = MultinomialNB()
    model.fit(tfidf_vectorized, y_train)
    preds = model.predict(vectorizer.transform(X_test))
    return preds








def tokenize(doc):
    snowball = SnowballStemmer('english')
    return [snowball.stem(word) for word in word_tokenize(doc.lower())]

def add_naive_bayes(df):
    y = df['fraud'].values
    X = df['name'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    vectorizer = TfidfVectorizer(stop_words='english', tokenizer=tokenize)
    tfidf_vectorized = vectorizer.fit_transform(X_train)
    model = MultinomialNB()
    model.fit(tfidf_vectorized, y_train)

    df['naive_bayes_name'] = model.predict(vectorizer.transform(X))

    return df
