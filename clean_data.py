import pandas as pd
import time
from datetime import datetime
try:
    from sklearn.model_selection import train_test_split
except:
    from sklearn.cross_validation import train_test_split
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from sklearn.naive_bayes import MultinomialNB
import re
import os
from bs4 import BeautifulSoup
import cPickle as pk
from pandas.io.json import json_normalize

def make_fraud_col(df_in, spammer=False):
    if spammer:
        fraud_accts = set(['fraudster_event', 'fraudster', 'fraudster_att', 'spammer_limited', 'spammer_warn', 'spammer_web', 'spammer'])
    else:
        fraud_accts = set(['fraudster_event', 'fraudster', 'fraudster_att'])

    new_df = df_in.copy()
    new_df['fraud'] = df_in['acct_type'].apply(lambda x: 1 if x in fraud_accts else 0)
    new_df.drop('acct_type', axis=1, inplace=True)
    return new_df

def make_dates_features(df):
    df['diff_event_user (hours)'] = (df['event_created'] - df['user_created']) / float(60**2)

    date_features = ['event_created', 'user_created']
    for feature in date_features:
        df[feature] = df[feature].apply(lambda x: time.gmtime(x))

    df['hour_event_created'] = df['event_created'].map(lambda x: x.tm_hour)
    df['hour_user_created'] = df['user_created'].map(lambda x: x.tm_hour)
    return df, ['hour_user_created', 'hour_event_created', 'diff_event_user (hours)']

def plot_hist(df, col):
    fraud = df[df['fraud'] == 1]
    nofraud = df[df['fraud'] == 0]
    fraud[col].hist(bins=30)
    plt.title('fraud')
    plt.show()
    #ax = plt.gca()
    plt.title('no fraud')
    nofraud[col].hist(bins=30)

def make_train_test(df, colstouse=None):
    if colstouse is not None:
        new_df = df[colstouse + ['fraud']].copy()
    else:
        new_df = df.copy()

    y = new_df.pop('fraud').values
    X = new_df.values
    columns = new_df.columns
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, stratify=y)
    return X_train, X_test, y_train, y_test, X, y, columns

def load_clean_df(path='data/data.json', cols=None, caps=True, refresh=False):
    '''
    loads, cleans, and extracts features from training dataset ('requires dataset to be unzipped first')
    if a zipped version of the pre-cleaned dataset exists, load that first

    caps is an option for extracting % of the title letters that are capitals
    for each event
    '''
    clean_filename = 'data/clean_data.gzip'
    if os.path.exists(clean_filename) and not refresh:
        df = pd.read_csv(clean_filename, compression='gzip')
        df.drop('Unnamed: 0', axis=1, inplace=True)
        return df

    df = pd.read_json(path)
    df = make_fraud_col(df)
    df = add_naive_bayes(df)
    df, extra_cols = make_dates_features(df)
    extra_cols = extra_cols + ['naive_bayes_name', 'naive_bayes_descr', 'fraud']
    if caps:
        df = get_pct_caps_in_title(df)
        df = binary_encode_filter(df, extra_cols=extra_cols + ['name_pct_upper'])
    else:
        df = binary_encode_filter(df, extra_cols=extra_cols)

    if cols is not None:
        # for selecting a subset of the final columns
        df = df[cols + ['fraud']]

    return df

def load_train_test(path='data/data.json', cols=None):
    df = load_clean_df(path, cols)
    return make_train_test(df, cols)

def binary_encode_filter(old_df, extra_cols=[]):
    '''
    * encodes certain fields into binary
    * excludes most fields and saves the ones we thought were appropriate
    * summary in feature_engineering_plan.ods
    '''
    df = old_df.copy()
    df['US_ind'] = np.where(df['country']=='US',1,0)
    df['delivery_ind'] = np.where(df['delivery_method']==0, 0, 1)
    df['org_desc_ind'] = np.where(df['org_desc']=="",0,1)
    df['org_fb_ind'] = np.where(df['org_facebook']!=0, 1, 0)
    df['org_twitter_ind'] = np.where(df['org_twitter'] != 0, 1, 0)
    df['payee_name_ind'] = np.where(df['payee_name']=="",0,1)
    df['venue_US_ind'] = np.where(df['venue_country']=='US',1,0)
    df['listed'] = np.where(df['listed']=='y',1,0)
    df_clean = df[['channels','gts','has_logo','listed','name_length',\
                    'num_payouts','sale_duration2','user_age',\
                    'user_type','US_ind','delivery_ind','org_desc_ind',\
                    'org_fb_ind','org_twitter_ind','payee_name_ind',\
                    'venue_US_ind'] + extra_cols]

    return df_clean

def tokenize(doc):
    snowball = SnowballStemmer('english')
    return [snowball.stem(word) for word in word_tokenize(doc.lower())]

def add_naive_bayes(df):
    y = df['fraud'].values
    X = df['name'].values
    X_body = []
    for descr in df['description']:
        html = BeautifulSoup(descr, 'html.parser')
        lst = [sub.text for sub in html.find_all('p')]
        X_body.append(''.join(lst))

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    X_body_train, X_body_test, y_body_train, y_body_test = train_test_split(X_body, y, random_state=42)

    vectorizer = TfidfVectorizer(stop_words='english', tokenizer=tokenize)
    name_tfidf_vectorized = vectorizer.fit_transform(X_train)
    pk.dump(vectorizer, open('name_vectorizer.pk', 'w'), 2)
    name_model = MultinomialNB()
    name_model.fit(name_tfidf_vectorized, y_train)
    pk.dump(name_model, open('name_model.pk', 'w'), 2)
    df['naive_bayes_name'] = name_model.predict(vectorizer.transform(X))

    vectorizer = TfidfVectorizer(stop_words='english', tokenizer=tokenize)
    body_tfidf_vectorized = vectorizer.fit_transform(X_body_train)
    pk.dump(vectorizer, open('body_vectorizer.pk', 'w'), 2)
    body_model = MultinomialNB()
    body_model.fit(body_tfidf_vectorized, y_body_train)
    pk.dump(body_model, open('body_model.pk', 'w'), 2)
    df['naive_bayes_descr'] = body_model.predict(vectorizer.transform(X_body))

    return df

def get_pct_caps_in_title(df):
    df['name_pct_upper'] = df['name'].apply(lambda x: np.mean([l.isupper() for l in "".join(re.findall("[a-zA-Z]+", x))]))
    df['name_pct_upper'] = df['name_pct_upper'].fillna(df['name_pct_upper'].mean())
    return df

def prepare_test_dataframe(stream_data, cols=None, caps=True):
    df = json_normalize(stream_data)

    name_model = pk.load(open('name_model.pk'))
    body_model = pk.load(open('body_model.pk'))
    name_vectorizer = pk.load(open('name_vectorizer.pk'))
    body_vectorizer = pk.load(open('body_vectorizer.pk'))
    df['naive_bayes_name'] = name_model.predict(name_vectorizer.transform(df['name'].values))
    X_body = []
    html = BeautifulSoup(df['description'][0], 'html.parser')
    lst = [sub.text for sub in html.find_all('p')]
    X_body.append(''.join(lst))
    df['naive_bayes_descr'] = body_model.predict(body_vectorizer.transform(X_body))

    df, extra_cols = make_dates_features(df)
    extra_cols = extra_cols + ['naive_bayes_name', 'naive_bayes_descr']
    if caps:
        df = get_pct_caps_in_title(df)
        df = binary_encode_filter(df, extra_cols=extra_cols + ['name_pct_upper'])
    else:
        df = binary_encode_filter(df, extra_cols)

    if cols is not None:
        # for selecting a subset of the final columns
        df = df[cols]

    return df

if __name__=="__main__":
    pass
