from pymongo import MongoClient
import pandas as pd
import json
import sys
import os.path
sys.path.append(
   os.path.abspath(os.path.join(os.path.dirname('__file__'), os.path.pardir)))
import clean_data as cd

DB_NAME = "eventbrite"
COLLECTION_NAME = "fraudPreds"

client = MongoClient()
db = client[DB_NAME]
coll = db[COLLECTION_NAME]

def store_training():
    df = cd.load_clean_df()
    # df['prediction'] = preds
    coll.insert_many(df.to_dict('records'))

def live_data_in_mongo(new_data):
    df = new_data
    # df = cd.load_clean_df().iloc[1]
    # df = pd.DataFrame(df)
    # df['prediction'] = pred
    coll.insert(df.to_dict('records'))
    # db.live_data.insert({'data': new_data, 'prediction': pred})
    # db.live_data.find().limit(10)
