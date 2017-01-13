import pandas as pd
from clean_data import make_fraud_col
import matplotlib.pyplot as plt

df = pd.read_json('data/data.json')
df = make_fraud_col(df)

feature_list_numerical = ['body_length', 'name_length', 'channels', 'fb_published', 'gts', 'has_logo', 'num_payouts', 'user_type']

# ommitted=['country', 'currency', 'listed',  'venue_country']
fraud_df = df[df['fraud']==1]
no_fraud_df = df[df['fraud']==0]

def compare_numericals(df, feature_list_numerical):
    for feature in feature_list_numerical:
        print feature
        fraud = fraud_df[feature]
        no_fraud = no_fraud_df[feature]

        fig = plt.figure(figsize=(8,8))
        ax1 = fig.add_subplot(1,2,1)
        ax2 = fig.add_subplot(1,2,2)
        ax1.hist(fraud)
        ax2.hist(no_fraud)
        ax1.set_title('Fraud: {}'.format(feature))
        ax2.set_title('No Fraud: {}'.format(feature))

        plt.savefig(feature)


if __name__ == '__main__':
    compare_numericals(df, feature_list_numerical)
