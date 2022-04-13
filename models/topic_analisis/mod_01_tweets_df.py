import pandas as pd
import ast
from pydash.objects import get


def get_tweets_df(tweets_fn_s3):
    tweets_df_ = pd.read_csv(tweets_fn_s3, low_memory=False, dtype={'user': object})

    # Here we collect the data we need from the tweets, neglecting the rest to easy up visualizations/explorations:
    tweets = tweets_df_[['id','created_at','text','lang']].copy()

    # Transform timestamps into valid datetime dtype:
    tweets['created_at_dt'] = pd.to_datetime(tweets['created_at'], format= '%a %b %d %H:%M:%S +0000 %Y')

    # In tweet['text'] we collect full text if existing:
    idx_full_text = tweets_df_[tweets_df_['extended_tweet'].notna()].index
    tweets.loc[idx_full_text, 'text'] = tweets_df_.loc[idx_full_text,'extended_tweet'].apply(ast.literal_eval).apply(pd.Series)['full_text']

    # Extract the user's name and user's location:
    tweets_df_['user'] = tweets_df_['user'].apply(ast.literal_eval)
    tweets['user_name'] = tweets_df_['user'].apply(pd.Series)['name']
    tweets['user_coordinates'] = tweets_df_['user'].map(
        lambda user: get(user, 'derived.locations.0.geo.coordinates'),
        na_action='ignore'
    )

    return tweets
