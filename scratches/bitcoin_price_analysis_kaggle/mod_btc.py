import matplotlib.pyplot as plt
from sklearn import preprocessing
import re
from nltk.tokenize import TweetTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
import nltk
from textblob import TextBlob


def crosscorr(detax, detay, lag=0, method=str):
    return detax.corrwith(detay.shift(lag), method=method)['score']


# correlation between tweets and bitcoin with different methods
def plot_correlation(tweets, bitcoin, method):
    correlation = [crosscorr(tweets, bitcoin, lag=i, method=method) for i in range(-20, 20)]
    plt.plot(range(-20, 20), correlation)
    plt.title(f"{method} cross-correlation")
    plt.xlabel("lag")
    plt.ylabel("correlation")
    plt.show()


def normalization_plot(tweets, price):
    min_max_scaler = preprocessing.StandardScaler()
    score_scaled = min_max_scaler.fit_transform(tweets['score'].values.reshape(-1, 1))
    tweets['normalized_score'] = score_scaled
    # crypto_used_grouped_scaled = min_max_scaler.fit_transform(crypto_usd_grouped.values.reshape(-1,1))
    crypto_used_grouped_scaled = price / max(price.max(), abs(price.min()))
    # crypto_usd_grouped['normalized_price'] = crypto_used_grouped_scaled

    fig, ax1 = plt.subplots(figsize=(20, 10))
    ax1.set_title("Normalized Bitcoin currency evolution compared to normalized twitter sentiment", fontsize=18)
    ax1.tick_params(labelsize=14)

    ax2 = ax1.twinx()
    ax1.plot_date(tweets.index, tweets['normalized_score'], 'g-')
    ax2.plot_date(price.index, crypto_used_grouped_scaled, 'b-')

    ax1.set_ylabel("Sentiment", color='g', fontsize=16)
    ax2.set_ylabel("Bitcoin normalized", color='b', fontsize=16)
    return plt.show()


def cleaning(data):
    lem = WordNetLemmatizer()
    stop_words = nltk.corpus.stopwords.words(['english'])
    tweet_without_url = re.sub(r'http\S+',' ', data)
    tweet_without_hashtag = re.sub(r'#\w+', ' ', tweet_without_url)
    tweet_without_mentions = re.sub(r'@\w+',' ', tweet_without_hashtag)
    precleaned_tweet = re.sub('[^A-Za-z]+', ' ', tweet_without_mentions)

    # Tokenization
    tweet_tokens = TweetTokenizer().tokenize(precleaned_tweet)
    tokens_without_punc = [w for w in tweet_tokens if w.isalpha()]
    tokens_without_sw = [t for t in tokens_without_punc if t not in stop_words]

    text_cleaned = [lem.lemmatize(t) for t in tokens_without_sw]

    return " ".join(text_cleaned)


def get_subjectivity(tweet):
    return TextBlob(tweet).sentiment.subjectivity


def get_polarity(tweet):
    return TextBlob(tweet).sentiment.polarity


def btc_price_cate(score):
    if score < 1:
        return 'negative'
    elif score == 1:
        return 'neutral'
    else:
        return 'positive'

