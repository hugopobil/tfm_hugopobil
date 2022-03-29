import re
from nltk.tokenize import TweetTokenizer
from nltk.stem.wordnet import WordNetLemmatizer

lem = WordNetLemmatizer()


def cleaning(data):
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