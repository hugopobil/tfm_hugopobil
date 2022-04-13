import numpy as np
from gensim.corpora import Dictionary
from nltk.tokenize import TweetTokenizer

from msmrh_btm.models.utilities import corpus_csr
from src.text_mining.text_processing.stopwords import stopwords
from src.utilities.duplicated_docs import unique_rows_csr


def tokenize_tweets(text_list, lang='es'):
    # We tokenize in a naive way
    stop_words = stopwords[lang]
    tokenizer = TweetTokenizer(strip_handles=True, preserve_case=False, reduce_len=True)
    tokens = [
        [token for token in  tokenizer.tokenize(text)
                    if not token.isnumeric() and
                    not token.startswith('http') and
                    len(token)>1 and
                    any(char.isalpha() for char in token) and
                    not token in stop_words]
        for text in text_list]
    return tokens

def dictionary_tweets(tokens):
    # We use Gensim's dictionary:
    dictionary = Dictionary(documents=tokens)
    # dictionary.filter_extremes() This is bad for BTM
    dictionary.compactify()
    return dictionary


def bow_tweets(tokens, dictionary):
    doc2bow = [dictionary.doc2bow(doc) for doc in tokens]
    X = corpus_csr(doc2bow)
    return X

def train_bow_tweets(X, min_n_words = 3):
    doc_count = X.sum(axis=1).A1
    X_ = X[np.argwhere(doc_count >= min_n_words)[:, 0]]
    return unique_rows_csr(X_)
