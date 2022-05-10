import pandas as pd
from IPython.display import display
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import calendar
import plotly.express as px


def get_topics_df(btm, dictionary, n_top_words=10):
    n_topics = btm.n_topics
    topic_word_order = btm.beta.argsort(axis=1)
    topic_words = [
        ','.join(
            [dictionary[word_id]
             for word_id in topic_word_order[topic_id, -n_top_words:][::-1]])
        for topic_id in range(n_topics)]

    topics = pd.DataFrame(topic_words, columns=['title'])
    topics.index.name = 'topic_id'
    return topics


def get_topic_tweets_df(btm, X, tweets, dictionary, min_tokens=3, min_topic_prob=0.3, return_theta=False):
    _, theta = btm.transform(X)
    # topic_prob = btm.get_theta()
    topic_prob = theta.sum(axis=0)
    topic_no_list = topic_prob.argsort()[::-1]
    topic_id2no = {
        topic_id: topic_no + 1
        for topic_no, topic_id in enumerate(topic_no_list)
    }

    tweets = tweets.copy()
    tweets['topic_id'] = theta.argmax(axis=1)
    tweets['topic_prob'] = theta.max(axis=1)

    if return_theta:
        theta_df = pd.DataFrame(theta).rename(columns=topic_id2no)
        tweets = pd.concat([tweets.set_index(theta_df.index), theta_df], axis=1)

    mask = (tweets.tokens.str.len() >= min_tokens) & (tweets.topic_prob >= min_topic_prob)
    tweets = tweets[mask].copy()
    topics = get_topics_df(btm, dictionary)
    topics['tweets_count'] = tweets.groupby('topic_id')['index'].count()
    topics['tweets_theta_count'] = theta[mask.values].sum(axis=0)
    topics['topic_no'] = topics.reset_index()['topic_id'].map(lambda x: topic_id2no[x])

    return topics, tweets


def show_topic(topic_no, topics, tweets, n_tweets=10, date_=None):
    topic = topics[topics.topic_no == topic_no]
    topic_id = topic.index[0]
    title = topics.at[topic_id, 'title']
    display(title)
    df = tweets[(tweets.topic_id == topic_id)]
    if date_ is not None:
        df = df.loc[df['date_clean'].dt.date == date_, :]
        display(f'Tweets in topic:{len(df)}')

    display(df.sample(min(n_tweets, len(df)))[['date_clean', 'cleaned_tweets', 'topic_prob']].style.set_properties(**{
        'text-align' : 'left',
        'white-space': 'wrap',
    }))


def get_daily_counts(topic_no, topics, tweets):
    topic_id = topics[topics.topic_no == topic_no].index[0]
    daily_counts = (
        tweets[tweets.topic_id == topic_id]
            .pipe(lambda df: df.groupby(df.topic_id)['index'].count())
    )
    return daily_counts


def plot_month(month, topic_no, topics, tweets):
    daily_counts = get_daily_counts(topic_no, topics, tweets)
    month_counts = daily_counts.loc[daily_counts.index.month == month]
    ax = month_counts.plot.bar(title=calendar.month_name[month])
    x_labels = month_counts.index.strftime('%d').to_list()
    ax.set_xticklabels(x_labels)
    plt.show()


def plot_daily_counts(topic_no, topics, tweets):
    daily_counts = get_daily_counts(topic_no, topics, tweets)
    fig = px.line(daily_counts)
    return fig
    # daily_counts.plot.line()
    # plt.show()


def plot_tweets_histogram(
        tweets, topics, topicno2label_dct,
        bin_size=1000 * 3600 * 3, topic_prob_cutoff=0.3):
    # bin size must be given in miliseconds
    fig = go.Figure()
    for topic_no in topicno2label_dct:
        topic_id = topics[topics['topic_no'] == topic_no].index.tolist()[0]
        publication_timestamps = tweets[
            (tweets['topic_id'] == topic_id) & (tweets['topic_prob'] > topic_prob_cutoff)][
            'created_at_dt']
        fig.add_trace(
            go.Histogram(
                x=publication_timestamps,
                xbins=dict(
                    start=str(publication_timestamps.min()),
                    end=str(publication_timestamps.max()),
                    size=bin_size),
                name=topicno2label_dct[topic_no]))
    fig.update_layout(showlegend=True)
    return fig
