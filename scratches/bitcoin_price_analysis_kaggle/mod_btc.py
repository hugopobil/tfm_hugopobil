import matplotlib.pyplot as plt
from sklearn import preprocessing


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
