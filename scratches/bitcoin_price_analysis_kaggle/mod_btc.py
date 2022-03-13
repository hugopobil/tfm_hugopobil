import matplotlib.pyplot as plt


def crosscorr(detax, detay, lag=0, method=str):
    return detax.corrwith(detay.shift(lag), method=method)['score']


# Correlation between tweets and bitcoin with different methods
def plot_correlation(tweets, bitcoin, method):
    correlation = [crosscorr(tweets, bitcoin, lag=i, method=method) for i in range(-20, 20)]
    plt.plot(range(-20, 20), correlation)
    plt.title(f"{method} cross-correlation")
    plt.xlabel("lag")
    plt.ylabel("correlation")
    plt.show()


# Plotting two axes and comparing two datasets
def plot_two_axes(tweets, bitcoin):
    fig, ax1 = plt.subplots(figsize=(20, 10))
    ax1.set_title('BTC evolution compared to Tweets Sentiment', fontsize=18)
    ax1.tick_params(labelsize=14)
    ax2 = ax1.twinx()
    ax1.plot(tweets, 'g-')
    ax2.plot(bitcoin, 'b-')
    # ax2.axis_date(btc_usd_grouped.index, btc_usd_grouped, 'b-')

    ax1.set_ylabel('Sentiment', fontsize=16)
    ax2.set_ylabel('Bitcoin', fontsize=16)
    plt.show()