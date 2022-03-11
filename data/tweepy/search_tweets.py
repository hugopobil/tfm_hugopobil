from searchtweets import load_credentials, gen_rule_payload, collect_results

def _search_api(query, from_date, to_date, count_bucket, max_requests, results_per_call):
    """
    call Twitter's search api and returns the rule (enriched query) and the stream of results
    Parameters
    ----------
    query : str
        A valid Twitter search API query
    from_date: str
        Date format as specified by `convert_utc_time` for the starting time of search.
    to_date: str
        Date format as specified by `convert_utc_time` for the end time of search.
    count_bucket: (str or None)
        If None, then download tweets, else if "day", "hour", "minute" download counts.
    max_requests: int
        Maximum number of requests
    results_per_call: int
        Number of results per request

    Returns
    -------
    rule: dict
        A valid json string for using as a payload to the API call
    tweets: list of tweets or counts
        List of results
    """
    max_results = max_requests * results_per_call
    result_stream_args = credential_args_data if count_bucket is None else credential_args_counts
    rule = gen_rule_payload(
        query,
        from_date=from_date,
        to_date=to_date,
        results_per_call=results_per_call,
        count_bucket=count_bucket
    )
    tweets = collect_results(
        rule,
        max_results=max_results,
        result_stream_args=result_stream_args
    )
    return rule, tweets


def get_tweets(query, from_date, to_date, max_requests=1, results_per_call=500):
    """
    Wrapper to _search_api to get tweets. See docstring there.
    """
    rule, tweets = _search_api(query, from_date, to_date, None, max_requests, results_per_call)
    return rule, tweets


def get_tweet_counts(query, from_date, to_date, count_bucket='day', max_requests=1, results_per_call=30):
    """
    Wrapper to _search_api to get tweet counts. See doctring there.
    """
    rule, tweets = _search_api(query, from_date, to_date, count_bucket, max_requests, results_per_call)
    return rule, tweets