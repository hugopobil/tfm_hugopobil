import re
from textblob import TextBlob


# Function to clean tweets
def clean_tweets(twt):
    twt = re.sub("#bitcoin", 'bitcoin', twt) # removes the '#' from bitcoin
    twt = re.sub("#Bitcoin", 'Bitcoin', twt) # removes the '#' from Bitcoin
    twt = re.sub('#[A-Za-z0-9]+', '', twt) # removes any string with a '#'
    twt = re.sub('\\n', '', twt) # removes the '\n' string
    twt = re.sub('https:\/\/\S+', '', twt) # removes any hyperlinks
    return twt


# Subjectivity
def get_subjectivity(tweet):
    return TextBlob(tweet).sentiment.subjectivity


# Polarity
def get_polarity(tweet):
    return TextBlob(tweet).sentiment.polarity

