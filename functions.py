from text_mining.text_processing.text2bow import Text2BowTransformerPhrases
from text_mining.text_processing.tokenizer import Tokenizer


# Transform tweets to BOW
def head_body2bow(df, language='en'):
    X = df['cleaned_tweets']
    tokenizer = Tokenizer(tokenizer_model='stanza', language=language)
    text2bow = Text2BowTransformerPhrases(tokenizer=tokenizer, form_phrases=False)

    print('Starting bow creation')

    corpus_bow = text2bow.fit_transform(X)
    df['bow'] = corpus_bow

    print('End bow creation')

    return df
