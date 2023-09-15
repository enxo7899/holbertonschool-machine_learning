import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

def bag_of_words(sentences, vocab=None):
    if vocab is None:
        vectorizer = CountVectorizer()
    else:
        vectorizer = CountVectorizer(vocabulary=vocab)

    embeddings = vectorizer.fit_transform(sentences).toarray()
    features = vectorizer.get_feature_names_out()

    return embeddings, features
