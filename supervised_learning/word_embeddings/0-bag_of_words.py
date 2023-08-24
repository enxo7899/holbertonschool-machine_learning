#!/usr/bin/env python3
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

def bag_of_words(sentences, vocab=None):
    if vocab is None:
        vectorizer = CountVectorizer()
    else:
        vectorizer = CountVectorizer(vocabulary=vocab)

    embeddings = vectorizer.fit_transform(sentences).toarray()
    features = vectorizer.get_feature_names()

    return embeddings, features

if __name__ == "__main__":
    sentences = ["Holberton school is Awesome!",
                 "Machine learning is awesome",
                 "NLP is the future!",
                 "The children are our future",
                 "Our children's children are our grandchildren",
                 "The cake was not very good",
                 "No one said that the cake was not very good",
                 "Life is beautiful"]
    E, F = bag_of_words(sentences)
    print(E)
    print(F)
