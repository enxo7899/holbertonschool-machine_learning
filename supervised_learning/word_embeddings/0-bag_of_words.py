#!/usr/bin/env python3

import numpy as np

def bag_of_words(sentences, vocab=None):
    if vocab is None:
        # Extract unique words from all sentences to build the vocabulary
        all_words = ' '.join(sentences).split()
        vocab = list(set(all_words))
    
    # Create a dictionary to map words to indices in the vocabulary
    word_to_index = {word: index for index, word in enumerate(vocab)}

    # Initialize the embeddings matrix with zeros
    embeddings = np.zeros((len(sentences), len(vocab)), dtype=int)

    # Fill in the embeddings matrix
    for i, sentence in enumerate(sentences):
        words = sentence.split()
        for word in words:
            if word in vocab:
                embeddings[i][word_to_index[word]] += 1

    return embeddings, vocab

if __name__ == "__main__":
    sentences = ["Holberton school is Awesome!",
                 "Machine learning is awesome",
                 "NLP is the future!",
                 "The children are our future",
                 "Our children's children are our grandchildren",
                 "The cake was not very good",
                 "No one said that the cake was not very good",
                 "Life is beautiful"]
    
    # Test with vocab=None (all words within sentences)
    E, F = bag_of_words(sentences)
    print("Vocab is None:")
    print(E)
    print(F)
    
    # Test with a custom vocab
    custom_vocab = ['awesome', 'future', 'cake', 'good', 'is', 'learning', 'beautiful']
    E, F = bag_of_words(sentences, vocab=custom_vocab)
    print("\nVocab is not None:")
    print(E)
    print(F)
