#!/usr/bin/env python3
"""
Defines function that calculates the n-gram BLEU score for a sentence
"""


import numpy as np


def transform_grams(references, sentence, n):
    """
    Transforms references and sentence based on grams
    """
    if n == 1:
        return references, sentence

    ngram_sentence = []
    sentence_length = len(sentence)

    # generate n-grams from a sentence by iterating through its words 
    # and creating n-grams of size n
    for i, word in enumerate(sentence):
        count = 0 # number of words that can be added to form an n-gram
        w = word  # accumulate the words to form an n-gram
        for j in range(1, n):
            if sentence_length > i + j: # index does not go beyond the length of the sentence
                w += " " + sentence[i + j]
                count += 1
        if count == n - 1 : # exactly n - 1 words have been added
            ngram_sentence.append(w)

    # generate n-grams from each reference translation in the references list
    ngram_references = []

    # for each reference, generate n-grams
    for ref in references:
        ngram_ref = []
        ref_length = len(ref)

        # iterate over words in reference, and generate n-gram
        for i, word in enumerate(ref):
            count = 0
            w = word
            for j in range(1, n):
                if ref_length > i + j:
                    w += " " + ref[i + j]
                    count += 1
            if count == n - 1:
                ngram_ref.append(w)
        ngram_references.append(ngram_ref)

    return ngram_references, ngram_sentence


def ngram_bleu(references, sentence, n):
    """
    Calculates the n-gram BLEU score for a sentence
    """
    
    ngram_references, ngram_sentence = transform_grams(references, sentence, n)

    """
    # to view the n=grams
    print("n-gram references")
    print(ngram_references)
    print("\n\nn-grame sentence")
    print(ngram_sentence)
    """
    
    ngram_sentence_length = len(ngram_sentence)
    sentence_length = len(sentence)

    # keys are words from the n-gram sentence 
    # and values are the counts of those words in the n-gram sentence
    sentence_dictionary = {word: ngram_sentence.count(word) for
                           word in ngram_sentence}
    # maximum counts of n-grams from the reference translations
    references_dictionary = {}

    # iterate through each n-gram reference 
    for ref in ngram_references:
        for gram in ref:
            # For each n-gram in the reference, it updates the count in the references_dictionary 
            # if the n-gram is absent or its count in the reference is higher than the stored count
            if references_dictionary.get(gram) is None or \
               references_dictionary[gram] < ref.count(gram):
                references_dictionary[gram] = ref.count(gram)

    # keys being words from the n-gram sentence and values initially set to 0. 
    # This dictionary will track how well the generated n-grams match the reference n-grams 
    matchings = {word: 0 for word in ngram_sentence}

    # If the n-gram is found in the matchings keys, it updates the corresponding value
    for ref in ngram_references:
        for gram in matchings.keys():
            if gram in ref:
                matchings[gram] = sentence_dictionary[gram]
    print(matchings)


    # ensuring that each word's value is capped at the minimum of its count in the reference n-grams
    # making sure that the matching words in the generated n-gram sentence 
    # are not counted more than the number of times they appear in the reference n-grams. 
    # This step ensures that the calculation of precision and 
    # the BLEU score are not skewed by an excessive number of repeated words.
    for gram in matchings.keys():
        if references_dictionary.get(gram) is not None:
            matchings[gram] = min(references_dictionary[gram], matchings[gram])

    precision = sum(matchings.values()) / ngram_sentence_length

    index = np.argmin([abs(len(word) - sentence_length) for
                       word in references])
    references_length = len(references[index])

    if sentence_length > references_length:
        BLEU = 1
    else:
        BLEU = np.exp(1 - float(references_length) / sentence_length)

    BLEU_score = BLEU * precision

    return BLEU_score
