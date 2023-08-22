#!/usr/bin/env python3
"""Function to calculate BLEU score"""


import numpy as np


def uni_bleu(references, sentence):
    """
    Calculates the unigram BLEU score for a sentence
    - references: a list of reference translations,
    where each reference is a list of words
    - sentence: the proposed sentence to be evaluated,
    represented as a list of words
    """
    # length of the translated sentences
    sentence_length = len(sentence)
    # a list to store the lengths of each reference translation
    references_length = []
    # store unique words present in both the reference translations
    words = {}

    for translation in references:
        # calculate and store the length of each reference translation
        references_length.append(len(translation))
        for word in translation:
            # if word is in sentence but not in dictionary
            # to ensure counting each unique word in the proposed sentence 
            if word in sentence and word not in words.keys():
                words[word] = 1

    # total count of unique words present in both the reference 
    total = sum(words.values())
    # index of the reference translation with the closest length to the 
    index = np.argmin([abs(len(i) - sentence_length) for i in references])
    # the length of the best matching reference translation
    best_match = len(references[index])

    """
     If the proposed sentence is longer than the best matching reference,
     the precision is set to 1, indicating perfect match.
     If the proposed sentence is shorter,
     it calculates the precision using an exponential decay formula
     that considers the ratio of lengths
    """
    if sentence_length > best_match:
        BLEU = 1
    else:
        BLEU = np.exp(1 - float(best_match) / float(sentence_length))

    BLEU_score = BLEU * np.exp(np.log(total / sentence_length))

    return BLEU_score
