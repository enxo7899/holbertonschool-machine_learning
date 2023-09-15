#!/usr/bin/env python3
"""
Defines function that calculates the n-gram BLEU score for a sentence
"""

import numpy as np

def ngram_bleu(references, sentence, n):
    def ngrams(tokens, n):
        ngrams_list = []
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i + n])
            ngrams_list.append(ngram)
        return ngrams_list

    def clipped_precision(candidate, references, n):
        candidate_ngrams = ngrams(candidate, n)
        max_clip = {}
        for ngram in candidate_ngrams:
            max_count = 0
            for ref in references:
                ref_ngrams = ngrams(ref, n)
                count = ref_ngrams.count(ngram)
                max_count = max(max_count, count)
            max_clip[ngram] = max_count
        clipped_count = sum(max_clip.values())
        total_count = len(candidate_ngrams)
        if total_count == 0:
            return 0
        precision = clipped_count / total_count
        return precision

    reference_lengths = [len(ref) for ref in references]
    candidate_length = len(sentence)

    closest_length = min(reference_lengths, key=lambda ref_len: abs(ref_len - candidate_length))

    if candidate_length > closest_length:
        brevity_penalty = 1
    else:
        brevity_penalty = np.exp(1 - closest_length / candidate_length)

    precisions = [clipped_precision(sentence, references, i) for i in range(1, n + 1)]
    log_precisions = np.log(precisions)
    
    # Calculate modified precision (BP = 1)
    if candidate_length >= closest_length:
        modified_precision = np.exp(np.mean(log_precisions))
    else:
        modified_precision = np.exp(np.mean(log_precisions) + (1 - closest_length / candidate_length))

    bleu = brevity_penalty * modified_precision

    return bleu

if __name__ == "__main__":
    references = [["the", "cat", "is", "on", "the", "mat"], ["there", "is", "a", "cat", "on", "the", "mat"]]
    sentence = ["there", "is", "a", "cat", "here"]
    n = 2

    bleu_score = ngram_bleu(references, sentence, n)
    print(bleu_score)
