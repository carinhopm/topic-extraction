import re

import numpy as np
from nltk.corpus import stopwords

STOPWORDS = {
    "en": stopwords.words('english'),
    "da": stopwords.words('danish'),
    "sv": stopwords.words('swedish'),
    "no": stopwords.words('norwegian')
}


def split_sentences(text, lowercase: bool = False):
    """
    Utility function to return a list of sentences.
    @param text The text that must be split in to sentences.
    """
    sentence_delimiters = re.compile(u'[.!?,;:\t\\\\"\\(\\)\\\'\u2019\u2013]|\\s\\-\\s')
    if lowercase:
        sentences = sentence_delimiters.split(text.lower())
    else:
        sentences = sentence_delimiters.split(text)
    return sentences



def separate_words(text, lowercase: bool = False):
    """
    Utility function to return a list of all words.
    @param text The text that must be split in to words.
    """
    splitter = re.compile(r'(?u)\W+')
    words = []
    for single_word in splitter.split(text):
        current_word = single_word.strip().lower()
        # leave numbers in phrase, but don't count as words, since they tend to invalidate scores of their phrases
        if current_word != '' and not is_number(current_word):
            if lowercase:
                words.append(current_word.lower())
            else:
                words.append(current_word)
    return words



def rescale_scores(cands_per_doc):
    """
    Utility function to rescale score values between 0 and 1.
    @param cands_per_doc
    """
    for i in range(len(cands_per_doc)):
        scores = np.asarray([cand[1] for cand in cands_per_doc[i]])
        normalized = (scores-min(scores))/(max(scores)-min(scores))
        for j in range(len(cands_per_doc[i])):
            cands_per_doc[i][j] = (cands_per_doc[i][j][0], normalized[j])
    return cands_per_doc


def is_number(s):
    try:
        float(s) if '.' in s else int(s)
        return True
    except ValueError:
        return False