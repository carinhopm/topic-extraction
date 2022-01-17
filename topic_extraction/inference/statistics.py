from typing import Dict, List

import math

from topic_extraction.data.utils import STOPWORDS, separate_words


# Score for stopwords in a keyphrase
STOPWORD_SCORE = 0.0


# Calculate statistical score per word
def calculate_word_scores(candidates: List[str],
                          sentences: List[str],
                          entities: List[str],
                          w_deg: float,
                          w_ent: float,
                          w_freq: float,
                          w_pos: float):
    word_degree = {}
    word_position, word_frequency = compute_word_positions_and_frequencies(sentences)
    for cand in candidates:
        word_list = separate_words(cand, lowercase=True)
        word_list_length = len(word_list)
        word_list_degree = word_list_length - 1
        for word in word_list:
            word_degree.setdefault(word, 0)
            word_degree[word] += word_list_degree
    # Normalize word freq. to remove bias for long texts
    avg = sum(list(word_frequency.values())) / len(word_frequency)
    word_frequency = {k: v/(avg) for k, v in word_frequency.items()}
    # Calculate word scores = (ent(w)*deg(w)) / (pos(w)*freq(w))
    word_score = {}
    for item in word_degree:
        num = (w_ent if item in entities else 0.0) + (w_deg*word_degree[item])
        num += w_pos/math.log2(word_position[item]+2)
        den =  w_freq*word_frequency[item]
        word_score[item] = num / den
    return word_score


# Computes statistical score per candidate
def generate_candidate_keyword_scores(cand_list: List[str],
                                      word_score: Dict,
                                      lang: str):
    keyword_candidates = {}
    stopwords = STOPWORDS[lang]
    for cand in cand_list:
        keyword_candidates.setdefault(cand, 0)
        word_list = separate_words(cand)
        candidate_score = 0
        for word in word_list:
            if word in stopwords: 
                candidate_score += STOPWORD_SCORE
            else:
                candidate_score += word_score[word]
        keyword_candidates[cand] = candidate_score
    keyword_candidates = {key:val for key, val in keyword_candidates.items() if val != 0}
    return keyword_candidates


# Computes word first appearance position by sentence number and word frequency
def compute_word_positions_and_frequencies(sentences: List[str]):
    positions, frequencies = {}, {}
    for pos, sentence in enumerate(sentences, 1):
        for word in separate_words(sentence, lowercase=True):
            if word not in positions:
                positions[word] = pos
            frequencies.setdefault(word, 0)
            frequencies[word] += 1
    return positions, frequencies
