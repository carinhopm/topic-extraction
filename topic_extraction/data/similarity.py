import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from topic_extraction.inference.bert_embedder import BertEmbedder


def word_similarity_score(a: str, b: str) -> float:
    words_a = a.lower().split()
    words_b = b.lower().split()
    if len(words_a)>0 and len(words_b)>0:
        if len(words_a)>=len(words_a):
            max_length = len(words_a)
        else:
            max_length = len(words_b)
        union = [word_a for word_a in words_a if word_a in words_b]
        return len(union) / max_length
    else:
        return 0.0


def letter_similarity_score(a: str, b: str) -> float:
    count = 0
    if len(a)>=len(b):
        max_length = len(a)
        diff = max_length - len(b)
    else:
        max_length = len(b)
        diff = max_length - len(a)
    for i in range(max_length):
            if a[i]==b[i]: count += 1
    return (count + diff) / max_length


def cosine_similarity_score(a: str, b: str,
                            embedder: BertEmbedder) -> float:
    tensor_a = np.atleast_2d(embedder.embed_sentence(a).numpy())
    tensor_b = np.atleast_2d(embedder.embed_sentence(b).numpy())
    return cosine_similarity(tensor_a, tensor_b)

