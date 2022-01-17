from typing import Dict, List

import numpy as np
import spacy
import spacy_stanza
import stanza
from string import punctuation
from tqdm import tqdm

from topic_extraction.data.utils import STOPWORDS, separate_words


SPACY_MODELS = {
    "en": "en_core_web_sm", # check tags here: https://spacy.io/models/en
    "da": "da_core_news_sm", # check tags here: https://spacy.io/models/da
    "no": "nb_core_news_sm" # check tags here: https://spacy.io/models/nb
    # SV -> check tags here: https://universaldependencies.org/treebanks/sv_talbanken/index.html
}
HOT_POS_TAGS = ['ADJ','NOUN','PROPN']
COLD_POS_TAGS = ['ADV','VERB']
ENTITY_TAGS = {
    "en": ['EVENT','FAC','GPE','LAW','NORP','ORG','PERSON','PRODUCT'],
    "da": ['LOC','ORG','PER','MISC'],
    "no": ['LOC','ORG','PER','MISC','PROD','EVT','DRV'],
    "sv": []
}


# Returns document avg. embedding with BERT
def embed_doc(docs: List[str], 
              models: Dict,
              lang: str):
    results = []
    for doc in tqdm(docs):
        results.append(np.atleast_2d(models[lang]["model"].embed_document(doc).numpy()))
    return results
    

# Returns candidate/document avg. embedding with BERT
def embed(candidates: List[str],
          models: Dict, 
          lang: str):
    results = []
    for candidate in candidates:
        cand_embedding = np.atleast_2d(models[lang]["model"].embed_sentence(candidate).numpy())
        results.append(cand_embedding)
    return results


class WordAnalyzer:

    def __init__(self,
                 lang: str):
        self.lang = lang.lower()
        self.stopwords = STOPWORDS[self.lang]
        self.hot_tags = HOT_POS_TAGS
        self.cold_tags = COLD_POS_TAGS
        self.entity_tags = ENTITY_TAGS[self.lang]
        self.nlp = self.load_model()
    
    def load_model(self):
        if self.lang=='sv':
            stanza.download("sv")
            nlp = spacy_stanza.load_pipeline("sv")
        else:
            nlp = spacy.load(SPACY_MODELS[self.lang])
        return nlp
    
    def get_doc(self, text: str):
        return self.nlp(text)
    
    
    def get_tagged_words(self, text: str, tags: List[str]):
        results = []
        doc = self.nlp(text)
        for token in doc:
            if token.text in punctuation:
                continue
            if token.pos_ in tags:
                results.append(token.text.lower())
        return results
    

    def get_relevant_words(self, text: str):
        hotwords, coldwords, entities = [], [], []
        doc = self.nlp(text)
        for token in doc:
            if token.text in punctuation:
                continue
            if token.pos_ in self.hot_tags:
                hotwords.append(token.text.lower())
            elif token.pos_ in self.cold_tags:
                coldwords.append(token.text.lower())
            if len(self.entity_tags)>0 and token.ent_type_ in self.entity_tags:
                entities.append(token.text.lower())
        hotwords = list(set(hotwords))
        coldwords = [x for x in list(set(coldwords)) if x not in hotwords]
        entities = list(set(entities))
        return hotwords, coldwords, entities
    

    def filter_candidates(self, 
                          candidates: List[str], 
                          hotwords: List[str],
                          coldwords: List[str]):
        results = []
        for cand in candidates:
            words = separate_words(cand, lowercase=True)
            if len(words)<1:
                continue

            # Stopwords at start/end filter
            if words[0] in self.stopwords:
                continue
            if len(words)>1 and words[-1] in self.stopwords:
                continue
            
            # Hot/cold words filter
            hot = False
            cold = False
            for word in words:
                if word in coldwords:
                    cold = True
                    break
                if word in hotwords:
                    hot = True
            if not hot or cold:
                continue

            results.append(cand)
        return results
