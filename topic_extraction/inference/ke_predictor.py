from typing import List, Dict

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from topic_extraction.data.preprocessing import preprocess_text
from topic_extraction.data.utils import split_sentences, rescale_scores
from topic_extraction.inference.bert_embedder import BertEmbedder
from topic_extraction.inference.linguistics import embed, embed_doc, WordAnalyzer
from topic_extraction.inference.statistics import calculate_word_scores, generate_candidate_keyword_scores



class KEPredictor(BaseEstimator):
    
    def __init__(self,
                 langs: List[str],
                 minFreq: int = 1,
                 w_deg: float = 2.0,
                 w_ent: float = 4.0,
                 w_freq: float = 5.0,
                 w_pos: float = 50.0,
                 k_plus: int = 50,
                 force_cpu: bool = False,
                 f_sem: float = 0.9):
        print(f'Initializing KE Predictor for {langs}...')
        self.models = {}
        self.langs = langs
        self.minFreq = minFreq
        self.w_deg = w_deg
        self.w_ent = w_ent
        self.w_freq = w_freq
        self.w_pos = w_pos
        self.k_plus = k_plus
        self.force_cpu = force_cpu
        self.f_sem = f_sem
        self.load_models(langs)
        print(f'Input parameters: {self.get_params()}\n')

    def load_models(self, langs: List[str]):
        for lang in langs:
            self.models[lang.lower()] = {
                "model": BertEmbedder(lang.lower())}
        return self

    def clear(self):
        del self.models
        self.models = {}
        return None
    
    @property
    def supported_languages(self) -> List[str]:
        return list(self.models.keys())
    
    def fit(self, X, y, **fit_params):
        return self # unsupervised

    
    def predict(self,
                docs: List[str],
                lang: str = 'en',
                keyword_len: int = 4,
                num_keywords: int = 10,
                incl_score: bool = False):
        lang = lang.lower()

        # Text cleaning
        print(f'Preprocessing {len(docs)} docs...')
        docs = [preprocess_text(doc) for doc in tqdm(docs)]

        # Extract candidates
        print(f'Generating keyphrase candidates...')
        cvs = [CountVectorizer(ngram_range=(1, keyword_len), 
                               min_df=self.minFreq,
                               lowercase=False).fit(split_sentences(doc))
                                               .get_feature_names() for doc in tqdm(docs)]
        print(f'Filtering  keyphrase candidates per document...')
        cands_per_doc, entities_per_doc = self._filter_candidates(docs, cvs, lang)

        # Score candidates via statistical evaluation
        print(f'Computing word scores via statistics...')
        word_scores_per_doc = [calculate_word_scores(cands, 
                                                     split_sentences(docs[i]),
                                                     entities_per_doc[i],
                                                     self.w_deg,
                                                     self.w_ent,
                                                     self.w_freq,
                                                     self.w_pos) for i, cands in tqdm(enumerate(cands_per_doc))]

        # Score candidates based on their words
        print(f'Computing keyword scores via statistics for filtering...')
        cand_scores_per_doc = [generate_candidate_keyword_scores(cands_per_doc[i], 
                                                                 scores,
                                                                 lang) for i, scores in tqdm(enumerate(word_scores_per_doc))]
        cands_per_doc = [list(dict(sorted(cand_scores.items(), 
                                          key=lambda item: item[1], 
                                          reverse=True)).items()) for cand_scores in cand_scores_per_doc]
        cands_per_doc = rescale_scores(cands_per_doc)

        # Case when the linguistic module is not required
        if self.f_sem==0.0:
            if incl_score:
                keywords = [[cand for cand in cands[:num_keywords]] for cands in cands_per_doc]
            else:
                keywords = [[cand[0] for cand in cands[:num_keywords]] for cands in cands_per_doc]
            return keywords

        # Generate doc. embeddings
        print(f'Generating doc-embeddings...')
        doc_embeddings = embed_doc(docs, self.models, lang)

        # Generate candidate embeddings
        print(f'Generating candidate-embeddings...')
        k_plus = self.k_plus if self.k_plus>=num_keywords else num_keywords	
        cands_per_doc = [cands[:k_plus] for cands in cands_per_doc]
        cand_embeddings = [embed([c[0] for c in cdocs], self.models, lang) for cdocs in tqdm(cands_per_doc)]

        # Extract and sort keywords
        keywords = []
        print(f'Filtering and ordering final keyphrases per doc...')
        for index, _ in tqdm(enumerate(docs)):
            # get string candidates from each doc
            doc_candidates = [c[0] for c in cands_per_doc[index]]
            if doc_candidates:
                # get document candidate embeddings
                doc_cand_embeddings = np.array(cand_embeddings[index])
                sims = {}
                # compute and order similarities (score) per candidate
                for cand_idx, cand in enumerate(doc_candidates):
                    sims[cand] = cosine_similarity(doc_embeddings[index], doc_cand_embeddings[cand_idx])[0]
                for cand in cands_per_doc[index]:
                    sims[cand[0]] = (1-self.f_sem)*cand[1] + self.f_sem*sims[cand[0]]
                # sort keywords by their global score
                sorted_sims = list(dict(sorted(sims.items(), key=lambda item: item[1], reverse=True)).items())
                # extract the requested num. of keywords
                if incl_score:
                    doc_keywords = [(sim[0], round(float(sim[1]), 4)) for sim in sorted_sims[:num_keywords]]
                else:
                    doc_keywords = [sim[0] for sim in sorted_sims[:num_keywords]]
                keywords.append(doc_keywords)
            else:
                keywords.append(["None Found"])

        return keywords
    

    # Applies different filters to remove useless keyword candidates; extracts entities
    def _filter_candidates(self, docs: List[str], doc_cands: List[str], lang: str):
        results, entities_per_doc = [], []
        analyzer = WordAnalyzer(lang)
        for i, cands in tqdm(enumerate(doc_cands)):
            hotwords, coldwords, entities = analyzer.get_relevant_words(docs[i])
            new_cands = analyzer.filter_candidates(cands, hotwords, coldwords)
            results.append(new_cands)
            entities_per_doc.append(entities)
        return results, entities_per_doc

    
