from typing import List

import torch
from transformers import AutoTokenizer, AutoModelForPreTraining

MODELS = {
    "en": "bert-base-uncased",
    "da": "Maltehb/danish-bert-botxo",
    "sv": "KB/bert-base-swedish-cased",
    "no": "NbAiLab/nb-bert-base"
}



class BertEmbedder:

    def __init__(self,
                 lang: str,
                 batch_size: int = 8,
                 max_length: int = 512,
                 force_cpu: bool = False):
        self.model = {}
        self.lang = lang.lower()
        self.max_length = max_length
        self.batch_size = batch_size
        self.force_cpu = force_cpu
        self.load_model()

    def load_model(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() and not self.force_cpu else "cpu")
        self.device = device
        self.model = {
            "tokenizer": AutoTokenizer.from_pretrained(MODELS[self.lang], use_fast=True),
            "model": AutoModelForPreTraining.from_pretrained(MODELS[self.lang],
                                                             output_hidden_states=True
                                                             ).to(device).eval()}
        return self
    
    def switch_language(self, new_lang: str):
        self.lang = new_lang
        self.clear()
        self.load_model()

    def clear(self):
        del self.model
        self.model = {}
        return None
    
    @property
    def supported_languages(self) -> List[str]:
        return list(self.models.keys())


    # Tokenizes some text
    def tokenize(self, text: str):
        tokenizer = self.model['tokenizer']
        marked_text = "[CLS] " + text + " [SEP]"
        tokenized = tokenizer.tokenize(marked_text, )
        return tokenized
    
    
    # Returns word embeddings
    def embed_word(self, word: str):
        hidden_states = self.embed(word)
        # Remove dimension 1, the "batches". Result -> Tensor[layers: 13, tokens: 3, features: 768]
        token_embeddings = torch.squeeze(hidden_states, dim=1)
        # Swap dimensions 0 and 1. Result -> Tensor[tokens: 3, layers: 13, features: 768]
        token_embeddings = token_embeddings.permute(1,0,2)
        token = token_embeddings[1] # get word we look for
        # At this point there are 2 methods to generate word vectors:
        # 1. Concatenating last 4 layers -> 4 x 768 = 3,072 vector length
        #wordvec = torch.cat((token[-1], token[-2], token[-3], token[-4]), dim=0)
        # 2. Summing together the last four layers = 768 vector length
        wordvec = torch.sum(token[-4:], dim=0)
        return wordvec
    

    # Returns sentence embeddings
    def embed_sentence(self, sentence: str):
        hidden_states = self.embed(sentence)
        # A simple approach is to average the second to last hidden layer of each token.
        # `token_vecs` is a tensor with shape [tokens x 768]
        token_vecs = hidden_states[-2][0]
        # Calculate the average of all token vectors.
        sentencevec = torch.mean(token_vecs, dim=0)
        return sentencevec
    

    # Returns document embeddings
    def embed_document(self, text: str):
        tokenizer = self.model['tokenizer']
        model = self.model['model']
        marked_text = "[CLS] " + text + " [SEP]"
        tokenized = tokenizer(marked_text, truncation=True, max_length=self.max_length)
        indexed_tokens = tokenized['input_ids']
        segments_ids = tokenized['attention_mask']
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])
        with torch.no_grad():
            if not self.force_cpu:
                tokens_tensor = tokens_tensor.to(self.device)
                segments_tensors = segments_tensors.to(self.device)
            outputs = model(tokens_tensor, segments_tensors)
            # Concatenate the tensors for all layers. We use `stack` here to create a new dimension in the tensor
            # since outputs[2] is a Tuple object. Result -> Tensor[layers: 13, batches: 1, tokens: 3, features: 768]
            hidden_states = torch.stack(outputs[2], dim=0)
            if not self.force_cpu:
                hidden_states = hidden_states.to("cpu")
        # A simple approach is to average the second to last hidden layer of each token.
        # `token_vecs` is a tensor with shape [tokens x 768]
        token_vecs = hidden_states[-2][0]
        # Calculate the average of all token vectors.
        docvec = torch.mean(token_vecs, dim=0)
        return docvec

    
    # Returns hidden states
    # based on https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial
    def embed(self, text: str):
        tokenizer = self.model['tokenizer']
        model = self.model['model']
        marked_text = "[CLS] " + text + " [SEP]"
        tokenized = tokenizer.tokenize(marked_text)
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized)
        segments_ids = [1] * len(tokenized)
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])
        with torch.no_grad():
            if not self.force_cpu:
                tokens_tensor = tokens_tensor.to(self.device)
                segments_tensors = segments_tensors.to(self.device)
            outputs = model(tokens_tensor, segments_tensors)
            # Concatenate the tensors for all layers. We use `stack` here to create a new dimension in the tensor
            # since outputs[2] is a Tuple object. Result -> Tensor[layers: 13, batches: 1, tokens: 3, features: 768]
            hidden_states = torch.stack(outputs[2], dim=0)
            if not self.force_cpu:
                hidden_states = hidden_states.to("cpu")
        return hidden_states
    
