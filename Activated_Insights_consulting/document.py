import pandas as pd
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import re

import spacy
from spacy.matcher import Matcher
from spacy.lang.en import English
from spacy.lang.en.stop_words import STOP_WORDS


#########################


class document(object):

    def __init__(self,
                 data_path,
                 model = 'en_core_web_sm'):

        self.data_path = data_path
        self.model = model

        self.load_csv()
        self.spacy_init()


    def load_csv(self):
        self.df = pd.read_csv(self.data_path, header=1)
        self.df = self.df.dropna(how='any')
        self.df = self.df.rename(columns={"Comment": "text"})
        self.df = self.df[:10000]


    def init_model(self):

        #self.nlp = spacy.load(self.model, disable=["tagger', 'ner'"])
        self.nlp = spacy.load(self.model)
        self.parser = English()


    def make_doc_df(self):

        self.df = pd.DataFrame()

        all_sentences = []
        all_sentence_entities = []
        all_tokens = []
        for r, comment in enumerate(self.df.text):
            doc = self.nlp(comment)
            tokens = [s.text for s in doc]
            sentences = [sent for sent in doc.sents]
            sentence_entities = [ent.text for ent in doc.ents]
            all_tokens.append(tokens)
            all_sentences.append(sentences)
            all_sentence_entities.append(sentence_entities)

        self.df['tokens'] = all_tokens
        self.df['sentences'] = all_sentences
        self.df['entities'] = all_sentence_entities