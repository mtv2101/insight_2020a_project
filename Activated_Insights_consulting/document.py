import pandas as pd
import spacy
from spacy.lang.en import English


#########################


class survey_doc(object):

    def __init__(self,
                 data_path,
                 header=1,
                 model = 'en_core_web_sm'):

        self.data_path = data_path
        self.header = header
        self.model = model

        self.load_csv()


    def load_csv(self):
        self.df = pd.read_csv(self.data_path, header=self.header)


    def filter_str_length(self, strng, thresh=2):
        #assert isinstance(strng, str)
        if len(strng)>thresh:
            return True
        else:
            return False


    def clean_unlabelled_data(self):
        self.df = self.df.dropna(how='any')
        self.df = self.df.rename(columns={'Comment': 'text'})
        self.df = self.df[['QID', 'text']]
        self.df = self.df[self.df['text'].apply(lambda x: isinstance(x, str))] # cut out non-string entries
        self.df = self.df[self.df['text'].apply(lambda x: self.filter_str_length(x))] # cut out strings less than 2 characters long


    def clean_labelled_data(self):
        self.df = self.df.dropna(how='any')
        self.df = self.df.rename(columns={'Comment': 'text'})
        self.df = self.df[['QID', 'text', 'JK label', 'JK sentiment']]
        self.df = self.df[self.df['JK label'] != 'No answer/Nothing']


    def init_model(self):
        self.nlp = spacy.load(self.model)
        self.parser = English()


    def make_doc_df(self):
        self.init_model()
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