import pandas as pd
import spacy
from spacy.lang.en import English


#########################


class survey_doc(object):

    def __init__(self,
                 data_path,
                 header = 1,
                 model = 'en_core_web_sm'):

        self.data_path = data_path
        self.header = header
        self.model = model

        if data_path[-4:] == '.csv':
            self.load_csv()
        elif data_path[-4:] == '.pkl':
            self.load_pickle()
        else:
            print('unrecognized file format')


    def load_csv(self):
        self.df = pd.read_csv(self.data_path, header=self.header)


    def load_pickle(self):
        self.df = pd.read_pickle(self.data_path)


    def filter_str_length(self, strng, thresh=2):
        #assert isinstance(strng, str)
        if len(strng)>thresh:
            return True
        else:
            return False


    def clean_context_free_data(self):
        # for text without headers or metadata columns
        self.df = self.df.dropna(how='any')
        self.df = self.df.rename(columns={'Comment': 'text'})
        self.df = self.df[self.df['text'].apply(lambda x: isinstance(x, str))] # cut out non-string entries
        self.df = self.df[self.df['text'].apply(lambda x: self.filter_str_length(x))] # cut out strings less than 2 characters long


    def clean_unlabelled_data(self):
        self.df = self.df.dropna(how='any')
        self.df = self.df.rename(columns={'Comment': 'text'})
        self.df = self.df[['QID', 'text']]
        self.df = self.df[self.df['text'].apply(lambda x: isinstance(x, str))] # cut out non-string entries
        self.df = self.df[self.df['text'].apply(lambda x: self.filter_str_length(x))] # cut out strings less than 2 characters long


    def clean_hand_labelled_data(self):
        self.df = pd.read_csv(self.data_path, converters={"JK label": lambda x: x.strip("[]").split(", ")})
        self.df = self.df.dropna(how='any')
        self.df = self.df.rename(columns={'Comment': 'text'})
        self.df = self.df[['Comment ID', 'QID', 'text', 'JK label', 'JK sentiment']]
        self.df = self.df[self.df['text'].apply(lambda x: isinstance(x, str))]  # cut out non-string entries
        self.df = self.df[self.df['text'].apply(lambda x: self.filter_str_length(x))]
        self.df = self.df[self.df['JK label'] != 'No answer/Nothing'] # We don't classify
        self.df = self.df[self.df['JK label'] != 'Other']
        onehot = pd.get_dummies(self.df['JK label'].apply(pd.Series).stack()).sum(level=0)
        self.df = pd.concat([self.df, onehot], axis=1)
        self.df = self.df.drop(['No answer/Nothing', 'Other'], axis=1)

        pd.to_pickle(self.df, 'hand_scored_df.pkl')


    def clean_regex_labelled_data(self):
        self.df = self.df.dropna(how='any')

        # merge rows by comment index, one-hot encode classes
        com_ids = self.df.comment_idx.unique()
        labels = []
        idx = []
        text = []
        for i,id in enumerate(com_ids):
            # for each regex match, often more than one match per comment
            idx.append(id)
            com_df = self.df[self.df['comment_idx'] == id]
            com_labs = com_df.topic.unique()
            labels.append(com_labs)
            text.append(self.df.iloc[i]['text'])

        print('regex provides ' + str(len(self.df)) + ' matches, ' + str(len(labels)) + ' unique labels')
        multilabel_df = pd.DataFrame()
        multilabel_df['comment_idx'] = idx
        multilabel_df['labels'] = labels
        multilabel_df['text'] = text

        onehot = pd.get_dummies(multilabel_df.labels.apply(pd.Series).stack()).sum(level=0)

        self.onehot = pd.concat([multilabel_df, onehot], axis=1)


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