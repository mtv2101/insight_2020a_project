
import numpy as np
import pandas as pd
import spacy

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from spacy_transformers import TransformersLanguage, TransformersWordPiecer, TransformersTok2Vec

from document import survey_doc


##############################

'''
embed.py takes .csv tables of comment text, 
or pickled files containing a pandas dataframe formatted 
using doucment.py.  

It embeds the text using tf/idf, or BERT
where each comment is treated as an independent document. 

It outputs a numpy array of rows(comments) by cols(vector features)

major dependencies are the spacy transformers library:
https://github.com/explosion/spacy-transformers

scikit-learn:
https://scikit-learn.org/stable/

requires python>3.6

'''


class embeddings(object):

    def __init__(self, question = 'both', model = 'bert'):

        self.question = question
        self.model = model
        #self.load_unlabeled_data()
        self.data = self.load_unlabeled_data()
        self.embed_data()


    def load_unlabeled_data(self):

        #paths = ['~/PycharmProjects/insight_2020a_project/Activated_Insights_consulting/AI_survey_data/2017 to mid 2018 comments.csv',
        #         '~/PycharmProjects/insight_2020a_project/Activated_Insights_consulting/AI_survey_data/2018 to mid 2019 comments.csv']
        paths = ['AI_data/2017_2018_comments.csv',
                 'AI_data/2018_2019_comments.csv']

        self.q1_df = pd.DataFrame()
        self.q2_df = pd.DataFrame()
        for data_path in paths:
            data = survey_doc(data_path)
            data.clean_unlabelled_data()
            data_q1 = data.df[(data.df['QID']==61) | (data.df['QID']=='Unique / Unusual')]
            self.q1_df = self.q1_df.append(data_q1, ignore_index=True)
            data_q2 = data.df[(data.df['QID']==62) | (data.df['QID']=='One Change')]
            self.q2_df = self.q2_df.append(data_q2, ignore_index=True)

        self.ul_df = self.q1_df.append(self.q2_df, ignore_index=True)

        return self.ul_df

    def load_regex_labeled_data(self):
        #data_path = ['~/PycharmProjects/insight_2020a_project/Activated_Insights_consulting/regex_scored_df.pkl']
        data_path = ['AI_data/regex_scored_df.pkl']
        labeled_data = survey_doc(data_path[0])
        labeled_data.clean_regex_labelled_data()
        self.l_df = labeled_data.df
        #self.l_data_q1 = labeled_data.df[labeled_data.df['QID'] == 61]
        #self.l_data_q2 = labeled_data.df[labeled_data.df['QID'] == 62]

        #self.l_df = self.l_data_q1.append(self.l_data_q2, ignore_index=True)

        return self.l_df

    def load_hand_labeled_data(self):
        #data_path = ['~/PycharmProjects/insight_2020a_project/Activated_Insights_consulting/hand_scored_df.pkl']
        data_path  =['AI_data/hand_scored_df.pkl']
        self.l_df = pd.read_pickle(data_path[0])
        return self.l_df


    def embed_data(self):
        models = ['tfidf', 'tfidf_by_class', 'bert']
        assert self.model in models

        if self.model == 'tfidf_by_class':
            return self.tfidf_by_class(self.data)
        if self.model == 'tfidf':
            return self.tfidf(self.data)
        if self.model == 'bert':
            self.l_bert = self.bert(self.data)


    def tfidf_by_class(self, df):
        '''
        get tfidf vectors on labeled and unlabelled data
        '''
        all_tfidf = []
        topics = df['topic'].unique()
        for topic in topics:
            temp_df = df[df['topic'] == topic]
            data = [sent for sent in df.comment_text]
            n_features = 200
            tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,
                                               max_features=n_features,
                                               stop_words='english')
            tfidf = tfidf_vectorizer.fit_transform(data)
            all_tfidf.append(tfidf)

        # format output from sparse matrix to something I can actually use
        tfidf_mat = np.empty((0, all_tfidf[0].shape[1]), dtype=all_tfidf[0].dtype)
        for t, topic_vecs in enumerate(all_tfidf):
            tfidf_mat = np.append(t, topic_vecs.toarray(), axis=0)

        return tfidf_mat


    def tfidf(self, df):
        '''
        get tfidf vectors on labeled and unlabelled data
        '''
        data = [comment for comment in df.text]
        n_features = 200
        tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,
                                           max_features=n_features,
                                           stop_words='english')
        tfidf = tfidf_vectorizer.fit_transform(data)


        # format output from sparse matrix to something I can actually use
        tfidf_mat = tfidf.toarray()

        return tfidf_mat


    def bert(self, df):

        model = 'bert-base-uncased'
        #nlp = spacy.load(model)

        nlp = TransformersLanguage(trf_name=model, meta={"lang": "en"})
        sentencizer = nlp.create_pipe("sentencizer")
        nlp.add_pipe(sentencizer, first=True)
        nlp.add_pipe(TransformersWordPiecer.from_pretrained(nlp.vocab, model))
        nlp.add_pipe(TransformersTok2Vec.from_pretrained(nlp.vocab, model))

        #data = [sent for sent in df.text]
        #docs = [nlp(d) for d in data]
        tensors = []
        print(len(df))
        for comment in df.text:
            doc = nlp(comment)
            #tensor = np.array(doc._.trf_last_hidden_state)
            tensor = doc._.trf_last_hidden_state
            print(tensor.shape)
            tensors.append(tensor.sum(axis=0))
        tensors = np.array(tensors)

        np.save('hand_labelled_bert_embeddings.npy', tensors)

        return tensors
