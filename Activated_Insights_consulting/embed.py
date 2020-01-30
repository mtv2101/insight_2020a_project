
import numpy as np
import pandas as pd
import spacy
from spacy.lang.en import English

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import metrics

from document import survey_doc


##############################


class embeddings(object):

    def __init__(self, question = 'both', model = 'tfidf'):

        self.question = question
        self.model = model
        self.load_unlabeled_data()
        #self.mat = self.embed_data()


    def load_unlabeled_data(self):

        paths = ['~/PycharmProjects/insight_2020a_project/Activated_Insights_consulting/AI_survey_data/2017 to mid 2018 comments.csv',
                 '~/PycharmProjects/insight_2020a_project/Activated_Insights_consulting/AI_survey_data/2018 to mid 2019 comments.csv']

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

    def load_labeled_data(self):
        labelled_data_path = ['~/PycharmProjects/insight_2020a_project/Activated_Insights_consulting/regex_scored_df.pkl']
        labeled_data = survey_doc(labelled_data_path[0])
        labeled_data.clean_labelled_data()
        self.l_df = labeled_data.df
        #self.l_data_q1 = labeled_data.df[labeled_data.df['QID'] == 61]
        #self.l_data_q2 = labeled_data.df[labeled_data.df['QID'] == 62]

        self.l_df = self.l_data_q1.append(self.l_data_q2, ignore_index=True)



    def embed_data(self):
        models = ['tfidf', 'tfidf_by_class', 'bert']
        assert self.model in models

        if self.model == 'tfidf_by_class':
            return self.tfidf_by_class(self.ul_df)
        if self.model == 'tfidf':
            return self.tfidf(self.ul_df)
        if self.model == 'bert':
            self.l_bert = self.bert(self.l_df)


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
        data = [sent for sent in df.text]
        n_features = 500
        tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,
                                           max_features=n_features,
                                           stop_words='english')
        tfidf = tfidf_vectorizer.fit_transform(data)

        # format output from sparse matrix to something I can actually use
        tfidf_mat = tfidf.toarray()

        return tfidf_mat


    def bert(self, df):

        # model = 'en_core_web_sm'  # for testing on laptop
        # model = 'en_core_web_lg'
        # model = 'en_vectors_web_lg' # many more words
        model = 'en_trf_bertbaseuncased_lg'
        nlp = spacy.load(model)

        data = [sent for sent in df.text]
        vectors = [nlp(d) for d in data]
        vectors = np.array(vectors)

        np.save('regex_labelled_bert_embeddings.npy', vectors)

        return vectors
