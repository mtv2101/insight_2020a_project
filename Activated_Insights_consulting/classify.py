
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import metrics

import document


##############################


class classify_labels(object):

    def __init__(self, question = 'both'):

        self.question = question
        self.load_data()


    def load_data(self):

        paths = ['~/PycharmProjects/insight_2020a_project/Activated_Insights_consulting/AI_survey_data/2017 to mid 2018 comments.csv',
                 '~/PycharmProjects/insight_2020a_project/Activated_Insights_consulting/AI_survey_data/2018 to mid 2019 comments.csv']

        self.q1_df = pd.DataFrame()
        self.q2_df = pd.DataFrame()
        for data_path in paths:
            data = document.survey_doc(data_path)
            data.clean_unlabelled_data()
            data_q1 = data.df[(data.df['QID']==61) | (data.df['QID']=='Unique / Unusual')]
            self.q1_df = self.q1_df.append(data_q1, ignore_index=True)
            data_q2 = data.df[(data.df['QID']==62) | (data.df['QID']=='One Change')]
            self.q2_df = self.q2_df.append(data_q2, ignore_index=True)

        labelled_data_path = ['~/PycharmProjects/insight_2020a_project/Activated_Insights_consulting/AI_survey_data/Labeled data and sentiments 2017-18 data2.csv']
        labeled_data = document.survey_doc(labelled_data_path[0], header=0)
        labeled_data.clean_labelled_data()
        self.l_data_q1 = labeled_data.df[labeled_data.df['QID'] == 61]
        self.l_data_q2 = labeled_data.df[labeled_data.df['QID'] == 62]

        self.choose_data()


    def choose_data(self):
        if self.question=='q1':
            self.ul_df = self.q1_df
            self.l_df = self.l_data_q1
        elif self.question=='q2':
            self.ul_df = self.q2_df
            self.l_df = self.l_data_q2
        elif self.question=='both':
            self.ul_df = self.q1_df.append(self.q2_df, ignore_index=True)
            self.l_df = self.l_data_q1.append(self.l_data_q2, ignore_index=True)


    def init_model(self, model = 'logit'):

        if model == 'logit':
            self.logit = LogisticRegression(class_weight='balanced',
                                       random_state=42,
                                       multi_class='multinomial',
                                       verbose=1,
                                       max_iter=1000)


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
        n_features = 200
        tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,
                                           max_features=n_features,
                                           stop_words='english')
        tfidf = tfidf_vectorizer.fit_transform(data)

        # format output from sparse matrix to something I can actually use
        tfidf_mat = tfidf.toarray()

        return tfidf_mat


    @staticmethod
    def train_test(self, model):

        #X = self.tfidf(self.ul_df)
        X = self.tfidf(self.l_df)
        y = self.l_df['JK label'].values
        print(len(self.ul_df), len(self.l_df))
        print(X.shape, y.shape)

        self.init_model(model)

        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index].T, y[test_index].T

            self.logit.fit(X_train, y_train)
            y_pred = self.logit.predict((X_test))

            print('accuracy = ' + str(metrics.accuracy_score(y_test, y_pred)))
            print('balenced accuracy = ' + str(metrics.balanced_accuracy_score(y_test, y_pred)))
            print('macro precision = ' + str(metrics.precision_score(y_test, y_pred, average='macro')))
