import numpy as np
import pandas as pd
import os
import timeit
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA

import document
import embed


############################################3


def main(num_iters=1):

    X, y, X_ul, categories = load_data()

    X_pca, X_ul_pca = pca_reduce(X, X_ul)

    models = setup_pipes()

    for n in range(num_iters):
        fit(X_pca,y,X_ul_pca,models,categories)



def load_data():

    y_path = '/home/matt_valley/PycharmProjects/insight_2020a_project/Activated_Insights_consulting/regex_scored_df.pkl'
    ydf = pd.read_pickle(y_path)
    categories = ydf.keys()[3:]
    ydf_onehot = ydf[categories]
    ydf_onehot.keys()
    y = ydf_onehot.to_numpy()
    print(y.shape)

    x_path = '/home/matt_valley/PycharmProjects/insight_2020a_project/Activated_Insights_consulting/tfidf_embeddings.npy'
    X_mat = np.load(x_path)
    # get X vectors that represent y data
    X = X_mat[ydf['comment_idx']]
    ul_idx = [i for i in range(X_mat.shape[0]) if i not in ydf['comment_idx'].values]
    X_ul = X_mat[ul_idx]
    print(X.shape, X_ul.shape)

    return X, y, X_ul, categories

def setup_pipes():

    LogReg_pipeline = Pipeline([('clf', OneVsRestClassifier(LogisticRegression(solver='sag',
                                                                               multi_class='multinomial',
                                                                               class_weight='balanced'),
                                                            n_jobs=-1)),])

    Forest_pipeline = Pipeline([('clf', OneVsRestClassifier(AdaBoostClassifier(n_estimators=50)))])

    MLP_pipeline = Pipeline([('clf', OneVsRestClassifier(MLPClassifier()))])

    return [LogReg_pipeline, Forest_pipeline, MLP_pipeline]


def pca_reduce(X, X_ul):

    X_merge = np.concatenate([X, X_ul], axis=0)
    pca = PCA(n_components=20)
    X_xform = pca.fit_transform(X_merge)
    print(X_xform.shape)
    #X_pca = pca.components_
    #plt.plot(pca.explained_variance_)
    #plt.show()

    X_pca = X_xform[:X.shape[0],:]
    X_ul_pca = X_xform[X.shape[0]:,:]
    print(X_pca.shape, X_ul_pca.shape)

    return X_pca, X_ul_pca


def fit(X,y,X_ul,models,categories):
    for model in models:
        start_time = timeit.default_timer()
        model.fit(X, y)
        process_time = timeit.default_timer() - start_time
        print(str(model) + ' training and predictions took: ' + str(process_time) + ' seconds')
'''
    for c,category in enumerate(categories):

        
        for model in models:

            #X_train, X_test, y_train, y_test = train_test_split(X, y,
            #                                                    random_state=42,
            #                                                    stratify=y,
            #                                                    test_size=0.1)

            start_time = timeit.default_timer()
            model.fit(X, y[:,c])


            # calculate metrics
            #y_pred = model.predict(X_ul)
            #print('accuracy = ' + str(metrics.accuracy_score(y_test, y_pred)))
            #print('balenced accuracy = ' + str(metrics.balanced_accuracy_score(y_test, y_pred)))
            #print('macro precision = ' + str(metrics.precision_score(y_test, y_pred,
            #                                                         average='macro')))
            #print('recall score = ' + str(metrics.recall_score(y_test, y_pred,
            #                                                   average='macro')))
            #print("\n")

            process_time = timeit.default_timer() - start_time
            print(str(model) + ' training and predictions took: ' + str(process_time) + ' seconds')
'''


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


def init_model(self, model = 'logit'):

    if model == 'logit':
        self.logit = LogisticRegression(class_weight='balanced',
                                   random_state=42,
                                   multi_class='multinomial',
                                   verbose=1,
                                   max_iter=1000)



if __name__ == "__main__":
    main()