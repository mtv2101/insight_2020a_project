import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier

import document
import embed


#################################


def one_hot(df):

    df = df.sort_values('comment_idx')
    df = df.reset_index(drop=True)
    print(df.keys())
    com_ids = df.comment_idx.unique()
    labels = []
    idx = []
    for id in com_ids:
        idx.append(id)
        com_df = df[df['comment_idx'] == id]
        com_labs = com_df.topic.unique()
        labels.append(com_labs)

    multilabel_df = pd.DataFrame()
    multilabel_df['comment_idx'] = idx
    multilabel_df['labels'] = labels

    onehot = pd.get_dummies(multilabel_df.labels.apply(pd.Series).stack()).sum(level=0)

    multilabel_df = pd.concat([multilabel_df, onehot], axis=1)

    return onehot


def main():



    categories = df.category.unique()

    LogReg_pipeline = Pipeline([('clf', OneVsRestClassifier(LogisticRegression(solver='sag'), n_jobs=-1)),])

    Forest_pipeline = Pipeline([('clf', OneVsRestClassifier(AdaBoostClassifier(n_estimators=50)))])

    NB_pipeline = Pipeline([('clf', OneVsRestClassifier(GaussianNB()))])

    for category in categories:

        LogReg_pipeline.fit(x_train, train[category])
        Forest_pipeline.fit(x_train, train[category])
        NB_pipeline.fit(x_train, train[category])

        # calculating test accuracy
        prediction = LogReg_pipeline.predict(x_test)
        print('Test accuracy is {}'.format(accuracy_score(test[category], prediction)))
        print("\n")



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
'''


if __name__ == "__main__":
    main()