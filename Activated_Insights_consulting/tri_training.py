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


#################################





    LogReg_pipeline = Pipeline([('clf', OneVsRestClassifier(LogisticRegression(solver='sag'), n_jobs=-1)),])

    Forest_pipeline = Pipeline([('clf', OneVsRestClassifier(AdaBoostClassifier(n_estimators=50)))])

    NB_pipeline = Pipeline([('clf', OneVsRestClassifier(GaussianNB))])


    for category in categories:
        print('**Processing {} comments...**'.format(category))

        LogReg_pipeline.fit(x_train, train[category])
        Forest_pipeline.fit(x_train, train[category])
        NB_pipeline.fit(x_train, train[category])

        # calculating test accuracy
        prediction = LogReg_pipeline.predict(x_test)
        print('Test accuracy is {}'.format(accuracy_score(test[category], prediction)))
        print("\n")