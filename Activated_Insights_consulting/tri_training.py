import numpy as np
import pandas as pd

import os
import timeit
import matplotlib.pyplot as plt
from random import choices
from numpy.random import default_rng

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.model_selection import GroupKFold, StratifiedKFold
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


def main():

    X, y, X_ul, categories = load_data()

    X_pca, X_ul_pca = pca_reduce(X, X_ul)

    models = setup_pipes()

    tri_fit(X_pca,y,X_ul_pca,models)



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
                                                                               multi_class='ovr',
                                                                               C=1.0,
                                                                               class_weight='None',
                                                                               random_state=42,
                                                                               max_iter=1000),
                                                            n_jobs=-1)),])

    Forest_pipeline = Pipeline([('clf', OneVsRestClassifier(AdaBoostClassifier(n_estimators=50,
                                                                               random_state=42),
                                                            n_jobs=-1))])

    #MLP_pipeline = Pipeline([('clf', OneVsRestClassifier(MLPClassifier(), n_jobs=-1))])
    KNN_pipeline = Pipeline([('clf', OneVsRestClassifier(KNeighborsClassifier(weights='uniform')
                                                         , n_jobs=-1))])

    return [LogReg_pipeline, Forest_pipeline, KNN_pipeline]


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


def generic_fit(model, X_train, y_train, X_test, y_test, X_ul):

    start_time = timeit.default_timer()

    model.fit(X_train, y_train)
    predictions = model.predict(X_ul)

    # calculate metrics
    y_pred = model.predict(X_test)
    acc = metrics.accuracy_score(y_test, y_pred)
    prec = metrics.precision_score(y_test, y_pred, average='samples')
    rec = metrics.recall_score(y_test, y_pred,average='macro')
    print('accuracy = ' + str(acc))
    print('macro precision = ' + str(prec))
    print('recall score = ' + str(rec))
    print("\n")

    process_time = timeit.default_timer() - start_time
    print(str(model) + ' training and predictions took: ' + str(process_time) + ' seconds')

    return predictions, acc, prec, rec


def find_consensus(preds, training_dat, training_labels, X_ul):

    X0, X1, X2 = training_dat
    y0, y1, y2 = training_labels

    assert (X0.shape[0] == y0.shape[0])
    assert (X1.shape[0] == y1.shape[0])
    assert (X2.shape[0] == y2.shape[0])

    p1, p2, p3 = preds
    ensemble = np.stack([p1, p2, p3], axis=0).astype('int16')

    consensus = np.mean(ensemble.astype('float'), axis=2)
    print(consensus.shape)
    for i in range(consensus.shape[1]):

        # if model 0 is the odd-one-out
        if (consensus[0, i] != consensus[1, i]) and (
                consensus[0, i] != consensus[2, i]) and (
                consensus[1, i] == consensus[2, i]):
            X0 = np.vstack([X0, X_ul[i, :]])
            y0 = np.vstack([y0, ensemble[1, i, :].squeeze()])

        # if model 1 is the odd-one-out
        elif (consensus[1, i] != consensus[0, i]) and (
                consensus[1, i] != consensus[2, i]) and (
                consensus[0, i] == consensus[2, i]):
            X1 = np.vstack([X1, X_ul[i, :]])
            y1 = np.vstack([y1, ensemble[0, i, :].squeeze()])

        # if model 2 is the odd-one-out
        elif (consensus[2, i] != consensus[0, i]) and (
                consensus[2, i] != consensus[1, i]) and (
                consensus[1, i] == consensus[0, i]):
            X2 = np.vstack([X2, X_ul[i, :]])
            y2 = np.vstack([y2, ensemble[1, i, :].squeeze()])

    print(X0.shape, X1.shape, X2.shape)
    print(y0.shape, y1.shape, y2.shape)

    training_dat = [X0, X1, X2]
    training_labels = [y0, y1, y2]

    return training_dat, training_labels




def tri_fit(X,y,X_ul,models,predict_size=1000):

    num_folds = 10

    X = preprocessing.scale(X)
    X_ul = preprocessing.scale(X_ul)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42,test_size=0.2)

    # initialize three copies of training data before accumulating model-specific examples
    training_dat = [X_train, X_train, X_train]
    training_labels = [y_train, y_train, y_train]

    all_acc = []
    all_prec = []
    all_rec = []

    for n in range(num_folds):

        rng = default_rng()
        rand_idx = rng.choice(np.arange(0,X_ul.shape[0], size=10000, replace=False))
        X_ul_sampled = X_ul[rand_idx, :]

        preds = []
        for m,model in enumerate(models):
            X_train = training_dat[m]
            y_train = training_labels[m]
            pred, acc, prec, rec = generic_fit(model, X_train, y_train, X_test, y_test, X_ul_sampled)
            preds.append(pred)
            all_acc.append(acc)
            all_prec.append(prec)
            all_rec.append(rec)
        training_dat, training_labels = find_consensus(preds, training_dat, training_labels, X_ul_sampled)


if __name__ == "__main__":
    main()