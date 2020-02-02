import numpy as np
import pandas as pd
import pickle

import os
import timeit
import matplotlib.pyplot as plt
from random import choices
from numpy.random import default_rng

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA



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

    x_path = '/home/matt_valley/PycharmProjects/insight_2020a_project/Activated_Insights_consulting/tfidf_embeddings.npy'
    X_mat = np.load(x_path)
    # get X vectors that represent y data
    X = X_mat[ydf['comment_idx']]
    ul_idx = [i for i in range(X_mat.shape[0]) if i not in ydf['comment_idx'].values]
    X_ul = X_mat[ul_idx]

    return X, y, X_ul, categories

def setup_pipes():

    LogReg = Pipeline([('clf', OneVsRestClassifier(LogisticRegression(solver='sag',
                                                                               multi_class='ovr',
                                                                               C=1.0,
                                                                               class_weight='None',
                                                                               random_state=42,
                                                                               max_iter=1000),
                                                            n_jobs=-1))])

    AdaBoost = Pipeline([('clf', OneVsRestClassifier(AdaBoostClassifier(n_estimators=50,
                                                                               random_state=42),
                                                            n_jobs=-1))])

    RForest = Pipeline([('clf', OneVsRestClassifier(RandomForestClassifier(n_estimators=50, random_state=42, max_depth=5, min_samples_leaf=3),
                                                         n_jobs=-1))])


    MLP = Pipeline([('clf', OneVsRestClassifier(MLPClassifier(tol=1e-3, max_iter=1000),
                                                      n_jobs=-1))])

    GP = Pipeline([('clf', OneVsRestClassifier(GaussianProcessClassifier(max_iter_predict = 1000, multi_class = 'one_vs_rest'),
                                                     n_jobs=-1))])

    KNN = Pipeline([('clf', OneVsRestClassifier(KNeighborsClassifier(n_neighbors=5,
                                                                              leaf_size=100,
                                                                              weights='uniform',
                                                                              algorithm='ball_tree'),
                                                           n_jobs=-1))])

    models = {'LogReg':LogReg, 'RForest':RForest, 'MLP':MLP}
    return models


def pca_reduce(X, X_ul):

    X_merge = np.concatenate([X, X_ul], axis=0)
    pca = PCA(n_components=20)
    X_xform = pca.fit_transform(X_merge)
    #X_pca = pca.components_
    #plt.plot(pca.explained_variance_)
    #plt.show()

    X_pca = X_xform[:X.shape[0],:]
    X_ul_pca = X_xform[X.shape[0]:,:]

    return X_pca, X_ul_pca


def generic_fit(model, X_train, y_train, X_test, y_test, X_ul):

    start_time = timeit.default_timer()
    model.fit(X_train, y_train)
    process_time = timeit.default_timer() - start_time
    print('training on ' + str(X_train.shape[0]) + ' took: ' + str(process_time) + ' seconds')

    start_time = timeit.default_timer()
    predictions = model.predict(X_ul)
    process_time = timeit.default_timer() - start_time
    print('predictions on ' + str(X_ul.shape[0]) + ' unlabeled data took: ' + str(process_time) + ' seconds')

    # calculate metrics
    y_pred = model.predict(X_test)
    acc = metrics.accuracy_score(y_test, y_pred)
    prec = metrics.precision_score(y_test, y_pred, average='samples')
    rec = metrics.recall_score(y_test, y_pred,average='macro')
    print('accuracy = ' + str(acc))
    print('macro precision = ' + str(prec))
    print('recall score = ' + str(rec))

    return predictions, acc, prec, rec


def find_consensus(preds, training_dat, training_labels, X_ul):

    X0, X1, X2 = training_dat
    y0, y1, y2 = training_labels

    assert (X0.shape[0] == y0.shape[0])
    assert (X1.shape[0] == y1.shape[0])
    assert (X2.shape[0] == y2.shape[0])

    p1, p2, p3 = preds
    print('predictions shape: ' + str(p1.shape) + ' ' + str(p2.shape) + ' ' + str(p3.shape))
    print('unlabeled data shape: ' + str(X_ul.shape))
    ensemble = np.stack([p1, p2, p3], axis=0).astype('int16')
    #print(ensemble.shape)

    X_ul_delete_list = []
    for i in range(ensemble.shape[1]):

        # if model 0 is the odd-one-out add labels to its training set
        if (ensemble[0, i, :] != ensemble[1, i, :]).all() and (
                ensemble[0, i, :] != ensemble[2, i, :]).all() and (
                ensemble[1, i, :] == ensemble[2, i, :]).all():
            #print(consensus[0,i], consensus[1,i], consensus[2,i])
            X0 = np.vstack([X0, X_ul[i, :]])
            X_ul_delete_list.append(i)
            y0 = np.vstack([y0, ensemble[1, i, :].squeeze()])

        # if model 1 is the odd-one-out add labels to its training set
        elif (ensemble[0, i, :] != ensemble[1, i, :]).all() and (
                ensemble[1, i, :] != ensemble[2, i, :]).all() and (
                ensemble[0, i, :] == ensemble[2, i, :]).all():
            #print(X1.shape, i)
            X1 = np.vstack([X1, X_ul[i, :]])
            X_ul_delete_list.append(i)
            y1 = np.vstack([y1, ensemble[0, i, :].squeeze()])

        # if model 2 is the odd-one-out add labels to its training set
        elif (ensemble[2, i, :] != ensemble[1, i, :]).all() and (
                ensemble[2, i, :] != ensemble[0, i, :]).all() and (
                ensemble[1, i, :] == ensemble[0, i, :]).all():
            #print(X2.shape, i)
            X2 = np.vstack([X2, X_ul[i, :]])
            X_ul_delete_list.append(i)
            y2 = np.vstack([y2, ensemble[1, i, :].squeeze()])

        # if all models agree, and predictions are not zero, add labels to all training sets
        elif (ensemble[0, i, :] == ensemble[1, i, :]).all() and (
                ensemble[0, i, :] == ensemble[2, i, :]).all() and (
                (np.sum(ensemble[0, i, :]) > 0)):

            X0 = np.vstack([X0, X_ul[i, :]])
            X1 = np.vstack([X1, X_ul[i, :]])
            X2 = np.vstack([X2, X_ul[i, :]])
            X_ul_delete_list.append(i)
            y0 = np.vstack([y0, ensemble[1, i, :].squeeze()])
            y1 = np.vstack([y1, ensemble[1, i, :].squeeze()])
            y2 = np.vstack([y2, ensemble[1, i, :].squeeze()])

    print(X0.shape, X_ul.shape)
    #X_ul.delete(X_ul_delete_list, axis=0)
    X_ul = np.delete(X_ul, X_ul_delete_list, axis=0)
    print(X0.shape, X1.shape, X2.shape)
    print(y0.shape, y1.shape, y2.shape)

    training_dat = [X0, X1, X2]
    training_labels = [y0, y1, y2]

    return training_dat, training_labels, X_ul


def tri_fit(X,y,X_ul,models):

    num_folds = 10

    X = preprocessing.scale(X)
    X_ul = preprocessing.scale(X_ul)

    #X_ul = X_ul[:50000,:]
    X = X[:1000,:]
    y = y[:1000,:]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42,test_size=0.2)
    print('training data size = ' + str(X_train.shape))
    print('testing data size = ' + str(X_train.shape))
    print('unlabeled data size = ' + str(X_ul.shape))

    # initialize three copies of training data before accumulating model-specific examples
    training_dat = [X_train, X_train, X_train]
    training_labels = [y_train, y_train, y_train]

    model_metrics = [{model:{'acc':[], 'prec':[], 'rec':[], 'train_size':[]}} for model in models.keys()]

    for n in range(num_folds):

        preds = []
        for i,(m,model) in enumerate(models.items()):
            #print('training ' + str(model))
            X_train = training_dat[i]
            y_train = training_labels[i]
            pred, acc, prec, rec = generic_fit(model, X_train, y_train, X_test, y_test, X_ul)
            preds.append(pred)
            model_metrics[i][m]['acc'].append(acc)
            model_metrics[i][m]['prec'].append(prec)
            model_metrics[i][m]['rec'].append(rec)
            model_metrics[i][m]['train_size'].append(X_train.shape[0])

        training_dat, training_labels, X_ul = find_consensus(preds, training_dat, training_labels, X_ul)

        pickle_out = open("tri_train_metrics.pkl", "wb")
        pickle.dump(model_metrics, pickle_out)
        pickle_out.close()

        pickle_out = open("tri_train_models.pkl", "wb")
        pickle.dump(models, pickle_out)
        pickle_out.close()


if __name__ == "__main__":
    main()