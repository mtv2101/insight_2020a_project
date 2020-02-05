import numpy as np
import pandas as pd
import pickle

import timeit

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA



############################################3


def main():

    X_train, y_train, X_ul = load_regex_data()

    X_test, y_test = load_hand_labelled_data()

    print(X_train.shape, X_test.shape, X_ul.shape)
    X_train, X_test, X_ul = pca_reduce(X_train, X_test, X_ul)
    print(X_train.shape, X_test.shape, X_ul.shape)

    models = setup_pipes()

    #tri_fit(X_train, X_test, y_train, y_test, X_ul, models)



def load_regex_data():

    y_path = 'regex_scored_df.pkl'
    ydf = pd.read_pickle(y_path)
    categories = ydf.keys()[3:]
    ydf_onehot = ydf[categories]
    ydf_onehot.keys()
    y = ydf_onehot.to_numpy()

    x_path = 'unlabelled_bert_embeddings.npy'
    X_mat = np.load(x_path)
    # get X vectors that represent y data
    X = X_mat[ydf['comment_idx']]
    ul_idx = [i for i in range(X_mat.shape[0]) if i not in ydf['comment_idx'].values]
    X_ul = X_mat[ul_idx]

    return X, y, X_ul


def get_unlabeled_data():

    x_path = 'unlabelled_bert_embeddings.npy'
    X_mat = np.load(x_path)

    return X_mat


def load_hand_labelled_data():

    #y_path = '/home/matt_valley/PycharmProjects/insight_2020a_project/Activated_Insights_consulting/regex_scored_df.pkl'
    y_path = 'hand_scored_df.pkl'
    # labels are encoded as a string, so here we seperate them as a list
    ydf = pd.read_pickle(y_path)
    categories = ydf.keys()[4:]
    ydf_onehot = ydf[categories]
    ydf_onehot.keys()
    y = ydf_onehot.to_numpy()

    #x_path = '/home/matt_valley/PycharmProjects/insight_2020a_project/Activated_Insights_consulting/tfidf_embeddings.npy'
    x_path = 'hand_labelled_bert_embeddings.npy'
    X_mat = np.load(x_path)

    return X_mat, y


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


def pca_reduce(X_train, X_test, X_ul):

    X_merge = np.concatenate([X_train, X_test, X_ul], axis=0)
    pca = PCA(n_components=20)
    X_xform = pca.fit_transform(X_merge)

    X_train = X_xform[:X_train.shape[0],:]
    X_test = X_xform[X_train.shape[0]:X_train.shape[0]+X_test.shape[0],:]
    X_ul = X_xform[X_train.shape[0]+X_test.shape[0]:,:]

    return X_train, X_test, X_ul


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


def find_consensus(preds, training_dat, training_labels, X_ul, training_method='classic'):

    X0, X1, X2 = training_dat
    y0, y1, y2 = training_labels

    assert (X0.shape[0] == y0.shape[0])
    assert (X1.shape[0] == y1.shape[0])
    assert (X2.shape[0] == y2.shape[0])

    p1, p2, p3 = preds
    print('unlabeled data shape: ' + str(X_ul.shape))
    ensemble = np.stack([p1, p2, p3], axis=0).astype('int16')
    #print(ensemble.shape)

    X_ul_delete_list = []
    for i in range(ensemble.shape[1]):
        if training_method == 'disagreement':
            # if model 0 is the odd-one-out add labels to its training set
            #print(ensemble[:,i,:].shape)
            #print(ensemble[0,i,:], ensemble[1,i,:], ensemble[2,i,:])
            if (ensemble[0, i, :] != ensemble[1, i, :]).any() and (
                    ensemble[0, i, :] != ensemble[2, i, :]).any() and (
                    ensemble[1, i, :] == ensemble[2, i, :]).all():
                #print(ensemble[0, i, :], ensemble[1, i, :])
                X0 = np.vstack([X0, X_ul[i, :]])
                X_ul_delete_list.append(i)
                y0 = np.vstack([y0, ensemble[1, i, :].squeeze()]) # append a correct label

            # if model 1 is the odd-one-out add labels to its training set
            elif (ensemble[0, i, :] != ensemble[1, i, :]).any() and (
                    ensemble[1, i, :] != ensemble[2, i, :]).any() and (
                    ensemble[0, i, :] == ensemble[2, i, :]).all():
                #print(ensemble[0, i, :], ensemble[1, i, :])
                X1 = np.vstack([X1, X_ul[i, :]])
                X_ul_delete_list.append(i)
                y1 = np.vstack([y1, ensemble[0, i, :].squeeze()])

            # if model 2 is the odd-one-out add labels to its training set
            elif (ensemble[2, i, :] != ensemble[1, i, :]).any() and (
                    ensemble[2, i, :] != ensemble[0, i, :]).any() and (
                    ensemble[1, i, :] == ensemble[0, i, :]).all():
                #print(ensemble[2, i, :], ensemble[1, i, :])
                X2 = np.vstack([X2, X_ul[i, :]])
                X_ul_delete_list.append(i)
                y2 = np.vstack([y2, ensemble[1, i, :].squeeze()])

        elif training_method == 'classic':
            # if all models agree, and predictions are not zero, add label to all training sets
            if (ensemble[0, i, :] == ensemble[1, i, :]).all() and (
                    ensemble[0, i, :] == ensemble[2, i, :]).all():# and (
                    #(ensemble[0, i, :]).any() == 1):
                X0 = np.vstack([X0, X_ul[i, :]])
                X1 = np.vstack([X1, X_ul[i, :]])
                X2 = np.vstack([X2, X_ul[i, :]])
                X_ul_delete_list.append(i)
                y0 = np.vstack([y0, ensemble[1, i, :].squeeze()])
                y1 = np.vstack([y1, ensemble[1, i, :].squeeze()])
                y2 = np.vstack([y2, ensemble[1, i, :].squeeze()])

    X_ul = np.delete(X_ul, X_ul_delete_list, axis=0)

    training_dat = [X0, X1, X2]
    training_labels = [y0, y1, y2]

    return training_dat, training_labels, X_ul


def tri_fit(X_train, X_test, y_train, y_test, X_ul, models, save_output=False):

    num_folds = 10

    X_train = preprocessing.scale(X_train)
    X_test = preprocessing.scale(X_test)
    X_ul = preprocessing.scale(X_ul)

    print(y_test)
    # optionally overwrite test data provided externally and test internally on a split
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, random_state=42,test_size=0.2)
    print('training data size = ' + str(X_train.shape))
    print('testing data size = ' + str(X_test.shape))
    print('target shape = ' + str(y_train.shape))
    print('unlabeled data size = ' + str(X_ul.shape))

    # initialize three copies of training data before accumulating model-specific examples
    training_dat = [X_train, X_train, X_train]
    training_labels = [y_train, y_train, y_train]

    model_metrics = [{model:{'acc':[], 'prec':[], 'rec':[], 'train_size':[]}} for model in models.keys()]

    for n in range(num_folds):

        preds = []
        for i,(m,model) in enumerate(models.items()):
            X_train = training_dat[i]
            y_train = training_labels[i]
            pred, acc, prec, rec = generic_fit(model, X_train, y_train, X_test, y_test, X_ul)
            preds.append(pred)
            model_metrics[i][m]['acc'].append(acc)
            model_metrics[i][m]['prec'].append(prec)
            model_metrics[i][m]['rec'].append(rec)
            model_metrics[i][m]['train_size'].append(X_train.shape[0])

        training_dat, training_labels, X_ul = find_consensus(preds, training_dat, training_labels, X_ul)

        if save_output:
            pickle_out = open("tri_train_metrics_disagree.pkl", "wb")
            pickle.dump(model_metrics, pickle_out)
            pickle_out.close()

            pickle_out = open("tri_train_models_disagree.pkl", "wb")
            pickle.dump(models, pickle_out)
            pickle_out.close()

            pickle_out = open("tri_train_predictions_disagree.pkl", "wb")
            pickle.dump(preds, pickle_out)
            pickle_out.close()


if __name__ == "__main__":
    main()