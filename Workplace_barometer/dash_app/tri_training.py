import numpy as np
import pandas as pd
import pickle
import os
import timeit
import time

from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA

from skmultilearn.model_selection import iterative_train_test_split

from dash_app.predict_and_plot import get_classes
from dash_app.load_text import load_unlabeled_data

############################################
# Matt Valley, Jan 2020
############################################


def main(data_source='local'):

    if data_source == 'remote':
        y_path = 'regex_scored_all_df.pkl'
        x_path = 'unlabelled_bert_embeddings.npy'
        hl_y_path = 'hand_scored_df.pkl'
        hl_x_path = 'hand_labelled_bert_embeddings.npy'

    elif data_source == 'local':
        y_path = '/home/matt_valley/PycharmProjects/insight_2020a_project/Workplace_barometer/output/regex_scores_20200216-033348.pkl'
        x_path = '/home/matt_valley/PycharmProjects/insight_2020a_project/Workplace_barometer/output/bert_mean_embeddings20200216-044222.npy'
        #hl_y_path = '/home/matt_valley/PycharmProjects/insight_2020a_project/Workplace_barometer/output/hand_scored_df.pkl'
        #hl_x_path = '/home/matt_valley/PycharmProjects/insight_2020a_project/Workplace_barometer/output/hand_labelled_bert_embeddings.npy'

    ul_df = load_unlabeled_data()
    X, y, ul_df = load_regex_data(x_path, y_path, ul_df)
    #X_test, y_test = load_hand_labelled_data(hl_x_path, hl_y_path)
    #X_train, y_train = load_hand_labelled_data(hl_x_path, hl_y_path)

    X = pca_reduce(X)

    models = setup_pipes()

    tri_fit(X, y, ul_df, models)
    #tri_fold_skf(X, y, ul_df, models)


def load_regex_data(x_path, y_path, ul_df):

    ydf = pd.read_pickle(y_path)
    # given ul_df is our master df of text and labels, merge df containing regex labels
    ul_df = ul_df.merge(ydf, right_on='comment_idx', how='left', left_index=True)
    ul_df = ul_df.reset_index()

    classes = get_classes()
    ul_df_onehot = ul_df[ul_df.columns.intersection(classes)]
    ul_df_onehot.keys()
    y = ul_df_onehot.to_numpy()

    X = np.load(x_path)

    return X, y, ul_df


def load_hand_labelled_data(hl_x_path, hl_y_path):

    # labels are encoded as a string, so here we seperate them as a list
    ydf = pd.read_pickle(hl_y_path)
    categories = ydf.keys()[4:]
    ydf_onehot = ydf[categories]
    ydf_onehot.keys()
    y = ydf_onehot.to_numpy()

    X_mat = np.load(hl_x_path)

    return X_mat, y


def setup_pipes():

    LogReg = Pipeline([('LogReg', OneVsRestClassifier(LogisticRegression(penalty='l2',
                                                                           solver='lbfgs',
                                                                           multi_class='ovr',
                                                                           C=0.001,
                                                                           class_weight='balanced',
                                                                           random_state=42,
                                                                           max_iter=1000,
                                                                           warm_start=True
                                                                      ),
                                                            n_jobs=-1))])

    SVC = Pipeline([('SVC', OneVsRestClassifier(LinearSVC(multi_class='ovr',
                                                   C=0.0001,
                                                   class_weight='balanced'),
                                                n_jobs=-1))])

    AdaBoost = Pipeline([('AdaBoost', OneVsRestClassifier(AdaBoostClassifier(n_estimators=50,
                                                                        random_state=42),
                                                            n_jobs=-1))])

    RForest = Pipeline([('RForest', OneVsRestClassifier(RandomForestClassifier(n_estimators=100,
                                                                           random_state=42,
                                                                           max_depth=5,
                                                                           min_samples_leaf=3,
                                                                           class_weight='balanced_subsample'
                                                                           ),
                                                            n_jobs=-1))])

    MLP = Pipeline([('MLP', OneVsRestClassifier(MLPClassifier(hidden_layer_sizes=(100),
                                                                tol=1e-4,
                                                                alpha=0.0001,
                                                                max_iter=1000,
                                                                activation='logistic'),
                                                            n_jobs=-1))])

    GP = Pipeline([('GP', OneVsRestClassifier(GaussianProcessClassifier(max_iter_predict = 1000,
                                                                        multi_class = 'one_vs_rest'),
                                                     n_jobs=-1))])

    KNN = Pipeline([('KNN', OneVsRestClassifier(KNeighborsClassifier(n_neighbors=5,
                                                                              leaf_size=100,
                                                                              weights='uniform',
                                                                              algorithm='ball_tree'),
                                                           n_jobs=-1))])

    #models = {'SVC':SVC, 'RForest':RForest, 'MLP':MLP}
    models = {'SVC': SVC, 'LogReg': LogReg, 'LogReg2': LogReg}
    return models


def pca_reduce(X):

    pca = PCA(n_components=20)
    X = pca.fit_transform(X)

    return X


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
    prec = metrics.precision_score(y_test, y_pred, average='weighted')
    prec_all = metrics.precision_score(y_test, y_pred, average=None)
    rec = metrics.recall_score(y_test, y_pred,average='macro')
    print('accuracy = ' + str(acc))
    print('macro precision = ' + str(prec))
    print('recall score = ' + str(rec))

    return predictions, acc, prec, prec_all, rec


def add_label(ul_df, idx, hot_lab):
    classes = get_classes()
    labels = [classes[idx] for idx,i in enumerate(hot_lab) if i==1]
    ul_df['labels'].iloc[idx] = labels
    return ul_df


def find_consensus(preds, training_dat, training_labels, X_ul, ul_df, unlabelled_idx, training_method='classic'):

    X0, X1, X2 = training_dat
    y0, y1, y2 = training_labels

    assert (X0.shape[0] == y0.shape[0])
    assert (X1.shape[0] == y1.shape[0])
    assert (X2.shape[0] == y2.shape[0])

    p1, p2, p3 = preds
    print('unlabeled data shape: ' + str(X_ul.shape))
    ensemble = np.stack([p1, p2, p3], axis=0).astype('int16')

    print(len(unlabelled_idx), ensemble.shape)
    assert len(unlabelled_idx) == ensemble.shape[1]

    # unlabelled index gives the indices within ul_df where label==np.nan
    for i,idx in enumerate(unlabelled_idx):
        if training_method == 'disagreement':
            # if model 0 is the odd-one-out add labels to its training set
            if (ensemble[0, i, :] != ensemble[1, i, :]).any() and (
                    ensemble[0, i, :] != ensemble[2, i, :]).any() and (
                    ensemble[1, i, :] == ensemble[2, i, :]).all():
                #print(ensemble[0, i, :], ensemble[1, i, :])
                X0 = np.vstack([X0, X_ul[i, :]])
                y0 = np.vstack([y0, ensemble[1, i, :].squeeze()]) # append a correct label
                X_ul = np.delete(X_ul, i, axis=0)
                del unlabelled_idx[i]
                ul_df = add_label(ul_df, idx, ensemble[1, i, :].squeeze())

            # if model 1 is the odd-one-out add labels to its training set
            elif (ensemble[0, i, :] != ensemble[1, i, :]).any() and (
                    ensemble[1, i, :] != ensemble[2, i, :]).any() and (
                    ensemble[0, i, :] == ensemble[2, i, :]).all():
                X1 = np.vstack([X1, X_ul[i, :]])
                y1 = np.vstack([y1, ensemble[0, i, :].squeeze()])
                X_ul = np.delete(X_ul, i, axis=0)
                del unlabelled_idx[i]
                ul_df = add_label(ul_df, idx, ensemble[0, i, :].squeeze())

            # if model 2 is the odd-one-out add labels to its training set
            elif (ensemble[2, i, :] != ensemble[1, i, :]).any() and (
                    ensemble[2, i, :] != ensemble[0, i, :]).any() and (
                    ensemble[1, i, :] == ensemble[0, i, :]).all():
                X2 = np.vstack([X2, X_ul[i, :]])
                y2 = np.vstack([y2, ensemble[1, i, :].squeeze()])
                X_ul = np.delete(X_ul, i, axis=0)
                del unlabelled_idx[i]
                ul_df = add_label(ul_df, idx, ensemble[1, i, :].squeeze())

        elif training_method == 'classic':
            # if all models agree, (and predictions are not zero?), add label to all training sets
            if (ensemble[0, i, :] == ensemble[1, i, :]).all() and (
                    ensemble[0, i, :] == ensemble[2, i, :]).all() and (
                    (ensemble[0, i, :]).any() == 1):
                X0 = np.vstack([X0, X_ul[i, :]])
                X1 = np.vstack([X1, X_ul[i, :]])
                X2 = np.vstack([X2, X_ul[i, :]])
                y0 = np.vstack([y0, ensemble[1, i, :].squeeze()])
                y1 = np.vstack([y1, ensemble[1, i, :].squeeze()])
                y2 = np.vstack([y2, ensemble[1, i, :].squeeze()])
                X_ul = np.delete(X_ul, i, axis=0)
                del unlabelled_idx[i]
                ul_df = add_label(ul_df, idx, ensemble[1, i, :].squeeze())

    training_dat = [X0, X1, X2]
    training_labels = [y0, y1, y2]

    return training_dat, training_labels, X_ul, unlabelled_idx, ul_df


def tri_fit(X, y, ul_df, models, save_output=True, skf=False):

    num_iters = 20

    print('X size = ' + str(X.shape))
    print('y size = ' + str(y.shape))

    # get X vectors that represent y data (labelled X)
    unlabelled_idx = [i for i, l in enumerate(ul_df['labels']) if isinstance(l, float)]
    labelled_idx = [i for i, l in enumerate(ul_df['labels']) if not isinstance(l, float)]

    X_l = X[labelled_idx,:]
    X_ul = X[unlabelled_idx,:]

    # y contains nans for unlabelled entries, remove these before training
    y_l = y[labelled_idx,:]

    if skf:
        X_train, y_train, X_test, y_test = iterative_train_test_split(X_l, y_l, test_size=0.2)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X_l, y_l, random_state=42,test_size=0.2)

    print('training data size = ' + str(X_train.shape))
    print('testing data size = ' + str(X_test.shape))
    print('training target shape = ' + str(y_train.shape))
    print('testing target shape = ' + str(y_test.shape))
    print('unlabeled data size = ' + str(X_ul.shape))

    # initialize three copies of training data before accumulating model-specific examples
    training_dat = [X_train, X_train, X_train]
    training_labels = [y_train, y_train, y_train]

    model_metrics = [{model:{'acc':[], 'prec':[], 'prec_all':[], 'rec':[], 'train_size':[]}} for model in models.keys()]

    for n in range(num_iters):

        preds = []
        for i,(m,model) in enumerate(models.items()):
            print(m)
            X_train = training_dat[i]
            y_train = training_labels[i]
            pred, acc, prec, prec_all, rec = generic_fit(model, X_train, y_train, X_test, y_test, X_ul)
            preds.append(pred)
            model_metrics[i][m]['acc'].append(acc)
            model_metrics[i][m]['prec'].append(prec)
            model_metrics[i][m]['prec_all'].append(prec_all)
            model_metrics[i][m]['rec'].append(rec)
            model_metrics[i][m]['train_size'].append(X_train.shape[0])
            print('class frequency: ' + str(np.mean(pred, axis=0)))

        training_dat, training_labels, X_ul, ul_df, unlabelled_idx = find_consensus(preds, training_dat, training_labels, X_ul, ul_df, unlabelled_idx)

    if save_output:
        save_things(model_metrics, models, ul_df)


def save_things(model_metrics, models, ul_df):
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    metric_path = "tri_train_metrics_disagree_" + str(timestamp) + '.pkl'
    pickle_out = open(metric_path, "wb")
    pickle.dump(model_metrics, pickle_out)
    pickle_out.close()

    model_path = "tri_train_models_disagree_" + str(timestamp) + '.pkl'
    pickle_out = open(model_path, "wb")
    pickle.dump(models, pickle_out)
    pickle_out.close()

    predictions_path = "all_predictions_" + str(timestamp) + '.pkl'
    pickle_out = open(predictions_path, "wb")
    pickle.dump(ul_df, pickle_out)
    pickle_out.close()


if __name__ == "__main__":
    main()