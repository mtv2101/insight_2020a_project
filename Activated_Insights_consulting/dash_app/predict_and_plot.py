import numpy as np
import pandas as pd
import pickle
from sklearn import metrics
import os
import timeit
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn import preprocessing
from sklearn.decomposition import PCA


def main():

    models = load_models()
    embeddings = load_embeddings()

    #predictions = [predict_new_labels(model, embeddings) for i,(m,model) in enumerate(models.items())]
    predictions = tri_predict(models, embeddings)
    gt, gt_indices = load_ground_truth()
    preds_matching_gt = predictions[gt_indices]
    print('ground truth shape = ' + str(gt.shape))
    print('predictions shape = ' + str(predictions.shape))
    print('predictions matching gt shape = ' + str(preds_matching_gt.shape))

    #class_prec = [score_predictions(predictions[m], gt) for m in range(len(models))]
    class_prec = score_predictions(preds_matching_gt, gt)
    plot_scores(class_prec)

    return predictions, gt, class_prec


def load_models():
    # expected input is a list of sklearn models, as output from tri_training.py
    model_path = '/home/matt_valley/PycharmProjects/insight_2020a_project/Activated_Insights_consulting/dash_app/tri_train_models_disagree_20200206-162059.pkl'
    models = pd.read_pickle(model_path)
    return models


def get_classes():
    classes = ['Co-workers/teamwork',
               'Schedule',
               'Management',
               'Benefits and leave',
               'Materials and resources',
               'Customers',
               'Pay',
               'Recognition',
               'Learning & Development',
               'Purpose',
               'Commmute',
               'Staffing level',
               'Communication',
               'Quality of care',
               'Employee relations',
               'Facility/setting']
    return classes


def load_embeddings():

    path = '/home/matt_valley/PycharmProjects/insight_2020a_project/Activated_Insights_consulting/unlabelled_bert_embeddings.npy'
    embeddings = np.load(path)
    embeddings = pca_reduce(embeddings)

    return embeddings


def pca_reduce(X):

    pca = PCA(n_components=20)
    X_xform = pca.fit_transform(X)
    return X_xform


def load_ground_truth():

    #gt_path = '/home/matt_valley/PycharmProjects/insight_2020a_project/Activated_Insights_consulting/hand_scored_df.pkl'
    gt_path = '/home/matt_valley/PycharmProjects/insight_2020a_project/Activated_Insights_consulting/regex_scores_20200206-221204.pkl'
    gt = pd.read_pickle(gt_path)

    # gt df contains more than one-hot labels, get just there
    classes = get_classes()
    print(classes)
    print(gt.keys())
    gt_one_hot = gt[gt.columns.intersection(classes)]
    gt_indices = gt['comment_idx']

    return gt_one_hot, gt_indices


def predict_new_labels(model, embeddings):
    # expected input is a single sklearn model
    #print(model)pred
    predictions = model.predict(embeddings)
    return predictions


def tri_predict(models, embeddings):

    pred = [model.predict(embeddings) for i,(m,model) in enumerate(models.items())]
    pred = np.stack(pred, axis=0).squeeze()
    pred == pred[0,:,:] # equate first model labels to the other two, TRUE if all agree
    all_labels = []
    # for each category
    for col in range(pred.shape[2]):
        all_labels.append([1 if pred[:,c,col].all()==True else 0 for c in range(pred.shape[1])])
    consensus_labels = np.array(all_labels).T
    return consensus_labels



def score_predictions(predictions, gt):

    assert (predictions.shape == gt.shape)
    class_precision = metrics.precision_score(gt, predictions, average=None)

    # code it myself for sanity check
    #class_precision2 = []
    #for n in range(gt.shape[1]):
      #  match = gt[:,n] - predictions[:,n]
      #  # 0 is either a hit (1-1), or a correct-reject (0-0)
      #  # -1 is a false-alarm (0-1)
      #  # 1 is a miss (1-0)
     #   unique, counts = np.unique(match, return_counts=True)
     #   hot_counts = dict(zip(unique, counts))
     #   class_precision2.append(hot_counts['0'] / (hot_counts['0'] + hot_counts['-1']))

    return class_precision


def plot_scores(class_precision):

    classes = get_classes()

    fig = go.Figure(data=[
        go.Bar(name='Topic precision', x=classes, y=class_precision)
    ])
    # Change the bar mode
    fig.update_layout(barmode='stack')
    fig.show()

    fig = go.Figure(data=[
        go.Bar(name='Class Frequency', x=classes, y=class_counts)
    ])
    fig.update_layout(barmode='stack')
    fig.show()


def get_class_frequency(df):

    classes = get_classes()
    print(df.keys())

    class_counts = [np.sum(df[c], axis=0) for c in classes]
    uncat_count = len(df) - sum(class_counts)

    return classes, class_counts, uncat_count




if __name__ == "__main__":
    main()