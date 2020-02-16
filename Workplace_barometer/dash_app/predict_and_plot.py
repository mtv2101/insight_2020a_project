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
    #predictions = tri_predict(models, embeddings)
    predictions = single_model_predict(models, embeddings)
    gt, gt_indices = load_ground_truth()
    preds_matching_gt = predictions[gt_indices]
    print('ground truth shape = ' + str(gt.shape))
    print('predictions shape = ' + str(predictions.shape))
    print('predictions matching gt shape = ' + str(preds_matching_gt.shape))

    #class_prec = [score_predictions(predictions[m], gt) for m in range(len(models))]
    class_prec = score_predictions(preds_matching_gt, gt)
    classes, gt_class_counts, uncat_count = get_class_frequency_df(gt)
    class_counts, uncat_count = get_class_frequency_preds(preds_matching_gt)
    all_class_counts, uncat_count = get_class_frequency_preds(predictions)

    gt_class_freq = [count/len(gt) for count in gt_class_counts]
    print(all_class_counts)
    all_class_freq = [count / len(predictions) for count in all_class_counts]
    plot_scores(gt_class_freq, 'Topic frequency in survey responses (Regex)', 'Topic Frequency')
    plot_scores(class_prec, 'Precision by class', 'Precision')
    plot_scores(all_class_freq, 'Topic frequency in survey responses (tri-train)', 'Topic Frequency')

    return predictions, gt, class_prec


def load_models():
    # expected input is a list of sklearn models, as output from tri_training.py
    model_path = '~/PycharmProjects/insight_2020a_project/Workplace_barometer/output/tri_train_models_disagree_20200210-132630.pkl'
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
               'Commute',
               'Staffing level',
               'Communication',
               'Quality of care',
               'Employee relations',
               'Facility/setting']
    return classes


def load_embeddings():

    #path = '/Workplace_barometer/output/unlabelled_bert_embeddings.npy'
    path = '/home/matt_valley/PycharmProjects/insight_2020a_project/Workplace_barometer/output/new_data_bert_sum_embeddings.npy'
    embeddings = np.load(path)
    embeddings = pca_reduce(embeddings)

    return embeddings


def pca_reduce(X):

    pca = PCA(n_components=20)
    X_xform = pca.fit_transform(X)
    return X_xform


def load_ground_truth():

    gt_path = '~/PycharmProjects/insight_2020a_project/Workplace_barometer/output/regex_scores_20200215-165137.pkl'
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


def single_model_predict(models, embeddings, model_name='MLP'):
    model = models[model_name]
    pred = model.predict(embeddings)
    return pred



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


def plot_scores(score, title, y_title):

    classes = get_classes()

    fig = go.Figure(data=[
        go.Bar(name='Topic precision', x=classes, y=score)
    ])
    fig.update_layout(barmode='stack',
                      title=title,
                      yaxis_title=y_title)
    fig.show()


def get_class_frequency_df(df):

    classes = get_classes()

    class_counts = [np.sum(df[c], axis=0) for c in classes]
    uncat_count = len(df) - sum(class_counts)

    return classes, class_counts, uncat_count


def get_class_frequency_preds(preds):

    class_counts = np.sum(preds, axis=0)
    uncat_count = preds.shape[0] - sum(class_counts)

    return class_counts, uncat_count


if __name__ == "__main__":
    main()