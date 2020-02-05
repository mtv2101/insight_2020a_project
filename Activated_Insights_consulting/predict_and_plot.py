import numpy as np
import pandas as pd
import pickle
from sklearn import metrics
import os
import timeit
import matplotlib.pyplot as plt


def main():

    models = load_models()
    embeddings = load_embeddings()
    predictions = predict_new_labels(models, embeddings)
    gt = load_ground_truth()
    scores = score_predictions(predictions, gt)
    plot_scores(predictions, gt, scores)


def load_models():
    # expected input is a list of sklearn models, as output from tri_training.py
    model_path = '/home/matt_valley/PycharmProjects/insight_2020a_project/Activated_Insights_consulting/tri_train_models.pkl'
    models = pd.read_pickle(model_path)
    return models


def load_embeddings():

    embeddings_path = '/home/matt_valley/PycharmProjects/insight_2020a_project/Activated_Insights_consulting/regex_labelled_bert_embeddings.npy'
    embeddings = np.load(embeddings_path)
    return embeddings


def load_ground_truth():

    gt_path = '/home/matt_valley/PycharmProjects/insight_2020a_project/Activated_Insights_consulting/hand_scored_df.pkl'
    gt = pd.read_pickle(gt_path)
    return gt


def predict_new_labels(model, embeddings):
    # expected input is a single sklearn model
    predictions = model.predict(embeddings)
    return predictions


def score_predictions(predictions, gt):

    macro_prec = metrics.precision_score(gt, predictions, average='samples')

    # first 4 cols are not one_hot encoded labels
    label_df = gt.loc[: 4:]
    labels = label_df.values()
    classes = label_df.keys()

    print(predictions.shape, classes)

    assert(predictions.shape == labels.shape)

    class_precision = []
    for n in range(labels.shape[1]):
        match = labels[:,n] - predictions[:,n]
        # 0 is either a hit (1-1), or a correct-reject (0-0)
        # -1 is a false-alarm (0-1)
        # 1 is a miss (1-0)
        unique, counts = np.unique(match, return_counts=True)
        hot_counts = dict(zip(unique, counts))
        class_precision.append(hot_counts['0'] / (hot_counts['0'] + hot_counts['-1']))





if __name__ == "__main__":
    main()