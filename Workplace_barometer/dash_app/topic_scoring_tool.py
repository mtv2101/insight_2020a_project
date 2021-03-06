
import pandas as pd
import numpy as np
import timeit
import spacy
import time

from dash_app.word_match import regex_matcher
from dash_app.embed import embeddings
import dash_app.load_text


######################################


def regex_find_topics(df, nlp, num_matches):
    start_time = timeit.default_timer()

    reg_match = regex_matcher()
    match_df = pd.DataFrame()

    if num_matches != -1:
        match_idx = np.random.choice(np.arange(0,len(df)), num_matches)
        for t in match_idx:
            text = df['text'].iloc[t]
            doc = nlp(text)
            out_df = reg_match.match_topics_with_spans(t, doc)
            match_df = match_df.append(out_df)

    else:
        for t in range(len(df)):
            text = df['text'].iloc[t]
            doc = nlp(text)
            out_df = reg_match.match_topics_with_spans(t, doc)
            match_df = match_df.append(out_df)

    com_ids = match_df.comment_idx.unique()
    labels = []
    idx = []
    text = []
    for id in com_ids:
        # match_df has several columns, here lets pull just one
        idx.append(int(id))
        com_df = match_df[match_df['comment_idx'] == id]
        com_labs = com_df.topic.unique()
        labels.append(com_labs)
        text.append(df.text.iloc[id])

    multilabel_df = pd.DataFrame()
    multilabel_df['comment_idx'] = idx
    multilabel_df['labels'] = labels
    multilabel_df['text'] = text

    onehot = pd.get_dummies(multilabel_df.labels.apply(pd.Series).stack()).sum(level=0)

    out_df = pd.concat([multilabel_df, onehot], axis=1)

    process_time = timeit.default_timer() - start_time
    print(str(len(df)) + ' submissions, query took ' + str(process_time) + ' seconds')

    #pd.to_pickle(out_df, 'regex_scored_all_df.pkl')

    return out_df


def read_matched_csv():
    match_df = pd.read_pickle('~/PycharmProjects/insight_2020a_project/Workplace_barometer/output/match_df.pkl')
    return match_df


def hand_score(df, num_examples=10):
    df['score'] = ''
    for t, topic in enumerate(df.topic.unique()):
        topic_df = df[df['topic']==topic]
        topic_idx = np.arange(0,len(topic_df))
        topic_rand_idx = np.random.choice(topic_idx, num_examples)
        for i,idx in enumerate(topic_rand_idx):
            try:
                text = topic_df['text'].iloc[idx]
            except:
                print('mislabeled column containing text')
            try:
                text = topic_df['comment_text'].iloc[idx]
            except:
                print('mislabeled column containing text')
            print('topic: ' + str(topic))
            print('text: ' + str(text))
            val = [input('class = ' + str(t) + '?') for t in df.topic.unique()]
            class_list = []
            for v,value in enumerate(val):
                if (value != 'y') and (value != 'n'):
                    print('wrong input - skipping to next one')
                    continue
                else:
                    if value == 'y':
                        class_list.append(df.topic.unique()[v])
                    else:
                        continue
                    df['score'].iloc[idx] = class_list
            print(class_list)

    pd.to_pickle(df, 'hand_scored_df.pkl')



def main(run_regex=True, do_hand_scoring=False, num_matches=-1):
    # randomly select "num_examples" of text from each class for hand-labelling

    model = 'en_core_web_sm'  # only minimal model needed
    nlp = spacy.load(model)

    df = load_text.load_unlabeled_data()
    #df = load_text.load_context_free_data()

    if run_regex:
        df = regex_find_topics(df, nlp, num_matches)
    else:
        df = read_matched_csv()

    if do_hand_scoring:
        hand_score(df, num_examples=10)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    save_path = "regex_scores_" + str(timestamp) + '.pkl'
    pd.to_pickle(df, save_path)

    return df


if __name__ == "__main__":
    main()