
import pandas as pd
import numpy as np
import timeit
import spacy

from word_match import regex_matcher
from embed import embeddings


######################################


def regex_find_topics(df, nlp, num_matches=20000):
    start_time = timeit.default_timer()

    match = regex_matcher()
    match_df = pd.DataFrame()

    if num_matches != -1:
        match_idx = np.random.choice(np.arange(0,len(df)), num_matches)
        for t in match_idx:
            text = df['text'].iloc[t]
            doc = nlp(text)
            out_df = match.match_topics(t, doc)
            match_df = match_df.append(out_df)
        sub_df = df[df.index.isin(match_idx)]
        out_df = pd.join([sub_df, match_df])
    else:
        # BROKEN!!!
        for t, text in enumerate(df['text']):
            doc = nlp(text)
            out_df = match.match_topics(t, doc)
            match_df = match_df.append(out_df)

    out_df.reset_index(inplace=True)

    process_time = timeit.default_timer() - start_time
    print(str(len(df)) + ' submissions, query took ' + str(process_time) + ' seconds')

    pd.to_pickle(out_df, 'regex_scored_df.pkl')

    return out_df


def read_matched_csv():
    match_df = pd.read_pickle('~/PycharmProjects/insight_2020a_project/Activated_Insights_consulting/match_df.pkl')
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



def main(run_regex=True, do_hand_scoring=False):
    # randomly select "num_examples" of text from each class for hand-labelling

    model = 'en_core_web_sm'  # only minimal model needed
    nlp = spacy.load(model)

    embeds = embeddings()
    embeds.load_data()
    df = embeds.ul_df # get unlabeled dataframe

    if run_regex:
        df = regex_find_topics(df, nlp)
    else:
        df = read_matched_csv()

    if do_hand_scoring:
        hand_score(df, num_examples=10)


if __name__ == "__main__":
    main()