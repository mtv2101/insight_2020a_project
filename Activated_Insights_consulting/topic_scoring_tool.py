
import pandas as pd
import numpy as np
import timeit
import spacy

from word_match import regex_matcher


######################################


def regex_find_topics(df, nlp):
    start_time = timeit.default_timer()

    match = regex_matcher()
    match_df = pd.DataFrame()
    for t, text in enumerate(df['comment_text']):
        doc = nlp(text)
        out_df = match.match_topics(t, doc)
        match_df = match_df.append(out_df)

    match_df.reset_index(drop=True, inplace=True)

    process_time = timeit.default_timer() - start_time
    print(str(len(df)) + ' submissions, query took ' + str(process_time) + ' seconds')

    return match_df


def read_matched_csv():
    match_df = pd.read_pickle('~/PycharmProjects/insight_2020a_project/Activated_Insights_consulting/match_df.pkl')
    return match_df


def main(run_regex=False, num_examples=10):
    # randomly select "num_examples" of text from each class for hand-labelling

    model = 'en_core_web_sm'  # only minimal model needed
    nlp = spacy.load(model)
    df = read_matched_csv()

    if run_regex:
        df = regex_find_topics(df, nlp)
    else:
        df = read_matched_csv()

    df['score'] = ''
    for t, topic in enumerate(df.topic.unique()):
        topic_df = df[df['topic']==topic]
        topic_idx = np.arange(0,len(topic_df))
        topic_rand_idx = np.random.choice(topic_idx, 1)
        for i,idx in enumerate(topic_rand_idx):
            text = topic_df['comment_text'].iloc[idx]
            print('topic: ' + str(topic))
            print('text: ' + str(text))
            val = input('correct class y/n?')
            if (val != 'y') and (val != 'n'):
                print('wrong input - skipping to next one')
                continue
            else:
                if val == 'y':
                    df['score'].iloc[idx] = topic
                else:
                    continue

    pd.to_pickle(df, 'hand_scored_df.pkl')


if __name__ == "__main__":
    main()