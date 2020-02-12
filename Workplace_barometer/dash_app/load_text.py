import pandas as pd
from dash_app.document import survey_doc

###############################
# different data loaders used throughout the package
###############################


def load_context_free_data():
    # sometime data comes without metadata like question ID, or column headers

    path = '~/PycharmProjects/insight_2020a_project/Workplace_barometer/AI_survey_data/data_20200212.csv'
    data = survey_doc(path, header=0)
    return data


def load_unlabeled_data():

    paths = ['~/PycharmProjects/insight_2020a_project/Workplace_barometer/AI_survey_data/2017 to mid 2018 comments.csv',
             '~/PycharmProjects/insight_2020a_project/Workplace_barometer/AI_survey_data/2018 to mid 2019 comments.csv']
    # paths = ['AI_data/2017_2018_comments.csv',
    #         'AI_data/2018_2019_comments.csv']

    q1_df = pd.DataFrame()
    q2_df = pd.DataFrame()
    for data_path in paths:
        data = survey_doc(data_path)
        data.clean_unlabelled_data()
        data_q1 = data.df[(data.df['QID' ]==61) | (data.df['QID' ]=='Unique / Unusual')]
        q1_df = q1_df.append(data_q1, ignore_index=True)
        data_q2 = data.df[(data.df['QID' ]==62) | (data.df['QID' ]=='One Change')]
        q2_df = q2_df.append(data_q2, ignore_index=True)

    ul_df = q1_df.append(q2_df, ignore_index=True)

    return ul_df

def load_regex_labeled_data():
    # data_path = ['~/PycharmProjects/insight_2020a_project/Workplace_barometer/output/regex_scored_df.pkl']
    data_path = ['AI_data/regex_scored_df.pkl']
    labeled_data = survey_doc(data_path[0])
    labeled_data.clean_regex_labelled_data()
    l_df = labeled_data.df
    # l_data_q1 = labeled_data.df[labeled_data.df['QID'] == 61]
    # l_data_q2 = labeled_data.df[labeled_data.df['QID'] == 62]

    # l_df = l_data_q1.append(l_data_q2, ignore_index=True)

    return l_df

def load_hand_labeled_data():
    # data_path = ['~/PycharmProjects/insight_2020a_project/Workplace_barometer/output/hand_scored_df.pkl']
    data_path =['AI_data/hand_scored_df.pkl']
    l_df = pd.read_pickle(data_path[0])
    return l_df