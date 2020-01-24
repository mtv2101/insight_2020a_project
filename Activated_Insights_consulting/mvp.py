import pandas as pd
import itertools
import matplotlib.pyplot as plt
import seaborn as sns


def find_words(col,word):
    # input column of text from dataframe
    # return row indices with matches to the word
    matches = [i for i,text in enumerate(col) if word in text]
    return matches


def plot_tf(seeds, topic_freq):
    plot_dat = pd.DataFrame(columns=seeds.keys())
    plot_dat.loc[0] = topic_freq

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(data=plot_dat, ax=ax)
    ax.set_ylabel('proportion of responses')
    plt.savefig('barplot.png')


def main():

    survey_path_1 = '/home/matt_valley/PycharmProjects/insight_2020a_project/Activated_Insights_consulting/AI_survey_data/2017 to mid 2018 comments.csv'

    df = pd.read_csv(survey_path_1, header=1)
    df = df.dropna(how='any')
    df = df.rename(columns={"Comment": "text"})
    df = df[:10000]

    seeds = {'Residents':['resident','patient','senior'],
            'Communication':['listen','communicate','complaint'],
            'Management':['supervisor','boss','rule','management','administration'],
            'Scheduling':['schedule','busy'],
            'Compensation':['compensation','money','pay','benefit'],
            'Colleagues':['colleague','employee','peer','coworker','associate','teammate'],
            }

    match_dict = dict((e,[]) for e in seeds)
    for key,val in seeds.items():
        matches = []
        for v in val:
            word_idx = find_words(df.text, v)
            matches.append(word_idx)
        match_dict[key] = matches

    total_count = 0
    topic_freq = []
    for key,val in match_dict.items():
        all_counts = list(itertools.chain.from_iterable(val))
        count = len(set(all_counts))
        total_count += count
        topic_freq.append(count/float(5000.0))

    plot_tf(seeds, topic_freq)