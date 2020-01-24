import pandas as pd
import itertools
import matplotlib.pyplot as plt
import seaborn as sns

class mvp(object):

    def __init__(self,
                 data_path):

        self.data_path = data_path
        self.topic_freq = []


    def find_words(self, word):
        # return row indices with matches to the word
        matches = [i for i,text in enumerate(self.df.text) if word in text]
        return matches


    def plot_tf(self):
        plot_dat = pd.DataFrame(columns=self.seeds.keys())
        plot_dat.loc[0] = self.topic_freq

        font = {'size': 14}

        plt.rc('font', **font)
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.set(font_scale=1.5)
        ax = sns.barplot(data=plot_dat, ax=ax)
        ax.set_ylabel('proportion of responses')
        plt.title('proportion of survey responses per topic')


    def print_examples(self, topic, num=5):
        for n in range(num):
            print(self.df.text.iloc[self.match_dict[topic][0][n]])


    def main(self):

        self.df = pd.read_csv(self.data_path, header=1)
        self.df = self.df.dropna(how='any')
        self.df = self.df.rename(columns={"Comment": "text"})
        self.df = self.df[:10000]

        self.seeds = {'Residents':['resident','patient','senior'],
                'Communication':['listen','communicate','complaint'],
                'Management':['supervisor','boss','rule','management','administration'],
                'Scheduling':['schedule','busy'],
                'Compensation':['compensation','money','pay','benefit'],
                'Colleagues':['colleague','employee','peer','coworker','associate','teammate'],
                }

        self.match_dict = dict((e,[]) for e in self.seeds)
        for key, val in self.seeds.items():
            matches = []
            for v in val:
                word_idx = self.find_words(v)
                matches.append(word_idx)
            self.match_dict[key] = matches

        total_count = 0
        for key, val in self.match_dict.items():
            all_counts = list(itertools.chain.from_iterable(val))
            count = len(set(all_counts))
            total_count += count
            self.topic_freq.append(count/float(5000.0))

        self.plot_tf()