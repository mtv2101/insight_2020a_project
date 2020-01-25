import pandas as pd
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import re

import spacy
from spacy.matcher import Matcher
from spacy.lang.en import English



class word_match(object):

    def __init__(self,
                 df,
                 model = 'en_core_web_sm',
                 match_method = 'spacy'):

        self.df = df
        self.model = model
        self.match_method = match_method
        self.topic_freq = []


    def init_model(self):

        #self.nlp = spacy.load(self.model, disable=["tagger', 'ner'"])
        self.nlp = spacy.load(self.model)
        self.parser = English()


    def setup_match_rules(self):

        {'resident': [{"TEXT": {"REGEX": "^[Rr](es.?d.?nt)$"}}],
         'patient': [{"TEXT": {"REGEX": "^[Pp](at\.?\.?nt)$"}}]}

    def match_seeds(self):

        self.topics = {'Co-workers/teamwork': [{"TEXT": {"REGEX": "^[Cc](ol.?.?.?g.?e.?)$"}}, #colleague
                                               {"TEXT": {"REGEX": "^[Cc](o.?w.?.?k.?er.?)$"}}, #coworker
                                               {"TEXT": {"REGEX": "^[Aa](s.?o.?.?.?te.?)$"}}, #associate
                                               {"TEXT": {"REGEX": "^[Tt](eam.?.?.?te.?)$"}}, #teammate
                                               {"TEXT": {"REGEX": "^[Tt](eam.?\work)$"}}, #teamwork
                                               {"TEXT": {"REGEX": "^[Ss](taf.?.?)$"}}, #staff
                                               {"TEXT": {"REGEX": "^[Ff](amily)$"}}, #family
                                               {"TEXT": {"REGEX": "^[Cc](ommunity)$"}} #community,
                                             ],
                  'Schedule': [{"TEXT": {"REGEX": "^[Ss](.?.?.?dule.?)$"}}, #schedule,
                               {"TEXT": {"REGEX": "^[Bb](usy)$"}}, #busy',
                               {"TEXT": {"REGEX": "^[Cc](al.?nd.?r.?)$"}}  #calendar
                               ],
                  'Management': [{"TEXT": {"REGEX": "^[Ss](upervis.?.?.?)$"}}, #'supervisor',
                                 {"TEXT": {"REGEX": "^[Bb](oss.?.?)$"}}, #'boss',
                                 {"TEXT": {"REGEX": "^[Rr](ule.?)$"}}, #'rule',
                                 {"TEXT": {"REGEX": "^[Mm](an.?g.?ment)$"}}, #'management',
                                 {"TEXT": {"REGEX": "^[Aa](dmin.?.?.?.?.?.?.?.?.?.?)$"}} #'administration'
                                 ],
                  'Benefits and leave': ['benefits', 'vacation', 'time off', 'personal time', 'sick days'],
                  'Support and resources': ['support', 'resources'],
                  'Customers': ['resident', 'patient', 'senior'],
                  'Pay': ['money', 'pay', 'paycheck', 'end of the month', 'debt', 'loans'],
                  'Recognition': ['recognition', 'appreciation'],
                  'Learning & Development': ['Staff development', 'teach', 'growth', 'training'],
                  'Purpose': ['purpose', 'meaning', 'mission'],
                  'No answer/Nothing': [],
                  'Location to home': ['commute', 'drive', 'location'],
                  'Staffing level': ['head-count', 'staffing', 'hire', 'employment'],
                  'Communication': ['listen', 'communicate'],
                  'Quality of care': ['care AND residents'],
                  'Employee relations': ['problem','harassment','abuse','lawyer','trouble','identity','sex','race','gender','bigot','ignorant','ignore','illegal','collapse','union','intervention'],
                  'Facility/setting': ['facilities','maintenance','clean', 'hygenic']
                      }

        '''
        self.seeds = {'Residents': ['resident', 'patient', 'senior'],
                      'Communication': ['listen', 'communicate', 'complaint'],
                      'Management': ['supervisor', 'boss', 'rule', 'management', 'administration'],
                      'Scheduling': ['schedule', 'busy'],
                      'Compensation': ['compensation', 'money', 'pay', 'benefit'],
                      'Colleagues': ['colleague', 'employee', 'peer', 'coworker', 'associate', 'teammate'],
                      }
        '''

        # apply spacy matcher
        if self.match_method == 'spacy':
            # setup spacy matcher
            matcher = Matcher(self.nlp.vocab)

            for topic,seeds in self.topics.items():
                for seed in seeds:
                    matcher.add(seed)


            for d, doc in self.df['doc']:
                matches = matcher(self.doc)
                for match in re.finditer(expression, doc.text):
                    start, end = match.span()
                    span = doc.char_span(start, end)
                    # This is a Span object or None if match doesn't map to valid token sequence
                    if span is not None:
                        print("Found match:", span.text)

        # ... or do simple text matching
        elif self.match_method == 'plaintext':
            self.match_dict = dict((e, []) for e in self.seeds)
            for key, val in self.seeds.items():
                matches = []
                for v in val:
                    word_idx = self.find_words(v)
                    matches.append(word_idx)
                self.match_dict[key] = matches

    def plot_match_stats(self):
        total_count = 0
        for key, val in self.match_dict.items():
            all_counts = list(itertools.chain.from_iterable(val))
            count = len(set(all_counts))
            total_count += count
            self.topic_freq.append(count / float(5000.0))

        plot_dat = pd.DataFrame(columns=self.seeds.keys())
        plot_dat.loc[0] = self.topic_freq

        font = {'size': 14}

        plt.rc('font', **font)
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.set(font_scale=1.5)
        ax = sns.barplot(data=plot_dat, ax=ax)
        ax.set_ylabel('proportion of responses')
        plt.title('proportion of survey responses per topic')