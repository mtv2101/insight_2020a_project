import pandas as pd
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import re

import spacy
from spacy.matcher import Matcher
from spacy.lang.en import English



class word_match(object):

    def __init__(self, model = 'en_core_web_sm'):

        self.model = model
        self.topic_freq = []

        self.init_model()
        self.setup_match_rules()


    def init_model(self):

        #self.nlp = spacy.load(self.model, disable=["tagger', 'ner'"])
        self.nlp = spacy.load(self.model)
        self.parser = English()


    def setup_match_rules(self):

        self.topics = {'Co-workers/teamwork':   [{"TEXT": {"REGEX": "^[Cc](ol.?.?.?g.?e.?)$"}}, #colleague
                                                {"TEXT": {"REGEX": "^[Cc](o.?w.?.?k.?er.?)$"}}, #coworker
                                                {"TEXT": {"REGEX": "^[Aa](s.?o.?.?.?te.?)$"}}, #associate
                                                {"TEXT": {"REGEX": "^[Tt](eam.?.?.?te.?)$"}}, #teammate
                                                {"TEXT": {"REGEX": "^[Tt](eam.?\work)$"}}, #teamwork
                                                {"TEXT": {"REGEX": "^[Ss](taf.?.?)$"}}, #staff
                                                {"TEXT": {"REGEX": "^[Ff](amily)$"}}, #family
                                                {"TEXT": {"REGEX": "^[Cc](ommunity)$"}} #community,
                                                ],
                  'Schedule':                   [{"TEXT": {"REGEX": "^[Ss](.?.?.?dule.?)$"}}, #schedule,
                                                {"TEXT": {"REGEX": "^[Bb](usy)$"}}, #busy',
                                                {"TEXT": {"REGEX": "^[Cc](al.?nd.?r.?)$"}}  #calendar
                                                ],
                  'Management':                 [{"TEXT": {"REGEX": "^[Ss](upervis.?.?.?)$"}}, #'supervisor',
                                                {"TEXT": {"REGEX": "^[Bb](oss.?.?)$"}}, #'boss',
                                                {"TEXT": {"REGEX": "^[Rr](ule.?)$"}}, #'rule',
                                                {"TEXT": {"REGEX": "^[Mm](an.?g.?ment)$"}}, #'management',
                                                {"TEXT": {"REGEX": "^[Aa](dmin.?.?.?.?.?.?.?.?.?.?)$"}} #'administration'
                                                ],
                  'Benefits and leave':         [{"TEXT": {"REGEX": "^[Bb](enefit.?)$"}}, #'benefits',
                                                {"TEXT": {"REGEX": "^[Vv](ac.?.?.?on.?)$"}}, #'vacation',
                                                {"TEXT": {"REGEX": "^[Tt](ime.?.?off.?)$"}}, #'time off',
                                                {"TEXT": {"REGEX": "^[Pp](erson.?.?.?time)$"}}, #'personal time',
                                                {"TEXT": {"REGEX": "^[Ss](ick.?day.?)$"}} #sick days
                                                ],
                  'Support and resources':      [{"TEXT": {"REGEX": "^[Ss](upport)$"}}, #'support',
                                                {"TEXT": {"REGEX": "^[Rr](eso.?rc.?.?.?)$"}} #resources, resourcing
                                                ],
                  'Customers':                  [{"TEXT": {"REGEX": "^[Rr](es.?d.?nt.?)$"}}, # resident,
                                                {"TEXT": {"REGEX": "^[Pp](at.?.?nt.?)$"}}, # patient
                                                {"TEXT": {"REGEX": "^[Ss](enior.?)$"}} # senior
                                                ],
                  'Pay':                        [{"TEXT": {"REGEX": "^[Mm](oney)$"}}, #'money',
                                                {"TEXT": {"REGEX": "^[Pp](ay.?.?.?.?.?)$"}}, #'pay', paycheck, payment
                                                {"TEXT": {"REGEX": "^[Rr](es.?d.?nt.?)$"}}, #'end of the month',
                                                {"TEXT": {"REGEX": "^[Dd](ebt)$"}}, #'debt',
                                                {"TEXT": {"REGEX": "^[Ll](oan.?)$"}} # loan
                                                ],
                  'Recognition':                [{"TEXT": {"REGEX": "^[Rr](ecognition.?)$"}}, #'recognition',
                                                {"TEXT": {"REGEX": "^[Aa](p.?rec.?.?.?tion)$"}} #'appreciation'],
                                                ],
                  'Learning & Development':     [{"TEXT": {"REGEX": "^[Ss](taf.?.?[Dd]evel.?.?ment)$"}}, #'Staff development',
                                                {"TEXT": {"REGEX": "^[Tt](each.?.?.?.?)$"}}, #'teach', teaching, teacher
                                                {"TEXT": {"REGEX": "^[Gg](rowth)$"}}, #'growth',
                                                {"TEXT": {"REGEX": "^[Tt](ra.?.?ing)$"}} #training
                                                ],
                  'Purpose':                    [{"TEXT": {"REGEX": "^[Pp](ur.?p.?.?se)$"}}, #'purpose',
                                                {"TEXT": {"REGEX": "^[Mm](eaning)$"}}, #'meaning',
                                                {"TEXT": {"REGEX": "^[Mm](ission)$"}}, #'mission'],
                                                ],
                  'No answer/Nothing':          [],
                  'Commmute':                   [{"TEXT": {"REGEX": "^[Cc](om.?ute)$"}}, #'commute',
                                                {"TEXT": {"REGEX": "^[Dd](riving)$"}}, #'driving',
                                                {"TEXT": {"REGEX": "^[Ll](ocation)$"}} # location'
                                                ],
                  'Staffing level':             [{"TEXT": {"REGEX": "^[Hh](ead.?count)$"}}, #'head-count',
                                                {"TEXT": {"REGEX": "^[Ss](taf.?.?.?.?)$"}}, #'staffing',
                                                {"TEXT": {"REGEX": "^[Hh](ire)$"}}, #'hire',
                                                {"TEXT": {"REGEX": "^[Ee](mployment)$"}}, #'employment'],
                                                ],
                  'Communication':              [{"TEXT": {"REGEX": "^[Ll](is.?en)$"}}, #'listen',
                                                {"TEXT": {"REGEX": "^[Cc](om.?un.?cat.?.?.?.?)$"}} #'communicate, communication
                                                ],
                  'Quality of care':            [{'TEXT': {'REGEX': "^[Cc](are)$"}}], #care AND residents,
                  'Employee relations':         [{"TEXT": {"REGEX": "^[Pp](roblem.?)$"}}, #'problem',
                                                {"TEXT": {"REGEX": "^[Hh](ar.?as.?.?.?.?.?)$"}}, #'harassment',
                                                {"TEXT": {"REGEX": "^[Aa](bus.?.?.?)$"}}, #'abuse, abusing
                                                {"TEXT": {"REGEX": "^[Ll](awyer)$"}}, #'lawyer',
                                                {"TEXT": {"REGEX": "^[Tt](rouble)$"}}, #'trouble',
                                                {"TEXT": {"REGEX": "^[Ss](ex)$"}}, #'sex',
                                                {"TEXT": {"REGEX": "^[Rr](ace)$"}}, #'race',
                                                {"TEXT": {"REGEX": "^[Gg](ender)$"}}, #'gender',
                                                {"TEXT": {"REGEX": "^[Bb](igot.?.?.?)$"}}, #'bigot',
                                                {"TEXT": {"REGEX": "^[Ig](nor.?.?.?)$"}}, #'ignorant', ignore
                                                {"TEXT": {"REGEX": "^(.?.?legal.?)$"}}, #'illegal', 'legal'
                                                {"TEXT": {"REGEX": "^[Cc](ollapse)$"}}, #'collapse',
                                                {"TEXT": {"REGEX": "^[Ii](nterven.?.?.?.?)$"}}, #'intervention', intervene
                                                ],
                  'Facility/setting':           [{"TEXT": {"REGEX": "^[Ff](ac.?l.?t.?.?.?)$"}}, #'facilities',
                                                {"TEXT": {"REGEX": "^[Mm](a.?nt.?n.?nce)$"}}, #'maintenance',
                                                {"TEXT": {"REGEX": "^[Cc](lean.?.?.?)$"}}, #'clean', cleanly, cleaning
                                                {"TEXT": {"REGEX": "^(.?.?hygen.?.?)$"}} #'hygenic', unhygenic
                                                ]
                      }


    def match_topics(self, doc):

        # setup spacy matcher
        self.match_dict = dict((e, []) for e in self.topics)
        for topic,rules in self.topics.items():
            matcher = Matcher(self.nlp.vocab)
            matcher.add(rules)
            matches = matcher(doc)
            self.match_dict[topic] = matches


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