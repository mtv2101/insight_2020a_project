import pandas as pd
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import re

import spacy
from spacy.matcher import Matcher
from spacy.lang.en import English



class regex_matcher(object):

    def __init__(self, model = 'en_core_web_sm', context_win=3):

        self.model = model
        self.topic_freq = []
        self.context_win = context_win

        self.init_model()
        self.setup_match_rules()


    def init_model(self):

        #self.nlp = spacy.load(self.model, disable=["tagger', 'ner'"])
        self.nlp = spacy.load(self.model)
        self.parser = English()


    def setup_match_rules(self):

        self.topics = {'Co-workers/teamwork':   [{"colleague": "[Cc](ol.?.?.?g.?e.?)"}, #colleague
                                                {"coworker": "[Cc](o.?w.?.?k.?er.?)"}, #coworker
                                                {"associate": "[Aa](s.?o.?.?.?te.?)"}, #associate
                                                {"teammate": "[Tt](eam.?.?.?te.?)"}, #teammate
                                                {"teamwork": "[Tt](eam.?\work)"}, #teamwork
                                                {"family": "[Ff](amily)"}, #family
                                                {"community": "[Cc](ommunity)"} #community,
                                                ],
                  'Schedule':                   [{"schedule": "[Ss](.?.?.?dule.?)"}, #schedule,
                                                {"busy": "[Bb](usy)"}, #busy',
                                                {"calendar": "[Cc](al.?nd.?r.?)"}  #calendar
                                                ],
                  'Management':                 [{"supervisor": "[Ss](upervis.?.?.?)"}, #'supervisor',
                                                {"boss": "[Bb](oss.?.?)"}, #'boss',
                                                {"rule": "[Rr](ule.?)"}, #'rule',
                                                {"management": "[Mm](an.?g.?ment)"}, #'management',
                                                {"administration": "[Aa](dmin.?.?.?.?.?.?.?.?.?.?)"} #'administration'
                                                ],
                  'Benefits and leave':         [{"benefits": "[Bb](enefit.?)"}, #'benefits',
                                                {"vacation": "[Vv](ac.?.?.?on.?)"}, #'vacation',
                                                {"time off": "[Tt](ime.?.?off.?)"}, #'time off',
                                                {"personal time": "[Pp](erson.?.?.?time)"}, #'personal time',
                                                {"sick day": "[Ss](ick.?day.?)"},
                                                {"maternity": "[Mm](aternity)"},
                                                {"paternity": "[Pp](aternity)"}#
                                                ],
                  'Materials and resources':    [{"support": "[Ss](upport)"}, #'support',
                                                {"resource": "[Rr](eso.?rc.?.?.?)"} #resources, resourcing
                                                ],
                  'Customers':                  [{"resident": "[Rr](es.?d.?nt.?)"}, # resident,
                                                {"patient": "[Pp](at.?.?nt.?)"}, # patient
                                                {"senior": "[Ss](enior.?)"} # senior
                                                ],
                  'Pay':                        [{"money": "[Mm](oney)"}, #'money',
                                                {"pay": "[Pp](ay.?.?.?.?.?)"}, #'pay', paycheck, payment
                                                {"debt": "[Dd](ebt)"}, #'debt',
                                                {"loan": "[Ll](oan.?)"},
                                                {"raise": "[Rr](aise.?)"}
                                                ],
                  'Recognition':                [{"recognition": "[Rr](ecognition.?)"}, #'recognition',
                                                {"appreciation": "[Aa](p.?rec.?.?t.?.?.?)"} #'appreciation'],
                                                ],
                  'Learning & Development':     [{"staff development": "[Ss](taf.?.?[Dd]evel.?.?ment)"}, #'Staff development',
                                                {"teach": "[Tt](each.?.?.?.?)"}, #'teach', teaching, teacher
                                                {"growth": "[Gg](rowth)"}, #'growth',
                                                {"training": "[Tt](ra.?.?ing)"} #training
                                                ],
                  'Purpose':                    [{"purpose": "[Pp](ur.?p.?.?se)"}, #'purpose',
                                                {"meaning": "[Mm](eaning)"}, #'meaning',
                                                {"mission": "[Mm](ission)"}, #'mission'],
                                                ],
                  'Commute':                    [{"commute": "[Cc](om.?ute.?.?)"}, #'commute', commuters
                                                {"drive": "[Dd](riv.?.?.?)"}, #'driving',
                                                {"location": "[Ll](ocation)"} # location'
                                                ],
                  'Staffing level':             [{"headcount": "[Hh](ead.?count)"}, #'head-count',
                                                {"staff": "[Ss](taff.?.?.?)"}, #'staffing',
                                                {"hire": "[Hh](ir.?.?.?)"}, #'hire',
                                                {"employ": "[Ee](mploy.?.?.?.?)"},
                                                {"turnover": "[Tt](urnover)"}#'employment'],
                                                ],
                  'Communication':              [{"listen": "[Ll](is.?en)"}, #'listen',
                                                {"communicate": "[Cc](om.?un.?cat.?.?.?.?)"}, #communicate, communication
                                                {'meetings': "[Mm](eetings)"},
                                                {'tone deaf': '[Tt](one.?deaf)'}
                                                ],
                  'Quality of care':            [{'care': "[Cc](are.?)"},
                                                {'compassion': "[Cc](compassion.?.?.?)"}#compassionate
                                                ],
                  'Employee relations':         [{"harass": "[Hh](ar.?as.?.?.?.?.?)"}, #'harassment',
                                                {"abuse": "[Aa](bus.?.?.?)"}, #'abuse, abusing
                                                {"lawyer": "[Ll](awyer)"},
                                                {"sue": "[Ss](ue)"}, #'sue',
                                                {"trouble": "[Tt](rouble)"}, #'trouble',
                                                {"sex": "[Ss](ex)"}, #'sex',
                                                {"race": "[Rr](ace)"}, #'race',
                                                {"gender": "[Gg](ender)"}, #'gender',
                                                {"bigot": "[Bb](igot.?.?.?)"}, #'bigot',
                                                {"ignore": "[Ig](nor.?.?.?)"}, #'ignorant', ignore
                                                {"legal": "(.?.?legal.?)"}, #'illegal', 'legal'
                                                {"collapse": "[Cc](ollapse)"}, #'collapse',
                                                {"intervene": "[Ii](nterven.?.?.?.?)"}, #'intervention', intervene
                                                ],
                  'Facility/setting':           [{"facility": "[Ff](ac.?l.?t.?.?.?)"}, #'facilities',
                                                {"maintain": "[Mm](a.?nt(ai|e)n.?.?.?.?)"}, #'maintain, maintenance',
                                                {"clean": "[Cc](lean.?.?.?)"}, #'clean', cleanly, cleaning
                                                {"hygene": "(.?.?hygen.?.?)"}, #hygeneic, unhygenic
                                                {"resort": "[Rr](resort)"},
                                                {"physical plant": "[Pp](hysical.?plant)"}#
                                                ],
                  #'N/A':                        [{'N/A': "[Nn](.?)[Aa]"},
                  #                              {'No comment': "[Nn](o.?)[Cc](omment)"}]
                      }

    def get_match_context(self, matches, doc):
        for match_id, tok_start, tok_end in matches:
            start = tok_start  # token position
        if start - self.context_win < 0:
            context_start = 0
        else:
            context_start = start - self.context_win
        if start + self.context_win >= len(doc):
            context_end = len(doc.text) - 1
        else:
            context_end = start + self.context_win
        context_span = doc[context_start:context_end].text
        #if context_span is None:
            #print('ERROR CONTEXT SPAN IS NONE')
        return context_span


    def match_topics(self, idx, doc):
        out_df = pd.DataFrame(columns = ['comment_idx', 'topic', 'conforming_text', 'matched_text', 'text'])
        self.match_dict = dict((e, []) for e in self.topics)
        for topic,rules in self.topics.items():
            for rule in rules:
                for key,regex in rule.items():
                    for match in re.finditer(regex, doc.text):
                        if match:
                            out_dict = {'comment_idx':idx, 'topic': topic, 'conforming_text': key, 'matched_text': match[0], 'text':doc.text}
                        out_df = out_df.append(out_dict, ignore_index=True)
        return out_df


    def match_topics_with_spans(self, idx, doc):
        # this extracts a span of words on either side of the matched word
        # Potentially useful if one wants to train a model on small text chunks, and convolve this model over a large hetergenous
        # text response (like a survey response with many topics)
        out_df = pd.DataFrame(columns = ['comment_idx', 'topic', 'conforming_text', 'matched_text', 'context_span', 'text'])
        # setup spacy matcher
        matcher = Matcher(self.nlp.vocab)
        self.match_dict = dict((e, []) for e in self.topics)
        for topic,rules in self.topics.items():
            for rule in rules:
                for key,regex in rule.items():
                    for match in re.finditer(regex, doc.text):
                        charstart, charend = match.span()
                        span = doc.char_span(charstart, charend)
                        if span is not None:
                            match_txt = span.text
                            matcher.add(match_txt, None, [{"TEXT": match_txt}])
                            matches = matcher(doc)
                            if matches:
                                context_span = self.get_match_context(matches, doc)
                            else:
                                #print('ERROR WITH SPACY TOKEN MATCHING!  ' + str(span.text))
                                context_span = match
                            out_dict = {'comment_idx':idx, 'topic': topic, 'conforming_text': key, 'matched_text': match_txt, 'context_span': context_span, 'text':doc.text}
                        else:
                            out_dict = pd.DataFrame(columns = ['comment_idx', 'topic', 'conforming_text', 'matched_text', 'context_span', 'text'])
                        # for eatch match append the output, unless there was an error else append ''
                        print(out_dict)
                        out_df = out_df.append(out_dict, ignore_index=True)
        return out_df


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