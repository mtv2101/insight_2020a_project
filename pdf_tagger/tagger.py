


def test_keyword_dict(self):
    test_keywords = {'construction date': ['the building was constructed in 1928 amidst a short-lived economic boom'],
                          'mortgage amount': ['remaining mortgage: $1,203,943',
                                              'An initial loan of $2.7M was used to finance the purchase'],
                          'maintenance cost': ['the yearly maintenance of the building is $80,000']
                        }
    return test_keywords


def get_tags():
    self.get_keywords()


def match_tags(text, tags):
    test_keywords = test_keyword_dict()
    for keys,tags in test_keywords.items():
        matches = ...