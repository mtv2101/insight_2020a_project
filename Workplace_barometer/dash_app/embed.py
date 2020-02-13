
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from spacy_transformers import TransformersLanguage, TransformersWordPiecer, TransformersTok2Vec
from load_text import load_context_free_data, load_unlabeled_data, load_regex_labeled_data


##############################

'''
embed.py takes .csv tables of comment text, 
or pickled files containing a pandas dataframe formatted 
using doucment.py.  

It embeds the text using tf/idf, or BERT
where each comment is treated as an independent document. 

It outputs a numpy array of rows(comments) by cols(vector features)

major dependencies are the spacy transformers library:
https://github.com/explosion/spacy-transformers

scikit-learn:
https://scikit-learn.org/stable/

requires python>3.6

'''


class embeddings(object):

    def __init__(self, question = 'both', model = 'bert'):

        self.question = question
        self.model = model


    def embed_data(self):
        models = ['tfidf', 'tfidf_by_class', 'bert']
        assert self.model in models

        #self.data = load_unlabeled_data()
        #self.data = load_regex_labeled_data()
        #self.data = load_hand_labeled_data()
        self.data = load_context_free_data()

        if self.model == 'tfidf_by_class':
            return self.tfidf_by_class(self.data)
        if self.model == 'tfidf':
            return self.tfidf(self.data)
        if self.model == 'bert':
            self.l_bert = self.bert(self.data)


    def tfidf_by_class(self, df):
        '''
        get tfidf vectors on labeled and unlabelled data
        '''
        all_tfidf = []
        topics = df['topic'].unique()
        for topic in topics:
            temp_df = df[df['topic'] == topic]
            data = [sent for sent in df.comment_text]
            n_features = 200
            tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,
                                               max_features=n_features,
                                               stop_words='english')
            tfidf = tfidf_vectorizer.fit_transform(data)
            all_tfidf.append(tfidf)

        # format output from sparse matrix to something I can actually use
        tfidf_mat = np.empty((0, all_tfidf[0].shape[1]), dtype=all_tfidf[0].dtype)
        for t, topic_vecs in enumerate(all_tfidf):
            tfidf_mat = np.append(t, topic_vecs.toarray(), axis=0)

        return tfidf_mat


    def tfidf(self, df):
        '''
        get tfidf vectors on labeled and unlabelled data
        '''
        data = [comment for comment in df.text]
        n_features = 200
        tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,
                                           max_features=n_features,
                                           stop_words='english')
        tfidf = tfidf_vectorizer.fit_transform(data)


        # format output from sparse matrix to something I can actually use
        tfidf_mat = tfidf.toarray()

        return tfidf_mat


    def bert(self, df):

        model = 'bert-base-uncased'
        #nlp = spacy.load(model)

        nlp = TransformersLanguage(trf_name=model, meta={"lang": "en"})
        sentencizer = nlp.create_pipe("sentencizer")
        nlp.add_pipe(sentencizer, first=True)
        nlp.add_pipe(TransformersWordPiecer.from_pretrained(nlp.vocab, model))
        nlp.add_pipe(TransformersTok2Vec.from_pretrained(nlp.vocab, model))

        tensors = []
        for comment in df.text:
            doc = nlp(comment)
            tensor = doc._.trf_last_hidden_state
            print(tensor.shape)
            tensors.append(tensor.mean(axis=0))
        tensors = np.array(tensors)

        np.save('new_data_bert_embeddings.npy', tensors)

        return tensors
