# -*- coding: utf-8 -*-
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF



class TopicMiner(object):
    def __init__(self, n_topics=10, max_features=1000, topics_per_document=3, top_words=15):
        self.n_topics = n_topics
        self.max_features = max_features
        self.topics_per_document = topics_per_document
        self.top_words = top_words
        self.topics = ["no topics inferred yet"]

    def get_top_words(self,model, feature_names, n_top_words):
        topics = []
        for topic_idx, topic in enumerate(model.components_):
            s = ("Topic #%d:" % topic_idx) + (" ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))
            topics.append(s)
        return topics

    def fit(self, data):
        # Convert data into vectorized tf-idf representation
        vectorizer = TfidfVectorizer(decode_error='ignore', max_df=0.5, min_df=3, max_features=self.max_features)
        self.tfidf = vectorizer.fit_transform(data)
        self.nmf = NMF(n_components=self.n_topics,
                  random_state=1,
                  alpha=.1, l1_ratio=.5) \
            .fit(self.tfidf)
        self.topics = self.get_top_words(self.nmf, vectorizer.get_feature_names(), self.top_words)
        return self.topics

    def predict(self, x):
        top_topics_for_document = self.nmf.transform(x).argsort()[0][:-self.topics_per_document - 1:-1]
        return top_topics_for_document


""""
# csv file path
csv_file_path = 'tag_transactions_2000.csv'
# array for documents and stop words wich are not during in the clustering
data = []
#stop_words = ['mann', 'frau', 'grau', 'schwarz', 'weiss', 'weiß', 'frauen', 'männer']
max_features = 3000
top_count = 25
cluster_per_image = 3
topics = 9 # it is better to use more than less topics but you have to find the sweet spot
# open csv file and convert the received data so that the vectorizer can tokenize it
with open(csv_file_path, 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    data = [(' '.join(row)) for row in reader]
"""
"""
# USAGE
tm = TopicMiner(9, 3000, 3, 10)
tm_topics = tm.fit(data)
for t in tm_topics:
    print t
print('prediction for a document')
print(tm.predict(tm.tfidf[12]))
"""




"""
tf_vectorizer = CountVectorizer(max_df=0.4, min_df=2,
                                max_features=max_features,
                                stop_words='english')
tf = tf_vectorizer.fit_transform(data)

lda = LatentDirichletAllocation(n_topics=topics, max_iter=5,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)
lda.fit(tf)

print("\nTopics in LDA model: ")
tf_feature_names = tf_vectorizer.get_feature_names()
print_top_words(lda, tf_feature_names, top_count)
"""