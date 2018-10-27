# -*- coding: utf-8 -*-
import csv
from time import time
import warnings
from collections import Counter
#import matplotlib.pyplot as plt
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn import metrics
    from sklearn.preprocessing import Normalizer
    from sklearn.pipeline import make_pipeline
    from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.cluster import DBSCAN


# csv file path
csv_file_path = 'tag_transactions_2000.csv'
# array for documents and stop words wich are not during in the clustering
data = []
k = 10
svd_dimensions = 300
number_of_top_words_in_centroid = 15

# open csv file and convert the received data so that the vectorizer can tokenize it
with open(csv_file_path, 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    data = [(' '.join(row)) for row in reader]

# Convert data into vectorized tf-idf representation
vectorizer = TfidfVectorizer(decode_error='ignore', max_df=0.4, min_df=3)
X = vectorizer.fit_transform(data)

# Dimensionality reduction using svd/lsa and a target dimensionality of 300
tr_svd = TruncatedSVD(svd_dimensions)
normalizer = Normalizer(copy=False)
lsa = make_pipeline(tr_svd, normalizer)
Y = lsa.fit_transform(X)
print("Explained variance of SVD : {}%".format(
    int(tr_svd.explained_variance_ratio_.sum() * 100)))

# lda

tf_vectorizer = CountVectorizer(max_df=0.4, min_df=2)
tf = tf_vectorizer.fit_transform(data)
lda = LatentDirichletAllocation(n_topics=k,max_iter=15, learning_method='online')
lda.fit(tf)

def print_top_words(model, feature_names, n_top_words):
    for topic_id, topic in enumerate(model.components_):
        print('\nTopic Nr.%d:' % int(topic_id + 1))
        print(''.join([feature_names[i] + ' ' + str(round(topic[i], 2))
              +' | ' for i in topic.argsort()[:-n_top_words - 1:-1]]))

print_top_words(lda, tf_vectorizer.get_feature_names(), number_of_top_words_in_centroid)
print(r'\n')

dbsc = DBSCAN(metric='cosine', n_jobs=-1, algorithm='brute',min_samples=4, eps=0.4)
labels = dbsc.fit_predict(Y)
print('#Clusters with DBSCAN: %i' % max(labels))
amout_of_noise = (len(filter(lambda x: x == -1, labels))/float(len(labels)))
print('Noise: %0.3f' % amout_of_noise)

# K-means clustering

km = MiniBatchKMeans(n_clusters=k, init='k-means++', n_init=1,
                     init_size=1000, batch_size=1000)
print("Clustering with K-means starts")
t0 = time()
km.fit(Y)
print("The clustering took %0.3fs" % (time() - t0))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(Y, km.labels_, sample_size=1000))


# Feature Extraction
original_space_centroids = tr_svd.inverse_transform(km.cluster_centers_)
order_centroids = original_space_centroids.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(k):
    print("Cluster %d:" % i)
    for ind in order_centroids[i, :number_of_top_words_in_centroid]:
        print(' %s' % terms[ind])

#print(Counter(km.labels_).values())
