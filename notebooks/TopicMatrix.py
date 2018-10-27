
# coding: utf-8

# In[1]:

import sys
sys.path.append('../lib')
sys.path.append('../src')
sys.path.append('../src/queries')
sys.path.append('../src/clustering')
sys.path.append('../data')
sys.path.append('../')

#%matplotlib inline

from lib import csv_reader as reader
import pandas as pd
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import math

from lib import year_classifier as year_classifier

from jinja2 import Environment, FileSystemLoader


# In[2]:

def append_mid_year(df, column='mid_year'):
    '''
    Adds a int column (default "mid_year") to the table. It is the mean (rounded up) of from_year and to_year.
    '''
    df[column] = (df['from_year']+df['to_year'])/2
    df[column] = (df['mid_year']+0.49).astype(int)
    return df


import db

#merge Data!!!
my_db = db.Db()
meta_tag_120_df = pd.merge(my_db.metadata_long_19, my_db.clusters_long_19)
#meta_tag_120_df['clusters_count'] = 4

append_mid_year(meta_tag_120_df);


# In[10]:

meta_tag_120_df


# In[14]:

topics = range(0,110)
matrixAll = [None] * 110

for topicA in topics:
    print(topicA)
    picture_id_with_topic = meta_tag_120_df[meta_tag_120_df['cluster_id']==topicA]['picture_id']
    #for topicB in 
    a = meta_tag_120_df[meta_tag_120_df['cluster_id'] != topicA]

    matrix= range(0,110)
    for x in matrix:
        matrix[x] = 0

    for pic_id in picture_id_with_topic:
        pictures = a[a['picture_id']== pic_id]
        for topic_id in pictures['cluster_id']:
            matrix[topic_id] = matrix[topic_id] +1

    matrixAll[topicA] = matrix

topic_martix_df = pd.DataFrame(matrixAll)

#topic_martix_df
#topic_martix_df.describe()
#topic_martix_df.as_matrix()


# In[29]:

import matplotlib.pyplot as plt
import numpy as np



# Display a random matrix with a specified figure number and a grayscale
# colormap

fig = plt.figure(figsize=(11, 11))
plt.matshow(topic_martix_df.as_matrix(), fignum=100, cmap=plt.cm.gray)

fig.show()


# In[135]:

- 


# In[185]:

topic_frequently_df.sort_values('count', ascending=False)[0:8]


# In[202]:


n = 10

df = topic_frequently_df.sort_values('count', ascending=False)[0:n]

topics= []*n

for i, x in enumerate(df['name']):
    topics.append( x)
    
y_pos = np.arange(len(topics))

count = df['count']

fig = plt.figure(figsize=(8, 4))
plt.barh(y_pos, count)
plt.yticks(y_pos, topics)
#plt.xlabel('Topic')
plt.xlabel('Count')
plt.title('Often combined top ' + str(n))
#plt.margins(0.2)
#plt.subplots_adjust(bottom=0.15)
#plt.set_xticklabels

plt.show()

