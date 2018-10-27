
# coding: utf-8

# In[1]:

import sys
sys.path.append('../lib')
sys.path.append('../src')
sys.path.append('../src/queries')
sys.path.append('../src/clustering')
sys.path.append('../src/visualization_lib')
sys.path.append('../data')
sys.path.append('../')

#%matplotlib inline


from lib import csv_reader as reader
import pandas as pd
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import math
import json

from lib import year_classifier as year_classifier

from jinja2 import Environment, FileSystemLoader


# In[3]:

from itertools import chain, combinations
def frequency_matrix(start, end):
    dimensions =    110 
    matrix = np.zeros(shape=(dimensions,dimensions), dtype=np.int)

    clusters = df.query('mid_year >= {0} & mid_year <= {1}'.format(start, end))

    cluster_list = clusters.groupby('picture_id')['cluster_id'].apply(list)
    cluster_list = np.array(cluster_list)
    for transaction in cluster_list:
        for z in chain.from_iterable(combinations(transaction, r) for r in range(len(transaction)+1)):
            if(len(z) == 2):
                i, j = z
                matrix[i, j] += 1
                matrix[j, i] += 1
    return matrix


#load Data!!!
import db

my_db = db.Db()
df = my_db.final_cluster_nation

time_ranges = [[1789,1847], [1848,1874], [1875,1914], [1789,1914]]

# nicht das gleiche ergebnis, da zeitraum kleiner
#df['mid_year'].max() = 1918


time_range_matrices = []

for x in time_ranges:
    mat = frequency_matrix(x[0], x[1])
    #time_range_matrices.append(pd.DataFrame(mat))
    time_range_matrices.append(mat)
    
time_range_matrices


# In[6]:

mat = time_range_matrices[3]
df_mat = pd.DataFrame(mat)
df_mat


# In[40]:




# In[95]:

# matrix to Nodes and links

jsondata = json.loads('{"nodes": [], "links": []}')

#"nodes": [
#    {"id": "Myriel", "group": 1},

valueMin = 80

topic_group_women=[3,9,45,97]
topic_group_men=[79, 1,43, 81, 55]
topic_group_war=[62,91,53,105,21]
topic_group_religion=[52, 85, 96, 15, 36, 67, 82, 88]
topic_group_landscape=[0, 27, 10, 51, 13, 17, 41, 60 ,63, 109, 77]
topic_group_portraits=[101, 103, 69, 92, 56, 69]

import json
def append_cluster_name(df, column='cluster_name'):
    # TODO!!!
    with open('../data/topics_benannt.txt') as topic_word_json:
        topic_word_dic = json.load(topic_word_json)
    
    df['cluster_name'] = df['cluster_id'].apply(lambda x: topic_word_dic[str(x)])
    return df

nodes = pd.DataFrame(range(0,110), columns=['cluster_id'])

nodes['group'] = nodes.cluster_id.map(lambda x:      1 if x in topic_group_women else      2 if x in topic_group_men else      3 if x in topic_group_war else      4 if x in topic_group_religion else      5 if x in topic_group_landscape else      6 if x in topic_group_portraits else 0 )

nodes = append_cluster_name(nodes)
json_nodes = json.loads(nodes.to_json(orient='records'))
jsondata['nodes'] = json_nodes

# links

#"links": [
#    {"source": "Napoleon", "target": "Myriel", "value": 1},



for x in range(110):
    for y in xrange(x+1,110):
        val = df_mat[x][y]
        if val >valueMin:
            jsondata["links"].append(json.loads('{'+'"source": "{0}", "target": "{1}", "value": {2}'.format(x,y,val)+'}'))

with open('graph/graph_data.js', 'w') as outfile:
    outfile.write("graph = ")
    json.dump(jsondata, outfile, sort_keys=True, indent=4)
    #json.dump(jsondata, outfile)    
print 'saved' + ' /graph/graph_data.js' + ' with valueMin: ' + str(valueMin) + '.'
jsondata

