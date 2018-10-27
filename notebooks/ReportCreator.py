import sys

from lib import csv_reader as reader
import pandas as pd
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import math

from lib import year_classifier as year_classifier

from jinja2 import Environment, FileSystemLoader


# In[ ]:

# gen html
# http://pbpython.com/pdf-reports.html

#reports folder
dir_reports = '../AnalysisTool/'
fsl = FileSystemLoader(dir_reports)
env = Environment(loader=fsl)


# In[ ]:

from scipy.interpolate import interp1d
from scipy.signal import medfilt
from scipy.signal import convolve

def append_mid_year(df, column='mid_year'):
    '''
    Adds a int column (default "mid_year") to the table. It is the mean (rounded up) of from_year and to_year.
    '''
    df[column] = (df['from_year']+df['to_year'])/2
    df[column] = (df['mid_year']+0.49).astype(int)
    return df
  

def plotTopic(topicsListsWithIDs, df, column='mid_year', smooth=True):
    '''
    topicsListsWithIDs takes a List with Lists of Topic-Cluster-IDs 
    like: [[52, 67, 85, 96],[62]]
    '''
    df = df.copy(deep=True)
    the_title = "Topics:"+str(topicsListsWithIDs)+' '+ column

    res= []
    for topic_ids in topicsListsWithIDs:
        topic_df = df[df['cluster_id'].isin(topic_ids)]

        #res_df = topic_df.groupby([df[column]]).count().add_suffix('_count').reset_index()[[column, 'cluster_id_count']]

        res_df = topic_df[[column, 'picture_id', 'cluster_id']].groupby([column, 'picture_id']).count().add_suffix('_count').reset_index()

        # problem topicsListsWithIDs = [[52, 67, 85, 96],[62]]
        # die erste topicslist hat mehr Einträge. Da diese ähnliche Cluster sind ist die warhscheinlichkeit hoch, 
        # dass ein Bild diese Ids aus als cluster enthält und somit wird das Bild öffter gezählt
        # Lösung:
        res_df = res_df[[column, 'picture_id']].groupby(column).count().add_suffix('_count').reset_index()
        
        
        #normalize
        df_all = df[[column, 'picture_id']].groupby(column).count().add_suffix('_count').reset_index()
        df_all['all_picture_id_count'] = df_all['picture_id_count'].map(lambda x: x/4)

        res_df = pd.merge(res_df, df_all[[column, 'all_picture_id_count']], on=column)
        
        
        #res_df['picture_id_count_normalized'] = res_df.map(lambda x: float(x['picture_id_count']) / x['all_picture_id_count'])
        res_df['picture_id_count_normalized'] = res_df['picture_id_count'] / res_df['all_picture_id_count']
        

        # smoothing
        #f = interp1d(test_x, medfilt(test_y, 7), kind='cubic')
        #xnew = np.linspace(1785, 1918, 20)
        #xnew, f(xnew), 'g-', 
        k2 = [0.5,0.5]
        k4 = [0.25,0.25,0.25,0.25]
        k5 = [0.2,0.2,0.2,0.2,0.2]
        k10 = [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]
        
        if smooth:
            res_df['picture_id_count_normalized'] = convolve(res_df['picture_id_count_normalized'], k4, mode='same')
                      
        legendname = 'Topic-Clusters:'+str(topic_ids)
        res_df.rename(columns={'picture_id_count_normalized': legendname}, inplace=True)
        res.append(res_df[[column,legendname]])

    # join them
    plot_df = pd.concat(res)
    #the_title = "Topics:"+str(topicIDs)+' '+ column
    
    data_y = np.array(plot_df[legendname])
    data_x = np.array(plot_df[column])
    
    fig = plt.figure(figsize=(11, 7))
    plt.title(the_title)
    #plt.plot(x=column, xticks=range(1785,1918,10), xlim=((1785,1918)), figsize=(11,5))
    plt.legend(loc='best')
    plt.plot(data_x, data_y)
    fig.show()
    
    return fig
    

def plotAllCount(df_o, column='mid_year'):   
    df = df_o.copy(deep=True)
   
    df = df[[column, 'picture_id']].groupby([column]).count().add_suffix('_count').reset_index()
    df['picture_id_count'] = df['picture_id_count'].map(lambda x: x/4)
    
    the_title = 'All Count '+column
    
    
    data_y = np.array(df['picture_id_count'])
    data_x = np.array(df[column])
    
    fig = plt.figure(figsize=(11, 5))
    plt.title(the_title)
    #plot = df[[column, 'picture_id_count']].plot(x=column, xticks=range(1785,1918,10), xlim=((1785,1918)), title=the_title, figsize=(11,5), legend=False)
    
    #, xticks=range(1785,1918,10)
    #, xlim=((1785,1918))
    plt.plot(data_x, data_y)
    fig.show()

    return fig

def plotMatrix(df_o):
    # Display a random matrix with a specified figure number and a grayscale
    # colormap
    #plt.clf() 
    
    #df = df_o.copy(deep=True)
    
    plt.clf()
    fig = plt.figure(figsize=(15, 15))
    plt.title('Topic Matrix')
    plt.matshow(topic_martix_df.as_matrix(), fignum=100, cmap=plt.cm.gray)

    topic_martix_path = dir_reports+"/plots/"+"topic_martix.jpg"
    #plt.show()

    return plt.gcf()

def plotTopicFrequently(df_o):
    df = df_o.copy(deep=True)
    n = 10
    
    df = df.sort_values('count', ascending=False)[0:n]
    topics= []*n
    for i, x in enumerate(df['name']):
        topics.append( x )
    for i, x in enumerate(df['topic_id']):
        topics[i] = str(x) + ': ' + str(topics[i])

    y_pos = np.arange(len(topics))

    count = df['count']

    fig = plt.figure(figsize=(11, 4))
    plt.barh(y_pos, count)
    plt.yticks(y_pos, topics)
    #plt.xlabel('Topic')
    plt.xlabel('Count')
    plt.title('Often combined top ' + str(n))
    #plt.margins(0.2)
    #plt.subplots_adjust(bottom=0.15)
    #plt.set_xticklabels
    plt.tight_layout()
    plt.show()
    return fig





# In[ ]:

#merge Data!!!
import db
my_db = db.Db()

# merge Tags
meta_tag_120_df = pd.merge(my_db.metadata_long_19, my_db.clusters_long_19)
#meta_tag_120_df['clusters_count'] = 4

topics_per_picture = 4

append_mid_year(meta_tag_120_df);

print ('-> merging clusters_long_19 lost %d that is %f p.' % (len(my_db.metadata_long_19) - len(meta_tag_120_df)/4, float(len(meta_tag_120_df)) / len(my_db.metadata_long_19)/4))
# pictures with tags only!
#meta_tag_120_df

# merge artists 
# on picture_id
meta_tag_120_artists_df = pd.merge(meta_tag_120_df, my_db.artist_origin[['picture_id','metadata_nationality','metadata_country','metadata_capital','metadata_longitude', 'metadata_latitude']], on='picture_id')


print ('-> merging artists lost %d that is %f p.' % (len(meta_tag_120_df)/topics_per_picture - len(meta_tag_120_artists_df)/topics_per_picture, float(len(meta_tag_120_artists_df)/topics_per_picture) / len(meta_tag_120_df)/topics_per_picture))


# In[ ]:

meta_tag_120_artists_df


# In[ ]:

# matrix, topics_frequently_df


#u jedem untersuchendem topic ein df
# 'topic_id', 'name', 'count'

import json
with open('../data/topics_benannt.txt') as topic_word_json:
        topic_word_dic = json.load(topic_word_json)

        
        
        
topics = range(0,110)
#opics = [3,4]


matrixAll = [None] * 110
topics_frequently_df = [None]*110

for topicA in topics:
    #rint(topicA)
    
    picture_id_with_topic = meta_tag_120_artists_df[meta_tag_120_artists_df['cluster_id']==topicA]['picture_id']
    #for topicB in 
    a = meta_tag_120_artists_df[meta_tag_120_artists_df['cluster_id'] != topicA]

    matrix= [0] * 110

    for pic_id in picture_id_with_topic:
        pictures = a[a['picture_id']== pic_id]
        for topic_id in pictures['cluster_id']:
            matrix[topic_id] = matrix[topic_id] +1

    matrixAll[topicA] = matrix
    
    all_dfs = []
    for i, count in enumerate(matrix):
        df = pd.DataFrame([[i, topic_word_dic[str(i)], count]])
        df.columns= ['topic_id', 'name', 'count']
        all_dfs.append(df)

        
    # topics_frequently_df
    topics_frequently_df[topicA] = all_dfs[topics[0]]
    topics_frequently_df[topicA].columns= ['topic_id', 'name', 'count']
    for x in all_dfs[1:]:
        topics_frequently_df[topicA] = topics_frequently_df[topicA].append(x, ignore_index=True)

# results:
        
# matrix 
topic_martix_df = pd.DataFrame(matrixAll)

# topics_frequently_df
topics_frequently_df = topics_frequently_df


# In[ ]:

#save

f = plotAllCount(meta_tag_120_artists_df);
AbsoluteTopicCount_path = dir_reports+"/plots/"+"AbsoluteTopicCount.jpg"
f.savefig(AbsoluteTopicCount_path)


f = plotMatrix(topic_martix_df);
topic_martix_path = dir_reports+"/plots/"+"topic_martix.jpg"
f.savefig(topic_martix_path)


# In[ ]:

for topic in range(0,109):
    f = plotTopicFrequently(topics_frequently_df[topic]);
    AbsoluteTopics_path = dir_reports+"/plots/"+"TopicFrequentlyPlot_"+str(topic)+".jpg"
    f.savefig(AbsoluteTopics_path)


# In[ ]:

for x in range(0,109):
    topic = x
    f = plotTopic([[topic]], meta_tag_120_artists_df);
    AbsoluteTopics_path = dir_reports+"/plots/"+"NormalizedTopicPlot_"+str(topic)+".jpg"
    f.savefig(AbsoluteTopics_path)


# In[ ]:

# from parallel plot
from src.queries import db
from matplotlib import pyplot as plt
from matplotlib import cm
import pandas as pd
import numpy as np
from pandas.tools.plotting import parallel_coordinates

def compare_cluster_epochs_nationality(clusters=[0],countries=['Germany', 'France','Britain'], epochs=[1789,1848,1875,1914]):
    my_db = db.Db()
    #combined = pd.merge(my_db.metadata_long_19, my_db.clusters_long_19)
    combined = my_db.final_cluster_nation
    epochs_window = [epochs[i:i+2] for i in xrange(len(epochs)-1)]

    results = []
    for nationality in countries:
        tmp = [nationality]
        for (beginning, end) in epochs_window:
            # This has to be renormalized later!
            art_per_nation_and_epoch = combined.query('metadata_country == "{0}" & mid_year >= {1} & mid_year < {2}'
                                                                       .format(nationality, beginning, end))
            #count = art_per_nation_and_epoch.groupby('cluster_id').count()
            #print count['metadata_surname']
            art_per_nation_epoch_and_cluster = art_per_nation_and_epoch[art_per_nation_and_epoch['cluster_id']
                .isin(clusters)].groupby('picture_id').count()

            #!!!!important step!!!!
            tmp.append(len(art_per_nation_epoch_and_cluster)/float(len(art_per_nation_and_epoch)))  
        results.append(tmp)
    return results

def plot_parallel(topic_group, topic_group_name):
    #plot
    df_plot = pd.DataFrame(compare_cluster_epochs_nationality(topic_group), columns=['country', '1789-1847', '1848-1874', '1875-1914'])

    fig = plt.figure(figsize=(11, 4))
    plt.ylabel('relative frequency')
    plt.xlabel('time range')
    plt.title('Topic: {0} {1}: Frequency over Time and by Nationalities'.format(topic_group_name,topic_group))
    #plt.suptitle('Cluster description: {0}'. format('to come ...'))
    parallel_coordinates(df_plot, 'country', colormap='jet', linewidth=5)

    plt.show()
    return fig

for x in range(0,109):
    topic = x
    #f = plotTopic([[topic]], meta_tag_120_artists_df);
    f = plot_parallel([topic], '')
    AbsoluteTopics_path = dir_reports+"/plots/"+"Parallel_Plot_"+str(topic)+".jpg"
    f.savefig(AbsoluteTopics_path)
    


# In[ ]:




# In[ ]:


# path
my_reader = reader.CSV_reader()
paths_df = pd.DataFrame(my_reader.get(my_reader.path_image_path), columns = ['picture_id', 'data_name', 'data_path'])

paths_df['picture_id'] = paths_df['picture_id'].astype(int)
paths_df['data_name'] = paths_df['data_name'].astype(str)
paths_df['data_path'] = paths_df['data_path'].astype(str)

#paths_df


# cluster words
my_reader = reader.CSV_reader()
cluster_words = pd.DataFrame(my_reader.get('../data/topics.txt'), columns = ['words'])

cluster_words['words'] = cluster_words['words'].astype(str)

#cluster_words['words'] = cluster_words['words'].map(lambda x: ("'" +x+"'").decode('utf-8'))
#cluster_words['words'] = cluster_words['words'].map(lambda x: u"'" +x)

cluster_words['words'] = cluster_words['words'].map(lambda x: x.split(':')[1])
#cluster_words['topic'] = cluster_words_tmp['words'].map(lambda x: x.split(':')[0])

#cluster_words


import json

# topic namen

with open('../data/topics_benannt.txt') as topic_word_json:
        topic_word_dic = json.load(topic_word_json)

#topic_word_dic



# merge img paths
#paths_df.query('picture_id == 100000')
df = pd.merge(meta_tag_120_artists_df, paths_df, on='picture_id')

df['data_path'] = df['data_path'].map(lambda x:x[11:])

df['my_path'] = 'data/artigo-images'+df['data_path']+df['data_name']
df


# In[ ]:

template_topics = []


# 0-109
cluster_topics = range(0,109)
#cluster_topics = [np.random.randint(109)]
#cluster_topics = [0,1,2,3,4,5,6]

max_num_images = 9
rank = 1
#folder = '/albertina/'
folder = '/koeln/'

folders = ['/albertina/','/amherst/','/inspektorx/','/koeln/', '/artemis/','/kunsthalle_karlsruhe/']

for cluster_topic in cluster_topics:
    #df21 = df[df['data_path'] == folder]
    df21 = df[df['data_path'].isin( folders)]
    df2 = df21[df21['cluster_id'] == cluster_topic]
    
    df3 = df2[df2['cluster_rank'] == rank]
    #images = np.array(df3['my_path'])
    image_ids = np.array(df3['picture_id'])
    cluster_ranks = np.array(df3['cluster_rank'])

    topic_dic={}
    
    if(len(image_ids) != 0):

        if len(image_ids) >= max_num_images:
            image_ids = image_ids[0:len(image_ids)-1]
            #np.random.shuffle(images)
            image_ids = image_ids[:max_num_images]

        words = np.array(cluster_words)[cluster_topic]
        
        images = []
        for picture_id in image_ids:
            image_df = df21[df21['picture_id'] == picture_id]
            image_df_first = image_df[image_df['cluster_rank'] == 1]
            
            #print(np.array(image_df_first['metadata_country'])[0])
            
            image = {
                'picture_id': picture_id,
                'data_name': np.array(image_df_first['data_name'])[0],
                'data_path': np.array(image_df_first['data_path'])[0],
                'my_path': np.array(image_df_first['my_path'])[0],
                
                'mid_year': np.array(image_df_first['mid_year'])[0],
                'metadata_country': str(np.array(image_df_first['metadata_country'])[0]).decode('utf-8'),
                'metadata_name': str(np.array(image_df_first['metadata_name'])[0]).decode('utf-8'),
                'metadata_surname': str(np.array(image_df_first['metadata_surname'])[0]).decode('utf-8'),
            }
            
            topics = []
            for topic_id in image_df['cluster_id']:
                cluster_rank = image_df[image_df['cluster_id']==topic_id]['cluster_rank']
                
                topic = {
                    'topic_id':topic_id,
                    'topic_name':topic_word_dic[str(topic_id)],
                    'cluster_rank': int(cluster_rank)
                }
                topics.append(topic)
            
            image['topics']= topics
            images.append(image)
        
        topic_dic = {
            'topic_id':cluster_topic,
            'topic_name': topic_word_dic[str(cluster_topic)],
            'words':words,
            'images': images
        }
        
        template_topics.append(topic_dic)

template_topics


# In[ ]:

# copy using images to folder
from shutil import copyfile
count = 0
destination = dir_reports + '/images'

# 24 missing
for x in range(107):
    print x, template_topics[x]['images'][0]['topics'][0]['topic_id']
    for img in template_topics[x]['images']:
        count+=1
        print img['data_path']
        copyfile('../'+img['my_path'],destination+'/'+img['data_name'])
count


# In[ ]:

# 
# Jinja

txt_merging_topics = 'Merging clusters_long_19 lost %d rows that is %f p.' % (len(my_db.metadata_long_19) - len(meta_tag_120_df)/4, float(len(meta_tag_120_df)) / len(my_db.metadata_long_19)/4)
txt_merging_artists = 'Merging artists lost %d rows that is %f p.' % (len(meta_tag_120_df)/topics_per_picture - len(meta_tag_120_artists_df)/topics_per_picture, float(len(meta_tag_120_artists_df)/topics_per_picture) / len(meta_tag_120_df)/topics_per_picture)


template_vars = {
    "title" : "Sales Funnel Report - National",
    "national_pivot_table": "test",
    "AbsoluteTopicCount": AbsoluteTopicCount_path,
    "topics": template_topics,
    "PathReportPlots": '../AnalysisTool/plots/',
    "topic_martrix": topic_martix_df,
    "txt_merging_topics": txt_merging_topics,
    "txt_merging_artists": txt_merging_artists,
    "num_pictures": len(meta_tag_120_artists_df)/4,
    }


#env.list_templates()

template = env.get_template("templates/mytemplate1.html")

#render
html_out = template.render(template_vars)
#save
with open(dir_reports+"Output.html", "w") as text_file:
    text_file.write(html_out.encode('utf-8'))
text_file.close()

