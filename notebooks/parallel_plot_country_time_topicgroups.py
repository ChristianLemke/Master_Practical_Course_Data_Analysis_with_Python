# coding: utf-8

# Eric Hobsbawm :
# das Zeitalter der Revolution (1789–1848)
# das Zeitalter des Kapitals (1848–1875) 
# das Zeitalter des Imperiums (1875–1914)
# 
# [3] Reinhart Koselleck prägte den Begriff der Sattelzeit, die etwa von 1770 bis 1830 gedauert habe.


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

# Frauen im wandel der zeit
topic_group_name= 'woman'
topic_group=[3,9,45,97]

# Der Mann im wandel der zeit
#topic_group_name= 'men'
#topic_group=[79, 1,43, 81, 55]

# Krieg im wandel der zeit
#topic_group_name= 'war'
#topic_group=[62,91,53,105,21]

# Religion im wandel der zeit
#topic_group_name= 'religion'
#topic_group=[52, 85, 96, 15, 36, 67, 82, 88]

# Landschaft
#topic_group_name= 'landscape'
#topic_group=[0, 27, 10, 51, 13, 17, 41, 60 ,63, 109, 77]

# Portraits im Wandel der Zeit
#topic_group_name= 'portraits'
#topic_group=[101, 103, 69, 92, 56, 69]


#plot
df_plot = pd.DataFrame(compare_cluster_epochs_nationality(topic_group), columns=['country', '1789-1847', '1848-1874', '1875-1914'])

plt.figure(figsize=(12,8))
plt.ylabel('relative frequency')
plt.xlabel('time range')
plt.title('Topic_group: {0} {1} Frequency over Time and by Nationalities'.format(topic_group_name,topic_group))
#plt.suptitle('Cluster description: {0}'. format('to come ...'))
parallel_coordinates(df_plot, 'country', colormap='jet', linewidth=5)

plt.show()