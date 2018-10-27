#-*- coding: utf-8 -*-
import csv
import lib.csv_reader as csvr
import pandas as pd
from src.clustering import topic_mining as tm
import year_classifier as yc
from time import time

t0 = time()
csv_reader = csvr.CSV_reader()
metadata = csv_reader.get_metadata_dic()
metadata_clean = yc.year_classifier().get_classified_years_dic(metadata)
metadata_clean_kv = [(k,v['metadata_name'], v['metadata_surname'], v['metadata_location'], v['from_year'], v['to_year']) for k, v in metadata_clean.iteritems()]
tags = csv_reader.get_tags()
print('reading csv files took: {}s'.format((time()-t0)))

# Create SQL-like tables
t0 = time()
metadata_table = pd.DataFrame(metadata_clean_kv, columns=['picture_id', 'metadata_name', 'metadata_surname',
                                                          'metadata_location', 'from_year', 'to_year'])
tag_table = pd.DataFrame(tags, columns=csv_reader.tags_labels)


# Tables restricted to the long 19th century
metadata_table_long_nineteenth_century = metadata_table.query('from_year >= 1785 and to_year <= 1918', inplace=False)
tag_table_long_nineteenth_century = tag_table[(tag_table['picture_id']
                                               .isin(metadata_table_long_nineteenth_century['picture_id']))]

# Tags only
tags_grouped = tag_table_long_nineteenth_century.groupby('picture_id', as_index=False)
tags_grouped = tags_grouped.aggregate(lambda x: list(x))
tags_recomputed = [sum(map(lambda t, c: int(c)*[t], a, b), []) for (a, b) in zip(tags_grouped['tag_tag'].values.tolist(),
                                                                                 tags_grouped['tag_count'].values.tolist())]
tags_recomputed_keys = tags_grouped['picture_id'].values
print('Setting up DataFrames took: {}s'.format((time()-t0)))

# Stats
print('Kept {0:0.2f} % of tag data'.format((tag_table_long_nineteenth_century.shape[0] /
                                                float(tag_table.shape[0])) * 100))

print('kept {0:0.2f} % of metadata'.format((metadata_table_long_nineteenth_century.shape[0] /
                                                 float(metadata_table.shape[0])) * 100))

# Topic Mining
t0 = time()
topic_miner = tm.TopicMiner(n_topics=110, max_features=3500, topics_per_document=4, top_words=15)
data = [(' '.join(row)) for row in tags_recomputed]
topics = topic_miner.fit(data)
print('Mining topics took: {}s'.format((time()-t0)))
print('TopicMiner extracted the following topics: \n')
for topic in topics:
    print(topic)

# Creating a cluster table
#(picture_id, clusterid)
t0 = time()
clusters_long_nineteenth_century = []
for idx, v in enumerate(topic_miner.tfidf):
    clusters = topic_miner.predict(v)
    picture_id = tags_recomputed_keys[idx]
    for indx, c in enumerate(clusters):
        clusters_long_nineteenth_century.append((picture_id, c, indx+1))

cluster_table_long_nineteenth_century = pd.DataFrame(clusters_long_nineteenth_century, columns=['picture_id', 'cluster_id', 'cluster_rank'])
print('Predicting topics took: {}s'.format((time()-t0)))


def write_tables_to_csv(table, path):
    with open(path, 'wb') as csv_file:
        csv_writer = csv.writer(csv_file)
        for row in table:
            row_ = row.encode('utf-8')
            csv_writer.writerow(row_)

t0 = time()
tag_table_long_nineteenth_century.to_csv(path_or_buf='tags_long_19.csv', index=False)
metadata_table_long_nineteenth_century.to_csv(path_or_buf='metadata_long_19.csv', index=False)
cluster_table_long_nineteenth_century.to_csv('clusters_long_19.csv', index=False)
write_tables_to_csv(topics, 'topics.txt')
print('Writing csv files took: {}s'.format((time()-t0)))