import lib.csv_reader as csvr
import pandas as pd
import csv


reader = csvr.CSV_reader()
tags = filter(lambda x: len(x) == 3, reader.get_tags())
tags_indices = reader.tags_labels
df = pd.DataFrame(data=tags, columns=reader.tags_labels)
grouped = df.groupby('picture_id')
df = grouped.aggregate(lambda x: tuple(x))
df['grouped'] = df['tag_tag']
#print(df)
i = 0
with open('tag_transactions.csv','wb') as out:
    csv_out = csv.writer(out)
    for group in df.grouped:
        csv_out.writerow(group)

