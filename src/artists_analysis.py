import lib.csv_reader as csvr
from collections import Counter

my_reader = csvr.CSV_reader()

# extract the artists names - schema: <name> " " <surname>
only_names = map(lambda row: "%s %s" % (row[-3], row[-2]), my_reader.get_metadata())
# count the artists
count_names = Counter(only_names)
# show a top-list and remove the most common artist, which is the "artist" where none was specified
top11 = count_names.most_common(11)
top10 = top11[1:]
print(top10)



