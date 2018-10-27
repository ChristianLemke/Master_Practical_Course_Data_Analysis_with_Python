import csv
import matplotlib.pyplot as plt
from ipywidgets import Color
from matplotlib import cm
import operator

#amount of nations in in the pie chart
n = 10
labels = []
sizes = []
colors = []
result_dict = dict()

with open("../data/artist_origin.csv", "rb") as file:
    file.readline()   # skip the first line
    data = csv.reader(file, delimiter=',', quotechar='|')
    rowsum = 0
    artistList = []

    for row in data:
        #check if Artist is already done
        if row[0]+' '+row[1] in artistList:
            continue

        # add country to dict or increment counter
        else:
            if row[4] in result_dict.keys():
                result_dict[row[4]] = result_dict[row[4]] + 1
            else:
                result_dict[row[4]] = 1


sorted_result_dict = sorted(result_dict.items(), key=operator.itemgetter(1), reverse=True)
for x in range(0,n):
    labels.append(sorted_result_dict[x][0])
    sizes.append(sorted_result_dict[x][1])
    colors.append(((float(x)/float(n)*255.)/255., (float(x)/float(n)*255.)/255., (float(x)/float(n)*255.)/255.))


patches, texts = plt.pie(sizes, colors=colors, shadow=True, startangle=90)
plt.legend(patches, labels, loc="best")
plt.axis('equal')
plt.tight_layout()
plt.show()