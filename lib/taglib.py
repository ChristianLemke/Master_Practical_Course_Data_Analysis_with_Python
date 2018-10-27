import csv_reader as csvr
import csv

allMetadata = csvr.CSV_reader().get_metadata()
allTags = csvr.CSV_reader().get_tags()
del allMetadata[:10]
del allTags[:10]

print(allTags[1])
print(allMetadata[2])

def getAllTags():
    tags=[]
    for element in allTags:
        if element[1] not in tags:
            tags.append(element[1])
            print(element[1] + " appended")
    return tags

#Returns all tags
def getAllTagsWithMentions(mentions):
    tags=[]
    for element in allTags:
        if element[1] not in tags:
            tags.append(element[1])
            print(element[1] + " appended")
    return tags

#returns all unique Tags with amount of mentions
def getNumberPerTag():
    res = {}
    for element in allTags:
        if element[1] not in res.keys():
            res[element[1]] = 1
        else:
            res[element[1]] = res[element[1]] + int(element[2])
    res = sorted(res.items(),key=lambda x:x[1])
    with open ('result.csv', 'wb') as f:
        writer = csv.writer(f)
        writer.writerows(res)
    return res

def getPictureWithAllTags():
    res = {}
    for element in allMetadata:
        res[element[0]] = []
    for tagElement in allTags:
        for key in res:
            if tagElement[0] == key:
                if tagElement[1] not in res[key]:
                 res[key].append(tagElement[1])
    return res



#returns Names of all pintors
def getAllPintorNames():
    pintors = []
    for element in allMetadata:
        if element[2] + " " +element[3] not in pintors:
            pintors.append(element[2] + " " + element[3])
            print(element[2] + " " +element[3] + " appended")
    return pintors

#returns a Dictionary with <Maler, #Bilder>
def getPicturesPerPintorAmount():
    res = {}
    for element in allMetadata:
        if element[2] + " " +element[3] not in res.keys():
            res[element[2] + " " +element[3]] = 1
        else:
            res[element[2] + " " +element[3]] = res[element[2] + " " +element[3]] + 1
    res = sorted(res.items(),key=lambda x:x[1])
    return res

#returns all Pictures for one pintor
def getPicturesPerPintor(name, surname):
    res = []
    for element in allMetadata:
        if element[2] == name and element[3] == surname:
            res.append(element)
    return res

print getNumberPerTag()
#print(getPictureWithAllTags())