import json


class json_reader:
    def __init__(self):
        self.dict = {}
        self.path = "../data/TagGroups.json"

#Opens the JSON-File
    def get(self, path):
        with open(path) as data_file:
            data = json.load(data_file)
            return data

    def getTagGroups(self):
        return self.get(self.path)

#dict = json_reader().getTagGroups()
#print(dict["TagGroups"])