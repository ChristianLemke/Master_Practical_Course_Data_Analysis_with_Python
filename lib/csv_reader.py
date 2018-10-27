# -*- coding: utf-8 -*-
import csv
from copy import deepcopy


class CSV_reader(object):

    def __init__(self):
        self.tags = []
        self.uniqueTags = []
        self.metadata = []
        self.metadata_dic = {}
        self.tags_dic = {}
        self.data_dic = {}
        self.path_tags = "../data/artigo-tags.csv"
        self.path_metadata = "../data/artigo-metadata.csv"
        self.path_image_path = "../data/artigo-path.csv"
        self.path_uniqueTags = "../data/uniqueTags.csv"
        self.metadata_labels = ['picture_id', 'metadata_year', 'metadata_name', 'metadata_surname', 'metadata_location']
        self.tags_labels = ['picture_id','tag_tag','tag_count']
        self.delimiter = ','

    def get(self, path):
        tmp = []
        with open(path, 'rb') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                tmp.append(row)
        return tmp

    def get_metadata(self):
        if(len(self.metadata) == 0):
            self.metadata = self.get(self.path_metadata)
        return self.metadata

    def get_metadata_dic(self):
        if not any(self.metadata_dic) :
            for data in self.get_metadata():
                self.metadata_dic[data[0]] = {}
                for i in range(1,len(self.metadata_labels)):
                    self.metadata_dic[data[0]].update({self.metadata_labels[i]: data[i]})
        return self.metadata_dic

    def get_tags(self):
        if(len(self.tags) == 0):
            self.tags = self.get(self.path_tags)
        return self.tags

    def get_uniqueTags(self):
        if(len(self.uniqueTags) == 0):
            self.uniqueTags = self.get(self.path_uniqueTags)
        return self.uniqueTags

    def get_tags_dic(self):
        if not any(self.tags_dic) :
            for data in self.get_tags():
                self.tags_dic[data[0]] = {data[1]: data[2]}
        return self.tags_dic

    def get_data_dic(self):
        """
        All data in a dictionary
        bsp: {5988 :{'metadata_location': 'Chicago (Illinois)',
        'metadata_name': 'Paul',
        'metadata_surname': 'Gauguin',
        'metadata_year': '1892',
        'tags': {'ABSTRAKT': '1',
            'AQUARELL': '2',
            'AST': '4',...}
        }, ...}
        """
        self.data_dic = deepcopy(self.get_metadata_dic())

        for entry in self.data_dic:
            self.data_dic[entry]["tags"] = {}

        for tag in self.get_tags():
            if tag[0] in self.data_dic:
                self.data_dic[tag[0]]["tags"].update({tag[1]: tag[2]})

        return self.data_dic
