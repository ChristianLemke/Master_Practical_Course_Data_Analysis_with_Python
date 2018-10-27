import year_classifier as yc
import csv_reader as csvr
import json

class nation_classifier(object):

    # Get the metadata dictionary from csv_reader
    def get_metadata(self):
        dic = csvr.CSV_reader().get_metadata_dic()
        return dic

    # Get the dictionary containing all relevant data including the classified years
    def get_dict_all_relevant_data(self):
        dic = yc.year_classifier().get_classified_years_dic(nation_classifier().get_metadata())
        return dic

    # Merge information from artist_nation to the dictionary with all relevant data
    def add_nation_to_dict(self):
        dic = self.get_dict_all_relevant_data()
        with open('../data/artist_nation.json') as data_file:
            nations = json.load(data_file)

        for entry in dic:
            # Search for key [query] in json and get the key (= artists nationality)
            if len(dic[entry]["metadata_name"]) != 0:
                query = "" + dic[entry]["metadata_surname"] + ", " + dic[entry]["metadata_name"]
            else:
                query = str(dic[entry]["metadata_surname"])

            # Add new field "nationality" to the dictionary
            try:
                nat = nations[str(query)]
                dic[entry]["nationality"] = str(nat)
            except:
                # In case the key "query" can't be found in the json file
                dic[entry]["nationality"] = "UNKNOWN"

            print dic[entry]
        return dic

nation_classifier().add_nation_to_dict()
