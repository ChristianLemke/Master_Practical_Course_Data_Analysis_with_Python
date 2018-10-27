from copy import deepcopy
import re

## '', '1783', '1586-1588', '1565/1570', '1639 - 1641', '1677-07-21 - 1677-07-21', '600 - 800',
## '1776-01-24 - 1801', 'um 1650', '1929, 99 (S 9)', 'letztes Viertel 15. Jahrhundert', '1682 oder 1683',
## 'Mitte 16. Jahrhundert', 'um 1810-20 / 1863 (1. Ausgabe)', '18th century', '18th-19th century',
## 'before 1913', 'mid-19th century', 'late 19th century', 'first quarter of the 19th century'
## '1929, 347 (3H47)', 'February, 1922', 'ca. 1911', '18th-early 20th century', 'Late 18th century', Erstes Drittel 16. Jh.


class year_classifier(object):

    def __init__(self):

        self.year_list = []
        self.year_with_text = []
        self.year_between_list = []
        self.year_century_list = []
        self.to_remove =[]
        self.re_pattern_year = re.compile('([\d]{4,4})')
        self.re_pattern_year_with_text = re.compile('[a-zA-Z.\s]*([\d]{4,4})[\s-]*\Z')
        #self.re_pattern_year_between = re.compile('[\s\w.]*([\d]{4})[-\s\w]*[-/und]+[-\s\w]*([\d]{4})[-\s\w]*')
        self.re_pattern_year_between = re.compile("^\D*(\d{1,4})(?:-\d{2}){0,2}[\s\-/und]+(\d{1,5})(?:-\d{2}){0,2}.*$")
        self.re_pattern_century = re.compile("^\D*(\d{2})\D*(Jahrhundert|Jh|Century|th)\D*$")


    def get_classified_years_dic(self, metadata_dic_para):
        """
        Takes a metadata_dic and copys it.
        Adding {from_year: x} and {to_year: y} and removes unclassified entrys.
        Prints out a short result.

        """
        self.metadata_dic = deepcopy(metadata_dic_para)

        for pic_key in self.metadata_dic.iterkeys():
            year_string = self.metadata_dic[pic_key]['metadata_year']

            if year_string != None or len(year_string) == 0 or 'B.C.' not in year_string:
                if len(year_string) == 4 and self.re_pattern_year.match(year_string):
                    # bsp: 1933
                    self.metadata_dic[pic_key]['from_year'] = int(year_string)
                    self.metadata_dic[pic_key]['to_year'] = int(year_string)
                    self.year_list.append(year_string)

                elif self.re_pattern_year_with_text.match(year_string):
                    # bsp: 'ca. 1959', 'um 1932', 'wohl 1955', 'vor 1844'
                    # keine exakte Zeitangaben
                    part = self.re_pattern_year_with_text.search(year_string).group(1)
                    self.metadata_dic[pic_key]['from_year'] = int(part)
                    self.metadata_dic[pic_key]['to_year'] = int(part)
                    self.year_list.append(year_string)

                elif self.re_pattern_year_between.match(year_string):
                    # bsp: '1841/1872', '1841-1872', 'um 1501-1502', '1841 - 1872', '1841-07 - 1842-07', '1671-07-21 - 1672-07-21'
                    from_year = int(self.re_pattern_year_between.search(year_string).group(1))
                    self.metadata_dic[pic_key]['from_year'] = from_year
                    to_year = int(self.re_pattern_year_between.search(year_string).group(2))
                    self.metadata_dic[pic_key]['to_year'] = (int(from_year)/100)*100 + int(to_year) if int(to_year)/100 is 0 else int(to_year)
                    self.year_between_list.append((year_string, from_year, self.metadata_dic[pic_key]['to_year']))

                elif self.re_pattern_century.match(year_string):
                    self.metadata_dic[pic_key]['from_year'] = (int(self.re_pattern_century.search(year_string).group(1)) -1) * 100 +1
                    self.metadata_dic[pic_key]['to_year'] = int(self.re_pattern_century.search(year_string).group(1)) * 100
                    self.year_century_list.append(year_string)

                else:
                    # remove entry (invalid)
                    self.to_remove.append(pic_key)

        #for pic_key in to_remove:
        for pic_key in self.to_remove:
            del self.metadata_dic[pic_key]

        return self.metadata_dic

    def print_results_stats(self):
        print "year_classifier.add_classified_years:"
        print "year_single: " + str(len(self.year_list))
        print "year_between: " + str(len(self.year_between_list))
        print "centuries: " + str(len(self.year_century_list))
        print "removed: " + str(len(self.to_remove))
        print "new size / old size: " + str(len(self.metadata_dic)) + "/" + str(len(self.metadata_dic)+len(self.to_remove))
        print "missclassification: " + str(float(len(self.to_remove))/(len(self.metadata_dic)+len(self.to_remove)) * 100) +" %"
