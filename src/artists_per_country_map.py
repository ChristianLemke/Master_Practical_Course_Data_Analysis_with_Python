"""
This script fetches a .csv file which contains information on paintings including the origin of the painting's artist.
Based on this information, the script draws a map of europe and plots the percentage of paintings in the dataset
coming from each country on the map using size-adjusted circles (bigger circle == more paintings from that country).
Only countries with at least 1.5% ("threshold") are taken into consideration.
"""


from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import csv

class nation_visualization(object):

	def __init__(self):
		self.countries = {}
		self.labels = []
		self.sizes = []

	def artists_per_country_map(self):
		plt.figure(figsize=(50, 50)) #Really large map to fill the screen

		
		#Determine circle size
		with open("../data/artist_origin.csv", "rb") as file:
			file.readline()   # skip the first line
			data = csv.reader(file, delimiter=',', quotechar='|')
			rowsum = 0

			for row in data:
				rowsum += 1
				if row[4] in self.countries.keys():
					self.countries[row[4]][0] += 1
				else:
					self.countries[row[4]] = [1 , row[6], row[7]]


			for key in self.countries.keys():
				self.countries[key][0] = round(float(self.countries[key][0]) / rowsum * 100, 2)


		#Basemap parameters to show europe
		x1 = -20.
		x2 = 40.
		y1 = 32.
		y2 = 64.
		 
		m = Basemap(resolution='l',projection='mill', llcrnrlat=y1, urcrnrlat=y2, llcrnrlon=x1, urcrnrlon=x2) #Europe
		#m = Basemap(resolution='l',projection='mill') #Whole world

		m.drawcoastlines(linewidth=0.2)
		m.drawcountries(linewidth=0.2)
		m.fillcontinents(color='#FCF5E1', lake_color='#EBFEFE') #Land color
		m.drawmapboundary(fill_color='#EBFEFE') #Water color
		#m.bluemarble() #Space view


		lat = []
		lon = []
		factor = 300
		threshold = 1.50 #Ignore countries with less than x % of images

		#Draw plots on map
		for entry in self.countries:
			if (self.countries[entry][0] < threshold):
				continue
			else:
				lon.append(float(self.countries[entry][1]))
				lat.append(float(self.countries[entry][2]))
				self.sizes.append(float(self.countries[entry][0]) * factor)
				self.labels.append(str(entry) + "\n" + str(self.countries[entry][0]) + "%")

		x,y = m(lon,lat)
		m.scatter(x,y, s = self.sizes, marker = 'o', zorder = 2, alpha=.3)
		for label, xpt, ypt in zip(self.labels, x, y):
			plt.text(xpt, ypt, label, fontsize=14, horizontalalignment='center', verticalalignment='center') #Add labels "NAME\n XX.XX%" to plots

		plt.title('Percentage of paintings coming from each country', fontsize=30)
		plt.show()

nation_visualization().artists_per_country_map()

