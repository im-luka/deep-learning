import collections
from gc import collect
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

csv_file = "csv_files/Spotify 2010 - 2019 Top 100.csv"
data = pd.read_csv(csv_file)

plt.figure(figsize=(12,8))

# FIRST GRAPH

year = data["year released"]
year_dicty = dict(collections.Counter(year))
year_dicty = collections.OrderedDict(sorted(year_dicty.items()))

plt.title("Amount of songs released in particular year")
plt.xlabel("Year Released", fontweight="bold", fontsize=20)
plt.ylabel("Amount",  fontweight="bold", fontsize=20)

plt.bar(range(len(year_dicty)), list(year_dicty.values()), align="center")
plt.xticks(range(len(year_dicty)), list(year_dicty.keys()))

plt.show()

# SECOND GRAPH

genre = data["top genre"]
genre_dicty = dict(collections.Counter(genre))

labels = []
counts = []

for key, item in genre_dicty.items():
  if(item > 15):
    labels.append(key)
    counts.append(item)

plt.title("The most popular genres")
plt.pie(counts, labels=labels)

plt.axis("equal")
plt.show()

# THIRD GRAPH

solo = []
duo = []
trio = []
band = []

top_year = list(set(data["top year"]))
top_year.sort()

for i in top_year:
  type = data[data["top year"] == i]["artist type"]
  type_dicty = dict(collections.Counter(type))

  if("Trio" not in type_dicty.keys()):
    trio.append(0)

  for i in type_dicty.keys():
    if i == 'Solo':
      solo.append(type_dicty[i])
    if i == 'Duo':
      duo.append(type_dicty[i])
    if i == 'Trio':
      trio.append(type_dicty[i])
    if i == 'Band/Group':
      band.append(type_dicty[i])

barWidth = 0.25
fig, ax = plt.subplots(figsize=(18,15))


barS = np.arange(len(solo))
barD = [x + barWidth for x in barS]
barT = [x + barWidth for x in barD]
barB = [x + barWidth for x in barT]

plt.bar(barS, solo, color="#00FFFF", width=barWidth, label="Solo")
plt.bar(barD, duo, color="#A52A2A", width=barWidth, label="Duo")
plt.bar(barT, trio, color="#CAE00D", width=barWidth, label="Trio")
plt.bar(barB, band, color="#1B4D3E", width=barWidth, label="Band")

plt.xlabel("Years", fontweight="bold", fontsize=20)
plt.ylabel("Amount of Artist Types", fontweight="bold", fontsize=20)
plt.xticks([y + barWidth for y in range(len(solo))], ["2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019"])
ax.tick_params(axis="both", which="major", labelsize=10)

x1, x2, y1, y2 = plt.axis()
plt.axis((x1,x2,y1,y2))

plt.legend(prop={"size": 20})
plt.show()