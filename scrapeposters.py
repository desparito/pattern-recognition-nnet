import pandas as pd
import urllib.request
import os.path

df = pd.read_csv("Data/MovieGenre.csv", encoding="latin-1")
for index, row in df.tail(4000).iterrows():
    try:
        if not (str(row[5]) == "nan" or os.path.isfile("Data/Posters/" + str(row[0]) + ".jpg")):
            urllib.request.urlretrieve(row[5], "Data/Posters/" + str(row[0]) + ".jpg")
    except urllib.error.HTTPError:
        pass #print(row[0])

