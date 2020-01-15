import pandas as pd
from numpy import nan
from collections import Counter
import os.path

# Read the csv
data = pd.read_csv("Data/MovieGenre.csv", encoding="latin-1")

# Fix the imbd link column as most do not work
data = data.drop("Imdb Link", 1)
data = data.assign(Link=["https://www.imdb.com/title/tt%07i"%i for i in data.imdbId])

# Remove movies that meet one of the following criteria:
# - Has no poster link
# - Has no genres at all
# - Is a duplicate of another movie
data = data[data.Poster.notna()]
data = data[data.Genre.notna()]
data = data[~data.duplicated("imdbId", "first")]

# Parse all genres and count each genre
data.Genre = [x.split("|") for x in data.Genre]
genres = Counter(y for x in data.Genre for y in x)

# Remove all genres that do not occur enough and remove movies that do not have any genre left
genres = [x for x in genres if genres[x] < 100]
data.Genre = [[x for x in r if x not in genres] for r in data.Genre]
data = data[data.Genre.apply(lambda x: len(x) > 0)]

# Remove year from the title and remove year from title if it is a correct year
data = data.assign(Year=[t[-5:-1] for t in data.Title])
data.loc[~data.Year.apply(lambda x: x.isdigit() and len(x) == 4), "Year"] = nan
data.loc[data.Year.notna(), "Title"] = data.loc[data.Year.notna(), "Title"].apply(lambda x: x[:-7])

# Remove all moves that do not have a year
#data = data[data.Year.notna()]

if os.path.isfile("Data/downloaderror.csv"):
    posters = list(pd.read_csv("Data/downloaderror.csv", header=None)[0])
    data = data[-data.imdbId.isin(posters)]

# Write the data to a new csv with genres concatinated as in the origional dataset
data.Genre = ['|'.join(i) for i in data.Genre]
data.to_csv("Data/cleaned.csv", index=False)
