import pandas as pd
from numpy import nan
from collections import Counter

data = pd.read_csv("Data/MovieGenre.csv", encoding="latin-1", )
data = data.drop("Imdb Link", 1)
data = data[data.Poster.notna()]
data = data[data.Genre.notna()]
data = data[~data.duplicated("imdbId", "first")]
data.Genre = [x.split("|") for x in data.Genre]

genres = Counter(y for x in data.Genre for y in x)
genres = [x for x in genres if genres[x] < 100]

data.Genre = [[x for x in r if x not in genres] for r in data.Genre]
data = data[data.Genre.apply(lambda x: len(x) > 0)]

data = data.assign(
    Year=[t[-5:-1] for t in data.Title], 
    Link=["https://www.imdb.com/title/tt%07i"%i for i in data.imdbId]
    )
data.loc[~data.Year.apply(lambda x: x.isdigit() and len(x) == 4), "Year"] = nan
#data = data[data.Year.notna()]
data.loc[data.Year.notna(), "Title"] = data.loc[data.Year.notna(), "Title"].apply(lambda x: x[:-7])

data.Genre = ['|'.join(i) for i in data.Genre]
data.to_csv("data.csv", index=False)
