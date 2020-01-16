import pandas as pd
import urllib.request
import os.path
from joblib import Parallel, delayed
import multiprocessing
     
df = pd.read_csv("Data/cleaned.csv", usecols=["imdbId", "Poster"])

if not os.path.isdir("Data/Posters"):
    os.makedirs("Data/Posters")

open("Data/downloaderror.csv", 'w').close()

def downloadPoster(row):
    name = "Data/Posters/" + str(row.imdbId) + ".jpg"
    if not os.path.isfile(name):
        try:
            urllib.request.urlretrieve(row.Poster, name)
        except urllib.error.HTTPError:
            errors = open("Data/downloaderror.csv", 'a')
            errors.write(str(row.imdbId)+"\n")
            errors.close()
 
num_cores = multiprocessing.cpu_count()
Parallel(n_jobs=num_cores)(delayed(downloadPoster)(i) for i in df.itertuples(index=False))
