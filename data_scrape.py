import pandas as pd
import urllib.request
import os.path

# Read the to-scrape posters
df = pd.read_csv("Data/cleaned.csv", usecols=["imdbId", "Poster"])

# Create a poster directory if it does not exist yet
if not os.path.isdir("Data/Posters"):
    os.makedirs("Data/Posters")

# Define a function for downloading a single poster, ignoring it if it cannot be downloaded
def downloadPoster(row):
    name = "Data/Posters/" + str(row.imdbId) + ".jpg"
    if not os.path.isfile(name):
        try:
            urllib.request.urlretrieve(row.Poster, name)
        except urllib.error.HTTPError:
            pass
 
# Run the download function in parallel for each poster.
from joblib import Parallel, delayed
import multiprocessing
num_cores = multiprocessing.cpu_count()
Parallel(n_jobs=num_cores)(delayed(downloadPoster)(i) for i in df.itertuples(index=False))
