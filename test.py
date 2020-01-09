
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output

#Needs to be changes to the correct path when the posters are there
path = 'Data/SampleMoviePosters/SampleMoviePosters'
import glob
import scipy.misc
from PIL import Image
image_glob = glob.glob(path+"/"+"*.jpg")
img_dict = {}
def get_id(filename):
    index_s = filename.rfind("/")+1
    index_f = filename.rfind(".jpg")
    return filename[index_s:index_f]
#Populate image dict
_ = [img_dict.update({get_id(fn):scipy.misc.imread(fn)}) for fn in image_glob]

#Reads the movie genres
df = pd.read_csv("Data/MovieGenre.csv",encoding="ISO-8859-1")
genres = []
length = len(df)
for n in range(len(df)):
    g = str(df.loc[n]["Genre"])
    genres.append(g)
    
classes = list(set(genres))
classes.sort()
num_classes = len(classes)

def get_classes_from_movie(movie_id):
    y = np.zeros(num_classes)
    g = str(df[df['imdbId']==movie_id]['Genre'].values[0])
    y[classes.index(g)] = 1
    return y  