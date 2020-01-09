
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output

#Needs to be changes to the correct path when the posters are there
path = 'Data/Posters'
import glob
import scipy.misc
import imageio

image_glob = glob.glob(path+"/"+"*.jpg")
img_dict = {}
def get_id(filename):
    index_s = max(filename.rfind("\\")+1, filename.rfind("/")+1)
    index_f = filename.rfind(".jpg")
    print(filename[index_s:index_f])
    return filename[index_s:index_f]
#Populate image dict
_ = [img_dict.update({get_id(fn):imageio.imread(fn)}) for fn in image_glob]
print(img_dict.keys())

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

import random

"""
Some relatively simple image preprocessing
"""
def preprocess(img,size=32):
    img = scipy.misc.imresize(img,(size,size))
    img = img.astype(np.float32)
    img = (img / 127.5) - 1.
    return img
    

def get_dataset(train_size,img_size=32):
        #indices = np.random.randint(0,len(list(img_dict.keys()))-1,batch_size)
        indices = random.sample(range(len(list(img_dict.keys()))),train_size)
        x = []
        y = []
        x_test = []
        y_test = []
        for i in range(len(list(img_dict.keys()))):
            id_key = int(list(img_dict.keys())[i])
            if i in indices:
                x.append(preprocess(img_dict[list(img_dict.keys())[i]],size=img_size))
                y.append(get_classes_from_movie(id_key))
            else:
                x_test.append(preprocess(img_dict[list(img_dict.keys())[i]],size=img_size))
                y_test.append(get_classes_from_movie(id_key))
        return x,y,x_test,y_test
        
#Constant to keep track of our image size
SIZE = 128   
x,y,x_test,y_test = get_dataset(900,img_size=SIZE)
x = np.asarray(x)
y = np.asarray(y)
x_test = np.asarray(x_test)
y_test = np.asarray(y_test)

print(x)