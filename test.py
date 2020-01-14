#https://www.kaggle.com/joshmiller656/classifying-movies-from-raw-image-using-convnets inspiration from this 
#https://github.com/benckx/dnn-movie-posters/blob/master/ interesting repo

import numpy as np # pip install numpy
import pandas as pd # pip install pandas

from subprocess import check_output

#Adjust the path to the posters here:
path = 'Data/SampleMoviePosters/SampleMoviePosters'
import glob #pip install glob
import scipy.misc #pip install ..
import imageio #pip install imageio
from PIL import Image #pip install Pillow

image_glob = glob.glob(path+"/"+"*.jpg")
img_dict = {}
def get_id(filename):
    index_s = max(filename.rfind("\\")+1, filename.rfind("/")+1)
    index_f = filename.rfind(".jpg")
    return filename[index_s:index_f]
#Populate image dicts
_ = [img_dict.update({get_id(fn):imageio.imread(fn, pilmode="RGB", as_gray=False)}) for fn in image_glob]

#Reads the movie genres
df = pd.read_csv("Data/MovieGenre.csv",encoding="ISO-8859-1")
genres = []
for n in range(len(df)):
    string = str(df.loc[n]["Genre"])
    gs = string.split("|")
    genres += gs
    
classes = list(set(genres))
classes.sort()
num_classes = len(classes)

def get_classes_from_movie(movie_id):
    y = np.zeros(num_classes)
    gs = str(df[df['imdbId']==movie_id]['Genre'].values[0])
    for g in gs.split("|"):
        y[classes.index(g)] = 1
    return y  

import random

"""
Some relatively simple image preprocessing
"""
def preprocess(img,size=32):
    img = np.array(Image.fromarray(img).resize((size,size)))
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
x,y,x_test,y_test = get_dataset(30,img_size=SIZE)
x = np.asarray(x)
y = np.asarray(y)
x_test = np.asarray(x_test)
y_test = np.asarray(y_test)

import keras
import vgg16
import resnet

model = vgg16.vggmodel(num_classes, SIZE)
#model = resnet.resnet50(num_classes, SIZE)

model.fit(x, y,
          batch_size=50,
          epochs=5,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

pred = model.predict(np.asarray([x_test[5]]))

print("pred")
print(pred) #predictions of all classes
print("np.argmax pred")
print(np.argmax(pred)) #Max class predicted
print("y_test[5]:")
print(np.argmax(y_test[5])) #The real class