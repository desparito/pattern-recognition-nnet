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

#Boolean to choose if we want to use YOLO:
USE_YOLO = True

print("Reading data")

image_glob = glob.glob(path+"/"+"*.jpg")

def get_id(filename):
    index_s = max(filename.rfind("\\")+1, filename.rfind("/")+1)
    index_f = filename.rfind(".jpg")
    return int(filename[index_s:index_f])

#Populate image dicts
img_dict = {get_id(fn):imageio.imread(fn, pilmode="RGB", as_gray=False) for fn in image_glob}

#Reads the movie genres
df = pd.read_csv("Data/cleaned.csv",index_col="imdbId")
if(USE_YOLO):
    yolo_df = pd.read_csv("Data/yolo.csv", index_col=0, encoding="utf-8-sig")
    yolo_df = yolo_df.fillna(0)

df = df.loc[(df['Year'] <= 1975)] #You can change this so remove old movies for now it is turned of because of the sample posters

print(df)

# Remove posters that do not occur in the csv and remove movies that have no poster
for id_key in list(img_dict):
    if id_key not in df.index:
        del img_dict[id_key] 
df = df.loc[list(img_dict)]

print(df)

df.Genre = [x.split("|") for x in df.Genre]
genres = sorted(set(y for x in df.Genre for y in x))

classes = pd.DataFrame(data={g:[g in r for r in df.Genre] for g in genres}, index=df.index)

import random

print("Processing data")

"""
Some relatively simple image preprocessing
"""
def preprocess(img,size=(32,32)):
    img = np.array(Image.fromarray(img).resize(size))
    img = img.astype(np.float32)
    img = (img / 127.5) - 1.
    return img

def get_dataset(train_size,img_size=(32,32)):
    #indices = np.random.randint(0,len(list(img_dict.keys()))-1,batch_size)
    images = list(img_dict)
    indices = random.sample(images,train_size)
    x = []
    y = []
    x_test = []
    y_test = []
    for id_key in images:
        if id_key in indices:
            x.append(preprocess(img_dict[id_key],size=img_size))
            y.append(classes.loc[id_key])
        else:
            x_test.append(preprocess(img_dict[id_key],size=img_size))
            y_test.append(classes.loc[id_key])
    return x,y,x_test,y_test

def get_dataset_yolo(train_size, img_size=(32,32)):
    images = list(img_dict)
    indices = random.sample(images,train_size)
    x_img = []
    x_yolo = []
    y = []
    x_img_test = []
    x_yolo_test = []
    y_test = []
    for id_key in images:
        if id_key in indices:
            x_img.append(preprocess(img_dict[id_key],size=img_size))
            x_yolo.append([yolo_df.loc[id_key]])
            y.append(classes.loc[id_key])
        else:
            x_img_test.append(preprocess(img_dict[id_key],size=img_size))
            x_yolo_test.append([yolo_df.loc[id_key]])
            y_test.append(classes.loc[id_key])
    return x_img,x_yolo,y,x_img_test,x_yolo_test,y_test

#Constant to keep track of our image size
SIZE = (128, 128)
if USE_YOLO:
    x_img,x_yolo,y,x_img_test,x_yolo_test,y_test = get_dataset_yolo(300,img_size=SIZE)
    x_img = np.asarray(x_img)
    x_yolo = np.asarray(x_yolo)
    x_img_test = np.asarray(x_img_test)
    x_yolo_test = np.asarray(x_yolo_test)
else:
    x,y,x_test,y_test = get_dataset(300,img_size=SIZE)
    x = np.asarray(x)
    x_test = np.asarray(x_test)

y = np.asarray(y)
y_test = np.asarray(y_test)

print("Using model")

import keras
import vgg16
import resnet
import fcnet

if USE_YOLO:
    img_model = vgg16.vggmodel(len(genres), SIZE, False)
    #img_model = resnet.resnet50(len(genres), SIZE, False) 
    model = fcnet.fcnmodel(len(genres), len(x_yolo[0][0]), img_model)

    model.fit([x_yolo,x_img], y,
          batch_size=50,
          epochs=5,
          validation_data=([x_yolo_test, x_img_test], y_test))
    score = model.evaluate([x_yolo_test, x_img_test], y_test)
else:
    model = vgg16.vggmodel(len(genres), SIZE)
    #model = resnet.resnet50(len(genres), SIZE)

    model.fit(x, y,
          batch_size=50,
          epochs=5,
          validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test)

#Print metrics:
for i in range(len(model.metrics_names)):
    print(model.metrics_names[i]+':', score[i])

#INSTEAD OF FITTING NEW MODEL YOU CAN LOAD A MODEL THIS WAY
#loadedmodel = vgg16.vggmodel(len(genres), SIZE)
#loadedmodel.load_weights("model.h5")
#pred = loadedmodel.predict(np.asarray([x_test[5]]))

#SAVE THE MODEL FOR FURTHER USE
#model.save_weights("model.h5")
#print("Saved model to disk")