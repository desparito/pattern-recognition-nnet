#https://www.kaggle.com/joshmiller656/classifying-movies-from-raw-image-using-convnets inspiration from this 
#https://github.com/benckx/dnn-movie-posters/blob/master/ interesting repo

import keras
from keras.callbacks import TensorBoard
from time import time
import vgg16
import resnet
import fcnet
import numpy as np # pip install numpy
import pandas as pd # pip install pandas
import random
import os
from subprocess import check_output

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#Adjust the path to the posters here:
path = 'Data/Posters/'
import glob #pip install glob
import scipy.misc #pip install ..
import imageio #pip install imageio
from PIL import Image #pip install Pillow

#Boolean to choose if we want to use YOLO:
USE_YOLO = True

print("Reading data")

def get_id(filename):
    index_s = max(filename.rfind("\\")+1, filename.rfind("/")+1)
    index_f = filename.rfind(".jpg")
    return int(filename[index_s:index_f])

# Populate image dicts
img_dict = {get_id(fn):fn for fn in glob.glob(path+"*.jpg")}

# Load yolo data
if(USE_YOLO):
    yolo_df = pd.read_csv("Data/yolo.csv", index_col=0, encoding="utf-8-sig")
    yolo_df = yolo_df.fillna(0)

#Reads the movie genres
df = pd.read_csv("Data/cleaned.csv",index_col="imdbId")
df.Genre = [x.split("|") for x in df.Genre]

# Remove posters that do not occur in the csv and remove movies that have no poster
for id_key in list(img_dict):
    if id_key not in df.index:
        del img_dict[id_key] 
    if USE_YOLO and id_key not in yolo_df.index:
        del img_dict[id_key]

df = df.loc[list(img_dict)]
if USE_YOLO:
    yolo_df = yolo_df.loc[list(img_dict)]

# Process genres
genres = sorted(set(y for x in df.Genre for y in x))
classes = pd.DataFrame(data={g:[g in r for r in df.Genre] for g in genres}, index=df.index)


print("Processing data")

"""
Some relatively simple image preprocessing
"""
def preprocess(img,size=(32,32)):
    img = imageio.imread(img, pilmode="RGB", as_gray=False)
    img = np.array(Image.fromarray(img).resize(size))
    img = img.astype(np.float32)
    img = (img / 127.5) - 1.
    return img

#Constant to keep track of our image size
IMG_SIZE   = (128, 128)
TRAIN_SIZE = round(len(img_dict)*0.2)

# Split and process the dataset
indices = random.sample(list(img_dict),TRAIN_SIZE)
x_img = []
x_img_test = []
y = []
y_test = []
x_yolo = []
x_yolo_test = []
for id_key in list(img_dict):
    if id_key in indices:
        x_img.append(preprocess(img_dict[id_key],size=IMG_SIZE))
        y.append(classes.loc[id_key])
        if USE_YOLO:
            x_yolo.append([yolo_df.loc[id_key]])
    else:
        x_img_test.append(preprocess(img_dict[id_key],size=IMG_SIZE))
        y_test.append(classes.loc[id_key])
        if USE_YOLO:
            x_yolo_test.append([yolo_df.loc[id_key]])

x_img       = np.asarray(x_img)
x_img_test  = np.asarray(x_img_test)
y           = np.asarray(y)
y_test      = np.asarray(y_test)
x_yolo      = np.asarray(x_yolo)
x_yolo_test = np.asarray(x_yolo_test)

print("Running Model")

tensorboard = TensorBoard(log_dir="logs/{}".format(time())) #initialise Tensorboard

# mode 0, 1, 2, 3
# translates to: vgg16, resnet50, vgg16-obj, resnet50-obj
def runmode(mode = 0, epochs = 5, batchsize = 50):
    modestr = ""
    
    if (mode < 2):
        if (mode == 0):
            modestr = "vgg16"
            model = vgg16.vggmodel(len(genres), IMG_SIZE)
        else:
            modestr = "resnet50"
            model = resnet.resnet50(len(genres), IMG_SIZE)
        
        print("Fitting " + modestr + ":")
        model.fit(x_img, y, batch_size=batchsize, epochs=epochs, validation_data=(x_img_test, y_test),callbacks=[tensorboard])
        score = model.evaluate(x_img_test, y_test)
    else: 
        if (mode == 2):
            modestr = "vgg16-objdet"
            img_model = vgg16.vggmodel(len(genres), IMG_SIZE, False)
        else:
            modestr = "resnet50-objdet"
            img_model = resnet.resnet50(len(genres), IMG_SIZE, False)

        model = fcnet.fcnmodel(len(genres), len(x_yolo[0][0]), img_model, modestr)
        print("Fitting " + modestr + ":")
        model.fit([x_yolo,x_img], y, batch_size=batchsize, epochs=epochs, validation_data=([x_yolo_test, x_img_test], y_test),callbacks=[tensorboard])
        score = model.evaluate([x_yolo_test, x_img_test], y_test)
    
    # print metrics
    print("Model metrics for " + modestr + ":")
    for i in range(len(model.metrics_names)):
        print(model.metrics_names[i]+':', score[i])

    # save model
    model.save_weights(modestr + ".h5")
    print("Saved model " + modestr + "to disk!")

def runmodeall(epochs = 5, batchsize = 50):
    #runmode(0, epochs, batchsize)
    #runmode(1, epochs, batchsize)
    #runmode(2, epochs, batchsize)
    runmode(3, epochs, batchsize)

runmodeall(20, 200)
