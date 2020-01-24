import vgg16
import resnet
import pandas as pd
import imageio
import numpy as np
from PIL import Image

GENRE_VECTOR_LENGTH = 22
SIZE = (128,128) 

#Adjust the path to the posters here:
path = 'Data/SampleMoviePosters/SampleMoviePosters/'

def preprocess(img,size=(32,32)):
    img = imageio.imread(img, pilmode="RGB", as_gray=False)
    img = np.array(Image.fromarray(img).resize(size))
    img = img.astype(np.float32)
    img = (img / 127.5) - 1.
    return img

def loadimages():
    import glob #pip install glob
    import scipy.misc #pip install ..
    import imageio #pip install imageio

    print("Reading data")

    image_glob = glob.glob(path+"*.jpg")

    def get_id(filename):
        index_s = max(filename.rfind("\\")+1, filename.rfind("/")+1)
        index_f = filename.rfind(".jpg")
        return int(filename[index_s:index_f])

    # Populate image dicts
    img_dict = {get_id(fn):fn for fn in image_glob}

    #Reads the movie genres
    df = pd.read_csv("Data/cleaned.csv",index_col="imdbId")
    #df = df.loc[(df['Year'] >= 2012)] #You can change this so remove old movies for now it is turned of because of the sample posters
    df.Genre = [x.split("|") for x in df.Genre]

    # Remove posters that do not occur in the csv and remove movies that have no poster
    for id_key in list(img_dict):
        if id_key not in df.index:
            del img_dict[id_key] 

    return img_dict

def buildmodel(mode = 0):
    modestr = ""
    
    if (mode < 2):
        if (mode == 0):
            modestr = "vgg16"
            model = vgg16.vggmodel(GENRE_VECTOR_LENGTH, SIZE)
        else:
            modestr = "resnet50"
            model = resnet.resnet50(GENRE_VECTOR_LENGTH, SIZE)
        
        return model
    else: 
        print("Heatmaps not implemented for YOLO models.")
    
img_dict = loadimages()
model = buildmodel(0)
from keras.models import load_model
model.load_weights("vgg16-70t-20e.h5")

#Visualise:
#https://github.com/nickbiso/Keras-Class-Activation-Map/blob/master/Class%20Activation%20Map(CAM).ipynb

visualise_keys = [4313614, 126029, 4048668, 5227516] #Add keys here to visualise
vis = []
for key in visualise_keys:
    vis.append(preprocess(img_dict[key],size=SIZE))
preds = model.predict(np.asarray(vis))

index = 0
print(preds)
for key in visualise_keys:
    argmax = np.argmax(preds[index])
    output = model.output[:, argmax]

    last_conv_layer = model.get_layer('last_conv') #for vgg16
    #last_conv_layer = model.get_layer('conv2d_53') #for resnet
    import keras.backend as K
    grads = K.gradients(output, last_conv_layer.output)[0]
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate(np.asarray([vis[index]]))

    for i in range(512):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
    heatmap = np.mean(conv_layer_output_value, axis=-1)
    print(heatmap)
    heatmap += np.abs(np.min(heatmap))#np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    import cv2 #pip install opencv-python
    img = cv2.imread(path + '/' + str(key) + '.jpg')
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    hif = .8
    superimposed_img = heatmap * hif + img
    output = './Heatmaps/' + str(key) + '.jpeg'
    cv2.imwrite(output, superimposed_img)
    img=imageio.imread(output)
    print('Wrote heatmap for label ' + str(argmax) + ' for poster with key ' + str(key))
    index += 1