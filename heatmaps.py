import vgg16
import resnet

GENRE_VECTOR_LENGTH = 22
SIZE = (128,128) 

def loadimages():
    #Adjust the path to the posters here:
    path = 'Data/SampleMoviePosters/SampleMoviePosters/'
    import glob #pip install glob
    import scipy.misc #pip install ..
    import imageio #pip install imageio
    from PIL import Image #pip install Pillow

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
        print("heatmaps not implemented for YOLO models.")
    
img_dict = loadimages()
model = buildmodel(0)
from keras.models import load_model
model.load_weights("vgg16-70t-20e.h5")
#model.load_weights("resnet50-70t-20e.h5")

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
    heatmap = np.abs(heatmap)
    heatmap /= np.max(heatmap)
    print(heatmap)

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