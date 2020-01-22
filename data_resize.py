import numpy as np
import os.path
import glob
import imageio
from PIL import Image
from joblib import Parallel, delayed
import multiprocessing

image_glob = glob.glob("Data/Posters/*.jpg")
size = (128,128)

if not os.path.isdir("Data/Arrays"):
    os.makedirs("Data/Arrays")

def get_id(filename):
    index_s = max(filename.rfind("\\")+1, filename.rfind("/")+1)
    index_f = filename.rfind(".jpg")
    return "Data/Arrays/" + filename[index_s:index_f]

def preprocess(filename):
    img = imageio.imread(filename, pilmode="RGB", as_gray=False)
    img = np.array(Image.fromarray(img).resize(size))
    img = img.astype(np.float32)
    img = (img / 127.5) - 1.
    np.save(get_id(filename), img)

num_cores = multiprocessing.cpu_count()
Parallel(n_jobs=num_cores)(delayed(preprocess)(i) for i in image_glob)
