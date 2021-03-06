# Running the classification
To run the full algoritm from scratch a few steps are required. These steps are listed below.

## Step 1: Setting up python
A few dependencies are requirement to run the code. These can be installed in an virtual environment with:

```
$ python -m venv env
$ source env/bin/activate
$ pip install -r requirements.txt
```

## Step 2: Creating the dataset
The origional dataset needs to be cleaned and all posters need to be downloaded. Firstly the data needs to be cleanedand then downloaded. After downloading each image will be changed to an numpy array.

```
$ python data_clean.py
$ python data_scrape.py
$ python data_resize.py
```

## Step 3: Creating YOLO data
If you want to use YOLO classification to increase accuracy the yolo data needs to be created beforehand. This requires the weight dataset as well, which can be downloaded with the script below. With these weights a model will be created which in turn can be used to do the actual classification.

```
$ wget https://pjreddie.com/media/files/yolov3.weights
$ python yolo_createmodel.py
$ python yolo_classify.py
```

## Step 4: Running the neural network
When all files are collected and created the network can be used.

```
$ python test.py
```
