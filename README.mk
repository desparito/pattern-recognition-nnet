# Running the classification
To run the full algoritm from scratch a few steps are required. These steps are listed below.

## Step 1: Setting up python
A few dependencies are requirement to run the code. These can be installed in an virtual environment with:
```bash
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## Step 2: Creating the dataset
The origional dataset needs to be cleaned and all posters need to be downloaded. Firstly the data needs to be cleanedand then downloaded.
```bash
python clean_data.py
python scrapeposters.py
```

## Step 3: Creating YOLO data
If you want to use YOLO classification to increase accuracy the yolo data needs to be created beforehand. This requires the weight dataset as well, which can be downloaded with the script below. With these weights a model will be created which in turn can be used to do the actual classification.
```bash
wget https://pjreddie.com/media/files/yolov3.weights
python yolomodel.py
python yolo.py
```

## Step 4: Running the neural network
When all files are collected and created the network can be used.
```bash
python test.py
```
