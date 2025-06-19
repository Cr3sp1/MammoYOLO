# Lesion detection and classification in mammograms with YOLOv1 and TinyYOLOv1 models using Keras

In this project we build YOLOv1 and TinyYOLOv1 models using the Keras api with the objective of detecting and classifying lesions in mammograms.

## Instructions

### Step 1 - Set the environment
Install the dependencies (preferably in a fresh vistual environment) by running:
`pip install -r requirements.txt`

### Step 2 - Get the data
Download the raw data in the data/ directory by running Manifest.tcia with [NBIA Data Retriever](https://wiki.cancerimagingarchive.net/display/NBIA/Downloading+TCIA+Images). Important: select the option "Classic Directory Names". From this directory then run the command:
`python format_data.py`

### Step 3 - Train the model
To train the models run the command:
`python training.py`

### Step 4 - Evaluate performance
To to evaluate the models performance via ROC and FROC curve run the command:
`python training.py`

### Step 5 - Visualize the results
The results of the training and model evaluation can be visualized in the Jupyter Notebook 'Results.ipynb'.
