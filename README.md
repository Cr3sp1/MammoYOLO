This wip project aims to build a neural network implementing the YOLO approach to object detection to detect and classify lesions in mammograms.

## Data Instructions

### Step 1
Download the raw data in the data/ directory by running Manifest.tcia with [NBIA Data Retriever](https://wiki.cancerimagingarchive.net/display/NBIA/Downloading+TCIA+Images). Important: select the option "Classic Directory Names".

### Step 2
Install the dependencies (preferably in a fresh vistual environment) by running:
`pip install -r requirements.txt`

### Step 3
From this directory run the command:
`python format_data.py`
