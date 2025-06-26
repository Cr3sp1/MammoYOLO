# Lesion Detection and Classification in Mammograms with YOLOv1 and TinyYOLOv1 Models Using Keras

In this project we build YOLOv1 and TinyYOLOv1 models as presented in the [original article](refs/Yolo.pdf) using the Keras API with the objective of detecting and classifying lesions in mammograms. We perform the training on the [DDSM dataset](refs/CBIS-DDSM.pdf), and compare our results with those obtained by [Ribli et al.](refs/CBIS-DDSM.pdf) using a Faster R-CNN.

## Training
Since the YOLO models take as input 448x448 images, we compress all the mammograms from the DDSM dataset to this resolution. We perform the training using Adam as an optimizer, with a learning rate of 0.0001 for the TinyYOLOv1 model and 0.00001 for the YOLOv1 model. Using a higher learning rate on YOLOv1 led to a lower loss function, but also to a fixed output independent of the input. To avoid overfitting we also perform early stopping, so we interrupt the training if the validation loss doesn't improve for 10 epochs straight.


## The YOLO model
The YOLOv1 (You Only Look Once) model is a convolutional neural network that performs object detection by dividing the input image into an *S×S* grid and predicting bounding boxes and class probabilities directly for each grid cell. Unlike traditional methods that apply region proposals or sliding windows, YOLOv1 formulates detection as a single regression problem, making it extremely fast.

The architecture consists of 24 convolutional layers followed by 2 fully connected layers. The convolutional layers extract spatial features from the image, while the fully connected layers output predictions. For each grid cell, the network predicts a fixed number of bounding boxes *B* (in our case 2), along with confidence scores and class probabilities. Each prediction includes:
- The coordinates of the *B* bounding boxes (x,y,w,h).
- *B* confidence scores (reflecting both the probability of object presence and the IoU with the ground truth).
- Class probabilities for each possible object category (e.g., malignant, benign). These describe the conditional valss probabilities given that an object is present in the cell.
The only difference in the TinyYOLOv1 variant is the much reduced number of convolutional layers used (8 instead of the original 24). Multiplying the confidence score of a box with the cell class probabilities gives the complexive class scores that represent the probability of finding an object of that class in the cell.

We use the YOLO architecture to detect lesions and classify them as either "malignant" (that indicate the presence of cancer) or 'benign" (that do not indicate the presence of cancer).

## Results
We evaluate the performance of our two models on the test set of the DDSM dataset, which was not used during training. We focus in particular on the ability of our models to find malignant lesions.
We consider that our models predicts the presence of cancer in a mammography with a certain threshold if there is at least one box with a malignant score above the threshold. This way we can classify whole mammograms as either indicating the presence of cancer or not.
We then measure the models' ability to correctly perform this classification task with the receiver operating characteristic (ROC) curve:
![roc](https://github.com/user-attachments/assets/fc4499c5-00cf-4e30-8132-a25ff36b74fe)
We see that the TinyYOLOv1 model somewhat works, while the full YOLO performs worse. This could be due to insufficient training. As a comparison, the Faster R-CNN model achieved an AUC of 0.95.
We also measure our models' ability to accurately localize the malignant lesions via the Free ROC (FROC) curve. On the y-axis is the percentage of malignant lesions detected within a certain threshold (ie: there exists at least one box with malignant score aboove the threshold whose center falls within a ground truth box), while on the x-axis is the number of false positives (ie: the number of boxes with malignant score above the threshold that do not fall within a ground truth box).
![froc](https://github.com/user-attachments/assets/1ad97da5-1a28-4681-9671-a58ea552a982)
We note that at 0.3 false positives per image our models achieve a sensitivity (true positive rate) of around 0.08, to compare with the sensitivity of 0.9 achieved by the Faster R-CNN.

## Example predictions
![pred1](https://github.com/user-attachments/assets/3c299e60-8bb7-44bf-854b-dd0bc849b6d7)
![pred2](https://github.com/user-attachments/assets/966e14b0-f062-4a31-a58f-f1f2c1439cb3)
![pred3](https://github.com/user-attachments/assets/e0e2d4ba-8e18-4f4e-93f1-55f29c02778a)
![pred4](https://github.com/user-attachments/assets/c94ec5ff-43fe-4e2c-befc-8a6ab9a1260d)

## Instructions for running the code

### Step 1 - Set up the environment
Install the dependencies (preferably in a fresh virtual environment) by running:
`pip install -r requirements.txt`

### Step 2 - Get the data
Download the raw data in the data/ directory by running Manifest.tcia with [NBIA Data Retriever](https://wiki.cancerimagingarchive.net/display/NBIA/Downloading+TCIA+Images). Important: select the option "Classic Directory Names". From this directory then run the command:
`python format_data.py`

### Step 3 - Train the model
To train the models run the command:
`python training.py`

### Step 4 - Evaluate performance
To to evaluate the models' performance via ROC and FROC curve run the command:
`python performance_eval.py`

### Step 5 - Visualize the results
The results of the training and model evaluation can be visualized in the [Jupyter Notebook](Results.ipynb), in which one can also perform prediction on specific images.
