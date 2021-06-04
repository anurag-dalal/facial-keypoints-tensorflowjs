# Facial Keypoints Detection using Tensorflowjs

This repository contain a Facial landmark detection, with inference in browser employing tensorflowjs.
The steps are explained in details

## Prerequisites
* 1\. Python
    * 1.1\. tensorflow 2.4.0
    * 1.2\. opencv
    * 1.3\. tensorflowjs_converter
    * 1.4\. anaconda or venv
* 2\. Node.js
* 3\. HTML
* 4\. Javascript, jQuery

Preferably a system with a GPU and cuda for faster training.

## Training Pipeline

This folder [facial-landmark-training](https://github.com/anurag-dalal/facial-keypoints-tensorflowjs/tree/main/facial-landmark-training) contain training pipeline for training images for facial landmark detection.
The dataset is used from kaggle, which can be found in the [Link](https://www.kaggle.com/c/facial-keypoints-detection/data)
This dataset contain 7049 training images of 96x96 resolution.
There are 15 facial landmarks that are annotated.

To train the images, you can run train.py. There are various augmentation methods used to augment the images.
The model used is MobileNet with a few layers added to the end of the model.
To change the dataset folder change this lines(line no: 25-27) in train.py:

```python
train_file = 'G:/Python codes/MLs/Dataset/facial-keypoints-detection/training.csv'
test_file = 'G:/Python codes/MLs/Dataset/facial-keypoints-detection/test.csv'
idlookup_file = 'G:/Python codes/MLs/Dataset/facial-keypoints-detection/IdLookupTable.csv'
```
The model reached a 92% accuracy on test data and 87% accuracy on validation data.
Note that this dataset from kaggle is for frontal view only, so to test we will use cropped images of faces present in [testimages](https://github.com/anurag-dalal/facial-keypoints-tensorflowjs/tree/main/testimages) folder.

To detect we can use detect.py. Here is a image produced by detect.py \
![Detected Image](/images/detected.png "detected image")

Once the model is trained, it can be converted to json format to be used for browser inference.
train.py will write a file called detectv1.h5, which is our saved model. Now in terminal or command prompt navigate to the folder containg the h5 file and run this command
```bash
mkdir tfjs
tensorflowjs_converter --input_format keras detectv1.h5 tfjs/
```
This will create a directory called tfjs and put the files of the converted model in the directory.

## Running the express app
To run the express app you need node.js installed.
Go inside the local-server directory from terminal, and run the following:
```bash
node .\server.js
```
This will start the server, now from the browser navigate to http://localhost:81/predict-with-tfjs.html, you will find this page:\
