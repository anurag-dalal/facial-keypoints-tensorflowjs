from flask import Flask, render_template, Response, request, jsonify
from werkzeug.utils import secure_filename
import numpy as np
import cv2
import os
import tensorflow as tf
import matplotlib.pyplot as plt

print('[INFO] Loading model...')
model = tf.keras.models.load_model('detectv1.h5', custom_objects=None, compile=True, options=None)
print('[INFO] Loading model... Done')

def plot_sample(image, keypoint, axis, title):
    image = image.reshape(96,96)
    axis.imshow(image, cmap='gray')
    axis.scatter(keypoint[0::2], keypoint[1::2], marker='x', s=20)
    plt.title(title)
    
img = cv2.imread('G:/Python codes/MLs/ModakaTech_node/facial-keypoints-tensorflowjs/testimages/test2.JPG')
HEIGHT = img.shape[0]
WIDTH = img.shape[1]
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.resize(img, (96,96), interpolation=cv2.INTER_AREA)
img_c = img.copy()
img = img/255.0
img = img[..., np.newaxis]
img = img[np.newaxis, ...]
img = img.astype('float32')
pred = model(img)
pred = pred.numpy()
pred = pred[0]
pred = pred.astype('int32')
for i in range(15):
    img_c = cv2.circle(img_c, (pred[i*2], pred[i*2+1]), 1, (0, 255, 0), 1)

#fig, axis = plt.subplots()
#plot_sample(img_c, pred[0], axis, "Sample image & keypoints")

plt.imshow(img_c, cmap='gray')
