MODEL_LOC = '../model/pneumonia_detection_cnn_model.h5'
DATA_DIR = '../data1/'
TRAINING_DATA_DIR = DATA_DIR + '/train/'
TEST_DATA_DIR = DATA_DIR + '/test/'
VAL_DATA_DIR = DATA_DIR + '/val/'
DETECTION_CLASSES = ('NORMAL', 'PNEUMONIA')
BATCH_SIZE = 32
EPOCHS = 100

# -------------------------------------------------------------------------
#                           Import Libraries
# -------------------------------------------------------------------------
import gradio as gr
import numpy as np
from keras.models import load_model
from keras.preprocessing import image

#from source.config import *

# load the trained CNN model

cnn_model = load_model(MODEL_LOC)

__author__ = "Baishali Dutta"
__copyright__ = "Copyright (c) 2021-2022 Baishali Dutta"
__license__ = "Apache License 2.0"
__version__ = "0.1"

# -------------------------------------------------------------------------
#                               Configurations
# -------------------------------------------------------------------------





def make_prediction(test_image):
    test_image = test_image.name
    test_image = image.load_img(test_image, target_size=(224, 224))
    test_image = image.img_to_array(test_image) / 255.
    test_image = np.expand_dims(test_image, axis=0)
    result = cnn_model.predict(test_image)
    return {"Normal": str(result[0][0]), "Pneumonia": str(result[0][1])}


image_input = gr.inputs.Image(type="file")

title = "Pneumonia Detection"
description = "This application uses a Convolutional Neural Network (CNN) model to predict whether " \
              "a chosen X-ray shows if the person has pneumonia disease or not. To check the model " \
              "prediction, here are the true labels of the provided examples below: the first 4 images " \
              "belong to normal whereas the last 4 images are of pneumonia category. More specifically, " \
              "the 5th and 6th images are of type viral pneumonia infection in nature whereas he last 2 images " \
              "are of bacterial infection in nature."

gr.Interface(fn=make_prediction,
             inputs=image_input,
             outputs="label",
             examples=[["NORMAL1.jpeg"],
                       ["NORMAL2.jpeg"],
                       ["NORMAL3.jpeg"],
                       ["NORMAL4.jpeg"],
                       ["PNEUMONIA1.jpeg"],
                       ["PNEUMONIA2.jpeg"],
                       ["PNEUMONIA3.jpeg"]],
             title=title,
             description=description) \
    .launch()
