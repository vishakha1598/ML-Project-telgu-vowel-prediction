# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 18:53:10 2021

@author: Vishakha
"""

#import libraries 
import pandas as pd
from tensorflow.keras.layers import Conv2D,Flatten,Dense,MaxPool2D,BatchNormalization,GlobalAveragePooling2D
from tensorflow.keras.applications.resnet50 import preprocess_input,decode_predictions
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from sklearn.model_selection import KFold
from keras.callbacks import EarlyStopping, TensorBoard
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import tensorflow as tf
from tensorflow import keras
import pandas as pd 
import seaborn as sns

#split the dataset into train,test,val as 60,20,20
"""import splitfolders
input_dataset="Vowel_Dataset"
output_dataset="output"
splitfolders.ratio(input_dataset,output_dataset,seed=42,ratio=(.6,.2,.2))"""


img_height,img_width = (224,224)
batch_size = 32

train_data_dir="output/train"
test_data_dir="output/test"
valid_data_dir="output/val"

#imagedata generator 
train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(train_data_dir,target_size=(img_height,img_width),batch_size=32,class_mode='categorical')
valid_generator = train_datagen.flow_from_directory(valid_data_dir,target_size=(img_height,img_width),batch_size=32,class_mode='categorical')
test_generator = train_datagen.flow_from_directory(test_data_dir,target_size=(img_height,img_width),batch_size=1,class_mode='categorical')

base_model = ResNet50(include_top=False,weights="imagenet")
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(train_generator.num_classes,activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics = ['accuracy'])
fit_history=model.fit(train_generator,epochs=10,validation_data=valid_generator)
test_loss, test_acc =model.evaluate(test_generator,verbose = 2)
print("Test accuracy: ",test_acc)

from keras.preprocessing import image
test_image = image.load_img('output/test/A/70.jpg', target_size = (224, 224))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
 
# Predicting the final class
result = model.predict(test_image)[0].argmax()
 
# Fetching the class labels
labels = train_generator.class_indices
labels = list(labels.items())
 
# Printing the final label
for label, i in labels:
    if i == result:
        print("The test image has: ", label)
        break