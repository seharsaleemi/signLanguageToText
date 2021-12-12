
#Folder Structure :- Download and extract data in "input" folder
# /input/sign-language-mnist

import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix,accuracy_score, classification_report

"""
We need to read the csv train and test inputs. Since we are training these as images, so we need to convert them to images and extract labels from it.
The 1st column of the csv has the label information and the rest are the image pixels.
We'll return the images and labels as numpy array.
"""

# Reading the dataset
def get_data(filename):
    with open(filename) as training_file:
        training_reader = csv.reader(training_file, delimiter=',')
        image = []
        labels = []
        line_count = 0
        for row in training_reader:
            if line_count == 0:
                line_count +=1
            else:
                labels.append(row[0])
                temp_image = row[1:785]
                image_data_as_array = np.array_split(temp_image, 28)
                image.append(image_data_as_array)
                line_count += 1
        images = np.array(image).astype('float')
        labels = np.array(labels).astype('float')
        print(f'Processed {line_count} lines.')

    return images, labels


training_images, training_labels = get_data("./input/sign-language-mnist/sign_mnist_train/sign_mnist_train.csv")
testing_images, testing_labels = get_data("./input/sign-language-mnist/sign_mnist_test/sign_mnist_test.csv")

print("Total Training images", training_images.shape)
print("Total Training labels",training_labels.shape)
print("Total Testing images",testing_images.shape)
print("Total Testing labels",testing_labels.shape)

alphabets = 'abcdefghijklmnopqrstuvwxyz'
mapping_letter = {}

for i,l in enumerate(alphabets):
    mapping_letter[l] = i
mapping_letter = {v:k for k,v in mapping_letter.items()}

"""
As you can see that there are 24 categories present in the labels, On careful observation we find that *j, *Z** is not present in the dataset.
Now we need to add another dimension in our images so that we can process it for the **ImageDataGenerator** and do the **Image Augmentation**
Read more [here](https://keras.io/api/preprocessing/image/)
"""

#Data Augmentation
training_images = np.expand_dims(training_images, axis = 3)
testing_images = np.expand_dims(testing_images, axis = 3)

print(training_images.shape)
print(testing_images.shape)

# Create an ImageDataGenerator and do Image Augmentation
train_datagen = ImageDataGenerator(rescale = 1.0/255.0,
                                   height_shift_range=0.1,
                                   width_shift_range=0.1,
                                   zoom_range=0.1,
                                   shear_range=0.1,
                                   rotation_range=10,
                                   fill_mode='nearest',
                                   horizontal_flip=True)


#Image Augmentation is not done on the testing data
validation_datagen = ImageDataGenerator(rescale=1.0/255)
train_datagenerator = train_datagen.flow(training_images,
                                         training_labels,
                                         batch_size = 32)
validation_datagenerator = validation_datagen.flow(testing_images,
                                                   testing_labels, 
                                                   batch_size=32)

"""
Now lets define a callback for avoiding the excess training and stopping the training based
on the predefined condition, in our case, we want training to stop once the req test accuracy
is reached above **99%**.
"""

# Define a Callback class that stops training once accuracy reaches 99%
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy')>0.995):
      print("\nReached 99.5% accuracy so cancelling training!")
      self.model.stop_training = True

# Define the model
model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape = (28,28,1)),
                                    tf.keras.layers.MaxPool2D(2,2),
                                    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
                                    tf.keras.layers.MaxPool2D(2,2),
                                    tf.keras.layers.Conv2D(512, (3,3), activation='relu'),
                                    tf.keras.layers.MaxPool2D(2,2),
                                    tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(1024, activation = 'relu'),
                                    tf.keras.layers.Dropout(0.2),
                                    tf.keras.layers.Dense(512, activation = 'relu'),
                                    tf.keras.layers.Dropout(0.2),
                                    tf.keras.layers.Dense(25, activation = 'softmax')])

model.summary()

"""I have used 3 Conv2D and 3 MaxPooling2D and the dropout of 0.2"""

# Compiling the Model. 
model.compile(loss = 'sparse_categorical_crossentropy',
             optimizer = tf.keras.optimizers.Adam(),
              metrics = ['accuracy'])

"""Learning Rate modification"""
learning_rate_reduction = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', 
                                            patience = 2, 
                                            verbose=1,factor=0.25, 
                                            min_lr=0.0001)

# Train the Model
callbacks = myCallback()
history = model.fit(train_datagenerator,
                    validation_data = validation_datagenerator,
                    steps_per_epoch = len(training_labels)//32,
                    epochs = 100,
                    validation_steps = len(testing_labels)//32,
                    callbacks = [callbacks, learning_rate_reduction])


evals = model.evaluate(testing_images, testing_labels, verbose=0)

print(evals)

model.save('sign_language.h5')

"""Visualise the model"""
tf.keras.utils.plot_model(model,
                          to_file="model.png",
                          show_shapes=True,
                          show_dtype=False,
                          show_layer_names=True,
                          rankdir="TB",                          
                          expand_nested=True,
                          dpi=96)
