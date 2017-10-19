'''
This program is used to detect the driver's status (10 statuses) by using a small convolutional neural network, which
is trained fram scatch using the training images.
'''

import os
import cv2 
import numpy as np

import csv
import shutil
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing import image

from keras import backend as K
K.set_image_dim_ordering('th')


model=Sequential([
Convolution2D(nb_filter=32, nb_row=3, nb_col=3, input_shape=(3,150,150)),
Activation('relu'),
MaxPooling2D(pool_size=(2,2)),
Convolution2D(32, 3, 3),
Activation('relu'),
MaxPooling2D(pool_size=(2,2)),
Convolution2D(nb_filter=64, nb_row=3, nb_col=3),
Activation('relu'),
MaxPooling2D(pool_size=(2,2)),
Flatten(),
Dense(64),
Activation('relu').
Dropout(0.5),
Dense(10),
Activation('softmax')])


model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])


train_datagen=ImageDataGenerator(rescale=1.0/255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)


test_datagen=ImageDataGenerator(rescale=1.0/255)

train_generator=train_datagen.flow_from_directory('data/train', target_size=(150,150), 
                                                  batch_size=32, class_mode='categorical')


validation_generator=test_datagen.flow_from_directory('data/validation', target_size=(150,150), 
                                                      batch_size=32, class_mode='categorical')


model.fit_generator(train_generator, samples_per_epoch=20924, nb_epoch=20, 
                   validation_data=validation_generator, nb_val_samples=800)


model.save_weights('weights.h5')


dir_im = 'data/test/test'
list_c = []
list_name = []
with open('results_distracted_driver.csv', 'a', newline='') as csvfile:
    csvwriter = csv.DictWriter(csvfile, fieldnames=["img", "c0", "c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9"])
    csvwriter.writeheader();


    for imageName in sorted(os.listdir(dir_im)):
      dir_n = dir_im + os.sep + imageName
      if dir_n.lower().endswith('.jpg'):
        img = image.load_img(dir_n, target_size=(150, 150))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        images = np.vstack([x])
        
        y_prob = model.predict(images) 
        y_classes = y_prob.argmax(axis=-1)
        classes = model.predict_classes(images)
        #print(classes)
        #print(type(y_prob[0]))
        #print(y_classes)
        list_c.append(classes[0])
        list_name.append(imageName)
        
        csvwriter.writerow({"img":imageName, "c0":y_prob[0][0], "c1":y_prob[0][1], "c2":y_prob[0][2], "c3":y_prob[0][3], "c4":y_prob[0][4], "c5":y_prob[0][5], "c6":y_prob[0][6], "c7":y_prob[0][7], "c8":y_prob[0][8], "c9":y_prob[0][9]})
        #print(classes)
#print(list_c)
#print(list_name)