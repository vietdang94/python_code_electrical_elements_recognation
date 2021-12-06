from os import listdir
import cv2
import numpy as np
import pickle
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.applications.vgg16 import VGG16
from keras.layers import Input, Flatten, Dense, Dropout
from keras.models import Model
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import random
from keras.models import  load_model
import sys

import pyfirmata as fir
import time


a = fir.Arduino('COM6')
a.digital[8].mode = fir.OUTPUT
a.digital[9].mode = fir.OUTPUT
a.digital[10].mode = fir.OUTPUT


cap = cv2.VideoCapture(0)

# объявление класса
class_name = ['00000','MICROCONTROLLER','MOTHERBOARD','REMOTE']

def get_model():
    model_vgg16_conv = VGG16(weights='imagenet', include_top=False)

    # замарозить слой
    for layer in model_vgg16_conv.layers:
        layer.trainable = False

    # создать model
    input = Input(shape=(128, 128, 3), name='image_input')
    output_vgg16_conv = model_vgg16_conv(input)

    # добавить слои FC va Dropout
    x = Flatten(name='flatten')(output_vgg16_conv)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dropout(0.5)(x)
    x = Dense(4, activation='softmax', name='predictions')(x)

    # компиляция
    my_model = Model(inputs=input, outputs=x)
    my_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return my_model

# Загрузить весы модели, которая была обучена
my_model = get_model()
my_model.load_weights("vggmodel.h5")

while(True):
    # Покадровая съемка

    ret, image_org = cap.read()
    if not ret:
        continue
    image_org = cv2.resize(image_org, dsize=None,fx=0.5,fy=0.5)
    # Изменить размер
    image = image_org.copy()
    image = cv2.resize(image, dsize=(128, 128))
    image = image.astype('float')*1./255
    # Преобразовать в тензор
    image = np.expand_dims(image, axis=0)

    # прогнозировать
    predict = my_model.predict(image)
    print("This picture is: ", class_name[np.argmax(predict[0])], (predict[0]))
    print(np.max(predict[0],axis=0))

    # a.digital[8].write(0)
    # a.digital[9].write(0)
    # a.digital[10].write(0)

    if (np.max(predict)>=0.8) and (np.argmax(predict[0])!=0):


        # показать фото
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (50, 50)
        fontScale = 0.5
        color = (0, 255, 0)
        thickness = 2

        cv2.putText(image_org, class_name[np.argmax(predict)], org, font,
                    fontScale, color, thickness, cv2.LINE_AA)
        if (class_name[np.argmax(predict)]=="MICROCONTROLLER"):
            a.digital[8].write(1)
        else:
            a.digital[8].write(0)
        if (class_name[np.argmax(predict)] == "MOTHERBOARD"):
            a.digital[9].write(1)
        else:
            a.digital[9].write(0)
        if (class_name[np.argmax(predict)] == "REMOTE"):
            a.digital[10].write(1)
        else:
            a.digital[10].write(0)

    cv2.imshow("Picture", image_org)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Когда все будет сделано, снимаем съемку
cap.release()
cv2.destroyAllWindows()

