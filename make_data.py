import numpy as np
import cv2
import time
import os

# Label: 00000 соответствует отсутствию элементов
label = "00000"

cap = cv2.VideoCapture(0)

# переменная, позволяющая сохранить картинки после 60 раз (исключить понятия и опущения элементов)
i = 0
while (True):
    # Покадровая съемка
    #
    i += 1
    ret, frame = cap.read()
    if not ret:
        continue
    frame = cv2.resize(frame, dsize=None, fx=0.3, fy=0.3)

    # показать фото
    cv2.imshow('frame', frame)

    # сохранить данные
    if i >= 60:
        print("число фото  = ", i - 60)
        # создать файл если он не существует
        if not os.path.exists('data/' + str(label)):
            os.mkdir('data/' + str(label))

        cv2.imwrite('data/' + str(label) + "/" + str(i) + ".png", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#когда все закончилось то остановить съемку
cap.release()
cv2.destroyAllWindows()
