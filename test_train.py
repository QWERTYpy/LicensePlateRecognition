import keras  # Для создания нейронной сети

import numpy as np
import cv2
import os
import shutil


model = keras.models.load_model('plate_stand2.h5')

# Путь к папке откуда брать для проверки
path = "./symbol"
str="0123456789ABCEHKMOPTXY"

# Путь куда складывать опознанные
path2 = "MainTrain"
for filename in os.listdir(path):

    # grayscale_image = cv2.imread(your_image, 0)
    color_image = cv2.imread(f"{path}/{filename}")
    gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    n_image = np.divide(gray_image, 255.0)
    n_image = n_image.reshape(1, 100, 100, 1).astype('float32')
    res = model.predict([n_image]).tolist()
    res = res[0]
    print([round(x,2) for x in res])
    print(f">>>>{str[res.index(max(res))]}")
    # cv2.imshow("crop", gray_image)
    # cv2.imshow("mask", mask)
    # cv2.waitKey(0)
    if max(res) >= 0.6:
        shutil.copyfile(f"{path}/{filename}",f"{path2}/{str[res.index(max(res))]}/{filename}")
        os.remove(f"{path}/{filename}")
    # print(f"{path}/{filename}->{path2}/{str[res.index(max(res))]}/{filename}")

