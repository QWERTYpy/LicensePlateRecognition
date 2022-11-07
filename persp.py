import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import threshold_local
from collections import Counter
import time
import random


def rotateImage(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR, borderValue=(255,255,255))
  return result


def Contrast(img):
    # Добавляем контрастность
    clahe = cv2.createCLAHE(clipLimit=3., tileGridSize=(8, 8))

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)  # convert from BGR to LAB color space
    l, a, b = cv2.split(lab)  # split on 3 different channels

    l2 = clahe.apply(l)  # apply CLAHE to the L-channel

    lab = cv2.merge((l2, a, b))  # merge channels
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)  # convert from LAB to BGR
    img = cv2.GaussianBlur(img, (3, 3), 0)
    # Избавляемся от шума
    img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    # ===
    return img


def Clear(img):
        # Очистка
    plate_img = img.copy()
    V = cv2.split(cv2.cvtColor(plate_img, cv2.COLOR_BGR2HSV))[2]

    T = threshold_local(V, 29, offset=15, method='gaussian')

    thresh = (V > T).astype('uint8') * 255

    img = cv2.bitwise_not(thresh)
    return img


def Rotate(img):
    img_copy = np.copy(img)
    img = cv2.GaussianBlur(img, (5, 5), 0)

    img = Contrast(img)
    img = Clear(img)

    low_threshold = 100
    high_threshold = 250
    img = cv2.Canny(img, low_threshold, high_threshold)
    # cv2.imshow("1", img)
    # cv2.waitKey(0)
    # Используйте преобразование Хафа, чтобы найти прямые линии
    # настройки параметров
    rho = 1
    theta = np.pi / 180
    threshold = 65
    min_line_length = 100
    max_line_gap = 25

    # Найдите прямую

    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
    line_image = np.copy(img_copy)
    # Нарисуйте прямую линию
    list_line = []
    list_angle = []
    try:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)
                _angle = round(np.arctan((y2 - y1) / (x2 - x1)) * 180 / np.pi, 0)
                list_line.append((_angle, (x2 - x1) ** 2 + (y2 - y1) ** 2))
                list_angle.append(_angle)

        # Считаем сколько линий обнаружено с одинаковыми углами
        count_angle = Counter(list_angle)
        # Находим наибольшее количество совпадений
        max_count_angle = max(count_angle.values())
        list_angle = []
        all_len = []
        for _val in count_angle:
            if count_angle[_val] == max_count_angle:
                max_len = 0
                for _tuple in list_line:
                    if _tuple[0] == _val and _tuple[1] > max_len:
                        max_len = _tuple[1]
                list_angle.append((_val, max_len))
                all_len.append(max_len)
        img2 = img_copy
        if len(list_angle) == 1:
            if list_angle[0][0] != 0.0:
                img2 = rotateImage(img_copy, list_angle[0][0])
                # print(f"Повернуто на: {list_angle[0][0]}")
        else:
            _max = max(all_len)
            for _val in list_angle:
                if _val[1] == _max:
                    if _val[0] != 0.0:
                        img2 = rotateImage(img_copy, _val[0])
                        # print(f"Повернуто на: {_val[0]}")
        return img2
    except:
        print("Не удалось определить наклон.")
        return img_copy

def Segment(img):
    img_orig = img.copy()
    img = Contrast(img)
    img = Clear(img)

    _y, _x = img.shape
    _summ = []
    # print(_y, _x)
    for __y in range(_y):
        _summ.append(0)
        for __x in range(_x):

            if img[__y, __x] == 255:
                _summ[__y] += 1
            else:
                _summ[__y] += 0
    x = _summ
    y = list(range(_y))
    # print(int(_y/2))
    # plt.plot(x, y)
    # plt.show()
    x[:20] = [0] * 20
    x[-20:] = [0] * 20
    x[int(_y/2)-20:int(_y/2)+40] = [0]*60
    # plt.plot(x, y)
    # plt.show()
    max_x1 = max(x[:int(_y/2)])
    max_x2 = max(x[int(_y/2):])
    max_y1 = x.index(max_x1, 0, int(_y/2))
    max_y2 = x.index(max_x2, int(_y/2))
    # print(max_x1, max_x2,max_y1,max_y2)
    # plt.plot(x, y)
    # plt.show()

    # cv2.line(img2, (0, max_y1), (_x, max_y1), (255, 0, 0), 5)
    # cv2.line(img2, (0, max_y2), (_x, max_y2), (255, 0, 0), 5)
    img = img[max_y1+10:max_y2-10, 0:_x]

    _y, _x = img.shape
    _summ = []
    # print(_y, _x)
    for __x in range(_x):
        _summ.append(0)
        for __y in range(_y):

            if img[__y,__x] == 255:
                _summ[__x]+=1
            else:
                _summ[__x] += 0


    x = list(range(_x))
    y = _summ

    max_y1 = max(y[:100])
    max_y2 = max(y[400:])
    max_x1 = y.index(max_y1, 0, 100)
    max_x2 = y.index(max_y2, 400)
    img = img[0:_y, max_x1+10:max_x2-10]
    img = cv2.resize(img, (500, int(_y * 500 / _x)))
    img2 = img.copy()
    _y, _x = img2.shape
    img2 = img2[10:_y-10,0:_x]
    # _xx=[30,93,148,198,248,303,358,383,426,469]
    # img = cv2.resize(img,(500,_y))
    # for __x in _xx:
    #     cv2.line(img, (__x, 0), (__x, _y), (255, 0, 0), 5)
    # plt.plot(x, y)
    #
    # plt.show()

    _y, _x = img2.shape
    cv2.rectangle(img2, pt1=(400, _y), pt2=(_x, int(_y*2.2/3)), color=(0,0 , 0), thickness=-1)
    _summ = []
    # print(_y, _x)
    for __x in range(_x):
        _summ.append(0)
        for __y in range(_y):

            if img2[__y, __x] == 255:
                _summ[__x] += 1
            else:
                _summ[__x] += 0

    x = list(range(_x))
    y = _summ
    list_line = []
    list_interval = [(15,50), (70,110), (130,160), (180,220), (230,270), (290,330), (360, 380), (390,410), (420,450), (460,480)]
    for _li in list_interval:
        min_y = min(y[_li[0]:_li[1]])
        min_x = y.index(min_y, _li[0], _li[1])
        list_line.append(min_x)

    _y, _x = img.shape
    plate_paths = "plate"
    symbol_list = []
    now_time = str(time.time())[0:10]
    # name_symbol = f"{now_time}_box.png"
    # cv2.imwrite(f"./{plate_paths}/{name_symbol}", img_orig)
    list_nbr = [1,2,3,4,5,6,7,8,9,10,11,12]
    for _ind in range(len(list_line)-1):
        if _ind == 6:
            continue
        # cv2.line(img, (__x+5, 0), (__x+5, _y), (255, 0, 0), 5)
        crop = img[0:_y,list_line[_ind]+5:list_line[_ind+1]+5]
        crop = cv2.resize(crop, (100,100))
        # cv2.imshow("ttt", crop)
        # cv2.waitKey(0)
        # Создаем маску слоя в которой отсутсвуют мелкие или очень крупные элементы
        _, labels = cv2.connectedComponents(crop)
        mask = np.zeros(crop.shape, dtype="uint8")
        # Устанавливаем минимальные и максимальные значения
        total_pixels = crop.shape[0] * crop.shape[1]  # Разрешение изображения
        lower = 500  # Меньше этого значения - удаляются с картинки
        upper = 10000  # Больше этого значения - удаляются с картинки
        # print(f"всего: {total_pixels}, min: {lower}, max: {upper}")
        # Проходим по всем уникальным компонентам
        for (i, label) in enumerate(np.unique(labels)):
            # Элементы относящиеся к фону пропускаем
            if label == 0:
                continue

            # Создаем маску
            labelMask = np.zeros(crop.shape, dtype="uint8")
            labelMask[labels == label] = 255
            numPixels = cv2.countNonZero(labelMask)
            # print(numPixels)
            if numPixels > lower and numPixels < upper:
                mask = cv2.add(mask, labelMask)
        crop = mask
        # cv2.imshow("crop", crop)
        # cv2.imshow("mask", mask)
        # cv2.waitKey(0)
        # name_symbol = f"{now_time}_{random.randint(100, 200)}.png"
        # cv2.imwrite(f"./{plate_paths}/{name_symbol}", crop)
        # Сохраняем найденный символ (для подготовки наборов для обучения)
        # now_time = str(time.time())[0:10]
        # name_symbol = f"{now_time}_{random.randint(100, 200)}.png"
        name_symbol = f"{now_time}_{list_nbr.pop()}.png"
        cv2.imwrite(f"./{plate_paths}/{name_symbol}", crop)
        symbol_list.append(name_symbol)

    # plt.plot(x, y)
    # plt.imshow(img)
    # plt.plot(x,y)
    #
    # plt.show()

    return symbol_list

# for filename in os.listdir("./detections"):
#     print(filename)
#     if "box" in filename:
#         img = cv2.imread(f"./detections/{filename}")
#         img2 = Rotate(img)
#         Segment(img2)
#         # cv2.imshow("orig", img)
#         # cv2.imshow("rot", img2)
#         # cv2.waitKey(0)