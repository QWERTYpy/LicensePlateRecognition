# Данный скрипт отыскивает номер на фотографии, с помощью обученной модели.
# Символы разделяются по контурам.
# К данному скрипту относятся:
# wpod-net.h5, wpod-net.json, local_utils.py,
import os.path
from os.path import splitext, basename
from keras.models import model_from_json
import cv2
from local_utils import detect_lp
import numpy as np
import glob
import time
import random
import keras


# Загружаем обученную модель https://github.com/quangnhat185/Plate_detect_and_recognize
def load_model(path):
    try:
        path = splitext(path)[0]
        with open('%s.json' % path, 'r') as json_file:
            model_json = json_file.read()
        model = model_from_json(model_json, custom_objects={})
        model.load_weights('%s.h5' % path)
        print("Модель успешно загружена...")
        return model
    except Exception as e:
        print(e)

# Подготовка изображения
def preprocess_image(image_path,resize=False):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255
    if resize:
        img = cv2.resize(img, (224,224))
    return img

# Передаем изображение в модель и получаем координаты пластины
# При ошибке "No Licensese plate is founded!" попробовать изменить Dmin
def get_plate(image_path, Dmax=608, Dmin=200):
    vehicle = preprocess_image(image_path)
    ratio = float(max(vehicle.shape[:2])) / min(vehicle.shape[:2])
    side = int(ratio * Dmin)
    bound_dim = min(side, Dmax)
    _ , LpImg, _, cor = detect_lp(wpod_net, vehicle, bound_dim, lp_threshold=0.5)
    return LpImg, cor

# Получаем изображение номерного знака и его координаты из изборажения
def covert_img(test_image):
    LpImg, cor = get_plate(test_image)  # LpImg[0] Получаем изображение номера
    # print("Detect %i plate(s) in"%len(LpImg), splitext(basename(test_image))[0])
    # print("Coordinate of plate(s) in image: \n", cor)
    if (len(LpImg)):  # Проверяем существует ли определенный номер
        plate_image = cv2.convertScaleAbs(LpImg[0], alpha=(255.0))
        # Преобразуем в серый и размываем
        gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7, 7), 0)
        # Сводим изображение в чб
        binary = cv2.adaptiveThreshold(blur, 255,
                                       cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 45, 8)
        # Создаем маску слоя в которой отсутсвуют мелкие или очень крупные элементы
        _, labels = cv2.connectedComponents(binary)
        mask = np.zeros(binary.shape, dtype="uint8")
        # Устанавливаем минимальные и максимальные значения
        total_pixels = plate_image.shape[0] * plate_image.shape[1]  # Разрешение изображения
        lower = 450  # Меньше этого значения - удаляются с картинки
        upper = 2000 # Больше этого значения - удаляются с картинки

        # Проходим по всем уникальным компонентам
        for (i, label) in enumerate(np.unique(labels)):
            # Элементы относящиеся к фону пропускаем
            if label == 0:
                continue

            # Создаем маску
            labelMask = np.zeros(binary.shape, dtype="uint8")
            labelMask[labels == label] = 255
            numPixels = cv2.countNonZero(labelMask)

            if numPixels > lower and numPixels < upper:
                mask = cv2.add(mask, labelMask)
        return binary, mask


# Сортируем найденные контуры по порядку
def sort_contours(cnts,reverse = False):
    i = 0
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))
    return cnts


def usr_value(contours, list_order_char_error):
    usr_small_h = []  # Среднее значение высоты мелких символов
    usr_w = []  # Среднее значение ширины всех символов
    usr_large_h = []  # Среднее значение высоты больших символов
    usr_sdvig = []  # Среднее значение расстояния между началами символов
    usr_small_y = []  # Среднее значение смещения для мелких символов
    usr_large_y = []  # Среднее значение смещения для больших символов

    for cont in contours:
        _idx = contours.index(cont)
        if cont:
            if not list_order_char_error[_idx]:
                if contours.index(cont) in [0, 4, 5, 6, 7]:
                    usr_small_h.append(cont[3])
                    if contours.index(cont) in [0, 4, 5]:
                        usr_small_y.append(cont[1])
                else:
                    usr_large_h.append(cont[3])
                    usr_large_y.append(cont[1])
                usr_w.append(cont[2])
        if cont:
            if _idx < 7:
                if contours[_idx + 1]:
                    usr_sdvig.append(contours[_idx + 1][0] - contours[_idx][0])
    #  Вычисляем средние значения
    usr_small_h = int(sum(usr_small_h) / len(usr_small_h))
    usr_small_y = int(sum(usr_small_y) / len(usr_small_y))
    usr_large_h = int(sum(usr_large_h) / len(usr_large_h))
    usr_large_y = int(sum(usr_large_y) / len(usr_large_y))
    usr_w = int(sum(usr_w) / len(usr_w))
    usr_sdvig = int(sum(usr_sdvig) / len(usr_sdvig))
    return usr_small_h, usr_small_y, usr_large_h, usr_large_y, usr_w, usr_sdvig


#  Заполняем отсутсвующие символы усредненными контурами
def correct_order_char(list_order_char, ideal_contours, usr_small_h, usr_small_y, usr_large_h, usr_large_y, usr_w, usr_sdvig):
    err_flag = False
    while sum(list_order_char) < 8:
        err_flag = not err_flag
        for err_ind in range(len(list_order_char)) if err_flag else range(len(list_order_char) - 1, 0, -1):
            if not list_order_char[err_ind]:
                w = usr_w
                if err_ind in [0, 4, 5]:
                    h = usr_small_h
                    y = usr_small_y
                elif err_ind in [6, 7]:
                    h = usr_small_h
                    y = usr_small_y - 10

                else:
                    h = usr_large_h
                    y = usr_large_y
                if err_ind < 7 and err_flag:
                    if list_order_char[err_ind + 1]:
                        x = ideal_contours[err_ind + 1][0] - usr_sdvig
                        ideal_contours[err_ind] = (x, y, w, h)
                        list_order_char[err_ind] = 1
                if err_ind > 0 and not err_flag:
                    if list_order_char[err_ind - 1]:
                        x = ideal_contours[err_ind - 1][0] + usr_sdvig
                        ideal_contours[err_ind] = (x, y, w, h)
                        list_order_char[err_ind] = 1
    return ideal_contours


def correct_order_char_error(list_order_char_error, ideal_contours, usr_small_h, usr_small_y, usr_large_h, usr_large_y, usr_w):
    for _ind in range(len(list_order_char_error)):
        _y = ideal_contours[_ind][1]
        _w = ideal_contours[_ind][2]
        _h = ideal_contours[_ind][3]

        if list_order_char_error[_ind]:
            if _ind in [0, 4, 5]:
                if abs(_y - usr_small_y) > 10: _y = usr_small_y
                if abs(_w - usr_w) > 10: _w = usr_w
                if abs(_h - usr_small_h) > 10: _h = usr_small_h
            elif _ind in [6, 7]:
                if abs(_y - usr_small_y - 10) > 10: _y = usr_small_y
                if abs(_w - usr_w) > 10: _w = usr_w
                if abs(_h - usr_small_h) > 10: _h = usr_small_h
            else:
                if abs(_y - usr_large_y) > 10: _y = usr_large_y
                if abs(_w - usr_w) > 10: _w = usr_w
                if abs(_h - usr_large_h) > 10: _h = usr_large_h
            ideal_contours[_ind] = (ideal_contours[_ind][0], _y, _w, _h)
    return  ideal_contours


def mask_rect(img, cnts):
    contours_list = []  # Список контуров
    # Отсеиваем контуры явно не подходящие под начертания символов
    for c in sort_contours(cnts):
        (x, y, w, h) = cv2.boundingRect(c)
        ratio = h / w
        if ratio <= 0.8 or ratio >= 4:
            continue
        contours_list.append((x, y, w, h))
    # Проводим первичную классификацию найденных символов
    list_order_char = [0] * 8  # Список: 1 - найден символ, 0 - не найден
    list_order_char_error = [0] * 8  # Список: 1 - множественный контур на месте символа, потребуется постобработка
    list_range = [80, 130, 180, 230, 280, 330, 380, 430]  # Примерные границы, в которые должны попасть символы
    ideal_contours = [0] * 8  # Список итоговых контуров
    # Ищем символы на своих местах
    for cont in contours_list:
        for lrange in list_range:
            if cont[0] < lrange:
                if not list_order_char[list_range.index(lrange)]:
                    ideal_contours[list_range.index(lrange)] = cont
                    list_order_char[list_range.index(lrange)] = 1
                    break
                else:
                    list_order_char_error[list_range.index(lrange)] = 1
                    break
    # Вычисляем усредненные параметры символов
    usr_small_h, usr_small_y, usr_large_h, usr_large_y, usr_w, usr_sdvig = usr_value(ideal_contours,
                                                                                     list_order_char_error)
    # Заполняем отсутствующие символы усредненными контурами
    ideal_contours = correct_order_char(list_order_char, ideal_contours, usr_small_h, usr_small_y, usr_large_h,
                                        usr_large_y, usr_w, usr_sdvig)
    # Проводим корректировку контуров с ошибками
    ideal_contours = correct_order_char_error(list_order_char_error, ideal_contours, usr_small_h, usr_small_y,
                                              usr_large_h, usr_large_y, usr_w)

    # Расширяем контуры
    for _ind in range(len(ideal_contours)):
        ideal_contours[_ind] = (ideal_contours[_ind][0] - 10, ideal_contours[_ind][1] - 10,
                                ideal_contours[_ind][2] + 20, ideal_contours[_ind][3] + 20)

    # return contours_list
    return ideal_contours

#  Загружаем модель
wpod_net_path = "wpod-net.json"
wpod_net = load_model(wpod_net_path)

model = keras.models.load_model('plate.h5')

image_paths = "images"
plate_paths = "plate"
symbol_paths = "train"
str_num = "012345567789ABCEHKMOPTXY"
# Создаем список изображений
image_paths = glob.glob(f"{image_paths}/*.jpg")
print(f"Найдено {len(image_paths)} изображений")

# if not os.path.exists(symbol_paths):
#     print("Папки для хранения символов не существует! ")
#     os.mkdir(symbol_paths)
#     for sbl in list(str_num):
#         os.mkdir(f"{symbol_paths}/{sbl}")
#     print("Создана новая структура папок.")
search_date = input("Введите дату обработки:")
html_file = open(f"{plate_paths}/{search_date}.html" , 'w')
html_text = f"<html><head><h1>Таблица за {search_date}</h1></head><table border=1>" \
            f"<tr><td>Распознанный номер</td><td>Определенный номер</td><td>Фото машины</td></tr>"
html_file.write(html_text)
for test_image in image_paths:
    try:
        bin_im, msk_im = covert_img(test_image)
        # Ищем контуры и ограничивающие из рамки
        cnts, _ = cv2.findContours(msk_im.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Если итоговые контуры выходят за границы, корректируем
        num_detect = ""
        symbol_list = []
        res_list = []
        for c in mask_rect(msk_im, cnts):
            (x, y, w, h) = c
            if x < 0 : x = 0
            if x > bin_im.shape[1] : x = bin_im.shape[1]
            if y < 0 : y = 0
            if y > bin_im.shape[0] : y = bin_im.shape[0]
            #  Вырезаем символ по контуру
            roi = bin_im[y:y+h, x:x+w]
            #  Меняем его размер, для дальнейшего использования в сети
            output = cv2.resize(roi, (100, 100), interpolation=cv2.INTER_AREA)
            # Передаем символ на распознование
            n_image = output.reshape(1, 100, 100, 1).astype('float32')
            res = model.predict([n_image]).tolist()
            #print(res)
            res = res[0]
            res_list.append(res)
            num_detect += str_num[res.index(max(res))]

            # Сохраняем найденный символ (для подготовки наборов для обучения)
            now_time = str(time.time())[0:10]
            name_symbol = f"{now_time}_{random.randint(100, 200)}.png"
            cv2.imwrite(f"./{plate_paths}/{name_symbol}", output)
            symbol_list.append(name_symbol)
        # Сохраняем полученную маску номера
        print(num_detect)
        now_time = str(time.time())[0:10]
        nnn = f"{now_time}_{random.randint(100, 200)}_bin.png"
        cv2.imwrite(f"{plate_paths}/{nnn}", bin_im)
        str_smb = ""
        for smb in symbol_list:
            str_smb += f"<img src='{smb}' width = 20>&nbsp"
        str_smb += f"<br>"
        for res in res_list:
            str_smb += f"{res}<br>"
        html_text = f"<tr><td>{num_detect}</td><td><img src='{nnn}' width = 200><br>{str_smb}</td><td><img src='..\{test_image}' width = 200></td></tr>"
        symbol_list.clear()
        # plt.figure()
        # plt.imshow(bin_im)
        # #plt.figure()
        # #plt.imshow(msk_im)
        # plt.tight_layout()
        # plt.show()
    except:
        print('Номер не распознан', test_image)
        html_text = f"<tr><td>НЕ РАСПОЗНАН</td><td>...</td><td><img src='..\{test_image}' width = 200></td></tr>"

    html_file.write(html_text)

html_text = "</table></html>"
html_file.write(html_text)
html_file.close()
