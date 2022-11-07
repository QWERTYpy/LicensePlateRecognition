# Скачиваем модель весов YOLOv4
# https://github.com/AlexeyAB/darknet#pre-trained-models
# https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights
# Конвертируем в веса h5
# convert.py yolov4.cfg yolov4.weights yolov4.h5
# Для данного скрипта нужны также:
# \common\backbones\layers.py
# \yolo4\models\layers.py
# Взято : https://github.com/david8862/keras-YOLOv3-model-set
# Проблема с весами custom TypeError: buffer is too small for requested array
# Пробуем родной метод.
# https://github.com/theAIGuysCode/yolov4-custom-functions
# Используем save_model.py
# python save_model.py --weights yolov4.weights --output ./checkpoints/yolov4-416 --input_size 416 --model yolov4
# Не забыть поменять в core/config.py путь к *.names
import os
#  Отоброжать только ошибки tensoflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.python.saved_model import tag_constants
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
import time
import colorsys
import random
import imutils
from skimage.filters import threshold_local
import persp
import keras




input_size = 416

# helper function to convert bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, xmax, ymax
def format_boxes(bboxes, image_height, image_width):
    for box in bboxes:
        ymin = int(box[0] * image_height)
        xmin = int(box[1] * image_width)
        ymax = int(box[2] * image_height)
        xmax = int(box[3] * image_width)
        box[0], box[1], box[2], box[3] = xmin, ymin, xmax, ymax
    return bboxes


def read_class_names(class_file_name):
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names


def recognize_plate(img, coords):
    # separate coordinates from box
    xmin, ymin, xmax, ymax = coords
    # get the subimage that makes up the bounded region and take an additional 5 pixels on each side
    box = img[int(ymin)-10:int(ymax)+10, int(xmin)-10:int(xmax)+10]

    # # grayscale region within bounding box
    # #gray = cv2.cvtColor(box, cv2.COLOR_RGB2GRAY)
    # # resize image to three times as large as original for better readability
    # #gray = cv2.resize(gray, None, fx = 3, fy = 3, interpolation = cv2.INTER_CUBIC)
    # # perform gaussian blur to smoothen image
    # #blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # # Контраст
    # # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # clahe = cv2.createCLAHE(clipLimit=3., tileGridSize=(8, 8))
    #
    # lab = cv2.cvtColor(box, cv2.COLOR_BGR2LAB)  # convert from BGR to LAB color space
    # l, a, b = cv2.split(lab)  # split on 3 different channels
    #
    # l2 = clahe.apply(l)  # apply CLAHE to the L-channel
    #
    # lab = cv2.merge((l2, a, b))  # merge channels
    # img2 = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)  # convert from LAB to BGR
    # #====
    # # Избавляемся от шума
    # img3 = cv2.fastNlMeansDenoisingColored(img2, None, 10, 10, 7, 21)
    # #===
    # # Очистка
    # """
    #     Extract Value channel from the HSV format of image and apply adaptive thresholding
    #     to reveal the characters on the license plate.
    #     """
    # plate_img = img3.copy()
    # V = cv2.split(cv2.cvtColor(plate_img, cv2.COLOR_BGR2HSV))[2]
    #
    # T = threshold_local(V, 29, offset=15, method='gaussian')
    #
    # thresh = (V > T).astype('uint8') * 255
    #
    # thresh = cv2.bitwise_not(thresh)
    fixed_width = 500
    #
    # """At this point, we tried applying the tried and tested method of opening
    # and closing the image using erosion and dilation, which usually works well for removing noise.
    # However, in this case, it was also separating characters like 'N', which a thinner connection
    # between the 2 vertical lines. Thus, removing noise like this became a risk which would not pay off.
    # We found segmenting the characters based on size to be a better approach than this, but this could
    # be feasible if the letters and numbers were clear in all the images."""

    # plate_img = imutils.resize(plate_img, width=fixed_width)
    # thresh = imutils.resize(thresh, width=fixed_width)
    thresh = box
    bool_im = True
    try:
        bool_im = True
        thresh = imutils.resize(box, width=fixed_width)
    except:
        bool_im = False
    # Makes the letters slightly thicker
    # kernel = np.ones((3, 3), np.uint8)
    # thresh = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel)
    #====
    # cv2.imshow("1", box)
    # cv2.imshow("2", img2)
    # cv2.imshow("3", img3)
    # cv2.imshow("4", plate_img)
    # cv2.imshow("5", thresh)
    #
    #
    # cv2.waitKey(0)
    # now_time = str(time.time())[0:10]
    # cv2.imwrite(f"./detections/box_ {now_time}.png", thresh)
    return thresh, bool_im


def draw_bbox(image, bboxes, counted_classes = None, allowed_classes=list(read_class_names("./custom.names").values()), read_plate = False):
    classes = read_class_names("./custom.names")
    num_classes = len(classes)
    image_h, image_w, _ = image.shape
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    random.seed(0)
    random.shuffle(colors)
    random.seed(None)
    bool_im = True
    out_boxes, out_scores, out_classes, num_boxes = bboxes
    # print(out_boxes, out_scores, out_classes, num_boxes)
    for i in range(num_boxes):
        if int(out_classes[i]) < 0 or int(out_classes[i]) > num_classes: continue
        coor = out_boxes[i]
        fontScale = 0.5
        score = out_scores[i]
        class_ind = int(out_classes[i])
        class_name = classes[class_ind]


        if class_name not in allowed_classes:
            continue
        else:
            if read_plate:
                height_ratio = int(image_h / 25)
                image, bool_im = recognize_plate(image, coor)



            # bbox_color = colors[class_ind]
            # bbox_thick = int(0.6 * (image_h + image_w) / 600)
            # c1, c2 = (int(coor[0]), int(coor[1])), (int(coor[2]), int(coor[3]))
            # cv2.rectangle(image,(int(c1[0]), int(c1[1])), (int(c2[0]), int(c2[1])), bbox_color, bbox_thick)
            # if counted_classes != None:
            #     height_ratio = int(image_h / 25)
            #     offset = 15
            #     for key, value in counted_classes.items():
            #         cv2.putText(image, "{}s detected: {}".format(key, value), (5, offset),
            #                     cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 2)
            #         offset += height_ratio
    return image, bool_im


def load_images(image_path, saved_model_loaded):

    #  Загружаем изображение
    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    image_data = cv2.resize(original_image, (input_size, input_size))
    image_data = image_data / 255.

    # get image name by using split method
    image_name = image_path.split('/')[-1]
    image_name = image_name.split('.')[0]

    images_data = []
    for i in range(1):
        images_data.append(image_data)
    images_data = np.asarray(images_data).astype(np.float32)
    infer = saved_model_loaded.signatures['serving_default']

    batch_data = tf.constant(images_data)
    pred_bbox = infer(batch_data)
    print(pred_bbox)
    for key, value in pred_bbox.items():
        boxes = value[:, :, 0:4]
        pred_conf = value[:, :, 4:]

    # run non max suppression on detections
    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
        scores=tf.reshape(
            pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
        max_output_size_per_class=50,
        max_total_size=50,
        iou_threshold=0.45,
        score_threshold=0.5
    )
    # print(scores.numpy()[0])
    # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, xmax, ymax
    original_h, original_w, _ = original_image.shape
    bboxes = format_boxes(boxes.numpy()[0], original_h, original_w)

    # hold all detection data in one variable
    pred_bbox = [bboxes, scores.numpy()[0], classes.numpy()[0], valid_detections.numpy()[0]]

    # read in all class names from config
    class_names = read_class_names("./custom.names")

    # by default allow all classes in .names file
    allowed_classes = list(class_names.values())

    # custom allowed classes (uncomment line below to allow detections for only people)
    # allowed_classes = ['person']


    image, bool_im = draw_bbox(original_image, pred_bbox, allowed_classes=allowed_classes,
                            read_plate=True)

    #image = Image.fromarray(image.astype(np.uint8))
    # if not False:
    #     image.show()
    #image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    #now_time = str(time.time())[0:10]
    #cv2.imwrite(f"./detections/{now_time}.png", image)
    return image, scores.numpy()[0][0], bool_im
# Обработка изображений всех

#  Загружаем обученную модель
saved_model_loaded = tf.saved_model.load("./checkpoints/custom-416", tags=[tag_constants.SERVING])
str_num = "0123456789ABCEHKMOPTXY"
image_paths = "images"
plate_paths = "plate"

model = keras.models.load_model('plate_stand2.h5')

search_date = "25" #input("Введите дату обработки:")
html_file = open(f"{plate_paths}/{search_date}.html" , 'w')
html_text = f"<html><head><h1>Таблица за {search_date}</h1></head><table border=1>" \
            f"<tr><td>Распознанный номер</td><td>Определенный номер</td><td>Фото машины</td></tr>"
html_file.write(html_text)

for filename in os.listdir(image_paths):
    num_detect = ""

    res_list = []
    img, sc, bool_im = load_images(f"{image_paths}/{filename}", saved_model_loaded)
    if sc == 0:
        print("Номер не обнаружен.")
        html_text = f"<tr><td>НЕ РАСПОЗНАН</td><td>...</td><td><img src='..\{image_paths}\{filename}' width = 200></td></tr>"
        html_file.write(html_text)
        continue
    if not bool_im:
        print("Ошибка детекции")
        html_text = f"<tr><td>ОШИБКА</td><td>...</td><td><img src='..\{image_paths}\{filename}' width = 200></td></tr>"
        html_file.write(html_text)
        continue
    # cv2.imshow("1", img)
    # cv2.waitKey(0)
    img = persp.Rotate(img)
    # Сохраняем полученную маску номера
    # print(num_detect)
    now_time = str(time.time())[0:10]
    nnn = f"{now_time}_box.png"
    cv2.imwrite(f"{plate_paths}/{nnn}", img)
    symbol_list = persp.Segment(img)

    str_smb = ""
    for smb in symbol_list:
        color_image = cv2.imread(f"{plate_paths}/{smb}")
        gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        n_image = np.divide(gray_image, 255.0)
        n_image = n_image.reshape(1, 100, 100, 1).astype('float32')
        res = model.predict([n_image]).tolist()
        # print(res)
        res = res[0]
        # res_list.append(res)
        if max(res) >=0.6:
            num_detect += str_num[res.index(max(res))]
        else:
            num_detect +="?"
        str_smb += f"<img src='{smb}' width = 20>&nbsp"
    str_smb += f"<br>"
    # for res in res_list:
    #     str_smb += f"<br>"
    html_text = f"<tr><td>{num_detect}</td><td><img src='{nnn}' width = 200><br>{str_smb}</td><td><img src='..\{image_paths}\{filename}' width = 200></td></tr>"
    symbol_list.clear()
    print(filename)
    html_file.write(html_text)

html_text = "</table></html>"
html_file.write(html_text)
html_file.close()

# model = keras.models.load_model('lenet.h5')
# # Обработка изображений тестовых
# for filename in os.listdir("./img_test"):
#     load_images(f"./images/{filename}", model)
#     print(filename)
