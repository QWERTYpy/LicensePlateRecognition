from keras.models import Sequential  # Простейший тип сети. Вся информация передается только последующему слою
from keras.layers import Dense, Dropout
import tensorflow as tf
from keras.layers import Flatten, Conv2D, MaxPooling2D # new!


data_dir = "MainTrain" # Путь к файлам для обучения
img_height = 100
img_width = 100
batch_size = 32

train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  label_mode='categorical',
  validation_split=0.2,
  subset="training",
  color_mode = "grayscale",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)


val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  label_mode='categorical',
  validation_split=0.2,
  subset="validation",
  color_mode = "grayscale",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_ds.class_names

# Подготовка изображений для передачи в сеть
normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
normalized_train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
normalized_val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
# Колличество классов (количество папок)
n_classes = 22


model = Sequential()
# первый сверточный слой:
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(100, 100, 1)))

# второй сверточный слой с субдискретизацией и прореживанием:
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())  # преобразует трехмерную карту активаций, сгенерированную слоем Conv2D(), в одномерный массив

# полносвязанный скрытый слой с прореживанием:
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

# выходной слой:
model.add(Dense(n_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(normalized_train_ds, epochs=8, validation_data=normalized_val_ds)
model.save('plate_stand.h5')

