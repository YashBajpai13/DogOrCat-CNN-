import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator


# Loading train data

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

training_set = train_datagen.flow_from_directory("dataset/training_set",
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

# Loading test data

test_datagen = ImageDataGenerator(rescale = 1./255)

test_set = test_datagen.flow_from_directory("dataset/test_set",
                                            target_size= (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

# Building the network

from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Dropout, Flatten

# The architecture is inspired by VGG net architecture

custom_vgg = Sequential()
custom_vgg.add(Conv2D(32, (3, 3), strides = 1, padding = "same", activation = "relu", input_shape = (64, 64, 3)))
custom_vgg.add(Dropout(0.4))
custom_vgg.add(MaxPool2D((2, 2)))

custom_vgg.add(Conv2D(64, (3, 3), strides = 1, padding = "same", activation = "relu"))
custom_vgg.add(Dropout(0.4))
custom_vgg.add(MaxPool2D((2, 2)))


custom_vgg.add(Flatten())
custom_vgg.add(Dense(1, activation = "sigmoid"))
print(custom_vgg.summary())

custom_vgg.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
custom_vgg.fit(x = training_set, validation_data = test_set, epochs = 35)

custom_vgg.save("saved_model")

