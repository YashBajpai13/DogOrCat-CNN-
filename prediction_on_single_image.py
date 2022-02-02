from tensorflow import keras
import numpy as np
from keras.preprocessing import image
model = keras.models.load_model(
    "saved_model")

test_image = image.load_img("dataset/single_prediction/cat_or_dog_2.jpg",
                            target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = model.predict(test_image/255.0)

if result[0][0] >= 0.5 :
    prediction = "dog"
else :
    prediction = "cat"
print(prediction)