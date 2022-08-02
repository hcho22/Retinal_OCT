from PIL import Image
from io import BytesIO
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import CustomObjectScope
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing import image
from src.image_augmentation import RandomColorAffine

import argparse
from pathlib import Path

input_shape = (224, 224)

model = load_model("models/10%_simclr_semi_supervised_model_bs64.33-0.27.h5",custom_objects={'RandomColorAffine': RandomColorAffine})

class_names = ['CNV','DME','DRUSEN','NORMAL']

#img_path = "CNV-732516-2.jpeg"

parser = argparse.ArgumentParser()
parser.add_argument("file_path", type=Path)
img_path = parser.parse_args()

def read_image(path):
	img = load_img(path, target_size=(224, 224))
	img_array = image.img_to_array(img)
	print(img_array.shape)
	img_batch = np.expand_dims(img_array, axis = 0)
	print(img_batch.shape)


	return img_batch

#def preprocess(image: Image.Image):
#	image = image.resize(input_shape)
#	image = np.asfarray(image)
#	image = image / 127.5 -1.0
#	image = np.expand_dims(image, 0)

#	return image

img = read_image(img_path.file_path)
predictions = model.predict(img)
print("Class Name: ", class_names[np.argmax(predictions[0])])
print(predictions[0])