from PIL import Image
from io import BytesIO
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import CustomObjectScope
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing import image
from image_augmentation import RandomColorAffine

def read_image(file) -> Image.Image:

	img = load_img(BytesIO(file), target_size=(224, 224))
	img_array = image.img_to_array(img)
	img_batch = np.expand_dims(img_array, axis = 0)

	return img_batch

def model():
	
	model = load_model("../models/10%_simclr_semi_supervised_model_bs64.33-0.27_080222.h5",custom_objects={'RandomColorAffine': RandomColorAffine})
	
	return model

def prediction(img, model):

	result = model.predict(img)

	return result
