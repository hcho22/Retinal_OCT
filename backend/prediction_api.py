from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import CustomObjectScope
from tensorflow.keras.preprocessing import image
from image_augmentation import RandomColorAffine

# From Frontend
def read_image(file):
	size = (224, 224)
	img_array = tf.keras.preprocessing.image.img_to_array(Image.open(file.file)) #image to array
	img_resized = tf.keras.preprocessing.image.smart_resize(img_array, size) #resize

    # Load the image into a batch of 1
	img_batch = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
	img_batch[0] = img_resized
	
	return img_batch


def model():
	
	model = load_model("../models/10%_simclr_semi_supervised_model_bs64.33-0.27_080222.h5",custom_objects={'RandomColorAffine': RandomColorAffine})
	
	return model

def prediction(img, model):

	result = model.predict(img)

	return result
