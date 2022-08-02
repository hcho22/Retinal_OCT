from fastapi import FastAPI
from fastapi import UploadFile, File
import uvicorn

from prediction_api import read_image, model, prediction

from PIL import Image
from io import BytesIO
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import CustomObjectScope
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing import image
from image_augmentation import RandomColorAffine

app = FastAPI()

class_names = ['CNV','DME','DRUSEN','NORMAL']

model1 = None

@app.get('/')
async def OCT():
    return "Welcome to OCT Classification!"

@app.post('/predict')
async def predict_image(file: UploadFile = File(...)):
	
	# Read the file uploaded by the user
	image = read_image(await file.read())
	print("Image loaded successfully")

	model1 = model()
	print("Model loaded successfully")

	# make predictions
	predictions = prediction(image, model1)
	print("Prediction: ", class_names[np.argmax(predictions[0])])
	print(predictions[0])
	
	return {"message": 'Success!'}


#if __name__ == "__main__":
#	uvicorn.run(app, port=8000, host='0.0.0.0')

