from fastapi import FastAPI
from fastapi import UploadFile, File
import uvicorn
from prediction_api import read_image, model, prediction
import numpy as np
from tensorflow.keras.preprocessing import image

app = FastAPI()

class_names = ['CNV','DME','DRUSEN','NORMAL']

model1 = None

@app.get('/')
async def OCT():
    return "Welcome to OCT Classification!"

@app.post('/predict')
async def predict_image(file: UploadFile = File(...)):

	# calls read_image function from prediction_api.py
	image_batch = read_image(file)
	print("Image loaded successfully")

	# calls model function from prediction_api.py
	model1 = model()
	print("Model loaded successfully")

	# make predictions
	predictions = prediction(image_batch, model1)
	print("Prediction: ", class_names[np.argmax(predictions[0])])
	print(predictions[0])
	
	return {"Prediction": class_names[np.argmax(predictions[0])]} # returns prediction class back to the front end

if __name__ == "__main__":
	uvicorn.run(app, port=8000, host='0.0.0.0')

