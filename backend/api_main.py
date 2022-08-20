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
	
	pred_name = class_names[np.argmax(predictions[0])]
	
	arr = np.array(predictions[0])
	pred_value = np.around(max(arr.tolist())*100,3)

	print("Prediction: ", pred_name)
	print("Accuracy: ", pred_value)

	return {"Prediction": pred_name, 
			"Accuracy": pred_value
			} # returns prediction class back to the front end


if __name__ == "__main__":
	uvicorn.run(app, port=8000, host='0.0.0.0')

