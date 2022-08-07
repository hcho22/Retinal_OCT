# Retinal OCT

##   Retinal Disease Classifier
![](https://i.imgur.com/2HBIIJd.png)

There are approximately 30 million OCT (optical coherence tomography) scans are performed each year. OCT is a non-invasive imaging test that uses light waves to take cross-section picture of human retina. Analysing, interpretating, and labeling these 30 million images can take significant amount of time and resource. 

This web app is able to predict three popular retina diseases and normal retina using unlabeled OCT images. These popular diseases are CNV (Choroidal Neovascularization), DME (Diabetic Mascular Edema), and DRUSEN. 

Resource
- [Publication](https://www.cell.com/cell/fulltext/S0092-8674(18)30154-5)


## Dataset
The Retinal OCT images can be downloaded in the following links:
- [Kaggle](https://www.kaggle.com/code/paultimothymooney/detect-retina-damage-from-oct-images/notebook)
- [Mendeley Data](https://data.mendeley.com/datasets/rscbjbr9sj/2)

## Model 
The model was created using the SimCLR (Simple Framework for Contrstive Learning). Please refer to acknowledgements for additional reading.  
![](https://i.imgur.com/QhfGuOV.gif)

## Architecture 
The following tools were used to setup the MLOPs pipeline:
- Fastapi: Main RestAPI Framework
- AWS EC2: Deployment
- Streamlit: Frontend UI

## Test Results
The results from the semi-supervised learning was highly comparable to the SOTA supervised learning models. 

Supervised Learning Model
| Model         | Train Acc. | Val Acc. | Test Acc. |
| --------      | --------   | -------- |-----------|
| VGG16         | 90.5%      | 89.1%    |93.1%      |
| InceptionV3   | 71.5%      | 75.7%    |67.6%      |
| Resnet50      | 91.7%      | 88.7%    |96.4%      |
| Resnet50V2    | 93.38%     | 87.9%    |98.14%     |



Semi-Supervised Learning
| Model                     | Train Acc.  | Val Acc. | Test Acc. |
| --------                  | --------    | -------- |-----------|
| simCLR 1% labeled images  | 98.97%      | 93.26%   |87.5%      |
| simCLR 10% labeled image  | 96.96%      | 93.23%   |97.10%     |


## Acknowledgements
- [SimCLR - Google Research](https://github.com/google-research/simclr)
- [SimCLR - Keras.io](https://keras.io/examples/vision/semisupervised_simclr/)

# Retinal OCT app installation procedure via cloning Git Repository

### Create new conda environment
Create a new conda environment by running the following command. 

conda create --name myenv python=3.9.2 

(python version 3.9.2 was used to create the app)

### Clone Samsung OCT repository
Clone the Samsung OCT repository by running the following command.

git clone git@github.com:(your profile)/Samsung_OCT.git

![](https://i.imgur.com/K4JTdIG.png)

### Install all the requirements

cd Samsung_OCT
pip install -U pip
pip install -r requirements.txt


### Activate api
cd backend

uvicorn api_ main:app --reload --workers 1 --host 0.0.0.0 --port 8000

if successful, the following outputs will appear.
![](https://i.imgur.com/7sG8rZ4.png)


### Activate frontend UI

Run the app by typing the following command from the terminal. 

cd frontend

streamlit run streamlit.py

browser will automatically and the app will be available locally. 

![](https://i.imgur.com/BzDMT4P.png)

From here you can upload your OCT scan image and the app will classify the disease (CNV, DME, or DRUSEN)

![](https://i.imgur.com/knwkrU2.jpg)

Uploaded image will be displayed, along with the result. 


### Run the app on AWS EC2

After launching the EC2, run the app by typing the following command from the terminal.

cd backend
uvicorn api_ main:app --reload --workers 1 --host 0.0.0.0 --port 8000

cd frontend
streamlit run streamlit.py

Network URL: http://(your ip address):8501
External URL: http://(your ip address):8501

App will launch on the External URL

### Limitations

Model 
- Extensive finetuning for SimCLR model has not been performed due to time and resource. Additional epochs and batch selections will be performed during the next phase release. 

Data
- Data tends to be slightly imbalanced towards to CNV and NORMAL classes. Additional training will be performed with class weights during the next phase release.

ML Pipeline
- Full automatic deployment has not been created. Web app should be able to launch automatically with a simple script that run and starts the docker container.  