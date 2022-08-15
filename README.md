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
The following tools were used to setup the MLOPs deployment pipeline:
- Fastapi: Main RestAPI Framework
- Docker: Images and Containers
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

# Retinal OCT web app installation procedure

### Create new conda environment
Create a new conda environment by running the following command. 

conda create --name myenv python=3.9.2 

(python version 3.9.2 was used to create the app)

### Clone Retinal OCT repository
Clone the Retinal OCT repository by running the following command.

git clone git@github.com:(your profile)/Retinal_OCT.git

![](https://i.imgur.com/gem7aqh.png)


### To build and run docker containers

cd Retinal_OCT

docker-compose build

if successful, the following outputs will appear.
![](https://i.imgur.com/VohZc1i.png)

docker-compose up

if successful, the following output will appear.
![](https://i.imgur.com/9a5OIQi.png)




### Testing via fastapi

User can test the model via fastapi using swaggerUI by visiting http://localhost:8000/docs

![](https://i.imgur.com/zWMuIG0.png)


### Testing via Streamlit
User can also use the app using Streamlit by visiting http://localhost:8501/

![](https://i.imgur.com/SyLOjfJ.png)

![](https://i.imgur.com/LoNkrtv.png)


### Limitations

Model 
- Extensive finetuning for SimCLR model has not been performed due to time and resource. Additional epochs and batch selections will be performed during the next phase release. 


ML Pipeline
- The web app has not been deployed on a cloud such as AWS or GCP yet.  