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
| Resnet50       | 91.7%      | 88.7%    |96.4%      |



Semi-Supervised Learning
| Model                     | Train Acc.  | Val Acc. | Test Acc. |
| --------                  | --------    | -------- |-----------|
| simCLR 1% labeled images  | 98.97%      | 93.26%   |87.5%      |
| simCLR 10% labeled image  | 96.96%      | 93.23%   |97.10%     |


## Acknowledgements
- [SimCLR - Google Research](https://github.com/google-research/simclr)
- [SimCLR - Keras.io](https://keras.io/examples/vision/semisupervised_simclr/)

