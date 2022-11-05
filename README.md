# Heroku-with-FastAPI-Example
Deploying a Machine Learning Model on Heroku with FastAPI
(https://github.com/FarizKesten/Heroku-with-FastAPI-Example)

# EDA.ipynb

Jupyter Notebook that works with the raw consensus data and save them to consensus_clean.csv

# DVC

DVC is used to save the consensus data and is saved to a S3 bucket

# Model Training
A model will be trained based on the cleaned up dataset and a LabelBinarizer(lb.joblib) , OneHotEncoder(encoder.joblib) and a RandomForest Model(model.joblib) will be saved

# Live API

API is built using FastAPI and is deployed on Heroku here:

https://deploy-ml-model-on-heroku.herokuapp.com/
