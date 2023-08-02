'''
This file contains utility functions that are used repeatedly in the data transformation and model training processes.
'''
import os
import sys

import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

'''
This function save_object is used to save objects (e.g., trained models, preprocessing objects) to a specified file path. 
It takes file_path and the object obj as input.
'''
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

'''
The evaluate_models function takes: 
1. input features of train & test, 
2. target variables, 
3. models: A dictionary containing different regression models as values, with their names as keys.
4. param: A dictionary containing hyperparameter values for each model, with the model names as keys.

A dictionary report is initialized to store the evaluation results (R-squared scores) of different models.
'''
def evaluate_models(X_train, y_train,X_test,y_test,models,param):
    try:
        report = {}

        '''
        For each model, it performs the following steps:
        1. Performs hyperparameter tuning using GridSearchCV with the specified hyperparameters from the param dictionary. 
           This step tries different combinations of hyperparameters and selects the best combination based on cross-validated performance using 3-fold cross-validation (cv=3).
        2. Sets the hyperparameters of the model to the best parameters found during hyperparameter tuning.
        3. Fits the model on the training data (X_train and y_train).
        4. Makes predictions on both the training and testing data.
        5. Calculates the R-squared score for the model's performance on the training data (train_model_score) and the testing data (test_model_score).
        6. Stores the R-squared score of the model on the testing data in the report dictionary with the model's name as the key.
        '''
        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            #model.fit(X_train, y_train)  # Train model

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

#The function returns the report dictionary, which contains the R-squared scores of each model on the testing data.
        return report

    except Exception as e:
        raise CustomException(e, sys)
