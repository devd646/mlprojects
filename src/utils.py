import os
import sys

import numpy as np
import pandas as pd

from src.exception import CustomException
import dill

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            dill.dump(obj, file_obj)
    
    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_model(X_train, y_train, X_test, y_test, models,param):
    try:
        train_report= {}
        test_report= {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=5)
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)
            
            # model.fit(X_train, y_train) # Train model

            y_train__pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train__pred)

            test_model_score = r2_score(y_test, y_test_pred)

            train_report[list(models.keys())[i]] = train_model_score
            
            test_report[list(models.keys())[i]] = test_model_score

        return (
            train_report, 
            test_report,
        )

    except Exception as e:
        raise CustomException(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    
    except Exception as e:
        raise CustomException(e, sys)