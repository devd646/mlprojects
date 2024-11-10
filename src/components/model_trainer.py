import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    RandomForestRegressor, 
    GradientBoostingRegressor, 
    AdaBoostRegressor,
)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
# from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import (
    save_object, 
    evaluate_model,
)

@dataclass
class ModelrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelrainerConfig()

    def initiate_model_traininer(self,train_array, test_array):
        try:
            logging.info(f"Splitting train and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1],
            )

            models = {
                "Random_Forest" : RandomForestRegressor(),
                "Decison_Tree" : DecisionTreeRegressor(),
                "Gradient_Boost" : GradientBoostingRegressor(),
                "Ada_Boost": AdaBoostRegressor(),
                # "XGBoost" : XGBRegressor(),
                "CatBoost": CatBoostRegressor(),
                "Linear_Regression": LinearRegression(),
                "KNN": KNeighborsRegressor()
            }

            params = {
                "Random_Forest": {
                    "n_estimators": [100,200,300,400,500],
                    "criterion": ["squared_error", "absolute_error", "friedman_mse", "poisson"],
                    "max_depth": [2,5,8,10],
                    "min_samples_split": [2, 5],
                    "max_features": ["sqrt", "log2", None],
                },
                "Decison_Tree": {
                    "criterion": ["squared_error", "absolute_error", "friedman_mse", "poisson"],
                    "splitter": ["best", "random"],
                    "max_depth": [2,5,8,10],
                    "min_samples_split": [2, 5],
                    "max_features": ["sqrt", "log2", None],
                },
                "Gradient_Boost": {
                    "loss": ["squared_error","absolute_error","huber","quantile"],
                    "learning_rate": [1,0.1,0.01,0.001],
                    "n_estimators": [100,200,300,400,500],
                    "criterion": ["friedman_mse", "squared_error"]
                },
                "Ada_Boost": {
                    "n_estimators": [100,200,300,400,500],
                    "learning_rate": [1,0.1,0.01,0.001],
                    "loss": ["linear","square","exponential"]
                },
                "CatBoost": {
                    "depth": [6,8,10],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "iterations": [30, 50, 100],
                },
                "Linear_Regression": {},
                "KNN": {
                    "n_neighbors": [3,5,7],
                    "weights": ["uniform", "distance"],
                    "algorithm": ["auto", "ball_tree", "kd_tree", "brute"]
                }
            }

            model_report: dict= evaluate_model(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, param=params)[1]

            # To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            #To get best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
                ]
            
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("no best model found")
            
            logging.info(f" Best found model on both train and test")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            predicted = best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)
            return best_model_name, r2_square

        except Exception as e:
            raise CustomException(e,sys)