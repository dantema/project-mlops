import pandas as pd
import numpy as np
import mlflow
import pickle

from catboost import CatBoostRegressor, Pool
from hyperopt import fmin, tpe, space_eval, STATUS_OK

from params import search_space, fit_params
from metrics import metrics_compute


# TODO. Metrics, read-prepare data, utils, operator
#TODO delete numba

def call_predictions(model, data: Pool) -> np.array:
    pred = model.predict(data)
    return np.array(pred)

def create_pools(df_train: pd.DataFrame,
                 df_test: pd.DataFrame,
                 y_train, y_test,
                 cat_features: list[str] = None,
                ) -> tuple[Pool, Pool]:
    
    train_pool = Pool(df_train, y_train, cat_features=cat_features)
    eval_pool = Pool(df_test, y_test, cat_features=cat_features)
    return train_pool, eval_pool


def train_model(df_train: pd.DataFrame, 
                df_test: pd.DataFrame, 
                target: str = "price",
                cat_features: list[str] = ["model", "transmission", "fuelType"]
                ) -> dict:

    train_pool, eval_pool = create_pools(df_train.drop(target, axis=1),
                                         df_test.drop(target, axis=1),
                                         y_train=df_train[target], 
                                         y_test=df_test[target], 
                                         cat_features=cat_features,
                                        )
    
    def objective(search_space: dict):
        with mlflow.start_run():

            mlflow.set_tag("model", "CatBoostRegressor")
            mlflow.log_params({**search_space})
    
            model = CatBoostRegressor(**search_space)      
            model.fit(X=train_pool, eval_set=eval_pool, **fit_params)  
            
            metrics = metrics_compute(df_test[target], call_predictions(model, eval_pool))       
            mlflow.log_metrics(metrics)

            mlflow.catboost.log_model(model, artifact_path= "models_mlflow") 
            with open('catboost_model.pkl', 'wb') as f_out:
                pickle.dump(model, f_out)
            mlflow.log_artifact(local_path= "catboost_model.pkl", artifact_path= "model_pickle")

            return {'loss': model.get_best_score()['validation']['RMSE'], 'status': STATUS_OK}

    best_params = fmin(
        fn = objective,
        space = search_space,
        algo = tpe.suggest,
        max_evals = 2)
    
    hyperparams = space_eval(search_space, best_params)
    return hyperparams









