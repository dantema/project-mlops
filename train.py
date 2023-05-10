import pandas as pd
import numpy as np
import mlflow
import pickle

from catboost import CatBoostRegressor, Pool
from hyperopt import fmin, space_eval, STATUS_OK

from params import fit_params, fmin_args
from metrics import metrics_compute

# TODO. Metrics, read-prepare data, utils, operator

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
                target: str,
                cat_features: list[str],
                search_space: dict
                ) -> dict:

    train_pool, eval_pool = create_pools(df_train.drop(target, axis=1),
                                         df_test.drop(target, axis=1),
                                         y_train=df_train[target], 
                                         y_test=df_test[target], 
                                         cat_features=cat_features,
                                        )
    
    def objective(search_space: dict):
        with mlflow.start_run():

            model = CatBoostRegressor(**search_space)      
            model.fit(X=train_pool, eval_set=eval_pool, **fit_params)             
            metrics = metrics_compute(df_test[target], 
                                      call_predictions(model, eval_pool))
            
            # logging
            with open('catboost_model.pkl', 'wb') as f_out:
                pickle.dump(model, f_out)
            mlflow.log_artifact(local_path= "catboost_model.pkl", 
                                artifact_path= "model_pickle")
            mlflow.log_metrics(metrics)
            mlflow.log_params({**search_space})
            mlflow.set_tag("model", type(model).__name__)
            mlflow.catboost.log_model(model, artifact_path= "models_mlflow") 

            return {'loss': model.get_best_score()['validation']['RMSE'], 
                    'status': STATUS_OK}

    best_params = fmin(fn = objective, **fmin_args)    
    hyperparams = space_eval(search_space, best_params)

    return hyperparams
