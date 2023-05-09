import pandas as pd
import numpy as np
import mlflow
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from catboost import CatBoostRegressor, Pool
from hyperopt import fmin, hp, tpe, Trials, space_eval, STATUS_OK
from hyperopt.pyll import scope

MLFLOW_TRACKING_URI = "file:///D:/Projects/mlflow_logs/mlruns"
MLFLOW_EXPERIMENT = "ford_prices_test"

#TODO delete numba

# define the search space for the hyperparameters
search_space = {'learning_rate': hp.choice('learning_rate', [0.05]),
                'iterations': hp.choice('iterations', [100]),
                'l2_leaf_reg': hp.choice('l2_leaf_reg', [0, 1, 2 ]),
                'depth': hp.choice('depth', [8]),
                'bootstrap_type' : hp.choice('bootstrap_type', ['Bayesian', 'Bernoulli'])}

def read_prepare_data(path: str) -> tuple[pd.DataFrame, pd.DataFrame]:

    df = pd.read_csv(path)

    df.loc[df["model"] == "Focus", "model"] = " Focus"
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

    return df_train, df_test

def array_convert(y) -> np.array:
    if isinstance(y, np.ndarray):
        return y
    return np.array(y)

def metrics_compute(y_true: np.array, y_pred: np.array) -> dict:
    
    y_true, y_pred = array_convert(y_true), array_convert(y_pred)
        
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    metrics_dict = {"RMSE": rmse,
                   "MAE": mae, 
                   "R2": r2}
    
    return metrics_dict

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


def train_model(df_train, df_test):

    cat_features = ["model", "transmission", "fuelType"]
    target = "price"    
    train_pool, eval_pool = create_pools(df_train.drop(target, axis=1),
                                         df_test.drop(target, axis=1),
                                         y_train=df_train[target], 
                                         y_test=df_test[target], 
                                         cat_features=cat_features,
                                        )
    
    def objective(search_space):
        with mlflow.start_run():
            # set a tag for easier classification and log the hyperparameters
            mlflow.set_tag("model", "CatBoostRegressor")
            params = {
                'random_seed': 1,
                'task_type': 'CPU',
                'thread_count': -1
                #'verbose': False
            }

            mlflow.log_params({**params, **search_space})
    
            model = CatBoostRegressor(**params, 
                                      **search_space, 
                                      allow_writing_files=False,
                                      silent=True)      
            model.fit(train_pool, eval_set=eval_pool, early_stopping_rounds=200, plot = False)  
            
            metrics = metrics_compute(df_test[target], call_predictions(model, eval_pool))       
            mlflow.log_metrics(metrics)
            mlflow.catboost.log_model(model, artifact_path= "models_mlflow") 
            with open('catboost_model.pkl', 'wb') as f_out:
                pickle.dump(model, f_out)
            mlflow.log_artifact(local_path= "catboost_model.pkl", artifact_path= "model")

            return {'loss': model.get_best_score()['validation']['RMSE'], 'status': STATUS_OK}


    best_params = fmin(
        fn = objective,
        space = search_space,
        algo = tpe.suggest,
        max_evals = 50)
    
    hyperparams = space_eval(search_space, best_params)

    return hyperparams
    
    
def main(path: str = "./data/ford.csv"):

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    df_train, df_test = read_prepare_data(path)
    params = train_model(df_train, df_test)

    print(params)

main()



