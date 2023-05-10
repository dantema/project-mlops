from prefect import flow
from prefect.task_runners import SequentialTaskRunner
from operators import read_prepare_data_operator, train_model_operator
from params import target_name, cat_features, search_space

import mlflow 

MLFLOW_TRACKING_URI = "file:///D:/Projects/mlflow_logs/mlruns"
MLFLOW_EXPERIMENT = "ford_prices_test"

@flow(task_runner = SequentialTaskRunner())    
def main(path: str = "./data/ford.csv"):

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    df_train, df_test = read_prepare_data_operator(path)
    best_params = train_model_operator(df_train= df_train,
                                       df_test= df_test, 
                                       target= target_name,
                                       cat_features= cat_features, 
                                       search_space= search_space)

if __name__ == "__main__":
    main()
