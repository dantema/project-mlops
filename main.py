from prefect import flow
from prefect.task_runners import SequentialTaskRunner
from operators import read_prepare_data_operator, train_model_operator

import mlflow 

MLFLOW_TRACKING_URI = "file:///D:/Projects/mlflow_logs/mlruns"
MLFLOW_EXPERIMENT = "ford_prices_test"

@flow(task_runner = SequentialTaskRunner())    
def main(path: str = "./data/ford.csv"):

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    df_train, df_test = read_prepare_data_operator(path)
    best_params = train_model_operator(df_train, df_test)

if __name__ == "__main__":
    main()
