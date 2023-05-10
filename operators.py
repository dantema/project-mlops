from data_preparation import read_prepare_data
from train import train_model
from prefect import task

@task 
def read_prepare_data_operator(*args, **kwargs):
    return read_prepare_data(*args, **kwargs)

@task 
def train_model_operator(*args, **kwargs):
    return train_model(*args, **kwargs)