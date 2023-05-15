from catboost import Pool
import mlflow
import pandas as pd
import numpy as np

car = {
    "" : [0],
    "model" : ["Focus"],
    "year" : [2018],
    "price": [12000],
    "transmission" : ["Manual"],
    "mileage" : [48000],
    "fuelType": ["Petrol"],
    "tax": [145],
    "mpg" : [61.4],
    "engineSize": [1.0]
}

cat_features = ["model", "transmission", "fuelType"]

def prepare_features(data):
    data = pd.DataFrame.from_dict(data)#.reset_index(drop = True)
    print(data)
    return Pool(data.drop(["price"], axis = 1), cat_features = cat_features)

def load_model():
    path ="file:///D:/Projects/mlflow_logs/mlruns/325982656137906787/8c71bce14846420c8d852bd422762a03/artifacts/models_mlflow"
    model = mlflow.pyfunc.load_model(path)
    return model

def test_predict():
    model = load_model()
    features = prepare_features(car)
    prediction = model.predict(features)
    assert prediction[0] > 0
    return prediction[0]

def test_return_type():
    model = load_model()
    features = prepare_features(car)
    prediction = model.predict(features)
    print(type(prediction[0]))
    assert type(prediction[0]) == np.float64

if __name__ == "__main__":
    prediction = test_predict()
    test_return_type()