import pickle
import pandas as pd
from catboost import Pool
from flask import Flask, jsonify, request

cat_features = ["model", "transmission", "fuelType"]
model_path = "../catboost_model.pkl"

def load_model(path: str):
    with open(path, "rb") as f_in:
        model = pickle.load(f_in)
    return model


def prepare_features(data: dict) -> Pool:
    df = pd.DataFrame.from_dict(data)
    return Pool(df, cat_features=cat_features)


def get_prediction(data: dict) -> float:
    pool = prepare_features(data)
    model = load_model(model_path)
    result = model.predict(pool)
    return float(result[0])


app = Flask('price-prediction')


@app.route('/predict', methods =['POST'])
def predict_endpoint():
    car = request.get_json()
    prediction = get_prediction(car)
    result = {
        "price": prediction
    }
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug = True, host = '0.0.0.0', port = 9696)
 