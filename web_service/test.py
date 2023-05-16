import requests

car = {
    "model" : ['Focus'],
    "year" : [2018],
    "transmission" : ['Manual'],
    "mileage" : [81000],
    "fuelType": ['Petrol'],
    "tax": [145],
    "mpg" : [61.4],
    "engineSize": [1.0]
}

url = "http://localhost:9696/predict"
response = requests.post(url, json = car)
print(response.json())
