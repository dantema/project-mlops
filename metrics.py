import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from utils import array_convert

def metrics_compute(y_true: np.array, 
                    y_pred: np.array) -> dict:
    
    y_true, y_pred = array_convert(y_true), array_convert(y_pred)
        
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    metrics_dict = {"RMSE": rmse,
                   "MAE": mae, 
                   "R2": r2}
    
    return metrics_dict