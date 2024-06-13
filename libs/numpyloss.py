import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
def MSE(y,y_pred):
    # return np.mean((y-y_pred)**2)
    return mean_squared_error(y,y_pred)
def MAE(y,y_pred):
    # return np.mean(np.abs(y-y_pred))
    return mean_absolute_error(y,y_pred)
def RMSE(y,y_pred):
    return np.sqrt(MSE(y,y_pred))
def MAPE(y,y_pred):
    return np.mean(np.abs((y_pred - y) / y))
    #return np.mean((y-y_pred)/y)
def WAPE(y,y_pred):
    # y = list(y[:, 0])
    #y_pred = list(y_pred[:, 0])
    diff=[abs(y[i]-y_pred[i]) for i in range(len(y))]
    return np.sum(diff)/np.sum(y)
