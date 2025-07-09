import numpy as np
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import datetime as dt
import pandas as pd
import yfinance as yf
import sklearn as sk
from sklearn import gaussian_process as gp
from sklearn.gaussian_process.kernels import DotProduct, RBF, WhiteKernel

def SMA(arr, n=10):
    """Compute the moving average of an array."""
    res = [sum(arr[i-n:i])/n for i in range(n, len(arr))]
    std = [np.std(arr[i-n:i]) for i in range(n, len(arr))]
    return res, std 

def sparse(arr, n=10, s=2):
    """Compute the sparse average of an array."""
    l = len(arr)
    res = []
    avg_ind = []
    for i in range(0, l-s*n, n):
        res.append(float(arr[i]))
        avg_ind.append(i+n)
    return res, avg_ind

def ml_model(start_date='2021-01-01', end_date='2021-01-01', index = 'SPY'):
    # Get the stock data
    stock = yf.download(index, start=start_date, end=end_date)
    print(start_date, end_date)
    data = pd.DataFrame.to_numpy(stock['Close'])
    data = data-data[0]
    days = np.atleast_2d(np.arange(len(data))).T

    #find the moving average
    day_avg = 15
    SMA_data, std_data = SMA(data, n=day_avg)

    # Downsample the data, s ensures there is unsampled data for prediction
    sparse_data, sparse_days = sparse(SMA_data, n=day_avg, s=4)
    sparse_std, _ = sparse(std_data, n=day_avg, s=4)
    sparse_days = np.atleast_2d(sparse_days).T

    # Create the Gaussian Process model
    kernel = 0.3*DotProduct()+0.3*RBF(length_scale=1e-2)  + 0.1*WhiteKernel(
        noise_level=1e-2, noise_level_bounds=(1e-10, 1e1))
    model = gp.GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=100)
    # Fit the model to the data
    model.fit(sparse_days, sparse_data)
    # Make predictions
    predictions = model.predict(days, return_std=True)
    return predictions[0], predictions[1]

