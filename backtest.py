import numpy as np
from gp_for_stocks import ml_model
import pandas as pd
from datetime import datetime
import yfinance as yf

dates = pd.date_range(start='2023-01-01',end='2025-07-09', freq='1ME')

mid_point = len(dates)//2 #Train model on the prev year before testing

## Strategy will be: predict the next week, if it's up, hold
## if it's predicted to be down, then sell

pnl = 100
index = 'AAPL'

for i in range(mid_point, len(dates)):
    p0, var = ml_model(dates[0], dates[i], index)
    actual = yf.download(index, start=dates[0], end=dates[i])
    ac_data = pd.DataFrame.to_numpy(actual['Close'])
    print(p0[-1]-p0[-20], (ac_data[-1]- ac_data[-20])/ac_data[-20])

    if p0[-1]-p0[-20]>0:
        pnl = pnl*(1+(ac_data[-1]- ac_data[-20])/ac_data[-20])

print(pnl)