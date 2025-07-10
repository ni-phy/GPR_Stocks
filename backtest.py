import numpy as np
from gp_for_stocks import ml_model
import pandas as pd
from datetime import datetime
import yfinance as yf

dates = pd.date_range(start='2023-01-01',end='2025-07-09', freq='1ME')

mid_point = len(dates)//2 #Train model on the prev year before testing

## Strategy will be: predict the next week, if it's up, hold
## if it's predicted to be down, then sell

pnl = 0
index = 'AAPL'

avail = [mid_point]*mid_point

for i in range(mid_point, len(dates)):

    p0, var = ml_model(dates[0], dates[i-1], index)
    actual = yf.download(index, start=dates[0], end=dates[i])
    ac_data = pd.DataFrame.to_numpy(actual['Close'])

    if p0[-1]-p0[-20]>0:
        # pnl = pnl+ac_data[-1]- ac_data[-20]
        avail[i-mid_point] = avail[i-mid_point-1] - 1
    if p0[-1]-p0[-20]<=0:
       avail[i-mid_point] = avail[i-mid_point-1] + 1
       pnl = pnl-(ac_data[-1]- ac_data[-20])
    
init_fund_alloc = mid_point*ac_data[0]
final_val = pnl + (mid_point-avail[-1])*ac_data[-1] + avail[-1]*ac_data[0]

print('Initial funds allocated:', init_fund_alloc)
print('Final value:', final_val)
print('PnL:', final_val - init_fund_alloc)
print('PnL (%):', 100*(final_val - init_fund_alloc)/init_fund_alloc)