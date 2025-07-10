import numpy as np
from gp_for_stocks import ml_model
import pandas as pd
from datetime import datetime
import yfinance as yf
import matplotlib.pyplot as plt

dates = pd.date_range(start='2024-01-01',end='2025-07-09', freq='1w')

mid_point = len(dates)//2 #Train model on the prev year before testing

## Strategy will be: predict the next week, if it's up, hold
## if it's predicted to be down, then sell

pnl = 0
index = 'AAPL'

# avail = [mid_point]*mid_point

accuracy = [False] * (len(dates) - mid_point)

for i in range(mid_point, len(dates)):
    
    ## Getting the data for this month
    real_data = yf.download(index, start=dates[0], end=dates[i])
    re_data = pd.DataFrame.to_numpy(real_data['Close'])
    re_data = re_data-re_data[0]
    days = np.atleast_2d(np.arange(len(re_data))).T

    ## Training on up to the month before
    p0, std_ = ml_model(dates[0], dates[i-1], days, index)

    # Test accuracy 
    diff = p0[-1] - re_data[-1]
    print(p0[-1], re_data[-1], std_[-1], diff)
    if abs(diff) <= std_[-1]:
        accuracy[i-mid_point] = True
    
    # plt.figure(figsize=(12, 6))
    # plt.plot(days, re_data, 'r.', markersize=10, label='Observed Data')
    # plt.plot(days, p0, 'b-', label='GP Mean')
    # plt.fill_between(np.arange(len(days)), p0 - std_, p0 + std_, alpha=0.2, color='blue', label='95% Confidence Interval')
    # plt.title(f'Gaussian Process Regression for {index}')
    # plt.xlabel('Date')
    # plt.ylabel('Stock Price')
    # plt.legend()
    # # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    # # plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    # plt.gcf().autofmt_xdate() 
    # plt.show()

    # ## If we expect positive change, we buy
    # if p0[-1]-p0[-5]>0:
    #     avail[i-mid_point] = avail[i-mid_point-1] - 1
    
    # ## If we expect negative change we sell, then we adjust pnl
    # if p0[-1]-p0[-5]<=0:
    #    avail[i-mid_point] = avail[i-mid_point-1] + 1
    #    pnl = pnl-(re_data[-1]- re_data[-20])

print('Prediction within 1std for one week:', 
      sum(accuracy)/(len(dates)-mid_point))

# # Initial funds assumed mid_point nums of available stock, this needs adjusting
# init_fund_alloc = mid_point*real_data[0] 

# ## Final funds are the pnl we kept in cash, the number of stocks we have
# ## and the number of available funds we didn't use
# final_val = pnl + (mid_point-avail[-1])*re_data[-1] + avail[-1]*re_data[0] 

# print('Initial funds allocated:', init_fund_alloc)
# print('Final value:', final_val)
# print('PnL:', final_val - init_fund_alloc)
# print('PnL (%):', 100*(final_val - init_fund_alloc)/init_fund_alloc)