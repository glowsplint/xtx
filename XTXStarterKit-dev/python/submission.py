import subprocess, sys, os
from core import Submission

sys.stdout = open(os.devnull, 'w')  # do NOT remove this code, place logic & imports below this line

import pandas as pd
import numpy as np
import ta
import lightgbm as lgb

from joblib import load

lgbm = lgb.Booster(model_file='booster.txt')
lasso = load('lasso.joblib')
ridge = load('ridge.joblib')

massive_df_length = 1565

bidSizeList = ['bidSize' + str(i) for i in range(0,15)]
askSizeList = ['askSize' + str(i) for i in range(0,15)]
bidRateList = ['bidRate' + str(i) for i in range(0,15)]
askRateList = ['askRate' + str(i) for i in range(0,15)]

"""
PYTHON submission

Implement the model below

##################################################### OVERVIEW ######################################################

1. Use get_next_data_as_string() OR get_next_data_as_list() OR get_next_data_as_numpy_array() to recieve the next row of data
2. Use the predict method to write the prediction logic, and return a float representing your prediction
3. Submit a prediction using self.submit_prediction(...)

################################################# OVERVIEW OF DATA ##################################################

1. get_next_data_as_string() accepts no input and returns a String representing a row of data extracted from data.csv
     Example output: '1619.5,1620.0,1621.0,,,,,,,,,,,,,1.0,10.0,24.0,,,,,,,,,,,,,1615.0,1614.0,1613.0,1612.0,1611.0,
     1610.0,1607.0,1606.0,1605.0,1604.0,1603.0,1602.0,1601.5,1601.0,1600.0,7.0,10.0,1.0,10.0,20.0,3.0,20.0,27.0,11.0,
     14.0,35.0,10.0,1.0,10.0,13.0'

2. get_next_data_as_list() accepts no input and returns a List representing a row of data extracted from data.csv,
   missing data is represented as NaN (math.nan)
     Example output: [1619.5, 1620.0, 1621.0, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, 1.0, 10.0,
     24.0, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, 1615.0, 1614.0, 1613.0, 1612.0, 1611.0, 1610.0,
     1607.0, 1606.0, 1605.0, 1604.0, 1603.0, 1602.0, 1601.5, 1601.0, 1600.0, 7.0, 10.0, 1.0, 10.0, 20.0, 3.0, 20.0,
     27.0, 11.0, 14.0, 35.0, 10.0, 1.0, 10.0, 13.0]

3. get_next_data_as_numpy_array() accepts no input and returns a Numpy Array representing a row of data extracted from
   data.csv, missing data is represented as NaN (math.nan)
   Example output: [1.6195e+03 1.6200e+03 1.6210e+03 nan nan nan nan nan nan nan nan nan nan nan nan 1.0000e+00
    1.0000e+01 2.4000e+01 nan nan nan nan nan nan nan nan nan nan nan nan 1.6150e+03 1.6140e+03 1.6130e+03 1.6120e+03
     1.6110e+03 1.6100e+03 1.6070e+03 1.6060e+03 1.6050e+03 1.6040e+03 1.6030e+03 1.6020e+03 1.6015e+03 1.6010e+03
      1.6000e+03 7.0000e+00 1.0000e+01 1.0000e+00 1.0000e+01 2.0000e+01 3.0000e+00 2.0000e+01 2.7000e+01 1.1000e+01
       1.4000e+01 3.5000e+01 1.0000e+01 1.0000e+00 1.0000e+01 1.3000e+01]

##################################################### IMPORTANT ######################################################

1. One of the methods get_next_data_as_string(), get_next_data_as_list(), or get_next_data_as_numpy_array() MUST be used and
   _prediction(pred) MUST be used to submit below in the solution implementation for the submission to work correctly.
2. get_next_data_as_string(), get_next_data_as_list(), or get_next_data_as_numpy_array() CANNOT be called more then once in a
   row without calling self.submit_prediction(pred).
3. In order to debug by printing do NOT call the default method `print(...)`, rather call self.debug_print(...)

"""


# class MySubmission is the class that you will need to implement
class MySubmission(Submission):

    """
    get_prediction(data) expects a row of data from data.csv as input and should return a float that represents a
       prediction for the supplied row of data
    """
    def get_prediction(self, data):
        X = data.replace([np.inf, -np.inf], np.nan).values
        lgbm_predictions = np.clip(lgbm.predict(np.atleast_2d(X)), -5, 5)[0]
        X[np.isnan(X)] = 100
        lasso_predictions = np.clip(lasso.predict(np.atleast_2d(X)), -5, 5)[0]
        return (np.vstack([lgbm_predictions, lasso_predictions]).T @ ridge.coef_)[0]

    """
    run_submission() will iteratively fetch the next row of data in the format
       specified (get_next_data_as_string, get_next_data_as_list, get_next_data_as_numpy_array)
       for every prediction submitted to self.submit_prediction()
    """
    def run_submission(self):

        self.debug_print("Use the print function `self.debug_print(...)` for debugging purposes, do NOT use the default `print(...)`")
        massive_df = pd.DataFrame()

        # only need last 15
        def create_limited_features(df):
            df = pd.DataFrame([df])
            df.columns = [*askRateList, *askSizeList, *bidRateList, *bidSizeList]
            df['midRate'] = (df.bidRate0 + df.askRate0) / 2 # necessary for ohlc
            df['OIR'] = (df.bidSize0 - df.askSize0)/(df.bidSize0 + df.askSize0)
            df['totalAskVol'] = df[askSizeList].sum(axis=1)
            df['totalBidVol'] = df[bidSizeList].sum(axis=1)
            df['OIR_total'] = (df.totalBidVol - df.totalAskVol)/(df.totalBidVol + df.totalAskVol)

            df['spread'] = df.askRate0 - df.bidRate0
            df['vwaBid'] = np.einsum('ij,ji->i', df[bidRateList], df[bidSizeList].T) / df[bidSizeList].sum(axis=1)
            df['vwaAsk'] = np.einsum('ij,ji->i', df[askRateList], df[askSizeList].T) / df[askSizeList].sum(axis=1)
            df['vwaBidDMid'] = df.midRate - df.vwaBid
            df['vwaAskDMid'] = df.vwaAsk - df.midRate
            df['diff_vwaBidAskDMid'] = df.vwaAskDMid - df.vwaBidDMid

            shift_val = 1
            b1, a1 = (df.bidRate0 < df.bidRate0.shift(shift_val)), (df.askRate0 < df.askRate0.shift(shift_val))
            b2, a2 = (df.bidRate0 == df.bidRate0.shift(shift_val)), (df.askRate0 == df.askRate0.shift(shift_val))
            valsB, valsA = [0, (df.bidSize0 - df.bidSize0.shift(shift_val))], [0, (df.askSize0 - df.askSize0.shift(shift_val))]
            defaultB, defaultA = df.bidSize0, df.askSize0
            df['deltaVBid'] = np.select([b1,b2], valsB, default=defaultB)
            df['deltaVAsk'] = np.select([a1,a2], valsA, default=defaultA)
            df['VOI'] = df.deltaVBid - df.deltaVAsk
            return df

        def append_to_df(massive_df, row):
            return massive_df.append(row, sort=False)

        def add_time_features(df, massive_df_length):
            tsi = [87, 261, 348, 435, 522]
            trix = [87, 174, 348, 435, 522]
            for t in tsi:        df['tsi' + str(t)] = ta.momentum.tsi(df.midRate, s=t, r=2.25*t)
            for t in trix:       df['trix' + str(t)] = ta.trend.trix(df.midRate, n=t)
            return df[-massive_df_length:]

        while(True):
            """
            NOTE: Only one of (get_next_data_as_string, get_next_data_as_list, get_next_data_as_numpy_array) can be used
            to get the row of data, please refer to the `OVERVIEW OF DATA` section above.

            Uncomment the one that will be used, and comment the others.
            """

            base_row = self.get_next_data_as_string()
            df = [float(x) if x else 0 for x in base_row.split(',')]
            row = create_limited_features(df)
            massive_df = append_to_df(massive_df, row)
            massive_df = add_time_features(massive_df, massive_df_length)
            data = pd.DataFrame([massive_df.iloc[-1]])
            prediction = self.get_prediction(data)

            """
            submit_prediction(prediction) MUST be used to submit your prediction for the current row of data
            """
            self.submit_prediction(prediction)

if __name__ == "__main__":
    MySubmission()
