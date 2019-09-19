import subprocess, sys, os
from core import Submission

sys.stdout = open(os.devnull, 'w')  # do NOT remove this code, place logic & imports below this line

import pandas as pd
import numpy as np
import ta

from collections import deque
from joblib import load
from sklearn.linear_model import LassoLarsCV
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from rolling import RollingWindowSplit

scaler = load('scaler10.joblib')
lasso = load('lasso_limited.joblib')

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
        X = data.values
        X_scaled = scaler.transform(X)
        sigmoid = (1/(1+np.exp(-0.22*lasso.predict(np.atleast_2d(X_pca))))-0.5)*20
        return sigmoid[0]

    """
    run_submission() will iteratively fetch the next row of data in the format
       specified (get_next_data_as_string, get_next_data_as_list, get_next_data_as_numpy_array)
       for every prediction submitted to self.submit_prediction()
    """
    def run_submission(self):

        self.debug_print("Use the print function `self.debug_print(...)` for debugging purposes, do NOT use the default `print(...)`")
        massive_df, resampled_df = pd.DataFrame(), pd.DataFrame()

        # only need last 15
        def create_limited_features(df):
            df['midRate'] = (df.askRate0 + df.bidRate0) / 2
            df['totalBidVol1'] = df.bidSize0 + df.bidSize1
            df['totalAskVol1'] = df.askSize0 + df.askSize1
            df['bidAskVol'] = df.askSize0 + df.bidSize0
            for i in range(2,5):
                df['totalBidVol' + str(i)] = df['totalBidVol' + str(i-1)] + df['bidSize' + str(i)]
                df['totalAskVol' + str(i)] = df['totalAskVol' + str(i-1)] + df['askSize' + str(i)]
            df['bidAskRatio4'] = df['totalBidVol' + str(4)] / df['totalAskVol' + str(4)]
            # df['OIR'] = (df.bidSize0 - df.askSize0)/(df.bidSize0 + df.askSize0)
            # def addTimeFeatures(i):
            #     df['daskRate' + str(i)] = df.askRate0.diff(i)
            #     df['dbidRate' + str(i)] = df.bidRate0.diff(i)
            # for i in range(6,11):
            #     addTimeFeatures(i)
            df['time'] = pd.date_range(start='1/1/1970', periods=2999999, freq='T')
            df.set_index('time', inplace=True)
            df_mid = df.midRate.resample('15Min').ohlc()
            df_mid['vol'] = df.bidAskVol.resample('15Min').mean()
            df_mid['volume_adi'] = ta.volume.acc_dist_index(df_mid.high, df_mid.low, df_mid.close, df_mid.vol, fillna=True)
            df_mid['others_dlr'] = ta.others.daily_log_return(df_mid.close, fillna=True)
            df = df.join(df_mid[['volume_adi', 'others_dlr']]).ffill().astype('float32')
            df.fillna(0, inplace=True)
            return df

        # Append new row to massive_df
        def append_to_df(massive_df, row):
            try: row.index = [massive_df.index[-1] + timedelta(minutes=1)]
            except IndexError: row.index = [datetime(1970,1,1)]
            return massive_df.append(row, sort=False)

        # Time series features
        def add_time_features(df):
            df['OIR'] = (df.bidSize0 - df.askSize0)/(df.bidSize0 + df.askSize0)
            return df

        # Manual time features
        def add_manual_time_features(df):
            def addTimeFeatures(i):
                df['daskRate' + str(i)] = df.askRate0.diff(i)
                df['dbidRate' + str(i)] = df.bidRate0.diff(i)
            for i in range(6,11):
                addTimeFeatures(i)
            df.fillna(0, inplace=True) # necessary
            return df[-15:]

        # Create time-based features + standardise
        def add_resample_features(massive_df, resampled_df):
            leftovers = len(massive_df) % 15
            a = pd.DataFrame()
            def pad_history():
                full_resampled = resampled_df.append(df_mid, sort=False)
                a = pd.DataFrame([full_resampled.iloc[0] for j in range(30+1-len(full_resampled))])
                a = a.append(full_resampled, sort=False)
                a.index = pd.date_range(start=df_mid.index[-1], periods=len(a), freq='-15Min').sort_values()
                df_mid_ta = ta.add_all_ta_features(a, "open", "high", "low", "close", "vol", fillna=True)

                # need to find a way to merge below (correct) and above (wrong)
                df_mid['volume_adi'] = ta.volume.acc_dist_index(df_mid.high, df_mid.low, df_mid.close, df_mid.vol, fillna=True)
                df_mid['others_dlr'] = ta.others.daily_log_return(df_mid.close, fillna=True)
                df = df.join(df_mid[['volume_adi', 'others_dlr']]).ffill().astype('float32')
                df.fillna(0, inplace=True)

                return df_mid_ta
            if leftovers == 0:
                df_mid = massive_df.tail(15).midRate.resample('15Min').ohlc()
                df_mid['vol'] = massive_df.tail(15).bidAskVol.resample('15Min').mean()
                df_mid_ta = pad_history()
                resampled_df = resampled_df.tail(30).append(df_mid_ta, sort=False)
            else:
                df_mid = massive_df.tail(leftovers).midRate.resample('15Min').ohlc()
                df_mid['vol'] = massive_df.tail(leftovers).bidAskVol.resample('15Min').mean()
                df_mid_ta = pad_history()

            if 'momentum_ao' in massive_df.columns:
                massive_df.update(df_mid_ta)
            else: massive_df = massive_df.join(df_mid_ta.tail(1))
            massive_df = massive_df.ffill().astype('float32')
            return massive_df, resampled_df

        while(True):
            """
            NOTE: Only one of (get_next_data_as_string, get_next_data_as_list, get_next_data_as_numpy_array) can be used
            to get the row of data, please refer to the `OVERVIEW OF DATA` section above.

            Uncomment the one that will be used, and comment the others.
            """
            # Pull data row
            # data = self.get_next_data_as_list()
            # base_row = self.get_next_data_as_numpy_array()
            base_row = self.get_next_data_as_string()
            df = [float(x) if x else 0 for x in base_row.split(',')]
            row = compute_cross_sectional(df)
            massive_df = append_to_df(massive_df, row)
            massive_df = add_time_features(massive_df)
            massive_df = add_manual_time_features(massive_df)
            massive_df, resampled_df = add_resample_features(massive_df, resampled_df)
            data = pd.DataFrame([massive_df.iloc[-1]])
            prediction = self.get_prediction(data)

            """
            submit_prediction(prediction) MUST be used to submit your prediction for the current row of data
            """
            self.submit_prediction(prediction)


if __name__ == "__main__":
    MySubmission()
