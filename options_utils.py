import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Dataset
import shap
import gdown
from sklearn.preprocessing import StandardScaler
import gc

import pandas as pd
import numpy as np
from scipy.stats import norm
import datetime
from collections import defaultdict
# Import the yfinance. If you get module not found error the run !pip install yfinance from your Jupyter notebook
import yfinance as yf
import pickle
import matplotlib.pyplot as plt
import itertools
from scipy.optimize import curve_fit
import statistics
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler
import warnings
from scipy.stats import invweibull, genextreme
import math as ma

# First import everthing you need
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D

from typing import Tuple, Union
import gdown
warnings.filterwarnings("ignore")

PUT_RIGHT = 0
CALL_RIGHT = 1

# Define our MLP model
class SimpleNN(nn.Module):
    def __init__(self, input_size, output_size):
        """
        IMPORTANT:
        if you add more fields, be sure to modify the "save_model" and "load_model" fields below.
        Fields that are not NN parameters are not saved with torch.save_state_dict or torch.load_state_dict
        """
        super(SimpleNN, self).__init__()
        hidden_size = 4 #1 + int(input_size//2)

        # Define the first layer
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        # Define the output layer
        self.fco = nn.Linear(hidden_size, output_size)

        self.mean = None
        self.std = None

    def fit_feature_scaler(self, train_features):
        # Don't want to fit the scaler twice!
        assert self.mean is None and self.std is None

        assert isinstance(train_features, torch.Tensor)

        self.mean = torch.mean(train_features, dim=0)
        self.std = torch.std(train_features, dim=0)

        # If a feature has no standard deviation,
        #  Let it be recorded as -1 to prevent divide-by-zero behavior
        # The feature will always be demeaned, so when passed into the NN, it will
        #  still always be zero.
        self.std[self.std==0] = -1

    def scale_features(self, features):
        return (features-self.mean) / self.std

    def forward(self, x):

        # If scaler is None, that means we want raw (unscaled) features.
        if self.mean is not None:
            x = self.scale_features(x)

        # Pass the input through the first layer, then apply ReLU activation function
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        # Pass through the output layer
        x = self.fco(x)
        return x

def assign_weights(df, right, moneyness, uniform=False):
    """
    uniform -- make all weights equal to one, irrespective of quantity of puts, calls, itm, otm, etc.
    """
    #df.drop(columns=["Unnamed: 0"], inplace=True)

    # df_p = df[[col for col in df.columns if "_call" not in col]].copy()
    # df_p.rename(columns={"mid_px_put" : "mid_px", "BSPrice_put": "BSPrice", 'spread_put': 'spread'}, inplace=True)

    # df_c = df[[col for col in df.columns if "_put" not in col]].copy()
    # df_c.rename(columns={"mid_px_call" : "mid_px", "BSPrice_call": "BSPrice", 'spread_call': 'spread'}, inplace=True)

    df_p = df[df["cp_flag"] == PUT_RIGHT].copy()
    df_p = df_p.rename(columns={"cp_flag": "right"})

    df_c = df[df["cp_flag"] == CALL_RIGHT].copy()
    df_c = df_c.rename(columns={"cp_flag": "right"})


    # print("columns of puts and calls extracted from merged DF:")
    # print(df_p.columns)
    # print(df_c.columns)

    # Free up space
    del df
    gc.collect()

    # df_p["right"] = PUT_RIGHT
    # df_c["right"] = CALL_RIGHT

    if right == "put":
        df = df_p
        df["right_wt"] = 1
    elif right == "call":
        df = df_c
        df["right_wt"] = 1
    elif right == "all":
        call_qty = len(df_c)
        put_qty = len(df_p)

        call_wt = (call_qty + put_qty) / (2*call_qty)
        put_wt = (call_qty + put_qty) / (2*put_qty)

        df_c["right_wt"] = call_wt
        df_p["right_wt"] = put_wt

        df = pd.concat([df_c, df_p], axis=0, ignore_index=True)
    else:
        print("Unrecognized Right")
        quit()

    # Free up space
    del df_c, df_p
    gc.collect()

    # Assign weights to dataset based on desired moneyness
    # "uok" represents "underlying over strike" (strike=K)
    # For OTM put options -- underlying must be GREATER than strike (exercising not desirable!)
    # OTM put options -- uok is > 1
    # ITM put options -- uok is <=1

    # ITM call options -- uok is >=1
    # OTM call options -- uok is < 1
    df["uok"] = df["SPX"] / df["strike_price"]

    df_itm = df[((df["uok"] <= 1) & (df["right"] == PUT_RIGHT)) | ((df["uok"] >= 1) & (df["right"] == CALL_RIGHT))].copy()
    df_otm = df[((df["uok"] > 1) & (df["right"] == PUT_RIGHT)) | ((df["uok"] < 1) & (df["right"] == CALL_RIGHT))].copy()
    if moneyness=="itm":
        df = df_itm
        df["moneyness_wt"] = 1
    elif moneyness=="otm":
        df = df_otm
        df["moneyness_wt"] = 1
    elif moneyness=="all":

        itm_qty = len(df_itm)
        otm_qty = len(df_otm)

        itm_wt = (itm_qty + otm_qty) / (2*itm_qty)
        otm_wt = (itm_qty + otm_qty) / (2*otm_qty)

        itm_condiiton = (df["uok"] <= 1) & (df["right"] == PUT_RIGHT) | (df["uok"] >= 1) & (df["right"] == CALL_RIGHT)
        df["moneyness_wt"] = np.where(itm_condiiton, itm_wt, otm_wt)
        #df["moneyness_wt"] = df["uok", "right"].apply(lambda uok, right: itm_wt if ((uok <=1 and right == PUT_RIGHT) or (uok >=1 and right==CALL_RIGHT)) else otm_wt)

    else:
        print("Unrecognized Moneyness")
        quit()

    # Free up space
    del df_itm, df_otm
    gc.collect()

    df["wt"] = df["moneyness_wt"] * df["right_wt"]
    df.drop(columns=["moneyness_wt", "right_wt"], inplace=True)

    #print(df)
    avgwt = df["wt"].mean()
    print(f"Average Weight: {avgwt}")

    if uniform:
        print(f"Assigned UNIFORM weights to a Dataset of length {len(df)}")
        df["wt"] = 1
        return df

    else:
        assert 0.95 < avgwt and avgwt < 1.05
        print(f"Assigned NONUNIFORM weights to a Dataset of length {len(df)}")
        return df

def split_data(merged, method, trainstart, teststart, testend):
    # Drop the 'Unnamed: 0' column
    # merged.drop(columns=["Unnamed: 0"], inplace=True)

    if method == "optionid":
        # Shuffle the unique option IDs
        oids = merged['optionid'].unique()
        random.shuffle(oids)
        l = len(oids)
        print(f"Number of OptionIDs: {l}")

        # 80% train
        train_ratio = 0.8

        cutoff = int(np.ceil(train_ratio * l))
        train_oids = oids[:cutoff]
        test_oids = oids[cutoff:]

        # Split the data into training and testing sets
        train = merged[merged["optionid"].isin(train_oids)]
        test = merged[merged["optionid"].isin(test_oids)]

    elif method == "temporal":

        # Step 1: Check data type
        #print("\nData type before conversion:", merged["date"].dtype)

        # Step 2: Convert to datetime
        merged["date"] = pd.to_datetime(merged["date"], errors='coerce')
        num_nat = merged['date'].isna().sum()

        assert num_nat == 0

        #print(f"\nNumber of dates converted to NaT: {num_nat}")

        # Step 3: Verify conversion
        #print("\nData type after conversion:", merged["date"].dtype)
        #print("\nDataFrame after conversion:")
        #print(merged)

        #print(len(merged))
        # Step 4: Filter based on date

        # test_csv["date"] = pd.to_datetime(test_csv["date"], errors='coerce')
        # teststart = pd.Timestamp(f'{teststart}-01-01')
        # testend = pd.Timestamp(f'{testend}-12-31')
        teststart = pd.Timestamp(f'{teststart}-01')
        testend = pd.Timestamp(f'{testend}-01')
        test = merged[(merged['date'] >= teststart) & (merged['date'] <= testend)]
        train = merged[(merged['date'] >= trainstart) & (merged['date'] < teststart)]


    else:
        assert False

    print(f"Train Length: {len(train)}")
    print(f"Test Length: {len(test)}")
    print('Percent Train:')
    print(100* len(train) / (len(train) + len(test)))
    return train, test

def load_model(model, fn):
  state = torch.load(fn)
  model.load_state_dict(state["state_dict"])
  model.mean = state["mean"]
  model.std = state["std"]

  return model

class FinancialDataset(Dataset):
    def __init__(self, features, weights, labels):
        self.features = torch.tensor(features.values, dtype=torch.float32)
        self.labels = torch.tensor(labels.values, dtype=torch.float32).view(-1, 1)
        self.weights = torch.tensor(weights.values, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.weights[idx], self.labels[idx]
