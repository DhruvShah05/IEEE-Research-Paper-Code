import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

class TrainOnlyScaler:
    """
    A scaler wrapper that ensures `.fit()` is only ever called once, 
    guarding against accidental fitting on validation or test sets (data leakage).
    """
    def __init__(self, use_zscore=True):
        self.scaler = StandardScaler() if use_zscore else None
        self._is_fit = False
        
    def fit(self, X: np.ndarray):
        if self._is_fit:
            raise RuntimeError("TrainOnlyScaler.fit() called more than once! You are likely leaking test data.")
        if self.scaler is not None:
            self.scaler.fit(X)
        self._is_fit = True
        return self
        
    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fit:
            raise RuntimeError("TrainOnlyScaler.transform() called before fit()!")
        if self.scaler is not None:
            return self.scaler.transform(X)
        return X

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)

def to_relative_price(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts raw crypto price levels to relative prices centered around the mid-price.
    Volumes are retained as-is.
    
    Expected df format:
    Col '2' to '21': 10 bid levels (price, vol, price, vol...)
    Col '22' to '41': 10 ask levels (price, vol, price, vol...)
    """
    res = df.copy()
    
    best_bid_col = '2'
    best_ask_col = '22'
    
    mid_price = (df[best_bid_col] + df[best_ask_col]) / 2.0
    
    # Bid columns: even index from 2 to 20 are prices
    for i in range(2, 22, 2):
        col = str(i)
        res[col] = df[col] - mid_price
        
    # Ask columns: even index from 22 to 40 are prices
    for i in range(22, 42, 2):
        col = str(i)
        res[col] = df[col] - mid_price
        
    return res
