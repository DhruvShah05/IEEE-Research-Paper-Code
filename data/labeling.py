import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def label_by_threshold(returns: np.ndarray, threshold: float) -> np.ndarray:
    """
    Given a 1D array of future returns, labels them:
    0: Down (return < -threshold)
    1: Stationary (-threshold <= return <= threshold)
    2: Up (return > threshold)
    """
    labels = np.ones_like(returns, dtype=np.int64) # Default to Stationary (1)
    labels[returns > threshold] = 2 # Up
    labels[returns < -threshold] = 0 # Down
    return labels

def apply_horizon_labeling(df: pd.DataFrame, horizon: int, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes future returns for a given horizon and applies thresholding.
    Returns the feature matrix (X) and label vector (y), truncating the last `horizon` rows.
    """
    best_bid_col = '2'
    best_ask_col = '22'
    
    mid_price = (df[best_bid_col] + df[best_ask_col]) / 2.0
    
    # Future mid-price after `horizon` events
    future_mid_price = mid_price.shift(-horizon)
    
    # Fractional return
    returns = (future_mid_price - mid_price) / mid_price
    
    # Drop the last `horizon` rows where future returns are NaN
    df_valid = df.iloc[:-horizon].copy()
    returns_valid = returns.iloc[:-horizon].values
    
    labels = label_by_threshold(returns_valid, threshold)
    
    # Extract just the 40 feature columns
    feature_cols = [str(i) for i in range(2, 42)]
    X = df_valid[feature_cols].values
    
    return X, labels

def run_threshold_sweep(df: pd.DataFrame, horizons: list, thresholds: list) -> pd.DataFrame:
    """
    Sweeps horizon x threshold combinations and computes class balances.
    Saves to results/threshold_sweep.csv
    """
    results = []
    
    best_bid_col = '2'
    best_ask_col = '22'
    mid_price = (df[best_bid_col] + df[best_ask_col]) / 2.0
    
    for h in horizons:
        future_mid_price = mid_price.shift(-h)
        returns = (future_mid_price - mid_price) / mid_price
        returns_valid = returns.iloc[:-h].values
        
        for t in thresholds:
            labels = label_by_threshold(returns_valid, t)
            unique, counts = np.unique(labels, return_counts=True)
            counts_dict = dict(zip(unique, counts))
            
            total = len(labels)
            p_down = counts_dict.get(0, 0) / total * 100
            p_stat = counts_dict.get(1, 0) / total * 100
            p_up = counts_dict.get(2, 0) / total * 100
            
            results.append({
                'horizon_events': h,
                'threshold': t,
                'pct_down': p_down,
                'pct_stationary': p_stat,
                'pct_up': p_up,
                'total_samples': total
            })
            
    res_df = pd.DataFrame(results)
    return res_df
