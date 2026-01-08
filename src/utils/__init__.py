"""Utility functions for pairs trading strategy."""

import random
import numpy as np
import pandas as pd
import torch
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings

warnings.filterwarnings("ignore")


def set_random_seeds(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device: str = "auto") -> torch.device:
    """Get the appropriate device for computation.
    
    Args:
        device: Device preference ("auto", "cpu", "cuda", "mps").
        
    Returns:
        torch.device: The selected device.
    """
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    else:
        return torch.device(device)


def validate_data_splits(
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
) -> None:
    """Validate that data splits are properly ordered in time.
    
    Args:
        train_start: Training data start date.
        train_end: Training data end date.
        test_start: Test data start date.
        test_end: Test data end date.
        
    Raises:
        ValueError: If data splits are not properly ordered.
    """
    train_start_dt = pd.to_datetime(train_start)
    train_end_dt = pd.to_datetime(train_end)
    test_start_dt = pd.to_datetime(test_start)
    test_end_dt = pd.to_datetime(test_end)
    
    if train_start_dt >= train_end_dt:
        raise ValueError("Training start date must be before training end date")
    
    if test_start_dt >= test_end_dt:
        raise ValueError("Test start date must be before test end date")
    
    if train_end_dt >= test_start_dt:
        raise ValueError("Training end date must be before test start date")


def calculate_returns(prices: pd.Series, method: str = "log") -> pd.Series:
    """Calculate returns from price series.
    
    Args:
        prices: Price series.
        method: Return calculation method ("log" or "simple").
        
    Returns:
        pd.Series: Returns series.
    """
    if method == "log":
        return np.log(prices / prices.shift(1))
    elif method == "simple":
        return (prices / prices.shift(1)) - 1
    else:
        raise ValueError("Method must be 'log' or 'simple'")


def calculate_volatility(returns: pd.Series, window: int = 252) -> pd.Series:
    """Calculate rolling volatility.
    
    Args:
        returns: Returns series.
        window: Rolling window size in days.
        
    Returns:
        pd.Series: Rolling volatility series.
    """
    return returns.rolling(window=window).std() * np.sqrt(252)


def calculate_sharpe_ratio(
    returns: pd.Series, 
    risk_free_rate: float = 0.02,
    annualization_factor: int = 252
) -> float:
    """Calculate Sharpe ratio.
    
    Args:
        returns: Returns series.
        risk_free_rate: Annual risk-free rate.
        annualization_factor: Days per year for annualization.
        
    Returns:
        float: Sharpe ratio.
    """
    excess_returns = returns - risk_free_rate / annualization_factor
    return excess_returns.mean() / returns.std() * np.sqrt(annualization_factor)


def calculate_max_drawdown(returns: pd.Series) -> Tuple[float, pd.Timestamp, pd.Timestamp]:
    """Calculate maximum drawdown.
    
    Args:
        returns: Returns series.
        
    Returns:
        Tuple of (max_drawdown, peak_date, trough_date).
    """
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    
    max_dd = drawdown.min()
    trough_date = drawdown.idxmin()
    peak_date = running_max.loc[:trough_date].idxmax()
    
    return max_dd, peak_date, trough_date


def calculate_half_life(spread: pd.Series) -> float:
    """Calculate half-life of mean reversion for a spread series.
    
    Args:
        spread: Spread series.
        
    Returns:
        float: Half-life in days.
    """
    spread_lag = spread.shift(1)
    spread_diff = spread - spread_lag
    
    # Remove NaN values
    valid_idx = ~(spread_lag.isna() | spread_diff.isna())
    spread_lag_clean = spread_lag[valid_idx]
    spread_diff_clean = spread_diff[valid_idx]
    
    if len(spread_lag_clean) < 2:
        return np.nan
    
    # OLS regression: spread_diff = alpha + beta * spread_lag
    X = sm.add_constant(spread_lag_clean)
    model = sm.OLS(spread_diff_clean, X).fit()
    
    beta = model.params[1]
    if beta >= 0:
        return np.nan  # No mean reversion
    
    half_life = -np.log(2) / beta
    return half_life


def create_time_based_splits(
    data: pd.DataFrame,
    train_end: str,
    test_start: str,
    validation_size: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create time-based train/validation/test splits.
    
    Args:
        data: Input data with datetime index.
        train_end: End date for training data.
        test_start: Start date for test data.
        validation_size: Fraction of training data to use for validation.
        
    Returns:
        Tuple of (train_data, val_data, test_data).
    """
    train_end_dt = pd.to_datetime(train_end)
    test_start_dt = pd.to_datetime(test_start)
    
    # Training data
    train_data = data[data.index <= train_end_dt].copy()
    
    # Test data
    test_data = data[data.index >= test_start_dt].copy()
    
    # Validation data (last portion of training data)
    val_size = int(len(train_data) * validation_size)
    val_data = train_data.tail(val_size).copy()
    train_data = train_data.head(len(train_data) - val_size).copy()
    
    return train_data, val_data, test_data


def check_data_leakage(
    features: pd.DataFrame,
    labels: pd.Series,
    feature_window: int = 1,
    label_window: int = 1
) -> bool:
    """Check for potential data leakage between features and labels.
    
    Args:
        features: Feature matrix.
        labels: Label series.
        feature_window: Feature computation window.
        label_window: Label computation window.
        
    Returns:
        bool: True if potential leakage detected.
    """
    # Check for overlapping time windows
    feature_end = features.index.max()
    label_start = labels.index.min()
    
    # Allow for some overlap due to windowing
    min_gap = max(feature_window, label_window)
    
    if (feature_end - label_start).days < min_gap:
        return True
    
    return False


def format_currency(value: float, currency: str = "USD") -> str:
    """Format currency values for display.
    
    Args:
        value: Numeric value.
        currency: Currency code.
        
    Returns:
        str: Formatted currency string.
    """
    if abs(value) >= 1e9:
        return f"{currency} ${value/1e9:.2f}B"
    elif abs(value) >= 1e6:
        return f"{currency} ${value/1e6:.2f}M"
    elif abs(value) >= 1e3:
        return f"{currency} ${value/1e3:.2f}K"
    else:
        return f"{currency} ${value:.2f}"


def format_percentage(value: float, decimals: int = 2) -> str:
    """Format percentage values for display.
    
    Args:
        value: Numeric value (as decimal).
        decimals: Number of decimal places.
        
    Returns:
        str: Formatted percentage string.
    """
    return f"{value * 100:.{decimals}f}%"
