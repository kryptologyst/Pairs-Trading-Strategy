"""Data loading and preprocessing utilities."""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import List, Dict, Optional, Tuple
import logging
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


class DataLoader:
    """Data loader for financial time series data."""
    
    def __init__(self, data_source: str = "yfinance"):
        """Initialize data loader.
        
        Args:
            data_source: Data source ("yfinance", "synthetic").
        """
        self.data_source = data_source
        
    def load_stock_data(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        fields: List[str] = None
    ) -> pd.DataFrame:
        """Load stock data for given symbols.
        
        Args:
            symbols: List of stock symbols.
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).
            fields: List of fields to retrieve (default: ["Close"]).
            
        Returns:
            pd.DataFrame: Multi-index DataFrame with symbols and fields.
        """
        if fields is None:
            fields = ["Close"]
            
        if self.data_source == "yfinance":
            return self._load_yfinance_data(symbols, start_date, end_date, fields)
        elif self.data_source == "synthetic":
            return self._generate_synthetic_data(symbols, start_date, end_date, fields)
        else:
            raise ValueError(f"Unsupported data source: {self.data_source}")
    
    def _load_yfinance_data(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        fields: List[str]
    ) -> pd.DataFrame:
        """Load data from Yahoo Finance."""
        logger.info(f"Loading data for {len(symbols)} symbols from {start_date} to {end_date}")
        
        try:
            # Download data for all symbols
            data = yf.download(
                symbols,
                start=start_date,
                end=end_date,
                group_by="ticker",
                progress=False
            )
            
            # Reshape data to have symbols as columns
            if len(symbols) == 1:
                # Single symbol case
                result = pd.DataFrame(index=data.index)
                for field in fields:
                    if field in data.columns:
                        result[f"{symbols[0]}_{field}"] = data[field]
            else:
                # Multiple symbols case
                result = pd.DataFrame(index=data[symbols[0]].index)
                for symbol in symbols:
                    for field in fields:
                        if field in data[symbol].columns:
                            result[f"{symbol}_{field}"] = data[symbol][field]
            
            # Forward fill missing values
            result = result.fillna(method='ffill').dropna()
            
            logger.info(f"Loaded {len(result)} rows of data")
            return result
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def _generate_synthetic_data(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        fields: List[str]
    ) -> pd.DataFrame:
        """Generate synthetic financial data for testing."""
        logger.info("Generating synthetic data")
        
        # Create date range
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        dates = dates[dates.weekday < 5]  # Only weekdays
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        result = pd.DataFrame(index=dates)
        
        for symbol in symbols:
            # Generate correlated random walk
            if symbol == symbols[0]:
                # First symbol: pure random walk
                returns = np.random.normal(0.0005, 0.02, len(dates))
            else:
                # Other symbols: correlated with first symbol
                correlation = 0.7 + (hash(symbol) % 30) / 100  # 0.7-1.0 correlation
                returns = correlation * result[f"{symbols[0]}_Close"].pct_change().fillna(0) + \
                         np.random.normal(0.0005, 0.02 * np.sqrt(1 - correlation**2), len(dates))
            
            # Generate price series
            prices = 100 * np.exp(np.cumsum(returns))
            
            for field in fields:
                if field == "Close":
                    result[f"{symbol}_{field}"] = prices
                elif field == "Open":
                    result[f"{symbol}_{field}"] = prices * (1 + np.random.normal(0, 0.001, len(dates)))
                elif field == "High":
                    result[f"{symbol}_{field}"] = prices * (1 + np.abs(np.random.normal(0, 0.005, len(dates))))
                elif field == "Low":
                    result[f"{symbol}_{field}"] = prices * (1 - np.abs(np.random.normal(0, 0.005, len(dates))))
                elif field == "Volume":
                    result[f"{symbol}_{field}"] = np.random.lognormal(15, 0.5, len(dates))
        
        logger.info(f"Generated {len(result)} rows of synthetic data")
        return result
    
    def save_data(self, data: pd.DataFrame, filepath: str) -> None:
        """Save data to file.
        
        Args:
            data: DataFrame to save.
            filepath: Output file path.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if filepath.suffix == '.csv':
            data.to_csv(filepath)
        elif filepath.suffix == '.parquet':
            data.to_parquet(filepath)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
        
        logger.info(f"Data saved to {filepath}")
    
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load data from file.
        
        Args:
            filepath: Input file path.
            
        Returns:
            pd.DataFrame: Loaded data.
        """
        filepath = Path(filepath)
        
        if filepath.suffix == '.csv':
            data = pd.read_csv(filepath, index_col=0, parse_dates=True)
        elif filepath.suffix == '.parquet':
            data = pd.read_parquet(filepath)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
        
        logger.info(f"Data loaded from {filepath}")
        return data


def preprocess_data(
    data: pd.DataFrame,
    remove_outliers: bool = True,
    outlier_threshold: float = 3.0,
    fill_method: str = "forward"
) -> pd.DataFrame:
    """Preprocess financial data.
    
    Args:
        data: Input DataFrame.
        remove_outliers: Whether to remove outliers.
        outlier_threshold: Z-score threshold for outlier detection.
        fill_method: Method for filling missing values ("forward", "backward", "interpolate").
        
    Returns:
        pd.DataFrame: Preprocessed data.
    """
    logger.info("Preprocessing data")
    
    processed_data = data.copy()
    
    # Remove outliers using Z-score
    if remove_outliers:
        for column in processed_data.columns:
            z_scores = np.abs((processed_data[column] - processed_data[column].mean()) / 
                            processed_data[column].std())
            processed_data = processed_data[z_scores < outlier_threshold]
    
    # Fill missing values
    if fill_method == "forward":
        processed_data = processed_data.fillna(method='ffill')
    elif fill_method == "backward":
        processed_data = processed_data.fillna(method='bfill')
    elif fill_method == "interpolate":
        processed_data = processed_data.interpolate()
    
    # Drop any remaining NaN values
    processed_data = processed_data.dropna()
    
    logger.info(f"Preprocessed data: {len(processed_data)} rows")
    return processed_data


def create_pairs_data(
    data: pd.DataFrame,
    symbols: List[str],
    price_field: str = "Close"
) -> Dict[str, pd.DataFrame]:
    """Create pairs data from individual stock data.
    
    Args:
        data: Multi-symbol DataFrame.
        symbols: List of stock symbols.
        price_field: Price field to use.
        
    Returns:
        Dict mapping pair names to DataFrames with price data.
    """
    pairs_data = {}
    
    for i, symbol1 in enumerate(symbols):
        for symbol2 in symbols[i+1:]:
            pair_name = f"{symbol1}_{symbol2}"
            
            col1 = f"{symbol1}_{price_field}"
            col2 = f"{symbol2}_{price_field}"
            
            if col1 in data.columns and col2 in data.columns:
                pair_data = pd.DataFrame({
                    'symbol1': data[col1],
                    'symbol2': data[col2]
                })
                pair_data.columns = [symbol1, symbol2]
                pairs_data[pair_name] = pair_data.dropna()
    
    logger.info(f"Created {len(pairs_data)} pairs")
    return pairs_data
