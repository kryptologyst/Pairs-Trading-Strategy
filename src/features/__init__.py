"""Feature engineering for pairs trading strategy."""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from typing import Dict, List, Tuple, Optional
import logging
from scipy import stats
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class PairsFeatureEngineer:
    """Feature engineer for pairs trading strategy."""
    
    def __init__(self, lookback_window: int = 252):
        """Initialize feature engineer.
        
        Args:
            lookback_window: Lookback window for rolling calculations.
        """
        self.lookback_window = lookback_window
        self.scaler = StandardScaler()
        
    def calculate_cointegration_features(
        self,
        pair_data: pd.DataFrame,
        min_pvalue: float = 0.05
    ) -> Dict[str, float]:
        """Calculate cointegration features for a pair.
        
        Args:
            pair_data: DataFrame with two price series.
            min_pvalue: Minimum p-value threshold for cointegration.
            
        Returns:
            Dict with cointegration statistics.
        """
        symbol1, symbol2 = pair_data.columns
        
        # Cointegration test
        try:
            coint_stat, pvalue, critical_values = sm.tsa.stattools.coint(
                pair_data[symbol1], pair_data[symbol2]
            )
            
            is_cointegrated = pvalue < min_pvalue
            
            # Calculate beta (hedge ratio)
            X = sm.add_constant(pair_data[symbol2])
            model = sm.OLS(pair_data[symbol1], X).fit()
            beta = model.params[symbol2]
            
            # Calculate spread
            spread = pair_data[symbol1] - beta * pair_data[symbol2]
            
            # Spread statistics
            spread_mean = spread.mean()
            spread_std = spread.std()
            spread_skew = stats.skew(spread)
            spread_kurtosis = stats.kurtosis(spread)
            
            # Half-life of mean reversion
            half_life = self._calculate_half_life(spread)
            
            return {
                'coint_stat': coint_stat,
                'coint_pvalue': pvalue,
                'is_cointegrated': is_cointegrated,
                'beta': beta,
                'spread_mean': spread_mean,
                'spread_std': spread_std,
                'spread_skew': spread_skew,
                'spread_kurtosis': spread_kurtosis,
                'half_life': half_life,
                'r_squared': model.rsquared
            }
            
        except Exception as e:
            logger.warning(f"Error calculating cointegration features: {e}")
            return {
                'coint_stat': np.nan,
                'coint_pvalue': 1.0,
                'is_cointegrated': False,
                'beta': np.nan,
                'spread_mean': np.nan,
                'spread_std': np.nan,
                'spread_skew': np.nan,
                'spread_kurtosis': np.nan,
                'half_life': np.nan,
                'r_squared': np.nan
            }
    
    def _calculate_half_life(self, spread: pd.Series) -> float:
        """Calculate half-life of mean reversion."""
        try:
            spread_lag = spread.shift(1)
            spread_diff = spread - spread_lag
            
            # Remove NaN values
            valid_idx = ~(spread_lag.isna() | spread_diff.isna())
            if valid_idx.sum() < 10:
                return np.nan
                
            spread_lag_clean = spread_lag[valid_idx]
            spread_diff_clean = spread_diff[valid_idx]
            
            # OLS regression: spread_diff = alpha + beta * spread_lag
            X = sm.add_constant(spread_lag_clean)
            model = sm.OLS(spread_diff_clean, X).fit()
            
            beta = model.params[1]
            if beta >= 0:
                return np.nan  # No mean reversion
            
            half_life = -np.log(2) / beta
            return half_life
            
        except Exception:
            return np.nan
    
    def calculate_technical_features(
        self,
        pair_data: pd.DataFrame,
        window: int = 20
    ) -> pd.DataFrame:
        """Calculate technical indicators for both stocks in the pair.
        
        Args:
            pair_data: DataFrame with two price series.
            window: Window for technical indicators.
            
        Returns:
            DataFrame with technical features.
        """
        features = pd.DataFrame(index=pair_data.index)
        
        for symbol in pair_data.columns:
            price = pair_data[symbol]
            
            # Price-based features
            features[f'{symbol}_returns'] = price.pct_change()
            features[f'{symbol}_log_returns'] = np.log(price / price.shift(1))
            features[f'{symbol}_volatility'] = price.pct_change().rolling(window).std()
            
            # Moving averages
            features[f'{symbol}_sma_{window}'] = price.rolling(window).mean()
            features[f'{symbol}_ema_{window}'] = price.ewm(span=window).mean()
            
            # Price ratios
            features[f'{symbol}_price_sma_ratio'] = price / features[f'{symbol}_sma_{window}']
            features[f'{symbol}_price_ema_ratio'] = price / features[f'{symbol}_ema_{window}']
            
            # Bollinger Bands
            bb_mean = price.rolling(window).mean()
            bb_std = price.rolling(window).std()
            features[f'{symbol}_bb_upper'] = bb_mean + 2 * bb_std
            features[f'{symbol}_bb_lower'] = bb_mean - 2 * bb_std
            features[f'{symbol}_bb_position'] = (price - bb_mean) / (2 * bb_std)
            
            # RSI
            features[f'{symbol}_rsi'] = self._calculate_rsi(price, window)
            
            # MACD
            macd_line, macd_signal, macd_hist = self._calculate_macd(price)
            features[f'{symbol}_macd'] = macd_line
            features[f'{symbol}_macd_signal'] = macd_signal
            features[f'{symbol}_macd_histogram'] = macd_hist
        
        return features.fillna(method='ffill').dropna()
    
    def _calculate_rsi(self, price: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = price.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(
        self, 
        price: pd.Series, 
        fast: int = 12, 
        slow: int = 26, 
        signal: int = 9
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD indicator."""
        ema_fast = price.ewm(span=fast).mean()
        ema_slow = price.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        macd_signal = macd_line.ewm(span=signal).mean()
        macd_histogram = macd_line - macd_signal
        return macd_line, macd_signal, macd_histogram
    
    def calculate_pair_features(
        self,
        pair_data: pd.DataFrame,
        features_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate pair-specific features.
        
        Args:
            pair_data: DataFrame with two price series.
            features_df: DataFrame with individual stock features.
            
        Returns:
            DataFrame with pair features.
        """
        pair_features = pd.DataFrame(index=pair_data.index)
        symbol1, symbol2 = pair_data.columns
        
        # Price ratio
        pair_features['price_ratio'] = pair_data[symbol1] / pair_data[symbol2]
        pair_features['log_price_ratio'] = np.log(pair_features['price_ratio'])
        
        # Returns correlation
        returns1 = features_df[f'{symbol1}_returns']
        returns2 = features_df[f'{symbol2}_returns']
        pair_features['returns_correlation'] = returns1.rolling(20).corr(returns2)
        
        # Volatility ratio
        vol1 = features_df[f'{symbol1}_volatility']
        vol2 = features_df[f'{symbol2}_volatility']
        pair_features['volatility_ratio'] = vol1 / vol2
        
        # Spread features
        beta = self._calculate_beta(pair_data[symbol1], pair_data[symbol2])
        spread = pair_data[symbol1] - beta * pair_data[symbol2]
        
        pair_features['spread'] = spread
        pair_features['spread_zscore'] = (spread - spread.rolling(20).mean()) / spread.rolling(20).std()
        pair_features['spread_ma_ratio'] = spread / spread.rolling(20).mean()
        
        # Momentum features
        pair_features['price_ratio_momentum'] = pair_features['price_ratio'].pct_change(5)
        pair_features['spread_momentum'] = spread.pct_change(5)
        
        return pair_features.fillna(method='ffill').dropna()
    
    def _calculate_beta(self, price1: pd.Series, price2: pd.Series) -> float:
        """Calculate beta (hedge ratio) between two price series."""
        try:
            X = sm.add_constant(price2)
            model = sm.OLS(price1, X).fit()
            return model.params[price2.name]
        except Exception:
            return 1.0
    
    def create_feature_matrix(
        self,
        pair_data: pd.DataFrame,
        include_technical: bool = True,
        include_pair: bool = True
    ) -> pd.DataFrame:
        """Create comprehensive feature matrix for a pair.
        
        Args:
            pair_data: DataFrame with two price series.
            include_technical: Whether to include technical indicators.
            include_pair: Whether to include pair-specific features.
            
        Returns:
            DataFrame with all features.
        """
        features_list = []
        
        # Technical features
        if include_technical:
            technical_features = self.calculate_technical_features(pair_data)
            features_list.append(technical_features)
        
        # Pair features
        if include_pair:
            if include_technical:
                pair_features = self.calculate_pair_features(pair_data, technical_features)
            else:
                # Create minimal features for pair calculations
                temp_features = pd.DataFrame(index=pair_data.index)
                for symbol in pair_data.columns:
                    temp_features[f'{symbol}_returns'] = pair_data[symbol].pct_change()
                    temp_features[f'{symbol}_volatility'] = pair_data[symbol].pct_change().rolling(20).std()
                pair_features = self.calculate_pair_features(pair_data, temp_features)
            features_list.append(pair_features)
        
        # Combine all features
        if features_list:
            feature_matrix = pd.concat(features_list, axis=1)
            return feature_matrix.fillna(method='ffill').dropna()
        else:
            return pd.DataFrame(index=pair_data.index)
    
    def scale_features(self, features: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Scale features using StandardScaler.
        
        Args:
            features: Feature DataFrame.
            fit: Whether to fit the scaler.
            
        Returns:
            Scaled feature DataFrame.
        """
        if fit:
            scaled_features = pd.DataFrame(
                self.scaler.fit_transform(features),
                index=features.index,
                columns=features.columns
            )
        else:
            scaled_features = pd.DataFrame(
                self.scaler.transform(features),
                index=features.index,
                columns=features.columns
            )
        
        return scaled_features
