"""Tests for pairs trading strategy components."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import tempfile
import os

# Import our modules
from src.data import DataLoader, preprocess_data, create_pairs_data
from src.features import PairsFeatureEngineer
from src.labels import PairsLabelGenerator, LabelConfig, LabelMethod
from src.models import CointegrationBaseline, KalmanFilterModel, MLPairsModel
from src.backtest import PairsBacktester, Trade, PositionType
from src.risk import RiskManager, RiskConfig, PositionSizingMethod
from src.utils import (
    set_random_seeds, get_device, validate_data_splits,
    calculate_returns, calculate_volatility, calculate_sharpe_ratio,
    calculate_max_drawdown, calculate_half_life
)


class TestDataLoader:
    """Test DataLoader class."""
    
    def test_init(self):
        """Test DataLoader initialization."""
        loader = DataLoader("yfinance")
        assert loader.data_source == "yfinance"
    
    def test_generate_synthetic_data(self):
        """Test synthetic data generation."""
        loader = DataLoader("synthetic")
        symbols = ["AAPL", "MSFT"]
        data = loader.load_stock_data(symbols, "2020-01-01", "2020-12-31")
        
        assert len(data) > 0
        assert all(f"{symbol}_Close" in data.columns for symbol in symbols)
        assert isinstance(data.index, pd.DatetimeIndex)
    
    def test_preprocess_data(self):
        """Test data preprocessing."""
        # Create test data with outliers
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        data = pd.DataFrame({
            "AAPL_Close": np.random.normal(100, 10, 100),
            "MSFT_Close": np.random.normal(200, 20, 100)
        }, index=dates)
        
        # Add outliers
        data.iloc[10, 0] = 1000  # Outlier
        data.iloc[20, 1] = 5000  # Outlier
        
        processed = preprocess_data(data, remove_outliers=True)
        
        assert len(processed) < len(data)  # Outliers should be removed
        assert not processed.isna().any().any()  # No NaN values
    
    def test_create_pairs_data(self):
        """Test pairs data creation."""
        dates = pd.date_range("2020-01-01", periods=50, freq="D")
        data = pd.DataFrame({
            "AAPL_Close": np.random.normal(100, 10, 50),
            "MSFT_Close": np.random.normal(200, 20, 50),
            "GOOGL_Close": np.random.normal(150, 15, 50)
        }, index=dates)
        
        pairs = create_pairs_data(data, ["AAPL", "MSFT", "GOOGL"])
        
        assert len(pairs) == 3  # AAPL_MSFT, AAPL_GOOGL, MSFT_GOOGL
        assert "AAPL_MSFT" in pairs
        assert pairs["AAPL_MSFT"].shape[1] == 2


class TestFeatureEngineer:
    """Test PairsFeatureEngineer class."""
    
    def test_init(self):
        """Test feature engineer initialization."""
        engineer = PairsFeatureEngineer(lookback_window=252)
        assert engineer.lookback_window == 252
    
    def test_cointegration_features(self):
        """Test cointegration feature calculation."""
        engineer = PairsFeatureEngineer()
        
        # Create cointegrated series
        np.random.seed(42)
        n = 100
        x = np.cumsum(np.random.randn(n))
        y = x + 0.5 * np.random.randn(n)
        
        pair_data = pd.DataFrame({
            "AAPL": x,
            "MSFT": y
        })
        
        features = engineer.calculate_cointegration_features(pair_data)
        
        assert "coint_pvalue" in features
        assert "beta" in features
        assert "is_cointegrated" in features
        assert isinstance(features["is_cointegrated"], bool)
    
    def test_technical_features(self):
        """Test technical indicator calculation."""
        engineer = PairsFeatureEngineer()
        
        # Create test data
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        pair_data = pd.DataFrame({
            "AAPL": 100 + np.cumsum(np.random.randn(100) * 0.1),
            "MSFT": 200 + np.cumsum(np.random.randn(100) * 0.1)
        }, index=dates)
        
        features = engineer.calculate_technical_features(pair_data)
        
        assert len(features) > 0
        assert "AAPL_returns" in features.columns
        assert "AAPL_volatility" in features.columns
        assert "AAPL_rsi" in features.columns


class TestModels:
    """Test model classes."""
    
    def test_cointegration_baseline(self):
        """Test cointegration baseline model."""
        model = CointegrationBaseline(entry_threshold=1.0, exit_threshold=0.5)
        
        # Create test features
        features = pd.DataFrame({
            "spread_zscore": np.random.randn(100)
        })
        labels = pd.Series(np.random.choice([-1, 0, 1], 100))
        
        model.fit(features, labels)
        model.set_cointegration_params(1.0, 0.0, 1.0, True)
        
        predictions = model.predict(features)
        probabilities = model.predict_proba(features)
        
        assert len(predictions) == len(features)
        assert probabilities.shape == (len(features), 3)
        assert np.all(np.isin(predictions, [-1, 0, 1]))
    
    def test_kalman_filter_model(self):
        """Test Kalman filter model."""
        model = KalmanFilterModel()
        
        # Create test features
        features = pd.DataFrame({
            "symbol1_price": 100 + np.random.randn(100),
            "symbol2_price": 200 + np.random.randn(100)
        })
        labels = pd.Series(np.random.choice([-1, 0, 1], 100))
        
        model.fit(features, labels)
        
        predictions = model.predict(features)
        probabilities = model.predict_proba(features)
        
        assert len(predictions) == len(features)
        assert probabilities.shape == (len(features), 3)
    
    def test_ml_model(self):
        """Test ML model."""
        model = MLPairsModel("xgboost", {"n_estimators": 10})
        
        # Create test features
        features = pd.DataFrame({
            "feature1": np.random.randn(100),
            "feature2": np.random.randn(100),
            "feature3": np.random.randn(100)
        })
        labels = pd.Series(np.random.choice([-1, 0, 1], 100))
        
        model.fit(features, labels)
        
        predictions = model.predict(features)
        probabilities = model.predict_proba(features)
        
        assert len(predictions) == len(features)
        assert probabilities.shape == (len(features), 3)


class TestBacktester:
    """Test backtesting functionality."""
    
    def test_backtester_init(self):
        """Test backtester initialization."""
        backtester = PairsBacktester(
            initial_capital=100000,
            transaction_cost=0.001,
            slippage=0.0005
        )
        
        assert backtester.initial_capital == 100000
        assert backtester.transaction_cost == 0.001
        assert backtester.slippage == 0.0005
    
    def test_run_backtest(self):
        """Test backtest execution."""
        backtester = PairsBacktester(initial_capital=100000)
        
        # Create test data
        dates = pd.date_range("2020-01-01", periods=50, freq="D")
        pair_data = pd.DataFrame({
            "AAPL": 100 + np.cumsum(np.random.randn(50) * 0.1),
            "MSFT": 200 + np.cumsum(np.random.randn(50) * 0.1)
        }, index=dates)
        
        signals = pd.Series(np.random.choice([-1, 0, 1], 50), index=dates)
        beta = 1.0
        
        result = backtester.run_backtest(pair_data, signals, beta, "AAPL", "MSFT")
        
        assert isinstance(result.equity_curve, pd.Series)
        assert isinstance(result.metrics, dict)
        assert "total_return" in result.metrics
        assert "sharpe_ratio" in result.metrics


class TestRiskManager:
    """Test risk management functionality."""
    
    def test_risk_manager_init(self):
        """Test risk manager initialization."""
        config = RiskConfig()
        risk_manager = RiskManager(config)
        
        assert risk_manager.config.max_drawdown == 0.15
        assert risk_manager.config.stop_loss == 0.05
    
    def test_position_sizing(self):
        """Test position sizing calculations."""
        config = RiskConfig()
        risk_manager = RiskManager(config)
        
        position_size = risk_manager.calculate_position_size(
            "AAPL_MSFT", 0.8, 0.02, 0.05, 100000
        )
        
        assert 0 <= position_size <= config.max_position_size
    
    def test_var_calculation(self):
        """Test VaR calculation."""
        config = RiskConfig()
        risk_manager = RiskManager(config)
        
        returns = pd.Series(np.random.normal(0, 0.02, 1000))
        var = risk_manager.calculate_var(returns, 0.05)
        
        assert isinstance(var, float)
        assert var < 0  # VaR should be negative


class TestUtils:
    """Test utility functions."""
    
    def test_set_random_seeds(self):
        """Test random seed setting."""
        set_random_seeds(42)
        # This is hard to test directly, but we can ensure it doesn't raise an error
        assert True
    
    def test_get_device(self):
        """Test device selection."""
        device = get_device("auto")
        assert device is not None
    
    def test_calculate_returns(self):
        """Test returns calculation."""
        prices = pd.Series([100, 105, 110, 108, 112])
        returns = calculate_returns(prices, method="simple")
        
        expected = pd.Series([np.nan, 0.05, 0.0476, -0.0182, 0.0370])
        pd.testing.assert_series_equal(returns, expected, atol=1e-3)
    
    def test_calculate_volatility(self):
        """Test volatility calculation."""
        returns = pd.Series(np.random.normal(0, 0.02, 300))
        volatility = calculate_volatility(returns, window=252)
        
        assert len(volatility) == len(returns)
        assert not volatility.iloc[-1].isna()
    
    def test_calculate_sharpe_ratio(self):
        """Test Sharpe ratio calculation."""
        returns = pd.Series(np.random.normal(0.05, 0.15, 252))
        sharpe = calculate_sharpe_ratio(returns)
        
        assert isinstance(sharpe, float)
        assert sharpe > 0  # Should be positive for positive expected return
    
    def test_calculate_max_drawdown(self):
        """Test maximum drawdown calculation."""
        returns = pd.Series([0.1, -0.05, 0.2, -0.15, 0.1])
        max_dd, peak_date, trough_date = calculate_max_drawdown(returns)
        
        assert isinstance(max_dd, float)
        assert max_dd <= 0  # Drawdown should be negative
        assert isinstance(peak_date, pd.Timestamp)
        assert isinstance(trough_date, pd.Timestamp)


class TestIntegration:
    """Integration tests."""
    
    def test_end_to_end_pipeline(self):
        """Test complete pipeline."""
        # This would be a more comprehensive test
        # For now, just test that components can be imported and initialized
        from src.data import DataLoader
        from src.features import PairsFeatureEngineer
        from src.models import CointegrationBaseline
        from src.backtest import PairsBacktester
        
        # Initialize components
        data_loader = DataLoader("synthetic")
        feature_engineer = PairsFeatureEngineer()
        model = CointegrationBaseline()
        backtester = PairsBacktester()
        
        # Test that they can be created without errors
        assert data_loader is not None
        assert feature_engineer is not None
        assert model is not None
        assert backtester is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
