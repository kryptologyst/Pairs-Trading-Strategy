"""Models for pairs trading strategy."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from abc import ABC, abstractmethod
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb
from scipy import stats
import statsmodels.api as sm

logger = logging.getLogger(__name__)


class BasePairsModel(ABC):
    """Base class for pairs trading models."""
    
    def __init__(self, name: str):
        """Initialize base model.
        
        Args:
            name: Model name.
        """
        self.name = name
        self.is_fitted = False
        
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit the model.
        
        Args:
            X: Feature matrix.
            y: Target labels.
        """
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions.
        
        Args:
            X: Feature matrix.
            
        Returns:
            Predictions array.
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities.
        
        Args:
            X: Feature matrix.
            
        Returns:
            Probability predictions array.
        """
        pass


class CointegrationBaseline(BasePairsModel):
    """Baseline cointegration-based pairs trading model."""
    
    def __init__(
        self,
        entry_threshold: float = 1.0,
        exit_threshold: float = 0.5,
        min_pvalue: float = 0.05
    ):
        """Initialize cointegration baseline model.
        
        Args:
            entry_threshold: Entry threshold in standard deviations.
            exit_threshold: Exit threshold in standard deviations.
            min_pvalue: Minimum p-value for cointegration.
        """
        super().__init__("cointegration_baseline")
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.min_pvalue = min_pvalue
        self.beta = None
        self.spread_mean = None
        self.spread_std = None
        self.is_cointegrated = False
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit the cointegration model.
        
        Args:
            X: Feature matrix (not used for this model).
            y: Target labels (not used for this model).
        """
        # This model doesn't use traditional ML features
        # It relies on cointegration statistics calculated separately
        self.is_fitted = True
        logger.info("Cointegration baseline model fitted")
    
    def set_cointegration_params(
        self,
        beta: float,
        spread_mean: float,
        spread_std: float,
        is_cointegrated: bool
    ) -> None:
        """Set cointegration parameters.
        
        Args:
            beta: Hedge ratio.
            spread_mean: Mean of spread.
            spread_std: Standard deviation of spread.
            is_cointegrated: Whether pair is cointegrated.
        """
        self.beta = beta
        self.spread_mean = spread_mean
        self.spread_std = spread_std
        self.is_cointegrated = is_cointegrated
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions based on spread z-scores.
        
        Args:
            X: Feature matrix containing spread_zscore.
            
        Returns:
            Predictions array (-1, 0, 1).
        """
        if not self.is_fitted or not self.is_cointegrated:
            return np.zeros(len(X))
        
        if 'spread_zscore' not in X.columns:
            logger.warning("spread_zscore not found in features")
            return np.zeros(len(X))
        
        spread_zscore = X['spread_zscore']
        predictions = np.zeros(len(X))
        
        # Entry signals
        predictions[spread_zscore < -self.entry_threshold] = 1  # Long spread
        predictions[spread_zscore > self.entry_threshold] = -1  # Short spread
        
        # Exit signals
        predictions[(spread_zscore > -self.exit_threshold) & 
                   (spread_zscore < self.exit_threshold)] = 0
        
        return predictions
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities based on distance from thresholds.
        
        Args:
            X: Feature matrix containing spread_zscore.
            
        Returns:
            Probability predictions array.
        """
        if not self.is_fitted or not self.is_cointegrated:
            return np.ones((len(X), 3)) / 3
        
        if 'spread_zscore' not in X.columns:
            return np.ones((len(X), 3)) / 3
        
        spread_zscore = X['spread_zscore']
        probabilities = np.zeros((len(X), 3))  # [short, neutral, long]
        
        # Calculate probabilities based on distance from thresholds
        for i, zscore in enumerate(spread_zscore):
            if zscore < -self.entry_threshold:
                # Strong long signal
                prob_long = min(1.0, abs(zscore) / self.entry_threshold)
                probabilities[i] = [0, 1 - prob_long, prob_long]
            elif zscore > self.entry_threshold:
                # Strong short signal
                prob_short = min(1.0, abs(zscore) / self.entry_threshold)
                probabilities[i] = [prob_short, 1 - prob_short, 0]
            else:
                # Neutral
                probabilities[i] = [0, 1, 0]
        
        return probabilities


class KalmanFilterModel(BasePairsModel):
    """Kalman filter-based pairs trading model."""
    
    def __init__(
        self,
        initial_state_covariance: float = 1.0,
        process_noise: float = 0.01,
        measurement_noise: float = 0.1
    ):
        """Initialize Kalman filter model.
        
        Args:
            initial_state_covariance: Initial state covariance.
            process_noise: Process noise variance.
            measurement_noise: Measurement noise variance.
        """
        super().__init__("kalman_filter")
        self.initial_state_covariance = initial_state_covariance
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.state_mean = None
        self.state_covariance = None
        self.beta_history = []
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit the Kalman filter model.
        
        Args:
            X: Feature matrix.
            y: Target labels.
        """
        if 'symbol1_price' not in X.columns or 'symbol2_price' not in X.columns:
            logger.error("Required price columns not found")
            return
        
        # Initialize Kalman filter
        self.state_mean = np.array([1.0])  # Initial beta
        self.state_covariance = np.array([[self.initial_state_covariance]])
        
        # Run Kalman filter on training data
        symbol1_prices = X['symbol1_price'].values
        symbol2_prices = X['symbol2_price'].values
        
        for i in range(len(symbol1_prices)):
            self._update_kalman_filter(symbol1_prices[i], symbol2_prices[i])
        
        self.is_fitted = True
        logger.info("Kalman filter model fitted")
    
    def _update_kalman_filter(self, price1: float, price2: float) -> None:
        """Update Kalman filter with new price observation."""
        # Prediction step
        predicted_covariance = self.state_covariance + self.process_noise
        
        # Update step
        measurement_residual = price1 - self.state_mean[0] * price2
        residual_covariance = predicted_covariance[0, 0] * price2**2 + self.measurement_noise
        
        kalman_gain = predicted_covariance[0, 0] * price2 / residual_covariance
        
        self.state_mean[0] += kalman_gain * measurement_residual
        self.state_covariance[0, 0] = (1 - kalman_gain * price2) * predicted_covariance[0, 0]
        
        self.beta_history.append(self.state_mean[0])
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using current Kalman filter state.
        
        Args:
            X: Feature matrix.
            
        Returns:
            Predictions array.
        """
        if not self.is_fitted:
            return np.zeros(len(X))
        
        if 'symbol1_price' not in X.columns or 'symbol2_price' not in X.columns:
            return np.zeros(len(X))
        
        predictions = np.zeros(len(X))
        current_beta = self.state_mean[0]
        
        symbol1_prices = X['symbol1_price'].values
        symbol2_prices = X['symbol2_price'].values
        
        for i in range(len(X)):
            spread = symbol1_prices[i] - current_beta * symbol2_prices[i]
            
            # Simple threshold-based signals
            if spread < -2 * np.sqrt(self.state_covariance[0, 0]):
                predictions[i] = 1  # Long spread
            elif spread > 2 * np.sqrt(self.state_covariance[0, 0]):
                predictions[i] = -1  # Short spread
        
        return predictions
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities using Kalman filter uncertainty.
        
        Args:
            X: Feature matrix.
            
        Returns:
            Probability predictions array.
        """
        if not self.is_fitted:
            return np.ones((len(X), 3)) / 3
        
        if 'symbol1_price' not in X.columns or 'symbol2_price' not in X.columns:
            return np.ones((len(X), 3)) / 3
        
        probabilities = np.zeros((len(X), 3))
        current_beta = self.state_mean[0]
        uncertainty = np.sqrt(self.state_covariance[0, 0])
        
        symbol1_prices = X['symbol1_price'].values
        symbol2_prices = X['symbol2_price'].values
        
        for i in range(len(X)):
            spread = symbol1_prices[i] - current_beta * symbol2_prices[i]
            spread_std = uncertainty * symbol2_prices[i]
            
            # Calculate probabilities based on spread z-score
            z_score = spread / spread_std
            
            if z_score < -2:
                probabilities[i] = [0, 0.2, 0.8]  # Strong long
            elif z_score > 2:
                probabilities[i] = [0.8, 0.2, 0]  # Strong short
            else:
                probabilities[i] = [0.1, 0.8, 0.1]  # Neutral
        
        return probabilities


class MLPairsModel(BasePairsModel):
    """Machine learning-based pairs trading model."""
    
    def __init__(
        self,
        model_type: str = "xgboost",
        model_params: Optional[Dict] = None
    ):
        """Initialize ML model.
        
        Args:
            model_type: Type of ML model ("xgboost", "lightgbm", "random_forest", "logistic").
            model_params: Model parameters.
        """
        super().__init__(f"ml_{model_type}")
        self.model_type = model_type
        self.model_params = model_params or {}
        self.model = None
        self.feature_importance = None
        
    def _create_model(self):
        """Create the ML model."""
        if self.model_type == "xgboost":
            self.model = xgb.XGBClassifier(**self.model_params)
        elif self.model_type == "lightgbm":
            self.model = lgb.LGBMClassifier(**self.model_params)
        elif self.model_type == "random_forest":
            self.model = RandomForestClassifier(**self.model_params)
        elif self.model_type == "logistic":
            self.model = LogisticRegression(**self.model_params)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit the ML model.
        
        Args:
            X: Feature matrix.
            y: Target labels.
        """
        self._create_model()
        
        # Handle missing values
        X_clean = X.fillna(X.mean())
        
        # Fit model
        self.model.fit(X_clean, y)
        
        # Store feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = pd.Series(
                self.model.feature_importances_,
                index=X_clean.columns
            ).sort_values(ascending=False)
        
        self.is_fitted = True
        logger.info(f"ML model {self.model_type} fitted")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions.
        
        Args:
            X: Feature matrix.
            
        Returns:
            Predictions array.
        """
        if not self.is_fitted:
            return np.zeros(len(X))
        
        X_clean = X.fillna(X.mean())
        return self.model.predict(X_clean)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities.
        
        Args:
            X: Feature matrix.
            
        Returns:
            Probability predictions array.
        """
        if not self.is_fitted:
            return np.ones((len(X), 3)) / 3
        
        X_clean = X.fillna(X.mean())
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X_clean)
        else:
            # Convert predictions to probabilities
            predictions = self.model.predict(X_clean)
            probabilities = np.zeros((len(X), 3))
            for i, pred in enumerate(predictions):
                probabilities[i, int(pred) + 1] = 1.0
            return probabilities


class PairsModelEnsemble:
    """Ensemble of pairs trading models."""
    
    def __init__(self, models: List[BasePairsModel], weights: Optional[List[float]] = None):
        """Initialize ensemble.
        
        Args:
            models: List of models to ensemble.
            weights: Model weights (default: equal weights).
        """
        self.models = models
        self.weights = weights or [1.0 / len(models)] * len(models)
        self.is_fitted = False
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit all models in the ensemble.
        
        Args:
            X: Feature matrix.
            y: Target labels.
        """
        for model in self.models:
            model.fit(X, y)
        
        self.is_fitted = True
        logger.info(f"Ensemble of {len(self.models)} models fitted")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make ensemble predictions.
        
        Args:
            X: Feature matrix.
            
        Returns:
            Ensemble predictions array.
        """
        if not self.is_fitted:
            return np.zeros(len(X))
        
        predictions = []
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
        
        # Weighted average
        ensemble_pred = np.zeros(len(X))
        for i, pred in enumerate(predictions):
            ensemble_pred += self.weights[i] * pred
        
        # Convert to discrete predictions
        ensemble_pred = np.where(ensemble_pred > 0.5, 1, 
                               np.where(ensemble_pred < -0.5, -1, 0))
        
        return ensemble_pred
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Make ensemble probability predictions.
        
        Args:
            X: Feature matrix.
            
        Returns:
            Ensemble probability predictions array.
        """
        if not self.is_fitted:
            return np.ones((len(X), 3)) / 3
        
        probabilities = []
        for model in self.models:
            prob = model.predict_proba(X)
            probabilities.append(prob)
        
        # Weighted average
        ensemble_prob = np.zeros((len(X), 3))
        for i, prob in enumerate(probabilities):
            ensemble_prob += self.weights[i] * prob
        
        # Normalize probabilities
        ensemble_prob = ensemble_prob / ensemble_prob.sum(axis=1, keepdims=True)
        
        return ensemble_prob
