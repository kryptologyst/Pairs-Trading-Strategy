"""Label generation for pairs trading strategy."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class LabelMethod(Enum):
    """Label generation methods."""
    THRESHOLD_BASED = "threshold_based"
    TRIPLE_BARRIER = "triple_barrier"
    REGIME_BASED = "regime_based"
    MOMENTUM_BASED = "momentum_based"


@dataclass
class LabelConfig:
    """Configuration for label generation."""
    method: LabelMethod
    entry_threshold: float = 1.0
    exit_threshold: float = 0.5
    stop_loss: float = 0.05
    take_profit: float = 0.10
    max_holding_period: int = 20
    min_holding_period: int = 1


class PairsLabelGenerator:
    """Generate trading labels for pairs trading strategy."""
    
    def __init__(self, config: LabelConfig):
        """Initialize label generator.
        
        Args:
            config: Label generation configuration.
        """
        self.config = config
        
    def generate_labels(
        self,
        pair_data: pd.DataFrame,
        spread: pd.Series,
        beta: float,
        symbol1: str,
        symbol2: str
    ) -> pd.Series:
        """Generate trading labels.
        
        Args:
            pair_data: DataFrame with price data.
            spread: Spread series.
            beta: Hedge ratio.
            symbol1: First symbol name.
            symbol2: Second symbol name.
            
        Returns:
            Series with trading labels (-1, 0, 1).
        """
        if self.config.method == LabelMethod.THRESHOLD_BASED:
            return self._generate_threshold_labels(spread)
        elif self.config.method == LabelMethod.TRIPLE_BARRIER:
            return self._generate_triple_barrier_labels(pair_data, spread, beta, symbol1, symbol2)
        elif self.config.method == LabelMethod.REGIME_BASED:
            return self._generate_regime_labels(spread)
        elif self.config.method == LabelMethod.MOMENTUM_BASED:
            return self._generate_momentum_labels(spread)
        else:
            raise ValueError(f"Unsupported label method: {self.config.method}")
    
    def _generate_threshold_labels(self, spread: pd.Series) -> pd.Series:
        """Generate threshold-based labels.
        
        Args:
            spread: Spread series.
            
        Returns:
            Series with labels.
        """
        labels = pd.Series(0, index=spread.index)
        
        # Calculate rolling statistics
        spread_mean = spread.rolling(window=20).mean()
        spread_std = spread.rolling(window=20).std()
        
        # Entry signals
        long_condition = spread < (spread_mean - self.config.entry_threshold * spread_std)
        short_condition = spread > (spread_mean + self.config.entry_threshold * spread_std)
        
        labels[long_condition] = 1  # Long spread
        labels[short_condition] = -1  # Short spread
        
        # Exit signals
        exit_long = spread > (spread_mean - self.config.exit_threshold * spread_std)
        exit_short = spread < (spread_mean + self.config.exit_threshold * spread_std)
        
        # Apply exit conditions
        labels[(labels == 1) & exit_long] = 0
        labels[(labels == -1) & exit_short] = 0
        
        return labels
    
    def _generate_triple_barrier_labels(
        self,
        pair_data: pd.DataFrame,
        spread: pd.Series,
        beta: float,
        symbol1: str,
        symbol2: str
    ) -> pd.Series:
        """Generate triple barrier labels.
        
        Args:
            pair_data: DataFrame with price data.
            spread: Spread series.
            beta: Hedge ratio.
            symbol1: First symbol name.
            symbol2: Second symbol name.
            
        Returns:
            Series with labels.
        """
        labels = pd.Series(0, index=spread.index)
        
        # Calculate rolling volatility for dynamic barriers
        spread_vol = spread.rolling(window=20).std()
        
        i = 0
        while i < len(spread):
            if i >= len(spread) - self.config.min_holding_period:
                break
                
            current_spread = spread.iloc[i]
            current_vol = spread_vol.iloc[i]
            
            if pd.isna(current_vol):
                i += 1
                continue
            
            # Define barriers
            upper_barrier = current_spread + self.config.take_profit * current_vol
            lower_barrier = current_spread - self.config.take_profit * current_vol
            stop_loss_upper = current_spread + self.config.stop_loss * current_vol
            stop_loss_lower = current_spread - self.config.stop_loss * current_vol
            
            # Determine initial signal
            if current_spread < -self.config.entry_threshold * current_vol:
                signal = 1  # Long spread
                target_barrier = upper_barrier
                stop_barrier = stop_loss_lower
            elif current_spread > self.config.entry_threshold * current_vol:
                signal = -1  # Short spread
                target_barrier = lower_barrier
                stop_barrier = stop_loss_upper
            else:
                i += 1
                continue
            
            # Find exit point
            exit_idx = self._find_exit_point(
                spread.iloc[i:i+self.config.max_holding_period],
                target_barrier,
                stop_barrier,
                signal
            )
            
            if exit_idx is not None:
                labels.iloc[i:i+exit_idx+1] = signal
                i += exit_idx + 1
            else:
                i += 1
        
        return labels
    
    def _find_exit_point(
        self,
        future_spread: pd.Series,
        target_barrier: float,
        stop_barrier: float,
        signal: int
    ) -> Optional[int]:
        """Find exit point for triple barrier method.
        
        Args:
            future_spread: Future spread values.
            target_barrier: Target barrier level.
            stop_barrier: Stop loss barrier level.
            signal: Trading signal (1 or -1).
            
        Returns:
            Exit index or None if no exit found.
        """
        for i, spread_val in enumerate(future_spread):
            if signal == 1:  # Long spread
                if spread_val >= target_barrier or spread_val <= stop_barrier:
                    return i
            elif signal == -1:  # Short spread
                if spread_val <= target_barrier or spread_val >= stop_barrier:
                    return i
        
        return None
    
    def _generate_regime_labels(self, spread: pd.Series) -> pd.Series:
        """Generate regime-based labels.
        
        Args:
            spread: Spread series.
            
        Returns:
            Series with labels.
        """
        labels = pd.Series(0, index=spread.index)
        
        # Detect regimes using rolling statistics
        spread_mean = spread.rolling(window=50).mean()
        spread_std = spread.rolling(window=50).std()
        
        # Define regime thresholds
        high_vol_threshold = spread_std.quantile(0.7)
        low_vol_threshold = spread_std.quantile(0.3)
        
        # Generate labels based on regime
        high_vol_regime = spread_std > high_vol_threshold
        low_vol_regime = spread_std < low_vol_threshold
        
        # In high volatility regime, use tighter thresholds
        high_vol_entry = self.config.entry_threshold * 0.5
        high_vol_exit = self.config.exit_threshold * 0.5
        
        # Entry signals
        long_condition = (
            (high_vol_regime & (spread < spread_mean - high_vol_entry * spread_std)) |
            (low_vol_regime & (spread < spread_mean - self.config.entry_threshold * spread_std))
        )
        
        short_condition = (
            (high_vol_regime & (spread > spread_mean + high_vol_entry * spread_std)) |
            (low_vol_regime & (spread > spread_mean + self.config.entry_threshold * spread_std))
        )
        
        labels[long_condition] = 1
        labels[short_condition] = -1
        
        # Exit signals
        exit_long = (
            (high_vol_regime & (spread > spread_mean - high_vol_exit * spread_std)) |
            (low_vol_regime & (spread > spread_mean - self.config.exit_threshold * spread_std))
        )
        
        exit_short = (
            (high_vol_regime & (spread < spread_mean + high_vol_exit * spread_std)) |
            (low_vol_regime & (spread < spread_mean + self.config.exit_threshold * spread_std))
        )
        
        labels[(labels == 1) & exit_long] = 0
        labels[(labels == -1) & exit_short] = 0
        
        return labels
    
    def _generate_momentum_labels(self, spread: pd.Series) -> pd.Series:
        """Generate momentum-based labels.
        
        Args:
            spread: Spread series.
            
        Returns:
            Series with labels.
        """
        labels = pd.Series(0, index=spread.index)
        
        # Calculate momentum indicators
        momentum_short = spread.pct_change(5)  # 5-day momentum
        momentum_long = spread.pct_change(20)  # 20-day momentum
        
        # Calculate momentum thresholds
        momentum_std = momentum_short.rolling(window=50).std()
        
        # Entry signals based on momentum
        long_condition = (
            (momentum_short < -self.config.entry_threshold * momentum_std) &
            (momentum_long < 0)  # Long-term momentum confirms
        )
        
        short_condition = (
            (momentum_short > self.config.entry_threshold * momentum_std) &
            (momentum_long > 0)  # Long-term momentum confirms
        )
        
        labels[long_condition] = 1
        labels[short_condition] = -1
        
        # Exit signals based on momentum reversal
        exit_long = momentum_short > -self.config.exit_threshold * momentum_std
        exit_short = momentum_short < self.config.exit_threshold * momentum_std
        
        labels[(labels == 1) & exit_long] = 0
        labels[(labels == -1) & exit_short] = 0
        
        return labels
    
    def create_feature_labels(
        self,
        features: pd.DataFrame,
        pair_data: pd.DataFrame,
        spread: pd.Series,
        beta: float,
        symbol1: str,
        symbol2: str
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Create features and labels for ML models.
        
        Args:
            features: Feature matrix.
            pair_data: DataFrame with price data.
            spread: Spread series.
            beta: Hedge ratio.
            symbol1: First symbol name.
            symbol2: Second symbol name.
            
        Returns:
            Tuple of (features_with_labels, labels).
        """
        # Generate labels
        labels = self.generate_labels(pair_data, spread, beta, symbol1, symbol2)
        
        # Align features and labels
        common_index = features.index.intersection(labels.index)
        features_aligned = features.loc[common_index]
        labels_aligned = labels.loc[common_index]
        
        # Add price information to features
        features_with_labels = features_aligned.copy()
        features_with_labels[f'{symbol1}_price'] = pair_data[symbol1].loc[common_index]
        features_with_labels[f'{symbol2}_price'] = pair_data[symbol2].loc[common_index]
        features_with_labels['spread'] = spread.loc[common_index]
        features_with_labels['beta'] = beta
        
        return features_with_labels, labels_aligned
    
    def validate_labels(self, labels: pd.Series) -> Dict[str, Any]:
        """Validate generated labels.
        
        Args:
            labels: Generated labels.
            
        Returns:
            Dictionary with validation statistics.
        """
        validation_stats = {
            'total_samples': len(labels),
            'long_signals': (labels == 1).sum(),
            'short_signals': (labels == -1).sum(),
            'neutral_signals': (labels == 0).sum(),
            'signal_ratio': (labels != 0).sum() / len(labels),
            'class_balance': {
                'long': (labels == 1).sum() / len(labels),
                'short': (labels == -1).sum() / len(labels),
                'neutral': (labels == 0).sum() / len(labels)
            }
        }
        
        return validation_stats
