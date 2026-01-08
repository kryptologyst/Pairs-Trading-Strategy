"""Risk management for pairs trading strategy."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from enum import Enum
from dataclasses import dataclass
import cvxpy as cp

logger = logging.getLogger(__name__)


class PositionSizingMethod(Enum):
    """Position sizing methods."""
    FIXED = "fixed"
    VOLATILITY_ADJUSTED = "volatility_adjusted"
    KELLY = "kelly"
    RISK_PARITY = "risk_parity"
    EQUAL_WEIGHT = "equal_weight"


@dataclass
class RiskConfig:
    """Risk management configuration."""
    max_drawdown: float = 0.15
    stop_loss: float = 0.05
    take_profit: float = 0.10
    position_sizing_method: PositionSizingMethod = PositionSizingMethod.KELLY
    max_position_size: float = 0.1
    max_correlation: float = 0.7
    var_confidence: float = 0.05
    lookback_window: int = 252


class RiskManager:
    """Risk management system for pairs trading."""
    
    def __init__(self, config: RiskConfig):
        """Initialize risk manager.
        
        Args:
            config: Risk management configuration.
        """
        self.config = config
        self.current_positions = {}
        self.portfolio_value = 0
        self.max_portfolio_value = 0
        
    def calculate_position_size(
        self,
        pair_name: str,
        signal_strength: float,
        volatility: float,
        expected_return: float,
        current_portfolio_value: float
    ) -> float:
        """Calculate position size for a pair.
        
        Args:
            pair_name: Name of the trading pair.
            signal_strength: Strength of the trading signal (-1 to 1).
            volatility: Volatility of the pair.
            expected_return: Expected return of the strategy.
            current_portfolio_value: Current portfolio value.
            
        Returns:
            Position size as fraction of portfolio.
        """
        if self.config.position_sizing_method == PositionSizingMethod.FIXED:
            return self.config.max_position_size * abs(signal_strength)
        
        elif self.config.position_sizing_method == PositionSizingMethod.VOLATILITY_ADJUSTED:
            # Adjust position size based on volatility
            base_size = self.config.max_position_size
            vol_adjustment = 1.0 / (1.0 + volatility)
            return base_size * abs(signal_strength) * vol_adjustment
        
        elif self.config.position_sizing_method == PositionSizingMethod.KELLY:
            # Kelly criterion for position sizing
            if expected_return <= 0 or volatility <= 0:
                return 0.0
            
            kelly_fraction = expected_return / (volatility ** 2)
            # Cap Kelly fraction to prevent over-leveraging
            kelly_fraction = min(kelly_fraction, self.config.max_position_size)
            return kelly_fraction * abs(signal_strength)
        
        elif self.config.position_sizing_method == PositionSizingMethod.RISK_PARITY:
            # Equal risk contribution
            num_pairs = len(self.current_positions) + 1
            risk_budget = 1.0 / num_pairs
            return risk_budget * abs(signal_strength)
        
        else:  # EQUAL_WEIGHT
            return self.config.max_position_size * abs(signal_strength)
    
    def check_risk_limits(
        self,
        pair_name: str,
        position_size: float,
        correlation_matrix: pd.DataFrame
    ) -> Tuple[bool, str]:
        """Check if position violates risk limits.
        
        Args:
            pair_name: Name of the trading pair.
            position_size: Proposed position size.
            correlation_matrix: Correlation matrix of all pairs.
            
        Returns:
            Tuple of (is_valid, reason).
        """
        # Check maximum position size
        if position_size > self.config.max_position_size:
            return False, f"Position size {position_size:.3f} exceeds maximum {self.config.max_position_size:.3f}"
        
        # Check correlation limits
        if pair_name in correlation_matrix.columns:
            max_corr = correlation_matrix[pair_name].max()
            if max_corr > self.config.max_correlation:
                return False, f"Correlation {max_corr:.3f} exceeds maximum {self.config.max_correlation:.3f}"
        
        # Check drawdown limit
        if self.portfolio_value > 0:
            current_drawdown = (self.max_portfolio_value - self.portfolio_value) / self.max_portfolio_value
            if current_drawdown > self.config.max_drawdown:
                return False, f"Current drawdown {current_drawdown:.3f} exceeds maximum {self.config.max_drawdown:.3f}"
        
        return True, "Risk limits satisfied"
    
    def calculate_var(
        self,
        returns: pd.Series,
        confidence_level: float = None
    ) -> float:
        """Calculate Value at Risk.
        
        Args:
            returns: Returns series.
            confidence_level: Confidence level for VaR calculation.
            
        Returns:
            VaR value.
        """
        if confidence_level is None:
            confidence_level = self.config.var_confidence
        
        return np.percentile(returns, confidence_level * 100)
    
    def calculate_expected_shortfall(
        self,
        returns: pd.Series,
        confidence_level: float = None
    ) -> float:
        """Calculate Expected Shortfall (Conditional VaR).
        
        Args:
            returns: Returns series.
            confidence_level: Confidence level for ES calculation.
            
        Returns:
            Expected Shortfall value.
        """
        if confidence_level is None:
            confidence_level = self.config.var_confidence
        
        var = self.calculate_var(returns, confidence_level)
        return returns[returns <= var].mean()
    
    def calculate_portfolio_risk(
        self,
        positions: Dict[str, float],
        returns_matrix: pd.DataFrame,
        correlation_matrix: pd.DataFrame
    ) -> Dict[str, float]:
        """Calculate portfolio risk metrics.
        
        Args:
            positions: Dictionary of position sizes.
            returns_matrix: Returns matrix for all pairs.
            correlation_matrix: Correlation matrix.
            
        Returns:
            Dictionary with risk metrics.
        """
        if not positions:
            return {
                'portfolio_volatility': 0.0,
                'portfolio_var': 0.0,
                'portfolio_es': 0.0,
                'diversification_ratio': 0.0
            }
        
        # Calculate portfolio weights
        total_exposure = sum(abs(size) for size in positions.values())
        if total_exposure == 0:
            return {
                'portfolio_volatility': 0.0,
                'portfolio_var': 0.0,
                'portfolio_es': 0.0,
                'diversification_ratio': 0.0
            }
        
        weights = {pair: size / total_exposure for pair, size in positions.items()}
        
        # Calculate portfolio volatility
        portfolio_variance = 0.0
        for pair1, weight1 in weights.items():
            for pair2, weight2 in weights.items():
                if pair1 in returns_matrix.columns and pair2 in returns_matrix.columns:
                    vol1 = returns_matrix[pair1].std()
                    vol2 = returns_matrix[pair2].std()
                    corr = correlation_matrix.loc[pair1, pair2] if pair1 in correlation_matrix.index and pair2 in correlation_matrix.columns else 0
                    portfolio_variance += weight1 * weight2 * vol1 * vol2 * corr
        
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        # Calculate portfolio returns
        portfolio_returns = pd.Series(0.0, index=returns_matrix.index)
        for pair, weight in weights.items():
            if pair in returns_matrix.columns:
                portfolio_returns += weight * returns_matrix[pair]
        
        # Calculate VaR and ES
        portfolio_var = self.calculate_var(portfolio_returns)
        portfolio_es = self.calculate_expected_shortfall(portfolio_returns)
        
        # Calculate diversification ratio
        weighted_avg_vol = sum(abs(weight) * returns_matrix[pair].std() 
                              for pair, weight in weights.items() 
                              if pair in returns_matrix.columns)
        diversification_ratio = weighted_avg_vol / portfolio_volatility if portfolio_volatility > 0 else 0
        
        return {
            'portfolio_volatility': portfolio_volatility,
            'portfolio_var': portfolio_var,
            'portfolio_es': portfolio_es,
            'diversification_ratio': diversification_ratio
        }
    
    def optimize_portfolio(
        self,
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame,
        risk_budget: float = 1.0
    ) -> pd.Series:
        """Optimize portfolio using mean-variance optimization.
        
        Args:
            expected_returns: Expected returns for each pair.
            covariance_matrix: Covariance matrix.
            risk_budget: Risk budget constraint.
            
        Returns:
            Optimal weights.
        """
        n = len(expected_returns)
        
        # Variables
        weights = cp.Variable(n)
        
        # Constraints
        constraints = [
            cp.sum(weights) == 1,  # Weights sum to 1
            weights >= 0,  # Long-only positions
            weights <= self.config.max_position_size  # Maximum position size
        ]
        
        # Objective: maximize Sharpe ratio (minimize negative Sharpe ratio)
        portfolio_return = expected_returns @ weights
        portfolio_variance = cp.quad_form(weights, covariance_matrix)
        portfolio_volatility = cp.sqrt(portfolio_variance)
        
        # Risk budget constraint
        constraints.append(portfolio_volatility <= risk_budget)
        
        # Objective function
        objective = cp.Maximize(portfolio_return - 0.5 * portfolio_variance)
        
        # Solve
        problem = cp.Problem(objective, constraints)
        problem.solve()
        
        if problem.status == cp.OPTIMAL:
            return pd.Series(weights.value, index=expected_returns.index)
        else:
            logger.warning("Portfolio optimization failed, using equal weights")
            return pd.Series(1.0 / n, index=expected_returns.index)
    
    def update_portfolio_value(self, new_value: float) -> None:
        """Update portfolio value and track maximum.
        
        Args:
            new_value: New portfolio value.
        """
        self.portfolio_value = new_value
        if new_value > self.max_portfolio_value:
            self.max_portfolio_value = new_value
    
    def calculate_stop_loss_level(
        self,
        entry_price: float,
        position_type: int,
        volatility: float
    ) -> float:
        """Calculate stop loss level.
        
        Args:
            entry_price: Entry price.
            position_type: Position type (1 for long, -1 for short).
            volatility: Volatility of the instrument.
            
        Returns:
            Stop loss level.
        """
        if position_type == 1:  # Long position
            return entry_price * (1 - self.config.stop_loss)
        else:  # Short position
            return entry_price * (1 + self.config.stop_loss)
    
    def calculate_take_profit_level(
        self,
        entry_price: float,
        position_type: int,
        volatility: float
    ) -> float:
        """Calculate take profit level.
        
        Args:
            entry_price: Entry price.
            position_type: Position type (1 for long, -1 for short).
            volatility: Volatility of the instrument.
            
        Returns:
            Take profit level.
        """
        if position_type == 1:  # Long position
            return entry_price * (1 + self.config.take_profit)
        else:  # Short position
            return entry_price * (1 - self.config.take_profit)
    
    def generate_risk_report(
        self,
        returns: pd.Series,
        positions: Dict[str, float],
        benchmark_returns: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """Generate comprehensive risk report.
        
        Args:
            returns: Portfolio returns.
            positions: Current positions.
            benchmark_returns: Benchmark returns for comparison.
            
        Returns:
            Dictionary with risk metrics.
        """
        report = {}
        
        # Basic risk metrics
        report['volatility'] = returns.std() * np.sqrt(252)
        report['sharpe_ratio'] = returns.mean() / returns.std() * np.sqrt(252)
        report['max_drawdown'] = self._calculate_max_drawdown(returns)
        
        # VaR and ES
        report['var_95'] = self.calculate_var(returns, 0.05)
        report['var_99'] = self.calculate_var(returns, 0.01)
        report['es_95'] = self.calculate_expected_shortfall(returns, 0.05)
        report['es_99'] = self.calculate_expected_shortfall(returns, 0.01)
        
        # Position analysis
        report['num_positions'] = len(positions)
        report['total_exposure'] = sum(abs(size) for size in positions.values())
        report['max_position'] = max(abs(size) for size in positions.values()) if positions else 0
        
        # Benchmark comparison
        if benchmark_returns is not None:
            aligned_returns = returns.align(benchmark_returns, join='inner')[0]
            aligned_benchmark = returns.align(benchmark_returns, join='inner')[1]
            
            report['beta'] = np.cov(aligned_returns, aligned_benchmark)[0, 1] / np.var(aligned_benchmark)
            report['alpha'] = aligned_returns.mean() - report['beta'] * aligned_benchmark.mean()
            report['information_ratio'] = report['alpha'] / (aligned_returns - aligned_benchmark).std()
        
        return report
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
