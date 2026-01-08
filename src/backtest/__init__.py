"""Backtesting engine for pairs trading strategy."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
from enum import Enum
import vectorbt as vbt

logger = logging.getLogger(__name__)


class PositionType(Enum):
    """Position types."""
    LONG_SPREAD = 1
    SHORT_SPREAD = -1
    NEUTRAL = 0


@dataclass
class Trade:
    """Trade record."""
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    symbol1: str
    symbol2: str
    position_type: PositionType
    entry_price1: float
    entry_price2: float
    exit_price1: float
    exit_price2: float
    quantity1: float
    quantity2: float
    pnl: float
    duration: int
    beta: float


@dataclass
class BacktestResults:
    """Backtest results container."""
    trades: List[Trade]
    equity_curve: pd.Series
    returns: pd.Series
    metrics: Dict[str, float]
    positions: pd.DataFrame
    drawdowns: pd.Series


class PairsBacktester:
    """Backtesting engine for pairs trading strategies."""
    
    def __init__(
        self,
        initial_capital: float = 100000,
        transaction_cost: float = 0.001,
        slippage: float = 0.0005,
        max_position_size: float = 0.1
    ):
        """Initialize backtester.
        
        Args:
            initial_capital: Initial capital amount.
            transaction_cost: Transaction cost as fraction of trade value.
            slippage: Slippage as fraction of trade value.
            max_position_size: Maximum position size as fraction of capital.
        """
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.max_position_size = max_position_size
        self.current_capital = initial_capital
        self.trades = []
        self.positions = {}
        
    def run_backtest(
        self,
        pair_data: pd.DataFrame,
        signals: pd.Series,
        beta: float,
        symbol1: str,
        symbol2: str
    ) -> BacktestResults:
        """Run backtest for a pairs trading strategy.
        
        Args:
            pair_data: DataFrame with price data for both symbols.
            signals: Trading signals (-1, 0, 1).
            beta: Hedge ratio.
            symbol1: First symbol name.
            symbol2: Second symbol name.
            
        Returns:
            BacktestResults object.
        """
        logger.info(f"Running backtest for {symbol1}-{symbol2} pair")
        
        # Initialize tracking variables
        equity_curve = pd.Series(index=pair_data.index, dtype=float)
        positions = pd.DataFrame(index=pair_data.index, columns=[
            'position1', 'position2', 'cash', 'total_value'
        ])
        
        current_position1 = 0
        current_position2 = 0
        cash = self.initial_capital
        
        for i, (date, row) in enumerate(pair_data.iterrows()):
            price1 = row[symbol1]
            price2 = row[symbol2]
            signal = signals.iloc[i] if i < len(signals) else 0
            
            # Calculate current portfolio value
            portfolio_value = current_position1 * price1 + current_position2 * price2 + cash
            
            # Execute trades based on signals
            if signal != 0:
                self._execute_trade(
                    date, price1, price2, signal, beta,
                    symbol1, symbol2, current_position1, current_position2,
                    cash, portfolio_value
                )
                
                # Update positions after trade
                current_position1, current_position2, cash = self._calculate_positions(
                    date, price1, price2, signal, beta, portfolio_value
                )
            
            # Record current state
            equity_curve[date] = portfolio_value
            positions.loc[date, 'position1'] = current_position1
            positions.loc[date, 'position2'] = current_position2
            positions.loc[date, 'cash'] = cash
            positions.loc[date, 'total_value'] = portfolio_value
        
        # Calculate returns
        returns = equity_curve.pct_change().fillna(0)
        
        # Calculate metrics
        metrics = self._calculate_metrics(returns, equity_curve)
        
        # Calculate drawdowns
        drawdowns = self._calculate_drawdowns(equity_curve)
        
        return BacktestResults(
            trades=self.trades,
            equity_curve=equity_curve,
            returns=returns,
            metrics=metrics,
            positions=positions,
            drawdowns=drawdowns
        )
    
    def _execute_trade(
        self,
        date: pd.Timestamp,
        price1: float,
        price2: float,
        signal: int,
        beta: float,
        symbol1: str,
        symbol2: str,
        current_position1: float,
        current_position2: float,
        cash: float,
        portfolio_value: float
    ) -> None:
        """Execute a trade based on signal."""
        # Calculate position sizes
        max_position_value = portfolio_value * self.max_position_size
        
        if signal == 1:  # Long spread (long symbol1, short symbol2)
            # Calculate quantities
            quantity1 = max_position_value / price1
            quantity2 = -quantity1 * beta  # Short position
            
            # Apply slippage
            effective_price1 = price1 * (1 + self.slippage)
            effective_price2 = price2 * (1 - self.slippage)
            
            # Calculate costs
            cost1 = quantity1 * effective_price1 * self.transaction_cost
            cost2 = abs(quantity2) * effective_price2 * self.transaction_cost
            
            # Check if we have enough capital
            required_capital = quantity1 * effective_price1 + cost1 + cost2
            if required_capital <= cash:
                # Execute trade
                cash -= required_capital
                
                # Record trade
                trade = Trade(
                    entry_date=date,
                    exit_date=pd.NaT,  # Will be filled when exiting
                    symbol1=symbol1,
                    symbol2=symbol2,
                    position_type=PositionType.LONG_SPREAD,
                    entry_price1=effective_price1,
                    entry_price2=effective_price2,
                    exit_price1=np.nan,
                    exit_price2=np.nan,
                    quantity1=quantity1,
                    quantity2=quantity2,
                    pnl=np.nan,
                    duration=0,
                    beta=beta
                )
                self.trades.append(trade)
                
        elif signal == -1:  # Short spread (short symbol1, long symbol2)
            # Calculate quantities
            quantity1 = -max_position_value / price1
            quantity2 = -quantity1 * beta  # Long position
            
            # Apply slippage
            effective_price1 = price1 * (1 - self.slippage)
            effective_price2 = price2 * (1 + self.slippage)
            
            # Calculate costs
            cost1 = abs(quantity1) * effective_price1 * self.transaction_cost
            cost2 = quantity2 * effective_price2 * self.transaction_cost
            
            # Check if we have enough capital
            required_capital = quantity2 * effective_price2 + cost1 + cost2
            if required_capital <= cash:
                # Execute trade
                cash -= required_capital
                
                # Record trade
                trade = Trade(
                    entry_date=date,
                    exit_date=pd.NaT,
                    symbol1=symbol1,
                    symbol2=symbol2,
                    position_type=PositionType.SHORT_SPREAD,
                    entry_price1=effective_price1,
                    entry_price2=effective_price2,
                    exit_price1=np.nan,
                    exit_price2=np.nan,
                    quantity1=quantity1,
                    quantity2=quantity2,
                    pnl=np.nan,
                    duration=0,
                    beta=beta
                )
                self.trades.append(trade)
    
    def _calculate_positions(
        self,
        date: pd.Timestamp,
        price1: float,
        price2: float,
        signal: int,
        beta: float,
        portfolio_value: float
    ) -> Tuple[float, float, float]:
        """Calculate current positions."""
        # This is a simplified version - in practice, you'd track positions more carefully
        max_position_value = portfolio_value * self.max_position_size
        
        if signal == 1:
            quantity1 = max_position_value / price1
            quantity2 = -quantity1 * beta
            cash = portfolio_value - quantity1 * price1 - quantity2 * price2
        elif signal == -1:
            quantity1 = -max_position_value / price1
            quantity2 = -quantity1 * beta
            cash = portfolio_value - quantity1 * price1 - quantity2 * price2
        else:
            quantity1 = 0
            quantity2 = 0
            cash = portfolio_value
        
        return quantity1, quantity2, cash
    
    def _calculate_metrics(
        self,
        returns: pd.Series,
        equity_curve: pd.Series
    ) -> Dict[str, float]:
        """Calculate performance metrics."""
        metrics = {}
        
        # Basic metrics
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
        metrics['total_return'] = total_return
        
        # Annualized metrics
        years = len(returns) / 252
        annualized_return = (1 + total_return) ** (1 / years) - 1
        metrics['annualized_return'] = annualized_return
        
        # Volatility
        annualized_volatility = returns.std() * np.sqrt(252)
        metrics['annualized_volatility'] = annualized_volatility
        
        # Sharpe ratio
        risk_free_rate = 0.02
        sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility
        metrics['sharpe_ratio'] = sharpe_ratio
        
        # Sortino ratio
        downside_returns = returns[returns < 0]
        downside_volatility = downside_returns.std() * np.sqrt(252)
        sortino_ratio = (annualized_return - risk_free_rate) / downside_volatility
        metrics['sortino_ratio'] = sortino_ratio
        
        # Maximum drawdown
        max_dd, _, _ = self._calculate_max_drawdown(equity_curve)
        metrics['max_drawdown'] = max_dd
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_dd)
        metrics['calmar_ratio'] = calmar_ratio
        
        # Hit rate
        profitable_trades = [t for t in self.trades if t.pnl > 0]
        hit_rate = len(profitable_trades) / len(self.trades) if self.trades else 0
        metrics['hit_rate'] = hit_rate
        
        # Profit factor
        gross_profit = sum(t.pnl for t in self.trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in self.trades if t.pnl < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
        metrics['profit_factor'] = profit_factor
        
        return metrics
    
    def _calculate_max_drawdown(self, equity_curve: pd.Series) -> Tuple[float, pd.Timestamp, pd.Timestamp]:
        """Calculate maximum drawdown."""
        cumulative = equity_curve / equity_curve.iloc[0]
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        max_dd = drawdown.min()
        trough_date = drawdown.idxmin()
        peak_date = running_max.loc[:trough_date].idxmax()
        
        return max_dd, peak_date, trough_date
    
    def _calculate_drawdowns(self, equity_curve: pd.Series) -> pd.Series:
        """Calculate drawdown series."""
        cumulative = equity_curve / equity_curve.iloc[0]
        running_max = cumulative.expanding().max()
        drawdowns = (cumulative - running_max) / running_max
        return drawdowns


class VectorBTBacktester:
    """VectorBT-based backtester for pairs trading."""
    
    def __init__(self, initial_capital: float = 100000):
        """Initialize VectorBT backtester.
        
        Args:
            initial_capital: Initial capital amount.
        """
        self.initial_capital = initial_capital
    
    def run_backtest(
        self,
        prices: pd.DataFrame,
        signals: pd.DataFrame,
        beta: float
    ) -> Dict[str, Any]:
        """Run backtest using VectorBT.
        
        Args:
            prices: DataFrame with price data.
            signals: DataFrame with trading signals.
            beta: Hedge ratio.
            
        Returns:
            Dictionary with backtest results.
        """
        # Create portfolio using VectorBT
        portfolio = vbt.Portfolio.from_signals(
            prices,
            signals,
            init_cash=self.initial_capital,
            fees=0.001,
            slippage=0.0005
        )
        
        # Extract results
        results = {
            'equity_curve': portfolio.value(),
            'returns': portfolio.returns(),
            'total_return': portfolio.total_return(),
            'sharpe_ratio': portfolio.sharpe_ratio(),
            'max_drawdown': portfolio.max_drawdown(),
            'trades': portfolio.trades.records_readable,
            'stats': portfolio.stats()
        }
        
        return results
