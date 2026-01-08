"""Streamlit demo for pairs trading strategy."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Any
import warnings

# Import our modules
from src.data import DataLoader, preprocess_data, create_pairs_data
from src.features import PairsFeatureEngineer
from src.labels import PairsLabelGenerator, LabelConfig, LabelMethod
from src.models import CointegrationBaseline, KalmanFilterModel, MLPairsModel
from src.backtest import PairsBacktester
from src.risk import RiskManager, RiskConfig, PositionSizingMethod
from src.utils import set_random_seeds, calculate_sharpe_ratio, calculate_max_drawdown

warnings.filterwarnings("ignore")

# Page configuration
st.set_page_config(
    page_title="Pairs Trading Strategy",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .disclaimer {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
        color: #856404;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Disclaimer
st.markdown("""
<div class="disclaimer">
    <h4>⚠️ IMPORTANT DISCLAIMER</h4>
    <p><strong>This is a research and educational demonstration only.</strong></p>
    <p>This application is for research and educational purposes only. It is NOT investment advice. 
    Past performance does not guarantee future results. Trading involves substantial risk of loss. 
    Always consult with a qualified financial advisor before making investment decisions.</p>
    <p>The models and strategies shown here may be inaccurate, incomplete, or unsuitable for your 
    specific situation. Use at your own risk.</p>
</div>
""", unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header">📈 Pairs Trading Strategy Demo</h1>', unsafe_allow_html=True)

# Sidebar configuration
st.sidebar.header("Configuration")

# Load default configuration
@st.cache_data
def load_config():
    """Load configuration file."""
    try:
        with open("configs/config.yaml", 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        # Default configuration if file not found
        return {
            'data': {
                'symbols': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX'],
                'start_date': '2020-01-01',
                'end_date': '2023-12-31',
                'data_source': 'yfinance'
            },
            'features': {
                'lookback_window': 252,
                'min_cointegration_pvalue': 0.05,
                'spread_threshold_multiplier': 1.0
            },
            'backtesting': {
                'initial_capital': 100000,
                'transaction_cost': 0.001,
                'slippage': 0.0005,
                'max_position_size': 0.1
            }
        }

config = load_config()

# Sidebar controls
st.sidebar.subheader("Data Configuration")
symbols = st.sidebar.multiselect(
    "Select Stock Symbols",
    options=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX', 'JPM', 'BAC', 'WMT', 'PG'],
    default=config['data']['symbols'][:4]
)

start_date = st.sidebar.date_input(
    "Start Date",
    value=pd.to_datetime(config['data']['start_date']).date()
)

end_date = st.sidebar.date_input(
    "End Date",
    value=pd.to_datetime(config['data']['end_date']).date()
)

st.sidebar.subheader("Strategy Parameters")
entry_threshold = st.sidebar.slider(
    "Entry Threshold (Std Dev)",
    min_value=0.5,
    max_value=3.0,
    value=config['features']['spread_threshold_multiplier'],
    step=0.1
)

exit_threshold = st.sidebar.slider(
    "Exit Threshold (Std Dev)",
    min_value=0.1,
    max_value=2.0,
    value=config['features']['spread_threshold_multiplier'] * 0.5,
    step=0.1
)

min_pvalue = st.sidebar.slider(
    "Min Cointegration P-value",
    min_value=0.01,
    max_value=0.1,
    value=config['features']['min_cointegration_pvalue'],
    step=0.01
)

st.sidebar.subheader("Backtesting Parameters")
initial_capital = st.sidebar.number_input(
    "Initial Capital ($)",
    min_value=10000,
    max_value=1000000,
    value=config['backtesting']['initial_capital'],
    step=10000
)

transaction_cost = st.sidebar.slider(
    "Transaction Cost (%)",
    min_value=0.0,
    max_value=0.01,
    value=config['backtesting']['transaction_cost'],
    step=0.0001,
    format="%.4f"
)

# Main content
if len(symbols) < 2:
    st.warning("Please select at least 2 symbols to analyze pairs.")
    st.stop()

# Load data
@st.cache_data
def load_stock_data(symbols, start_date, end_date):
    """Load stock data."""
    data_loader = DataLoader("yfinance")
    try:
        data = data_loader.load_stock_data(
            symbols=symbols,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d')
        )
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

with st.spinner("Loading stock data..."):
    stock_data = load_stock_data(symbols, start_date, end_date)

if stock_data is None:
    st.error("Failed to load data. Please try different symbols or date range.")
    st.stop()

# Process data
processed_data = preprocess_data(stock_data)
pairs_data = create_pairs_data(processed_data, symbols)

if not pairs_data:
    st.error("No valid pairs found. Please select different symbols.")
    st.stop()

# Select pair for analysis
st.subheader("📊 Pair Analysis")
pair_names = list(pairs_data.keys())
selected_pair = st.selectbox("Select Pair to Analyze", pair_names)

if selected_pair:
    pair_data = pairs_data[selected_pair]
    symbol1, symbol2 = pair_data.columns
    
    # Calculate cointegration
    feature_engineer = PairsFeatureEngineer()
    coint_features = feature_engineer.calculate_cointegration_features(
        pair_data, min_pvalue
    )
    
    # Display cointegration results
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Cointegration P-value",
            f"{coint_features['coint_pvalue']:.4f}",
            delta="✓ Cointegrated" if coint_features['is_cointegrated'] else "✗ Not Cointegrated"
        )
    
    with col2:
        st.metric("Beta (Hedge Ratio)", f"{coint_features['beta']:.3f}")
    
    with col3:
        st.metric("R-squared", f"{coint_features['r_squared']:.3f}")
    
    with col4:
        st.metric("Half-life (days)", f"{coint_features['half_life']:.1f}")
    
    if not coint_features['is_cointegrated']:
        st.warning(f"Pair {selected_pair} is not cointegrated. Results may not be reliable.")
    
    # Calculate spread
    beta = coint_features['beta']
    spread = pair_data[symbol1] - beta * pair_data[symbol2]
    
    # Generate trading signals
    spread_mean = spread.rolling(window=20).mean()
    spread_std = spread.rolling(window=20).std()
    
    buy_signal = spread < (spread_mean - entry_threshold * spread_std)
    sell_signal = spread > (spread_mean + entry_threshold * spread_std)
    exit_long = spread > (spread_mean - exit_threshold * spread_std)
    exit_short = spread < (spread_mean + exit_threshold * spread_std)
    
    signals = pd.Series(0, index=spread.index)
    signals[buy_signal] = 1
    signals[sell_signal] = -1
    signals[(signals == 1) & exit_long] = 0
    signals[(signals == -1) & exit_short] = 0
    
    # Plot spread and signals
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Price Spread and Trading Signals', 'Stock Prices'),
        vertical_spacing=0.1
    )
    
    # Spread plot
    fig.add_trace(
        go.Scatter(
            x=spread.index,
            y=spread,
            name='Spread',
            line=dict(color='blue')
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=spread_mean.index,
            y=spread_mean,
            name='Mean',
            line=dict(color='green', dash='dash')
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=spread_mean.index,
            y=spread_mean - entry_threshold * spread_std,
            name='Buy Threshold',
            line=dict(color='red', dash='dash')
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=spread_mean.index,
            y=spread_mean + entry_threshold * spread_std,
            name='Sell Threshold',
            line=dict(color='purple', dash='dash')
        ),
        row=1, col=1
    )
    
    # Add signal markers
    buy_points = spread[buy_signal]
    sell_points = spread[sell_signal]
    
    if len(buy_points) > 0:
        fig.add_trace(
            go.Scatter(
                x=buy_points.index,
                y=buy_points.values,
                mode='markers',
                marker=dict(symbol='triangle-up', size=10, color='green'),
                name='Buy Signal'
            ),
            row=1, col=1
        )
    
    if len(sell_points) > 0:
        fig.add_trace(
            go.Scatter(
                x=sell_points.index,
                y=sell_points.values,
                mode='markers',
                marker=dict(symbol='triangle-down', size=10, color='red'),
                name='Sell Signal'
            ),
            row=1, col=1
        )
    
    # Price plot
    fig.add_trace(
        go.Scatter(
            x=pair_data.index,
            y=pair_data[symbol1],
            name=symbol1,
            line=dict(color='blue')
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=pair_data.index,
            y=pair_data[symbol2],
            name=symbol2,
            line=dict(color='orange')
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        height=800,
        showlegend=True,
        title_text=f"Pairs Trading Analysis: {selected_pair}"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Run backtest
    st.subheader("🚀 Backtest Results")
    
    if st.button("Run Backtest", type="primary"):
        with st.spinner("Running backtest..."):
            backtester = PairsBacktester(
                initial_capital=initial_capital,
                transaction_cost=transaction_cost,
                slippage=0.0005,
                max_position_size=0.1
            )
            
            backtest_result = backtester.run_backtest(
                pair_data,
                signals,
                beta,
                symbol1,
                symbol2
            )
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Total Return",
                    f"{backtest_result.metrics['total_return']:.1%}",
                    delta=f"{backtest_result.metrics['annualized_return']:.1%} annualized"
                )
            
            with col2:
                st.metric(
                    "Sharpe Ratio",
                    f"{backtest_result.metrics['sharpe_ratio']:.2f}"
                )
            
            with col3:
                st.metric(
                    "Max Drawdown",
                    f"{backtest_result.metrics['max_drawdown']:.1%}"
                )
            
            with col4:
                st.metric(
                    "Hit Rate",
                    f"{backtest_result.metrics['hit_rate']:.1%}"
                )
            
            # Plot equity curve
            fig_equity = go.Figure()
            
            fig_equity.add_trace(
                go.Scatter(
                    x=backtest_result.equity_curve.index,
                    y=backtest_result.equity_curve.values,
                    name='Portfolio Value',
                    line=dict(color='blue', width=2)
                )
            )
            
            fig_equity.update_layout(
                title="Portfolio Equity Curve",
                xaxis_title="Date",
                yaxis_title="Portfolio Value ($)",
                height=400
            )
            
            st.plotly_chart(fig_equity, use_container_width=True)
            
            # Display trades
            if backtest_result.trades:
                st.subheader("📋 Trade History")
                
                trades_data = []
                for trade in backtest_result.trades:
                    trades_data.append({
                        'Entry Date': trade.entry_date.strftime('%Y-%m-%d'),
                        'Exit Date': trade.exit_date.strftime('%Y-%m-%d') if pd.notna(trade.exit_date) else 'Open',
                        'Position': 'Long Spread' if trade.position_type.value == 1 else 'Short Spread',
                        'P&L': f"${trade.pnl:.2f}" if pd.notna(trade.pnl) else 'N/A',
                        'Duration': f"{trade.duration} days"
                    })
                
                trades_df = pd.DataFrame(trades_data)
                st.dataframe(trades_df, use_container_width=True)
            else:
                st.info("No trades executed during the backtest period.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.8rem;">
    <p>This is a research and educational demonstration only. Not investment advice.</p>
    <p>Built with Streamlit | Pairs Trading Strategy Demo</p>
</div>
""", unsafe_allow_html=True)
