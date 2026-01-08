# Pairs Trading Strategy

A research-ready implementation of pairs trading strategy using cointegration and advanced machine learning models.

## ⚠️ IMPORTANT DISCLAIMER

**This is a research and educational demonstration only. It is NOT investment advice.**

- Past performance does not guarantee future results
- Trading involves substantial risk of loss
- Always consult with a qualified financial advisor before making investment decisions
- The models and strategies shown here may be inaccurate, incomplete, or unsuitable for your specific situation
- Use at your own risk

## Overview

This project implements a comprehensive pairs trading strategy that:

- Identifies cointegrated pairs of stocks using statistical tests
- Generates trading signals using multiple approaches (threshold-based, Kalman filter, ML models)
- Performs realistic backtesting with transaction costs and slippage
- Includes risk management and position sizing
- Provides interactive visualization and analysis tools

## Features

### Core Functionality
- **Cointegration Analysis**: Engle-Granger cointegration tests to identify suitable pairs
- **Multiple Models**: Baseline cointegration, Kalman filter, and ML-based approaches
- **Realistic Backtesting**: Transaction costs, slippage, and position sizing
- **Risk Management**: Drawdown control, stop-loss, take-profit, and portfolio-level risk metrics
- **Interactive Demo**: Streamlit-based web application for strategy exploration

### Advanced Features
- **Feature Engineering**: Technical indicators, spread statistics, and pair-specific features
- **Label Generation**: Multiple labeling methods (threshold-based, triple-barrier, regime-based)
- **Ensemble Methods**: Combination of multiple models for improved performance
- **Comprehensive Evaluation**: Both ML metrics and financial performance metrics
- **Reproducible Research**: Deterministic seeding and proper data splits

## Installation

### Prerequisites
- Python 3.10 or higher
- pip or conda package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/kryptologyst/Pairs-Trading-Strategy.git
cd Pairs-Trading-Strategy
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Or using conda:
```bash
conda env create -f environment.yml
conda activate pairs-trading
```

3. Install the package in development mode:
```bash
pip install -e .
```

## Quick Start

### 1. Run the Complete Pipeline

```bash
python scripts/run_pipeline.py
```

This will:
- Load stock data for configured symbols
- Identify cointegrated pairs
- Generate features and labels
- Train multiple models
- Run backtests
- Generate evaluation results

### 2. Launch Interactive Demo

```bash
streamlit run demo/streamlit_app.py
```

This opens a web interface where you can:
- Select different stock pairs
- Adjust strategy parameters
- Visualize spread and signals
- Run backtests interactively
- View performance metrics

### 3. Use Individual Components

```python
from src.data import DataLoader
from src.features import PairsFeatureEngineer
from src.models import CointegrationBaseline

# Load data
data_loader = DataLoader("yfinance")
data = data_loader.load_stock_data(["AAPL", "MSFT"], "2020-01-01", "2023-12-31")

# Create pairs
pairs_data = create_pairs_data(data, ["AAPL", "MSFT"])

# Analyze cointegration
feature_engineer = PairsFeatureEngineer()
coint_features = feature_engineer.calculate_cointegration_features(pairs_data["AAPL_MSFT"])

# Generate signals
model = CointegrationBaseline()
# ... (see examples in notebooks/)
```

## Project Structure

```
pairs-trading-strategy/
├── src/                    # Source code
│   ├── data/              # Data loading and preprocessing
│   ├── features/          # Feature engineering
│   ├── labels/            # Label generation
│   ├── models/            # Trading models
│   ├── backtest/          # Backtesting engine
│   ├── risk/              # Risk management
│   └── utils/             # Utility functions
├── configs/               # Configuration files
├── scripts/               # Pipeline scripts
├── notebooks/             # Jupyter notebooks
├── tests/                 # Unit tests
├── assets/                # Output files and results
├── demo/                  # Streamlit demo
├── data/                  # Data storage
│   ├── raw/              # Raw data files
│   └── processed/        # Processed data files
├── requirements.txt       # Python dependencies
├── pyproject.toml        # Project configuration
└── README.md             # This file
```

## Configuration

The strategy behavior is controlled through `configs/config.yaml`:

```yaml
# Data Configuration
data:
  symbols: ["AAPL", "MSFT", "GOOGL", "AMZN"]
  start_date: "2020-01-01"
  end_date: "2023-12-31"
  data_source: "yfinance"

# Feature Engineering
features:
  lookback_window: 252
  min_cointegration_pvalue: 0.05
  spread_threshold_multiplier: 1.0

# Models
models:
  baseline:
    name: "cointegration_baseline"
    params:
      entry_threshold: 1.0
      exit_threshold: 0.5

# Backtesting
backtesting:
  initial_capital: 100000
  transaction_cost: 0.001
  slippage: 0.0005
  max_position_size: 0.1

# Risk Management
risk:
  max_drawdown: 0.15
  stop_loss: 0.05
  take_profit: 0.10
  position_sizing: "kelly"
```

## Models

### 1. Cointegration Baseline
- Uses Engle-Granger cointegration test
- Generates signals based on spread z-scores
- Simple threshold-based entry/exit rules

### 2. Kalman Filter
- Dynamic hedge ratio estimation
- Adapts to changing market conditions
- Uses prediction uncertainty for position sizing

### 3. Machine Learning Models
- XGBoost, LightGBM, Random Forest
- Uses technical indicators and spread features
- Trained on historical signal performance

### 4. Ensemble Methods
- Combines multiple models
- Weighted voting or averaging
- Improved robustness and performance

## Evaluation Metrics

### Financial Metrics
- **Total Return**: Overall portfolio return
- **Sharpe Ratio**: Risk-adjusted return
- **Sortino Ratio**: Downside risk-adjusted return
- **Calmar Ratio**: Return to max drawdown ratio
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Hit Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profit / gross loss

### Machine Learning Metrics
- **Accuracy**: Correct signal predictions
- **Precision/Recall**: Signal quality metrics
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under ROC curve

## Risk Management

### Position Sizing Methods
- **Fixed**: Constant position size
- **Volatility Adjusted**: Size inversely proportional to volatility
- **Kelly Criterion**: Optimal position size based on expected return and volatility
- **Risk Parity**: Equal risk contribution across positions

### Risk Controls
- Maximum position size limits
- Correlation-based position limits
- Drawdown controls
- Stop-loss and take-profit levels
- Portfolio-level risk monitoring

## Data Sources

### Supported Data Sources
- **Yahoo Finance**: Free historical data via yfinance
- **Synthetic Data**: Generated data for testing and development

### Data Schema
```
market_data.csv:
- datetime: Trading date
- symbol: Stock symbol
- open, high, low, close: OHLC prices
- volume: Trading volume
```

## Development

### Running Tests
```bash
pytest tests/
```

### Code Formatting
```bash
black src/ scripts/ demo/
ruff check src/ scripts/ demo/
```

### Pre-commit Hooks
```bash
pre-commit install
pre-commit run --all-files
```

## Examples and Tutorials

See the `notebooks/` directory for detailed examples:

- `01_data_loading.ipynb`: Data loading and preprocessing
- `02_cointegration_analysis.ipynb`: Cointegration testing and pair selection
- `03_feature_engineering.ipynb`: Feature creation and selection
- `04_model_training.ipynb`: Model training and evaluation
- `05_backtesting.ipynb`: Backtesting and performance analysis
- `06_risk_management.ipynb`: Risk management and position sizing

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{pairs_trading_strategy,
  title={Pairs Trading Strategy: A Modern Implementation},
  author={Kryptologyst},
  year={2026},
  url={https://github.com/kryptologyst/Pairs-Trading-Strategy}
}
```

## Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Check the documentation in `docs/`
- Review the example notebooks

## Acknowledgments

- Built with modern Python data science stack
- Inspired by academic research on pairs trading
- Uses industry-standard risk management practices
- Designed for reproducibility and research use

---

**Remember: This is for research and educational purposes only. Not investment advice.**
# Pairs-Trading-Strategy
