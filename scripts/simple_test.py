#!/usr/bin/env python3
"""Simple script to test the pairs trading strategy."""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings

# Import our modules
from src.data import DataLoader, preprocess_data, create_pairs_data
from src.features import PairsFeatureEngineer
from src.labels import PairsLabelGenerator, LabelConfig, LabelMethod
from src.models import CointegrationBaseline
from src.backtest import PairsBacktester
from src.utils import set_random_seeds

warnings.filterwarnings("ignore")

def main():
    """Run a simple test of the pairs trading strategy."""
    print("Pairs Trading Strategy - Simple Test")
    print("=" * 50)
    
    # Set random seeds
    set_random_seeds(42)
    
    # 1. Load data
    print("\n1. Loading data...")
    data_loader = DataLoader("synthetic")  # Use synthetic data for testing
    symbols = ["AAPL", "MSFT"]
    
    try:
        stock_data = data_loader.load_stock_data(
            symbols=symbols,
            start_date="2020-01-01",
            end_date="2023-12-31"
        )
        print(f"✓ Loaded data: {stock_data.shape}")
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return
    
    # 2. Preprocess data
    print("\n2. Preprocessing data...")
    processed_data = preprocess_data(stock_data)
    pairs_data = create_pairs_data(processed_data, symbols)
    print(f"✓ Created {len(pairs_data)} pairs")
    
    # 3. Analyze first pair
    pair_name = list(pairs_data.keys())[0]
    pair_data = pairs_data[pair_name]
    symbol1, symbol2 = pair_data.columns
    
    print(f"\n3. Analyzing pair: {pair_name}")
    
    # Calculate cointegration
    feature_engineer = PairsFeatureEngineer()
    coint_features = feature_engineer.calculate_cointegration_features(pair_data)
    
    print(f"✓ Cointegration p-value: {coint_features['coint_pvalue']:.4f}")
    print(f"✓ Is cointegrated: {coint_features['is_cointegrated']}")
    print(f"✓ Beta: {coint_features['beta']:.3f}")
    
    if not coint_features['is_cointegrated']:
        print("⚠️  Pair is not cointegrated, but continuing for demo...")
    
    # 4. Generate features and labels
    print("\n4. Generating features and labels...")
    features = feature_engineer.create_feature_matrix(pair_data)
    
    # Calculate spread
    beta = coint_features['beta']
    spread = pair_data[symbol1] - beta * pair_data[symbol2]
    
    # Generate labels
    label_config = LabelConfig(
        method=LabelMethod.THRESHOLD_BASED,
        entry_threshold=1.0,
        exit_threshold=0.5
    )
    label_generator = PairsLabelGenerator(label_config)
    labels = label_generator.generate_labels(pair_data, spread, beta, symbol1, symbol2)
    
    print(f"✓ Generated features: {features.shape}")
    print(f"✓ Generated labels: {labels.value_counts().to_dict()}")
    
    # 5. Train model
    print("\n5. Training model...")
    model = CointegrationBaseline(entry_threshold=1.0, exit_threshold=0.5)
    model.fit(features, labels)
    model.set_cointegration_params(
        coint_features['beta'],
        coint_features['spread_mean'],
        coint_features['spread_std'],
        coint_features['is_cointegrated']
    )
    print("✓ Model trained")
    
    # 6. Generate predictions
    print("\n6. Generating predictions...")
    predictions = model.predict(features)
    print(f"✓ Predictions: {np.unique(predictions, return_counts=True)}")
    
    # 7. Run backtest
    print("\n7. Running backtest...")
    backtester = PairsBacktester(
        initial_capital=100000,
        transaction_cost=0.001,
        slippage=0.0005
    )
    
    backtest_result = backtester.run_backtest(
        pair_data,
        pd.Series(predictions, index=features.index),
        beta,
        symbol1,
        symbol2
    )
    
    print("✓ Backtest completed")
    print(f"✓ Total return: {backtest_result.metrics['total_return']:.2%}")
    print(f"✓ Sharpe ratio: {backtest_result.metrics['sharpe_ratio']:.2f}")
    print(f"✓ Max drawdown: {backtest_result.metrics['max_drawdown']:.2%}")
    print(f"✓ Number of trades: {len(backtest_result.trades)}")
    
    # 8. Create simple visualization
    print("\n8. Creating visualization...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot spread and signals
    ax1.plot(spread.index, spread, label='Spread', alpha=0.7)
    ax1.scatter(spread.index[predictions == 1], spread[predictions == 1], 
                color='green', marker='^', label='Buy Signal', s=50)
    ax1.scatter(spread.index[predictions == -1], spread[predictions == -1], 
                color='red', marker='v', label='Sell Signal', s=50)
    ax1.set_title(f'Spread and Trading Signals - {pair_name}')
    ax1.set_ylabel('Spread')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot equity curve
    ax2.plot(backtest_result.equity_curve.index, backtest_result.equity_curve.values)
    ax2.set_title('Portfolio Equity Curve')
    ax2.set_ylabel('Portfolio Value ($)')
    ax2.set_xlabel('Date')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_dir = Path("assets")
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / "simple_test_results.png", dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved to {output_dir / 'simple_test_results.png'}")
    
    plt.show()
    
    print("\n" + "=" * 50)
    print("Simple test completed successfully!")
    print("=" * 50)

if __name__ == "__main__":
    main()
