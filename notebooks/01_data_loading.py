"""Example notebook: Data Loading and Preprocessing."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

# Import our modules
from src.data import DataLoader, preprocess_data, create_pairs_data
from src.utils import set_random_seeds

warnings.filterwarnings("ignore")

# Set random seeds for reproducibility
set_random_seeds(42)

# Configure plotting
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("Pairs Trading Strategy - Data Loading Example")
print("=" * 50)

# 1. Initialize Data Loader
print("\n1. Initializing Data Loader...")
data_loader = DataLoader("yfinance")

# 2. Load Stock Data
print("\n2. Loading Stock Data...")
symbols = ["AAPL", "MSFT", "GOOGL", "AMZN"]
start_date = "2020-01-01"
end_date = "2023-12-31"

try:
    stock_data = data_loader.load_stock_data(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        fields=["Close", "Volume"]
    )
    print(f"✓ Loaded data for {len(symbols)} symbols")
    print(f"✓ Data shape: {stock_data.shape}")
    print(f"✓ Date range: {stock_data.index.min()} to {stock_data.index.max()}")
except Exception as e:
    print(f"✗ Error loading data: {e}")
    print("Using synthetic data instead...")
    stock_data = data_loader.load_stock_data(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        fields=["Close"]
    )

# 3. Display Data Info
print("\n3. Data Information:")
print(stock_data.info())
print("\nFirst few rows:")
print(stock_data.head())

# 4. Preprocess Data
print("\n4. Preprocessing Data...")
processed_data = preprocess_data(
    stock_data,
    remove_outliers=True,
    outlier_threshold=3.0,
    fill_method="forward"
)
print(f"✓ Processed data shape: {processed_data.shape}")

# 5. Create Pairs Data
print("\n5. Creating Pairs Data...")
pairs_data = create_pairs_data(processed_data, symbols)
print(f"✓ Created {len(pairs_data)} pairs")

for pair_name, pair_data in pairs_data.items():
    print(f"  - {pair_name}: {len(pair_data)} observations")

# 6. Visualize Data
print("\n6. Creating Visualizations...")

# Plot individual stock prices
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.flatten()

for i, symbol in enumerate(symbols):
    col_name = f"{symbol}_Close"
    if col_name in processed_data.columns:
        axes[i].plot(processed_data.index, processed_data[col_name])
        axes[i].set_title(f"{symbol} Price")
        axes[i].set_ylabel("Price ($)")
        axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("assets/stock_prices.png", dpi=300, bbox_inches='tight')
plt.show()

# Plot price correlations
price_cols = [f"{symbol}_Close" for symbol in symbols if f"{symbol}_Close" in processed_data.columns]
correlation_matrix = processed_data[price_cols].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title("Stock Price Correlations")
plt.tight_layout()
plt.savefig("assets/price_correlations.png", dpi=300, bbox_inches='tight')
plt.show()

# Plot pairs data
if pairs_data:
    pair_name = list(pairs_data.keys())[0]
    pair_data = pairs_data[pair_name]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot individual prices
    ax1.plot(pair_data.index, pair_data.iloc[:, 0], label=pair_data.columns[0])
    ax1.plot(pair_data.index, pair_data.iloc[:, 1], label=pair_data.columns[1])
    ax1.set_title(f"Price Comparison: {pair_name}")
    ax1.set_ylabel("Price ($)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot price ratio
    price_ratio = pair_data.iloc[:, 0] / pair_data.iloc[:, 1]
    ax2.plot(pair_data.index, price_ratio)
    ax2.set_title(f"Price Ratio: {pair_name}")
    ax2.set_ylabel("Price Ratio")
    ax2.set_xlabel("Date")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("assets/pair_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()

# 7. Save Processed Data
print("\n7. Saving Processed Data...")
output_dir = Path("data/processed")
output_dir.mkdir(parents=True, exist_ok=True)

processed_data.to_csv(output_dir / "stock_data.csv")
print(f"✓ Saved processed data to {output_dir / 'stock_data.csv'}")

# Save pairs data
for pair_name, pair_data in pairs_data.items():
    pair_data.to_csv(output_dir / f"pair_{pair_name}.csv")
    print(f"✓ Saved {pair_name} data to {output_dir / f'pair_{pair_name}.csv'}")

print("\n" + "=" * 50)
print("Data Loading Example Completed Successfully!")
print("=" * 50)
