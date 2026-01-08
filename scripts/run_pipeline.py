#!/usr/bin/env python3
"""Main pipeline for pairs trading strategy."""

import pandas as pd
import numpy as np
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any
import warnings
from datetime import datetime

# Import our modules
from src.data import DataLoader, preprocess_data, create_pairs_data
from src.features import PairsFeatureEngineer
from src.labels import PairsLabelGenerator, LabelConfig, LabelMethod
from src.models import (
    CointegrationBaseline, KalmanFilterModel, MLPairsModel, PairsModelEnsemble
)
from src.backtest import PairsBacktester, BacktestResults
from src.risk import RiskManager, RiskConfig, PositionSizingMethod
from src.utils import (
    set_random_seeds, get_device, validate_data_splits,
    calculate_sharpe_ratio, calculate_max_drawdown
)

warnings.filterwarnings("ignore")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PairsTradingPipeline:
    """Main pipeline for pairs trading strategy."""
    
    def __init__(self, config_path: str):
        """Initialize pipeline.
        
        Args:
            config_path: Path to configuration file.
        """
        self.config = self._load_config(config_path)
        self.data_loader = DataLoader(self.config['data']['data_source'])
        self.feature_engineer = PairsFeatureEngineer(
            self.config['features']['lookback_window']
        )
        
        # Initialize risk manager
        risk_config = RiskConfig(
            max_drawdown=self.config['risk']['max_drawdown'],
            stop_loss=self.config['risk']['stop_loss'],
            take_profit=self.config['risk']['take_profit'],
            position_sizing_method=PositionSizingMethod(self.config['risk']['position_sizing'])
        )
        self.risk_manager = RiskManager(risk_config)
        
        # Initialize models
        self.models = self._initialize_models()
        
        # Set random seeds
        set_random_seeds(self.config['random_state'])
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def _initialize_models(self) -> Dict[str, Any]:
        """Initialize all models."""
        models = {}
        
        # Baseline model
        models['baseline'] = CointegrationBaseline(
            entry_threshold=self.config['models']['baseline']['params']['entry_threshold'],
            exit_threshold=self.config['models']['baseline']['params']['exit_threshold'],
            min_pvalue=self.config['features']['min_cointegration_pvalue']
        )
        
        # Kalman filter model
        models['kalman'] = KalmanFilterModel(
            initial_state_covariance=self.config['models']['advanced']['params']['initial_state_covariance'],
            process_noise=self.config['models']['advanced']['params']['process_noise'],
            measurement_noise=self.config['models']['advanced']['params']['measurement_noise']
        )
        
        # ML model
        models['ml'] = MLPairsModel(
            model_type=self.config['models']['ml_model']['name'],
            model_params=self.config['models']['ml_model']['params']
        )
        
        # Ensemble model
        models['ensemble'] = PairsModelEnsemble([
            models['baseline'],
            models['kalman'],
            models['ml']
        ])
        
        return models
    
    def load_and_preprocess_data(self) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """Load and preprocess data."""
        logger.info("Loading and preprocessing data")
        
        # Load stock data
        stock_data = self.data_loader.load_stock_data(
            symbols=self.config['data']['symbols'],
            start_date=self.config['data']['start_date'],
            end_date=self.config['data']['end_date']
        )
        
        # Preprocess data
        processed_data = preprocess_data(stock_data)
        
        # Create pairs data
        pairs_data = create_pairs_data(
            processed_data,
            self.config['data']['symbols']
        )
        
        logger.info(f"Loaded {len(processed_data)} rows of data for {len(pairs_data)} pairs")
        return processed_data, pairs_data
    
    def generate_features_and_labels(
        self,
        pairs_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Tuple[pd.DataFrame, pd.Series, Dict[str, float]]]:
        """Generate features and labels for all pairs."""
        logger.info("Generating features and labels")
        
        results = {}
        
        for pair_name, pair_data in pairs_data.items():
            logger.info(f"Processing pair: {pair_name}")
            
            # Calculate cointegration features
            coint_features = self.feature_engineer.calculate_cointegration_features(
                pair_data,
                self.config['features']['min_cointegration_pvalue']
            )
            
            # Skip if not cointegrated
            if not coint_features['is_cointegrated']:
                logger.warning(f"Pair {pair_name} is not cointegrated, skipping")
                continue
            
            # Generate features
            features = self.feature_engineer.create_feature_matrix(pair_data)
            
            # Generate labels
            label_config = LabelConfig(
                method=LabelMethod.THRESHOLD_BASED,
                entry_threshold=self.config['features']['spread_threshold_multiplier'],
                exit_threshold=self.config['features']['spread_threshold_multiplier'] * 0.5
            )
            label_generator = PairsLabelGenerator(label_config)
            
            # Calculate spread for labels
            symbol1, symbol2 = pair_data.columns
            beta = coint_features['beta']
            spread = pair_data[symbol1] - beta * pair_data[symbol2]
            
            # Generate labels
            labels = label_generator.generate_labels(
                pair_data, spread, beta, symbol1, symbol2
            )
            
            # Align features and labels
            common_index = features.index.intersection(labels.index)
            features_aligned = features.loc[common_index]
            labels_aligned = labels.loc[common_index]
            
            results[pair_name] = (features_aligned, labels_aligned, coint_features)
        
        logger.info(f"Generated features and labels for {len(results)} pairs")
        return results
    
    def train_models(
        self,
        features_labels: Dict[str, Tuple[pd.DataFrame, pd.Series, Dict[str, float]]]
    ) -> Dict[str, Any]:
        """Train all models."""
        logger.info("Training models")
        
        trained_models = {}
        
        for pair_name, (features, labels, coint_features) in features_labels.items():
            logger.info(f"Training models for pair: {pair_name}")
            
            # Create time-based splits
            train_end = self.config['data']['train_end']
            test_start = self.config['data']['test_start']
            
            train_features = features[features.index <= train_end]
            train_labels = labels[labels.index <= train_end]
            test_features = features[features.index >= test_start]
            test_labels = labels[labels.index >= test_start]
            
            if len(train_features) == 0 or len(test_features) == 0:
                logger.warning(f"Insufficient data for pair {pair_name}, skipping")
                continue
            
            pair_models = {}
            
            # Train baseline model
            baseline_model = CointegrationBaseline(
                entry_threshold=self.config['features']['spread_threshold_multiplier'],
                exit_threshold=self.config['features']['spread_threshold_multiplier'] * 0.5,
                min_pvalue=self.config['features']['min_cointegration_pvalue']
            )
            baseline_model.fit(train_features, train_labels)
            baseline_model.set_cointegration_params(
                coint_features['beta'],
                coint_features['spread_mean'],
                coint_features['spread_std'],
                coint_features['is_cointegrated']
            )
            pair_models['baseline'] = baseline_model
            
            # Train Kalman filter model
            kalman_model = KalmanFilterModel()
            kalman_model.fit(train_features, train_labels)
            pair_models['kalman'] = kalman_model
            
            # Train ML model
            ml_model = MLPairsModel(
                model_type=self.config['models']['ml_model']['name'],
                model_params=self.config['models']['ml_model']['params']
            )
            ml_model.fit(train_features, train_labels)
            pair_models['ml'] = ml_model
            
            # Train ensemble
            ensemble_model = PairsModelEnsemble([
                pair_models['baseline'],
                pair_models['kalman'],
                pair_models['ml']
            ])
            ensemble_model.fit(train_features, train_labels)
            pair_models['ensemble'] = ensemble_model
            
            trained_models[pair_name] = {
                'models': pair_models,
                'train_features': train_features,
                'train_labels': train_labels,
                'test_features': test_features,
                'test_labels': test_labels,
                'coint_features': coint_features
            }
        
        logger.info(f"Trained models for {len(trained_models)} pairs")
        return trained_models
    
    def run_backtests(
        self,
        trained_models: Dict[str, Any],
        pairs_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Dict[str, Any]]:
        """Run backtests for all models and pairs."""
        logger.info("Running backtests")
        
        backtest_results = {}
        
        for pair_name, model_data in trained_models.items():
            logger.info(f"Running backtest for pair: {pair_name}")
            
            pair_data = pairs_data[pair_name]
            coint_features = model_data['coint_features']
            
            pair_results = {}
            
            for model_name, model in model_data['models'].items():
                logger.info(f"Backtesting {model_name} model")
                
                # Get test data
                test_features = model_data['test_features']
                test_labels = model_data['test_labels']
                
                # Generate predictions
                predictions = model.predict(test_features)
                
                # Run backtest
                backtester = PairsBacktester(
                    initial_capital=self.config['backtesting']['initial_capital'],
                    transaction_cost=self.config['backtesting']['transaction_cost'],
                    slippage=self.config['backtesting']['slippage'],
                    max_position_size=self.config['backtesting']['max_position_size']
                )
                
                # Align test data with predictions
                test_data = pair_data.loc[test_features.index]
                
                backtest_result = backtester.run_backtest(
                    test_data,
                    pd.Series(predictions, index=test_features.index),
                    coint_features['beta'],
                    pair_data.columns[0],
                    pair_data.columns[1]
                )
                
                pair_results[model_name] = backtest_result
            
            backtest_results[pair_name] = pair_results
        
        logger.info(f"Completed backtests for {len(backtest_results)} pairs")
        return backtest_results
    
    def evaluate_results(
        self,
        backtest_results: Dict[str, Dict[str, Any]]
    ) -> pd.DataFrame:
        """Evaluate and compare results."""
        logger.info("Evaluating results")
        
        evaluation_data = []
        
        for pair_name, pair_results in backtest_results.items():
            for model_name, result in pair_results.items():
                metrics = result.metrics
                
                evaluation_data.append({
                    'pair': pair_name,
                    'model': model_name,
                    'total_return': metrics['total_return'],
                    'annualized_return': metrics['annualized_return'],
                    'annualized_volatility': metrics['annualized_volatility'],
                    'sharpe_ratio': metrics['sharpe_ratio'],
                    'sortino_ratio': metrics['sortino_ratio'],
                    'max_drawdown': metrics['max_drawdown'],
                    'calmar_ratio': metrics['calmar_ratio'],
                    'hit_rate': metrics['hit_rate'],
                    'profit_factor': metrics['profit_factor'],
                    'num_trades': len(result.trades)
                })
        
        evaluation_df = pd.DataFrame(evaluation_data)
        
        # Sort by Sharpe ratio
        evaluation_df = evaluation_df.sort_values('sharpe_ratio', ascending=False)
        
        logger.info("Evaluation completed")
        return evaluation_df
    
    def save_results(
        self,
        evaluation_df: pd.DataFrame,
        backtest_results: Dict[str, Dict[str, Any]],
        output_dir: str = "assets"
    ) -> None:
        """Save results to files."""
        logger.info("Saving results")
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save evaluation results
        evaluation_df.to_csv(output_path / "evaluation_results.csv")
        
        # Save detailed backtest results
        for pair_name, pair_results in backtest_results.items():
            pair_dir = output_path / pair_name
            pair_dir.mkdir(exist_ok=True)
            
            for model_name, result in pair_results.items():
                # Save equity curve
                result.equity_curve.to_csv(pair_dir / f"{model_name}_equity_curve.csv")
                
                # Save metrics
                metrics_df = pd.DataFrame([result.metrics])
                metrics_df.to_csv(pair_dir / f"{model_name}_metrics.csv")
                
                # Save trades
                if result.trades:
                    trades_data = []
                    for trade in result.trades:
                        trades_data.append({
                            'entry_date': trade.entry_date,
                            'exit_date': trade.exit_date,
                            'symbol1': trade.symbol1,
                            'symbol2': trade.symbol2,
                            'position_type': trade.position_type.value,
                            'pnl': trade.pnl,
                            'duration': trade.duration
                        })
                    trades_df = pd.DataFrame(trades_data)
                    trades_df.to_csv(pair_dir / f"{model_name}_trades.csv")
        
        logger.info(f"Results saved to {output_dir}")
    
    def run_full_pipeline(self) -> pd.DataFrame:
        """Run the complete pipeline."""
        logger.info("Starting pairs trading pipeline")
        
        # Load and preprocess data
        stock_data, pairs_data = self.load_and_preprocess_data()
        
        # Generate features and labels
        features_labels = self.generate_features_and_labels(pairs_data)
        
        # Train models
        trained_models = self.train_models(features_labels)
        
        # Run backtests
        backtest_results = self.run_backtests(trained_models, pairs_data)
        
        # Evaluate results
        evaluation_df = self.evaluate_results(backtest_results)
        
        # Save results
        self.save_results(evaluation_df, backtest_results)
        
        logger.info("Pipeline completed successfully")
        return evaluation_df


def main():
    """Main function."""
    # Load configuration
    config_path = "configs/config.yaml"
    
    # Initialize pipeline
    pipeline = PairsTradingPipeline(config_path)
    
    # Run pipeline
    results = pipeline.run_full_pipeline()
    
    # Print results
    print("\n" + "="*80)
    print("PAIRS TRADING STRATEGY RESULTS")
    print("="*80)
    print(results.to_string(index=False))
    print("="*80)
    
    # Print best performing model
    best_result = results.iloc[0]
    print(f"\nBest performing model: {best_result['model']} for pair {best_result['pair']}")
    print(f"Sharpe Ratio: {best_result['sharpe_ratio']:.3f}")
    print(f"Total Return: {best_result['total_return']:.3f}")
    print(f"Max Drawdown: {best_result['max_drawdown']:.3f}")


if __name__ == "__main__":
    main()
