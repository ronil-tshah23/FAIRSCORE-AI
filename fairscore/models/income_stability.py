"""
Income Stability Model (Is) for FairScore.
Uses LightGBM for predicting income stability score.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any
from pathlib import Path
import joblib

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report

from fairscore.config import Config


class IncomeStabilityModel:
    """
    Income Stability Model using LightGBM.
    
    Evaluates the regularity and predictability of income inflows.
    Higher scores indicate more stable and predictable income patterns.
    
    Features used:
    - is_income_regularity
    - is_income_variance
    - is_income_frequency
    - is_salary_consistency
    - is_income_growth
    """
    
    FEATURE_NAMES = [
        "is_income_regularity",
        "is_income_variance", 
        "is_income_frequency",
        "is_salary_consistency",
        "is_income_growth",
    ]
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize Income Stability Model."""
        self.config = config or Config()
        self.model = None
        self.is_fitted = False
        
        # LightGBM parameters optimized for memory efficiency
        self.params = {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "num_leaves": self.config.model.is_num_leaves,
            "max_depth": self.config.model.is_max_depth,
            "learning_rate": self.config.model.is_learning_rate,
            "n_estimators": self.config.model.is_n_estimators,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "random_state": self.config.data.random_seed,
            "n_jobs": 2,  # Limit for memory efficiency
            "verbose": -1,
        }
    
    def _create_target(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        Create target variable for training.
        
        Target is binary: 1 for stable income (top 50%), 0 for unstable.
        Based on composite score of income features.
        """
        # Calculate composite income stability score
        composite = (
            features_df["is_income_regularity"] * 0.3 +
            (1 - features_df["is_income_variance"]) * 0.25 +
            features_df["is_income_frequency"] * 0.2 +
            features_df["is_salary_consistency"] * 0.15 +
            features_df["is_income_growth"].clip(0, 1) * 0.1
        )
        
        # Convert to binary (median split)
        threshold = composite.median()
        return (composite > threshold).astype(int).values
    
    def train(
        self,
        features_df: pd.DataFrame,
        test_size: float = 0.2,
    ) -> Dict[str, Any]:
        """
        Train the Income Stability Model.
        
        Args:
            features_df: DataFrame with engineered features
            test_size: Fraction for validation
            
        Returns:
            Dictionary with training metrics
        """
        if lgb is None:
            raise ImportError("LightGBM is required for IncomeStabilityModel")
        
        # Prepare features and target
        X = features_df[self.FEATURE_NAMES].values
        y = self._create_target(features_df)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=self.config.data.random_seed
        )
        
        # Create LightGBM datasets
        train_data = lgb.Dataset(X_train, label=y_train, feature_name=self.FEATURE_NAMES)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # Train with early stopping
        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=self.params["n_estimators"],
            valid_sets=[train_data, val_data],
            valid_names=["train", "valid"],
            callbacks=[
                lgb.early_stopping(stopping_rounds=20),
                lgb.log_evaluation(period=0),  # Suppress logging
            ],
        )
        
        self.is_fitted = True
        
        # Evaluate
        y_pred_proba = self.model.predict(X_val)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        metrics = {
            "model_name": "Income Stability (Is)",
            "algorithm": "LightGBM",
            "auc_score": roc_auc_score(y_val, y_pred_proba),
            "accuracy": accuracy_score(y_val, y_pred),
            "best_iteration": self.model.best_iteration,
            "feature_importance": dict(zip(
                self.FEATURE_NAMES,
                self.model.feature_importance(importance_type="gain").tolist()
            )),
        }
        
        return metrics
    
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """
        Predict income stability score.
        
        Args:
            features: DataFrame with feature columns
            
        Returns:
            Array of scores between 0 and 1
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call train() first or load a saved model.")
        
        X = features[self.FEATURE_NAMES].values
        return self.model.predict(X)
    
    def predict_single(self, feature_dict: Dict[str, float]) -> float:
        """
        Predict score for a single user from manual input.
        
        Args:
            feature_dict: Dictionary with feature values
            
        Returns:
            Score between 0 and 1
        """
        df = pd.DataFrame([feature_dict])
        return self.predict(df)[0]
    
    def save(self, path: Optional[Path] = None):
        """Save model to disk."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Nothing to save.")
        
        path = path or self.config.models_dir / "income_stability_model.pkl"
        joblib.dump(self.model, path)
        print(f"Saved Income Stability Model to: {path}")
    
    def load(self, path: Optional[Path] = None):
        """Load model from disk."""
        path = path or self.config.models_dir / "income_stability_model.pkl"
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        
        self.model = joblib.load(path)
        self.is_fitted = True
        print(f"Loaded Income Stability Model from: {path}")
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from trained model."""
        if not self.is_fitted:
            raise ValueError("Model not fitted.")
        
        importance = self.model.feature_importance(importance_type="gain")
        return dict(zip(self.FEATURE_NAMES, importance.tolist()))
