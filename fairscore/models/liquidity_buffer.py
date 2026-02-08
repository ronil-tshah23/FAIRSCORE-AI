"""
Liquidity & Buffer Model (Ls) for FairScore.
Uses Logistic Regression for predicting liquidity and savings strength.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any
from pathlib import Path
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler

from fairscore.config import Config


class LiquidityBufferModel:
    """
    Liquidity & Buffer Model using Logistic Regression.
    
    Evaluates savings strength and emergency fund coverage.
    Higher scores indicate better financial cushion.
    
    Features used:
    - ls_savings_ratio
    - ls_avg_balance_ratio
    - ls_min_balance_rate
    - ls_emergency_fund
    - ls_balance_volatility
    """
    
    FEATURE_NAMES = [
        "ls_savings_ratio",
        "ls_avg_balance_ratio",
        "ls_min_balance_rate",
        "ls_emergency_fund",
        "ls_balance_volatility",
    ]
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize Liquidity Buffer Model."""
        self.config = config or Config()
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        # Logistic Regression parameters
        self.model = LogisticRegression(
            max_iter=self.config.model.ls_max_iter,
            C=self.config.model.ls_C,
            solver="lbfgs",
            class_weight="balanced",
            random_state=self.config.data.random_seed,
            n_jobs=2,
        )
    
    def _create_target(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        Create target variable for training.
        
        Target is binary: 1 for good liquidity, 0 for poor.
        """
        # Calculate composite liquidity score
        composite = (
            features_df["ls_savings_ratio"] * 0.25 +
            features_df["ls_avg_balance_ratio"].clip(0, 1) * 0.20 +
            features_df["ls_min_balance_rate"] * 0.20 +
            (features_df["ls_emergency_fund"] / 12).clip(0, 1) * 0.20 +
            features_df["ls_balance_volatility"] * 0.15
        )
        
        # Convert to binary
        threshold = composite.median()
        return (composite > threshold).astype(int).values
    
    def train(
        self,
        features_df: pd.DataFrame,
        test_size: float = 0.2,
    ) -> Dict[str, Any]:
        """
        Train the Liquidity Buffer Model.
        
        Args:
            features_df: DataFrame with engineered features
            test_size: Fraction for validation
            
        Returns:
            Dictionary with training metrics
        """
        # Prepare features and target
        X = features_df[self.FEATURE_NAMES].values
        y = self._create_target(features_df)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=self.config.data.random_seed
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        self.is_fitted = True
        
        # Evaluate
        y_pred_proba = self.model.predict_proba(X_val_scaled)[:, 1]
        y_pred = self.model.predict(X_val_scaled)
        
        # Get feature coefficients
        coefficients = dict(zip(
            self.FEATURE_NAMES,
            self.model.coef_[0].tolist()
        ))
        
        metrics = {
            "model_name": "Liquidity & Buffer (Ls)",
            "algorithm": "Logistic Regression",
            "auc_score": roc_auc_score(y_val, y_pred_proba),
            "accuracy": accuracy_score(y_val, y_pred),
            "feature_coefficients": coefficients,
            "feature_importance": {k: abs(v) for k, v in coefficients.items()},
        }
        
        return metrics
    
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """
        Predict liquidity buffer score.
        
        Args:
            features: DataFrame with feature columns
            
        Returns:
            Array of scores between 0 and 1
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call train() first or load a saved model.")
        
        X = features[self.FEATURE_NAMES].values
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]
    
    def predict_single(self, feature_dict: Dict[str, float]) -> float:
        """Predict score for a single user from manual input."""
        df = pd.DataFrame([feature_dict])
        return self.predict(df)[0]
    
    def save(self, path: Optional[Path] = None):
        """Save model and scaler to disk."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Nothing to save.")
        
        path = path or self.config.models_dir / "liquidity_buffer_model.pkl"
        scaler_path = path.parent / "liquidity_buffer_scaler.pkl"
        
        joblib.dump(self.model, path)
        joblib.dump(self.scaler, scaler_path)
        print(f"Saved Liquidity Buffer Model to: {path}")
    
    def load(self, path: Optional[Path] = None):
        """Load model and scaler from disk."""
        path = path or self.config.models_dir / "liquidity_buffer_model.pkl"
        scaler_path = path.parent / "liquidity_buffer_scaler.pkl"
        
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        
        self.model = joblib.load(path)
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)
        self.is_fitted = True
        print(f"Loaded Liquidity Buffer Model from: {path}")
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance (absolute coefficients) from trained model."""
        if not self.is_fitted:
            raise ValueError("Model not fitted.")
        
        coefficients = self.model.coef_[0]
        return {name: abs(coef) for name, coef in zip(self.FEATURE_NAMES, coefficients)}
