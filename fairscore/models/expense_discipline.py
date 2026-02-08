"""
Expense Discipline Model (Bs) for FairScore.
Uses Random Forest for predicting expense discipline score.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any
from pathlib import Path
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

from fairscore.config import Config


class ExpenseDisciplineModel:
    """
    Expense Discipline Model using Random Forest.
    
    Evaluates spending patterns and budgeting consistency.
    Higher scores indicate better expense management.
    
    Features used:
    - bs_expense_variance
    - bs_essential_ratio
    - bs_discretionary_ratio
    - bs_recurring_rate
    - bs_budget_adherence
    """
    
    FEATURE_NAMES = [
        "bs_expense_variance",
        "bs_essential_ratio",
        "bs_discretionary_ratio",
        "bs_recurring_rate",
        "bs_budget_adherence",
    ]
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize Expense Discipline Model."""
        self.config = config or Config()
        self.model = None
        self.is_fitted = False
        
        # Random Forest parameters optimized for memory
        self.model = RandomForestClassifier(
            n_estimators=self.config.model.bs_n_estimators,
            max_depth=self.config.model.bs_max_depth,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features="sqrt",
            n_jobs=self.config.model.bs_n_jobs,
            random_state=self.config.data.random_seed,
            warm_start=False,
            class_weight="balanced",
        )
    
    def _create_target(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        Create target variable for training.
        
        Target is binary: 1 for disciplined spending, 0 for undisciplined.
        """
        # Calculate composite expense discipline score
        composite = (
            (1 - features_df["bs_expense_variance"]) * 0.25 +
            features_df["bs_essential_ratio"] * 0.25 +
            (1 - features_df["bs_discretionary_ratio"]) * 0.15 +
            features_df["bs_recurring_rate"] * 0.15 +
            features_df["bs_budget_adherence"] * 0.20
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
        Train the Expense Discipline Model.
        
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
        
        # Train model
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        
        # Evaluate
        y_pred_proba = self.model.predict_proba(X_val)[:, 1]
        y_pred = self.model.predict(X_val)
        
        metrics = {
            "model_name": "Expense Discipline (Bs)",
            "algorithm": "Random Forest",
            "auc_score": roc_auc_score(y_val, y_pred_proba),
            "accuracy": accuracy_score(y_val, y_pred),
            "n_estimators": self.model.n_estimators,
            "feature_importance": dict(zip(
                self.FEATURE_NAMES,
                self.model.feature_importances_.tolist()
            )),
        }
        
        return metrics
    
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """
        Predict expense discipline score.
        
        Args:
            features: DataFrame with feature columns
            
        Returns:
            Array of scores between 0 and 1
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call train() first or load a saved model.")
        
        X = features[self.FEATURE_NAMES].values
        return self.model.predict_proba(X)[:, 1]
    
    def predict_single(self, feature_dict: Dict[str, float]) -> float:
        """Predict score for a single user from manual input."""
        df = pd.DataFrame([feature_dict])
        return self.predict(df)[0]
    
    def save(self, path: Optional[Path] = None):
        """Save model to disk."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Nothing to save.")
        
        path = path or self.config.models_dir / "expense_discipline_model.pkl"
        joblib.dump(self.model, path)
        print(f"Saved Expense Discipline Model to: {path}")
    
    def load(self, path: Optional[Path] = None):
        """Load model from disk."""
        path = path or self.config.models_dir / "expense_discipline_model.pkl"
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        
        self.model = joblib.load(path)
        self.is_fitted = True
        print(f"Loaded Expense Discipline Model from: {path}")
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from trained model."""
        if not self.is_fitted:
            raise ValueError("Model not fitted.")
        
        return dict(zip(self.FEATURE_NAMES, self.model.feature_importances_.tolist()))
