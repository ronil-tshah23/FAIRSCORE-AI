"""
Risk & Anomaly Detection Model (Rf) for FairScore.
Uses Isolation Forest for detecting financial anomalies.
Memory-efficient alternative to Autoencoder for 8GB RAM systems.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any
from pathlib import Path
import joblib

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from fairscore.config import Config


class RiskAnomalyModel:
    """
    Risk & Anomaly Detection Model using Isolation Forest.
    
    Identifies irregular or suspicious financial behavior.
    Higher scores indicate MORE risk/anomalies (inverted for FAIR-SCORE).
    
    Features used:
    - rf_overdraft_count
    - rf_large_txn_ratio
    - rf_irregular_time
    - rf_category_entropy
    - rf_spike_score
    """
    
    FEATURE_NAMES = [
        "rf_overdraft_count",
        "rf_large_txn_ratio",
        "rf_irregular_time",
        "rf_category_entropy",
        "rf_spike_score",
    ]
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize Risk Anomaly Model."""
        self.config = config or Config()
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        # Isolation Forest parameters optimized for memory
        self.model = IsolationForest(
            n_estimators=self.config.model.rf_n_estimators,
            contamination=self.config.model.rf_contamination,
            max_samples="auto",
            max_features=1.0,
            bootstrap=False,
            n_jobs=self.config.model.rf_n_jobs,
            random_state=self.config.data.random_seed,
            warm_start=False,
        )
    
    def train(
        self,
        features_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Train the Risk Anomaly Model.
        
        Unlike other models, this is unsupervised anomaly detection.
        
        Args:
            features_df: DataFrame with engineered features
            
        Returns:
            Dictionary with training metrics
        """
        # Prepare features
        X = features_df[self.FEATURE_NAMES].values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled)
        self.is_fitted = True
        
        # Get predictions for training data to compute statistics
        predictions = self.model.predict(X_scaled)
        scores = self.model.decision_function(X_scaled)
        
        # Calculate metrics
        num_anomalies = (predictions == -1).sum()
        anomaly_rate = num_anomalies / len(predictions)
        
        # Feature importance approximation (based on score variance per feature)
        feature_importance = {}
        for i, name in enumerate(self.FEATURE_NAMES):
            # Compute importance as correlation with anomaly score
            correlation = np.corrcoef(X[:, i], scores)[0, 1]
            feature_importance[name] = abs(correlation) if not np.isnan(correlation) else 0.0
        
        metrics = {
            "model_name": "Risk & Anomaly (Rf)",
            "algorithm": "Isolation Forest",
            "total_samples": len(predictions),
            "anomalies_detected": int(num_anomalies),
            "anomaly_rate": anomaly_rate,
            "contamination_param": self.config.model.rf_contamination,
            "score_mean": float(scores.mean()),
            "score_std": float(scores.std()),
            "feature_importance": feature_importance,
        }
        
        return metrics
    
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """
        Predict risk/anomaly score.
        
        Args:
            features: DataFrame with feature columns
            
        Returns:
            Array of scores between 0 and 1 (higher = more risky)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call train() first or load a saved model.")
        
        X = features[self.FEATURE_NAMES].values
        X_scaled = self.scaler.transform(X)
        
        # decision_function returns negative for outliers
        # We convert to 0-1 range where higher = more risky
        raw_scores = self.model.decision_function(X_scaled)
        
        # Normalize to 0-1 (invert so outliers have higher scores)
        min_score = raw_scores.min()
        max_score = raw_scores.max()
        
        if max_score > min_score:
            normalized = 1 - (raw_scores - min_score) / (max_score - min_score)
        else:
            normalized = np.full_like(raw_scores, 0.5)
        
        return normalized
    
    def predict_single(self, feature_dict: Dict[str, float]) -> float:
        """Predict score for a single user from manual input."""
        df = pd.DataFrame([feature_dict])
        return self.predict(df)[0]
    
    def is_anomaly(self, features: pd.DataFrame) -> np.ndarray:
        """
        Check if samples are anomalies.
        
        Args:
            features: DataFrame with feature columns
            
        Returns:
            Boolean array (True = anomaly)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted.")
        
        X = features[self.FEATURE_NAMES].values
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        return predictions == -1
    
    def save(self, path: Optional[Path] = None):
        """Save model and scaler to disk."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Nothing to save.")
        
        path = path or self.config.models_dir / "risk_anomaly_model.pkl"
        scaler_path = path.parent / "risk_anomaly_scaler.pkl"
        
        joblib.dump(self.model, path)
        joblib.dump(self.scaler, scaler_path)
        print(f"Saved Risk Anomaly Model to: {path}")
    
    def load(self, path: Optional[Path] = None):
        """Load model and scaler from disk."""
        path = path or self.config.models_dir / "risk_anomaly_model.pkl"
        scaler_path = path.parent / "risk_anomaly_scaler.pkl"
        
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        
        self.model = joblib.load(path)
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)
        self.is_fitted = True
        print(f"Loaded Risk Anomaly Model from: {path}")
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance approximation."""
        if not self.is_fitted:
            raise ValueError("Model not fitted.")
        
        # Isolation Forest doesn't have direct feature importance
        # Return equal weights as placeholder
        return {name: 1.0 / len(self.FEATURE_NAMES) for name in self.FEATURE_NAMES}
