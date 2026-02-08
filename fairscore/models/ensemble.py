"""
Ensemble Scoring Engine for FairScore.
Combines outputs from all four models to compute the final FAIR-SCORE.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass

from fairscore.config import Config
from fairscore.models.income_stability import IncomeStabilityModel
from fairscore.models.expense_discipline import ExpenseDisciplineModel
from fairscore.models.liquidity_buffer import LiquidityBufferModel
from fairscore.models.risk_anomaly import RiskAnomalyModel


@dataclass
class ScoreBreakdown:
    """Breakdown of FAIR-SCORE components."""
    fair_score: int
    income_stability: float  # Is
    expense_discipline: float  # Bs
    liquidity_buffer: float  # Ls
    risk_anomaly: float  # Rf
    category: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "fair_score": self.fair_score,
            "income_stability_score": self.income_stability,
            "expense_discipline_score": self.expense_discipline,
            "liquidity_buffer_score": self.liquidity_buffer,
            "risk_anomaly_score": self.risk_anomaly,
            "category": self.category,
            "weights": {
                "Is": 0.35,
                "Bs": 0.30,
                "Ls": 0.25,
                "Rf": 0.10,
            }
        }


class EnsembleScorer:
    """
    Ensemble Scoring Engine for computing FAIR-SCORE.
    
    Formula:
    FAIR-SCORE = 300 + 600 * (0.35*Is + 0.30*Bs + 0.25*Ls + 0.10*(1-Rf))
    
    Where:
    - Is: Income Stability score (0-1)
    - Bs: Expense Discipline score (0-1)
    - Ls: Liquidity & Buffer score (0-1)
    - Rf: Risk & Anomaly score (0-1, higher = more risky)
    
    Final score range: 300-900
    """
    
    SCORE_MIN = 300
    SCORE_MAX = 900
    
    CATEGORIES = {
        (300, 499): "POOR",
        (500, 649): "FAIR",
        (650, 749): "GOOD",
        (750, 900): "EXCELLENT",
    }
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize Ensemble Scorer."""
        self.config = config or Config()
        
        # Model weights
        self.weight_is = self.config.model.weight_is  # 0.35
        self.weight_bs = self.config.model.weight_bs  # 0.30
        self.weight_ls = self.config.model.weight_ls  # 0.25
        self.weight_rf = self.config.model.weight_rf  # 0.10
        
        # Individual models
        self.is_model = IncomeStabilityModel(config)
        self.bs_model = ExpenseDisciplineModel(config)
        self.ls_model = LiquidityBufferModel(config)
        self.rf_model = RiskAnomalyModel(config)
        
        self.models_loaded = False
    
    def load_models(self):
        """Load all trained models from disk."""
        self.is_model.load()
        self.bs_model.load()
        self.ls_model.load()
        self.rf_model.load()
        self.models_loaded = True
    
    def compute_score(
        self,
        is_score: float,
        bs_score: float,
        ls_score: float,
        rf_score: float,
    ) -> int:
        """
        Compute FAIR-SCORE from individual model scores.
        
        Args:
            is_score: Income Stability score (0-1)
            bs_score: Expense Discipline score (0-1)
            ls_score: Liquidity Buffer score (0-1)
            rf_score: Risk Anomaly score (0-1, higher = riskier)
            
        Returns:
            FAIR-SCORE between 300 and 900
        """
        # Apply formula: 300 + 600 * weighted_sum
        weighted_sum = (
            self.weight_is * is_score +
            self.weight_bs * bs_score +
            self.weight_ls * ls_score +
            self.weight_rf * (1 - rf_score)  # Invert risk score
        )
        
        score = self.SCORE_MIN + (self.SCORE_MAX - self.SCORE_MIN) * weighted_sum
        return int(np.clip(score, self.SCORE_MIN, self.SCORE_MAX))
    
    def get_category(self, score: int) -> str:
        """Get category label for a FAIR-SCORE."""
        for (low, high), category in self.CATEGORIES.items():
            if low <= score <= high:
                return category
        return "UNKNOWN"
    
    def predict(self, features: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Predict FAIR-SCORE for multiple users.
        
        Args:
            features: DataFrame with all feature columns
            
        Returns:
            Tuple of (scores array, breakdown DataFrame)
        """
        if not self.models_loaded:
            self.load_models()
        
        # Get individual model predictions
        is_scores = self.is_model.predict(features)
        bs_scores = self.bs_model.predict(features)
        ls_scores = self.ls_model.predict(features)
        rf_scores = self.rf_model.predict(features)
        
        # Compute FAIR-SCORES
        fair_scores = []
        for i in range(len(features)):
            score = self.compute_score(
                is_scores[i], bs_scores[i], ls_scores[i], rf_scores[i]
            )
            fair_scores.append(score)
        
        fair_scores = np.array(fair_scores)
        
        # Create breakdown DataFrame
        breakdown = pd.DataFrame({
            "user_id": features["user_id"] if "user_id" in features.columns else range(len(features)),
            "fair_score": fair_scores,
            "is_score": is_scores,
            "bs_score": bs_scores,
            "ls_score": ls_scores,
            "rf_score": rf_scores,
            "category": [self.get_category(s) for s in fair_scores],
        })
        
        return fair_scores, breakdown
    
    def predict_single(
        self,
        is_features: Dict[str, float],
        bs_features: Dict[str, float],
        ls_features: Dict[str, float],
        rf_features: Dict[str, float],
    ) -> ScoreBreakdown:
        """
        Predict FAIR-SCORE for a single user from manual input.
        
        Args:
            is_features: Income Stability features
            bs_features: Expense Discipline features
            ls_features: Liquidity Buffer features
            rf_features: Risk Anomaly features
            
        Returns:
            ScoreBreakdown with all details
        """
        if not self.models_loaded:
            self.load_models()
        
        # Get individual predictions
        is_score = self.is_model.predict_single(is_features)
        bs_score = self.bs_model.predict_single(bs_features)
        ls_score = self.ls_model.predict_single(ls_features)
        rf_score = self.rf_model.predict_single(rf_features)
        
        # Compute FAIR-SCORE
        fair_score = self.compute_score(is_score, bs_score, ls_score, rf_score)
        
        return ScoreBreakdown(
            fair_score=fair_score,
            income_stability=is_score,
            expense_discipline=bs_score,
            liquidity_buffer=ls_score,
            risk_anomaly=rf_score,
            category=self.get_category(fair_score),
        )
    
    def predict_from_raw_scores(
        self,
        is_score: float,
        bs_score: float,
        ls_score: float,
        rf_score: float,
    ) -> ScoreBreakdown:
        """
        Compute FAIR-SCORE directly from model scores (no model inference).
        
        Useful for manual entry mode where features are already computed.
        """
        fair_score = self.compute_score(is_score, bs_score, ls_score, rf_score)
        
        return ScoreBreakdown(
            fair_score=fair_score,
            income_stability=is_score,
            expense_discipline=bs_score,
            liquidity_buffer=ls_score,
            risk_anomaly=rf_score,
            category=self.get_category(fair_score),
        )
    
    def explain_score(self, breakdown: ScoreBreakdown) -> str:
        """
        Generate human-readable explanation of the score.
        
        Args:
            breakdown: ScoreBreakdown object
            
        Returns:
            Multi-line explanation string
        """
        lines = [
            f"FAIR-SCORE: {breakdown.fair_score} ({breakdown.category})",
            "",
            "Component Breakdown:",
            f"  [Is] Income Stability:    {breakdown.income_stability:.2%} (weight: 35%)",
            f"  [Bs] Expense Discipline:  {breakdown.expense_discipline:.2%} (weight: 30%)",
            f"  [Ls] Liquidity & Buffer:  {breakdown.liquidity_buffer:.2%} (weight: 25%)",
            f"  [Rf] Risk & Anomaly:      {breakdown.risk_anomaly:.2%} (weight: 10%)",
            "",
        ]
        
        # Add recommendations based on lowest scores
        recommendations = []
        
        if breakdown.income_stability < 0.5:
            recommendations.append("- Improve income consistency by maintaining regular income sources")
        if breakdown.expense_discipline < 0.5:
            recommendations.append("- Focus on budgeting and reducing discretionary spending")
        if breakdown.liquidity_buffer < 0.5:
            recommendations.append("- Build emergency fund and maintain higher savings ratio")
        if breakdown.risk_anomaly > 0.5:
            recommendations.append("- Reduce irregular transactions and maintain normal spending patterns")
        
        if recommendations:
            lines.append("Recommendations:")
            lines.extend(recommendations)
        
        return "\n".join(lines)
