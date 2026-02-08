"""
SHAP Explainability Module for FairScore.
Provides interpretable explanations for credit score decisions.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Any, Tuple
from pathlib import Path

from fairscore.config import Config


class ShapExplainer:
    """
    SHAP-based explainability for FairScore models.
    
    Provides feature-level explanations for each model's predictions,
    helping users understand what factors influence their credit score.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize SHAP explainer."""
        self.config = config or Config()
        self._shap_available = self._check_shap()
    
    def _check_shap(self) -> bool:
        """Check if SHAP is available."""
        try:
            import shap
            return True
        except ImportError:
            return False
    
    def explain_score_simple(
        self,
        is_score: float,
        bs_score: float,
        ls_score: float,
        rf_score: float,
        is_features: Optional[Dict[str, float]] = None,
        bs_features: Optional[Dict[str, float]] = None,
        ls_features: Optional[Dict[str, float]] = None,
        rf_features: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Generate simple explanation without SHAP (rule-based).
        
        This is a fallback when SHAP is not available or for quick explanations.
        
        Args:
            is_score: Income Stability score (0-1)
            bs_score: Expense Discipline score (0-1)
            ls_score: Liquidity Buffer score (0-1)
            rf_score: Risk Anomaly score (0-1)
            *_features: Optional feature dictionaries for detailed breakdown
            
        Returns:
            Dictionary with explanation components
        """
        # Calculate FAIR-SCORE
        fair_score = 300 + 600 * (
            0.35 * is_score +
            0.30 * bs_score +
            0.25 * ls_score +
            0.10 * (1 - rf_score)
        )
        fair_score = int(np.clip(fair_score, 300, 900))
        
        # Determine category
        if fair_score < 500:
            category = "POOR"
        elif fair_score < 650:
            category = "FAIR"
        elif fair_score < 750:
            category = "GOOD"
        else:
            category = "EXCELLENT"
        
        # Calculate component contributions
        contributions = [
            ("Income Stability", is_score * 0.35 * 600, is_score, "positive" if is_score > 0.5 else "negative"),
            ("Expense Discipline", bs_score * 0.30 * 600, bs_score, "positive" if bs_score > 0.5 else "negative"),
            ("Liquidity Buffer", ls_score * 0.25 * 600, ls_score, "positive" if ls_score > 0.5 else "negative"),
            ("Risk Assessment", (1 - rf_score) * 0.10 * 600, 1 - rf_score, "positive" if rf_score < 0.5 else "negative"),
        ]
        
        # Sort by impact
        contributions.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # Generate waterfall data
        waterfall = {
            "base": 300,
            "final": fair_score,
            "contributions": [
                {
                    "name": name,
                    "contribution": round(contrib, 1),
                    "score": round(score, 3),
                    "impact": impact,
                }
                for name, contrib, score, impact in contributions
            ]
        }
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            is_score, bs_score, ls_score, rf_score,
            is_features, bs_features, ls_features, rf_features
        )
        
        # Generate text explanations
        top_positive = [c for c in contributions if c[3] == "positive"][:2]
        top_negative = [c for c in contributions if c[3] == "negative"][:2]
        
        strengths = [f"{c[0]} ({c[2]:.0%})" for c in top_positive]
        weaknesses = [f"{c[0]} ({c[2]:.0%})" for c in top_negative]
        
        return {
            "fair_score": fair_score,
            "category": category,
            "waterfall": waterfall,
            "strengths": strengths,
            "weaknesses": weaknesses,
            "recommendations": recommendations,
            "component_scores": {
                "income_stability": is_score,
                "expense_discipline": bs_score,
                "liquidity_buffer": ls_score,
                "risk_anomaly": rf_score,
            },
        }
    
    def _generate_recommendations(
        self,
        is_score: float,
        bs_score: float,
        ls_score: float,
        rf_score: float,
        is_features: Optional[Dict[str, float]] = None,
        bs_features: Optional[Dict[str, float]] = None,
        ls_features: Optional[Dict[str, float]] = None,
        rf_features: Optional[Dict[str, float]] = None,
    ) -> List[Dict[str, str]]:
        """Generate personalized recommendations based on scores."""
        recommendations = []
        
        # Income Stability recommendations
        if is_score < 0.5:
            if is_features and is_features.get("is_income_regularity", 0) < 0.5:
                recommendations.append({
                    "category": "Income",
                    "priority": "high",
                    "text": "Establish more regular income sources to improve stability score",
                    "impact": "+50-80 points potential"
                })
            if is_features and is_features.get("is_income_variance", 0) > 0.5:
                recommendations.append({
                    "category": "Income",
                    "priority": "medium",
                    "text": "Reduce income volatility by diversifying income streams",
                    "impact": "+30-50 points potential"
                })
        
        # Expense Discipline recommendations
        if bs_score < 0.5:
            if bs_features and bs_features.get("bs_discretionary_ratio", 0) > 0.3:
                recommendations.append({
                    "category": "Spending",
                    "priority": "high",
                    "text": "Reduce discretionary spending on non-essential categories",
                    "impact": "+40-70 points potential"
                })
            recommendations.append({
                "category": "Spending",
                "priority": "medium",
                "text": "Set up automatic bill payments to improve consistency",
                "impact": "+20-40 points potential"
            })
        
        # Liquidity Buffer recommendations
        if ls_score < 0.5:
            if ls_features and ls_features.get("ls_savings_ratio", 0) < 0.2:
                recommendations.append({
                    "category": "Savings",
                    "priority": "high",
                    "text": "Increase monthly savings to at least 20% of income",
                    "impact": "+50-80 points potential"
                })
            if ls_features and ls_features.get("ls_emergency_fund", 0) < 3:
                recommendations.append({
                    "category": "Savings",
                    "priority": "high",
                    "text": "Build an emergency fund covering 3-6 months of expenses",
                    "impact": "+30-60 points potential"
                })
        
        # Risk & Anomaly recommendations
        if rf_score > 0.5:
            if rf_features and rf_features.get("rf_overdraft_count", 0) > 0.3:
                recommendations.append({
                    "category": "Risk",
                    "priority": "high",
                    "text": "Avoid overdraft situations by maintaining minimum balance",
                    "impact": "+20-40 points potential"
                })
            if rf_features and rf_features.get("rf_large_txn_ratio", 0) > 0.3:
                recommendations.append({
                    "category": "Risk",
                    "priority": "medium",
                    "text": "Spread large purchases over time to reduce spending spikes",
                    "impact": "+15-30 points potential"
                })
        
        # Sort by priority
        priority_order = {"high": 0, "medium": 1, "low": 2}
        recommendations.sort(key=lambda x: priority_order.get(x["priority"], 2))
        
        return recommendations[:5]  # Return top 5 recommendations
    
    def create_waterfall_ascii(self, explanation: Dict[str, Any], width: int = 50) -> str:
        """
        Create ASCII waterfall visualization for TUI.
        
        Args:
            explanation: Dictionary from explain_score_simple
            width: Width of the bars
            
        Returns:
            Multi-line ASCII art string
        """
        waterfall = explanation["waterfall"]
        base = waterfall["base"]
        final = waterfall["final"]
        
        lines = [
            "FAIR-SCORE Breakdown",
            "=" * 50,
            "",
            f"Base Score:           {base}",
            "-" * 50,
        ]
        
        running_total = base
        for contrib in waterfall["contributions"]:
            name = contrib["name"]
            value = contrib["contribution"]
            score = contrib["score"]
            impact = contrib["impact"]
            
            # Create bar
            bar_length = int(abs(value) / 10)
            if value >= 0:
                bar = "+" * min(bar_length, 20)
                sign = "+"
            else:
                bar = "-" * min(bar_length, 20)
                sign = ""
            
            running_total += value
            
            # Format line
            icon = "[+]" if impact == "positive" else "[-]"
            line = f"{icon} {name:20} {sign}{value:>6.1f} {bar}"
            lines.append(line)
        
        lines.extend([
            "-" * 50,
            f"Final FAIR-SCORE:     {final} ({explanation['category']})",
            "=" * 50,
        ])
        
        return "\n".join(lines)
    
    def create_feature_bars(self, explanation: Dict[str, Any]) -> str:
        """
        Create ASCII feature contribution bars for TUI.
        
        Args:
            explanation: Dictionary from explain_score_simple
            
        Returns:
            Multi-line ASCII art string
        """
        scores = explanation["component_scores"]
        
        lines = [
            "Component Scores",
            "-" * 40,
        ]
        
        labels = [
            ("Is", "Income Stability", scores["income_stability"]),
            ("Bs", "Expense Discipline", scores["expense_discipline"]),
            ("Ls", "Liquidity Buffer", scores["liquidity_buffer"]),
            ("Rf", "Risk Level", 1 - scores["risk_anomaly"]),  # Invert for display
        ]
        
        for code, name, score in labels:
            bar_length = int(score * 20)
            bar = "#" * bar_length + "." * (20 - bar_length)
            percent = score * 100
            lines.append(f"[{code}] [{bar}] {percent:5.1f}%")
        
        return "\n".join(lines)


def compute_manual_features(
    monthly_income: float,
    income_frequency: str,
    monthly_expenses: float,
    essential_ratio: float,
    avg_balance: float,
    monthly_savings: float,
    overdraft_count: int,
    irregular_txn_count: int,
) -> Tuple[Dict, Dict, Dict, Dict]:
    """
    Compute model features from manual user input.
    
    This function converts user-provided financial data into
    the feature format expected by each model.
    
    Args:
        monthly_income: Average monthly income
        income_frequency: "monthly", "weekly", or "irregular"
        monthly_expenses: Average monthly expenses
        essential_ratio: Ratio of essential to total expenses (0-1)
        avg_balance: Average account balance
        monthly_savings: Average monthly savings
        overdraft_count: Number of overdrafts in last 6 months
        irregular_txn_count: Number of irregular/large transactions
        
    Returns:
        Tuple of (is_features, bs_features, ls_features, rf_features)
    """
    # Income Stability features
    freq_scores = {"monthly": 0.9, "weekly": 0.7, "irregular": 0.4}
    freq_variance = {"monthly": 0.1, "weekly": 0.2, "irregular": 0.5}
    
    is_features = {
        "is_income_regularity": freq_scores.get(income_frequency, 0.5),
        "is_income_variance": freq_variance.get(income_frequency, 0.3),
        "is_income_frequency": 0.8 if income_frequency in ["monthly", "weekly"] else 0.5,
        "is_salary_consistency": 0.9 if income_frequency == "monthly" else 0.5,
        "is_income_growth": 0.5,  # Neutral without historical data
    }
    
    # Expense Discipline features
    discretionary_ratio = 1 - essential_ratio
    expense_to_income = monthly_expenses / (monthly_income + 1)
    
    bs_features = {
        "bs_expense_variance": max(0, 1 - expense_to_income),
        "bs_essential_ratio": essential_ratio,
        "bs_discretionary_ratio": discretionary_ratio,
        "bs_recurring_rate": essential_ratio * 0.8,  # Estimate
        "bs_budget_adherence": max(0, 1 - abs(expense_to_income - 0.7)),
    }
    
    # Liquidity Buffer features
    savings_ratio = monthly_savings / (monthly_income + 1)
    balance_to_expense = avg_balance / (monthly_expenses + 1)
    
    ls_features = {
        "ls_savings_ratio": min(1, savings_ratio),
        "ls_avg_balance_ratio": min(1, balance_to_expense / 3),
        "ls_min_balance_rate": 1.0 if avg_balance > 1000 else 0.5,
        "ls_emergency_fund": min(1, balance_to_expense / 3),
        "ls_balance_volatility": 0.7,  # Assume moderate stability
    }
    
    # Risk & Anomaly features
    rf_features = {
        "rf_overdraft_count": min(1, overdraft_count / 5),
        "rf_large_txn_ratio": min(1, irregular_txn_count / 10),
        "rf_irregular_time": 0.1,  # Assume normal
        "rf_category_entropy": 0.5,  # Assume moderate diversity
        "rf_spike_score": min(1, irregular_txn_count / 5),
    }
    
    return is_features, bs_features, ls_features, rf_features
