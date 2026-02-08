"""
Feature Engineering Pipeline for FairScore.
Transforms raw transaction data into behavioral indicators for ML models.
OPTIMIZED: Uses vectorized pandas operations for speed.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
import gc

from fairscore.config import Config


class FeatureEngineer:
    """
    Extracts behavioral features from user and transaction data.
    
    Features are organized into four groups:
    - Income Stability (Is): Regularity and predictability of income
    - Expense Discipline (Bs): Spending patterns and budgeting
    - Liquidity & Buffer (Ls): Savings and emergency fund strength
    - Risk & Anomaly (Rf): Irregular or suspicious activity
    """
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize feature engineer."""
        self.config = config or Config()
    
    def engineer_features(
        self,
        users_df: pd.DataFrame,
        transactions_df: pd.DataFrame,
        save: bool = True
    ) -> pd.DataFrame:
        """
        Engineer all features from raw data.
        
        Args:
            users_df: Users dataframe
            transactions_df: Transactions dataframe
            save: Whether to save features to disk
            
        Returns:
            DataFrame with engineered features per user
        """
        print("Engineering features (optimized)...")
        
        # Convert timestamp once
        transactions_df = transactions_df.copy()
        transactions_df["timestamp"] = pd.to_datetime(transactions_df["timestamp"])
        transactions_df["month"] = transactions_df["timestamp"].dt.to_period("M")
        transactions_df["hour"] = transactions_df["timestamp"].dt.hour
        
        # Process transactions to get aggregated metrics
        print("  Aggregating transactions...")
        txn_features = self._aggregate_transactions(transactions_df)
        
        # Merge with user data
        features_df = users_df.merge(txn_features, on="user_id", how="left")
        
        # Calculate all features
        print("  Computing Income Stability (Is) features...")
        features_df = self._compute_income_stability_features(features_df, transactions_df)
        
        print("  Computing Expense Discipline (Bs) features...")
        features_df = self._compute_expense_discipline_features(features_df, transactions_df)
        
        print("  Computing Liquidity & Buffer (Ls) features...")
        features_df = self._compute_liquidity_buffer_features(features_df, transactions_df)
        
        print("  Computing Risk & Anomaly (Rf) features...")
        features_df = self._compute_risk_anomaly_features(features_df, transactions_df)
        
        # Normalize features
        print("  Normalizing features...")
        features_df = self._normalize_features(features_df)
        
        if save:
            features_path = self.config.data_dir / "features.csv"
            features_df.to_csv(features_path, index=False)
            print(f"Saved features to: {features_path}")
        
        gc.collect()
        return features_df
    
    def _aggregate_transactions(self, transactions_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate transaction-level data to user level."""
        
        # Basic aggregations
        agg = transactions_df.groupby("user_id").agg({
            "amount": ["sum", "mean", "std", "count"],
            "balance_after": ["mean", "std", "min", "max"],
        }).reset_index()
        
        # Flatten column names
        agg.columns = ["user_id", "total_amount", "avg_amount", "std_amount", 
                       "txn_count", "avg_balance", "std_balance", "min_balance", "max_balance"]
        
        # Credit vs Debit splits
        credits = transactions_df[transactions_df["type"] == "credit"].groupby("user_id")["amount"].sum().reset_index()
        credits.columns = ["user_id", "total_credits"]
        
        debits = transactions_df[transactions_df["type"] == "debit"].groupby("user_id")["amount"].sum().reset_index()
        debits.columns = ["user_id", "total_debits"]
        
        agg = agg.merge(credits, on="user_id", how="left")
        agg = agg.merge(debits, on="user_id", how="left")
        agg = agg.fillna(0)
        
        return agg
    
    def _compute_income_stability_features(
        self,
        features_df: pd.DataFrame,
        transactions_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Compute Income Stability (Is) features - VECTORIZED."""
        df = features_df.copy()
        
        # Filter income transactions
        income_categories = ["salary", "freelance_payment", "upi_received", "transfer_in"]
        income_txns = transactions_df[
            (transactions_df["type"] == "credit") & 
            (transactions_df["category"].isin(income_categories))
        ]
        
        # Monthly income aggregation
        monthly_income = income_txns.groupby(["user_id", "month"])["amount"].sum().reset_index()
        
        # Income stats - vectorized
        income_stats = monthly_income.groupby("user_id")["amount"].agg(["mean", "std", "count"]).reset_index()
        income_stats.columns = ["user_id", "monthly_income_mean", "monthly_income_std", "income_months"]
        income_stats["monthly_income_std"] = income_stats["monthly_income_std"].fillna(0)
        income_stats["is_income_variance"] = income_stats["monthly_income_std"] / (income_stats["monthly_income_mean"] + 1)
        income_stats["is_income_regularity"] = 1 / (1 + income_stats["is_income_variance"])
        income_stats["is_income_frequency"] = income_stats["income_months"] / 12
        
        # Salary consistency - vectorized
        salary_txns = transactions_df[transactions_df["category"] == "salary"]
        salary_stats = salary_txns.groupby("user_id")["amount"].agg(["std", "mean"]).reset_index()
        salary_stats.columns = ["user_id", "salary_std", "salary_mean"]
        salary_stats["salary_std"] = salary_stats["salary_std"].fillna(0)
        salary_stats["is_salary_consistency"] = 1 / (1 + salary_stats["salary_std"] / (salary_stats["salary_mean"] + 1))
        
        # Income growth - vectorized using groupby apply with numpy
        def calc_growth(group):
            if len(group) > 1:
                amounts = group.sort_values("month")["amount"].values
                x = np.arange(len(amounts))
                slope = np.polyfit(x, amounts, 1)[0]
                return slope / (amounts.mean() + 1)
            return 0.0
        
        income_growth = monthly_income.groupby("user_id").apply(calc_growth, include_groups=False).reset_index()
        income_growth.columns = ["user_id", "is_income_growth"]
        
        # Merge all income features
        df = df.merge(income_stats[["user_id", "is_income_regularity", "is_income_variance", "is_income_frequency"]], 
                      on="user_id", how="left")
        df = df.merge(salary_stats[["user_id", "is_salary_consistency"]], on="user_id", how="left")
        df = df.merge(income_growth, on="user_id", how="left")
        
        # Fill NaN with neutral values
        for col in ["is_income_regularity", "is_income_variance", "is_income_frequency", "is_salary_consistency"]:
            df[col] = df[col].fillna(0.5)
        df["is_income_growth"] = df["is_income_growth"].fillna(0)
        
        return df
    
    def _compute_expense_discipline_features(
        self,
        features_df: pd.DataFrame,
        transactions_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Compute Expense Discipline (Bs) features - VECTORIZED."""
        df = features_df.copy()
        
        # Filter expense transactions
        expense_txns = transactions_df[transactions_df["type"] == "debit"]
        
        # Monthly expense aggregation
        monthly_expense = expense_txns.groupby(["user_id", "month"])["amount"].sum().reset_index()
        
        # Expense variance - vectorized
        expense_stats = monthly_expense.groupby("user_id")["amount"].agg(["mean", "std"]).reset_index()
        expense_stats.columns = ["user_id", "monthly_expense_mean", "monthly_expense_std"]
        expense_stats["monthly_expense_std"] = expense_stats["monthly_expense_std"].fillna(0)
        expense_stats["bs_expense_variance"] = expense_stats["monthly_expense_std"] / (expense_stats["monthly_expense_mean"] + 1)
        
        # Category ratios - vectorized
        essential_categories = ["rent", "utilities", "groceries", "emi", "medical", "education"]
        discretionary_categories = ["shopping", "entertainment"]
        
        user_totals = expense_txns.groupby("user_id")["amount"].sum().reset_index()
        user_totals.columns = ["user_id", "total_expense"]
        
        essential_totals = expense_txns[expense_txns["category"].isin(essential_categories)].groupby("user_id")["amount"].sum().reset_index()
        essential_totals.columns = ["user_id", "essential_expense"]
        
        discretionary_totals = expense_txns[expense_txns["category"].isin(discretionary_categories)].groupby("user_id")["amount"].sum().reset_index()
        discretionary_totals.columns = ["user_id", "discretionary_expense"]
        
        ratios = user_totals.merge(essential_totals, on="user_id", how="left")
        ratios = ratios.merge(discretionary_totals, on="user_id", how="left")
        ratios = ratios.fillna(0)
        ratios["bs_essential_ratio"] = ratios["essential_expense"] / (ratios["total_expense"] + 1)
        ratios["bs_discretionary_ratio"] = ratios["discretionary_expense"] / (ratios["total_expense"] + 1)
        
        # Recurring payment rate - vectorized
        recurring = expense_txns.groupby("user_id")["is_recurring"].mean().reset_index()
        recurring.columns = ["user_id", "bs_recurring_rate"]
        
        # Budget adherence - vectorized using groupby
        def calc_adherence(group):
            if len(group) > 1:
                mean_exp = group["amount"].mean()
                deviations = np.abs(group["amount"] - mean_exp) / (mean_exp + 1)
                return 1 - deviations.mean()
            return 0.5
        
        adherence = monthly_expense.groupby("user_id").apply(calc_adherence, include_groups=False).reset_index()
        adherence.columns = ["user_id", "bs_budget_adherence"]
        
        # Merge all expense features
        df = df.merge(expense_stats[["user_id", "bs_expense_variance"]], on="user_id", how="left")
        df = df.merge(ratios[["user_id", "bs_essential_ratio", "bs_discretionary_ratio"]], on="user_id", how="left")
        df = df.merge(recurring, on="user_id", how="left")
        df = df.merge(adherence, on="user_id", how="left")
        
        # Fill NaN
        df["bs_expense_variance"] = df["bs_expense_variance"].fillna(0.5)
        df["bs_essential_ratio"] = df["bs_essential_ratio"].fillna(0.5)
        df["bs_discretionary_ratio"] = df["bs_discretionary_ratio"].fillna(0.3)
        df["bs_recurring_rate"] = df["bs_recurring_rate"].fillna(0.5)
        df["bs_budget_adherence"] = df["bs_budget_adherence"].fillna(0.5)
        
        return df
    
    def _compute_liquidity_buffer_features(
        self,
        features_df: pd.DataFrame,
        transactions_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Compute Liquidity & Buffer (Ls) features - VECTORIZED."""
        df = features_df.copy()
        
        # Savings ratio - vectorized (already have total_credits and total_debits)
        df["ls_savings_ratio"] = (df["total_credits"] - df["total_debits"]) / (df["total_credits"] + 1)
        df["ls_savings_ratio"] = df["ls_savings_ratio"].clip(lower=0)
        
        # Average balance ratio - vectorized
        monthly_expense = df["total_debits"] / 12
        df["ls_avg_balance_ratio"] = df["avg_balance"] / (monthly_expense + 1)
        df["ls_avg_balance_ratio"] = df["ls_avg_balance_ratio"].clip(upper=10)
        
        # Min balance rate - vectorized
        min_balance_threshold = 1000
        df["ls_min_balance_rate"] = (df["min_balance"] > min_balance_threshold).astype(float)
        
        # Emergency fund - vectorized
        avg_monthly_expense = df["total_debits"].mean() / 12
        df["ls_emergency_fund"] = df["avg_balance"] / (avg_monthly_expense + 1)
        df["ls_emergency_fund"] = df["ls_emergency_fund"].clip(0, 12)
        
        # Balance volatility - vectorized
        df["ls_balance_volatility"] = df["std_balance"] / (df["avg_balance"] + 1)
        df["ls_balance_volatility"] = 1 / (1 + df["ls_balance_volatility"])
        
        # Fill NaN
        df["ls_savings_ratio"] = df["ls_savings_ratio"].fillna(0.1)
        df["ls_avg_balance_ratio"] = df["ls_avg_balance_ratio"].fillna(1)
        df["ls_min_balance_rate"] = df["ls_min_balance_rate"].fillna(0.5)
        df["ls_emergency_fund"] = df["ls_emergency_fund"].fillna(1)
        df["ls_balance_volatility"] = df["ls_balance_volatility"].fillna(0.5)
        
        return df
    
    def _compute_risk_anomaly_features(
        self,
        features_df: pd.DataFrame,
        transactions_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Compute Risk & Anomaly (Rf) features - VECTORIZED."""
        df = features_df.copy()
        
        # Overdraft detection - vectorized
        overdraft_threshold = 100
        transactions_df["is_overdraft"] = transactions_df["balance_after"] < overdraft_threshold
        overdraft_counts = transactions_df.groupby("user_id")["is_overdraft"].sum().reset_index()
        overdraft_counts.columns = ["user_id", "rf_overdraft_count"]
        
        # Large transaction ratio - vectorized
        user_txn_stats = transactions_df.groupby("user_id")["amount"].agg(["mean", "std"]).reset_index()
        user_txn_stats.columns = ["user_id", "amount_mean", "amount_std"]
        user_txn_stats["amount_std"] = user_txn_stats["amount_std"].fillna(0)
        
        txn_with_stats = transactions_df.merge(user_txn_stats, on="user_id", how="left")
        txn_with_stats["is_large"] = txn_with_stats["amount"] > (txn_with_stats["amount_mean"] + 2 * txn_with_stats["amount_std"])
        large_txn_ratio = txn_with_stats.groupby("user_id")["is_large"].mean().reset_index()
        large_txn_ratio.columns = ["user_id", "rf_large_txn_ratio"]
        
        # Irregular time transactions - vectorized
        transactions_df["is_irregular_time"] = (transactions_df["hour"] >= 23) | (transactions_df["hour"] <= 5)
        irregular_time = transactions_df.groupby("user_id")["is_irregular_time"].mean().reset_index()
        irregular_time.columns = ["user_id", "rf_irregular_time"]
        
        # Category entropy - vectorized with lambda
        def calc_entropy(x):
            counts = x.value_counts(normalize=True)
            return -np.sum(counts * np.log(counts + 1e-10))
        
        category_entropy = transactions_df.groupby("user_id")["category"].apply(calc_entropy).reset_index()
        category_entropy.columns = ["user_id", "rf_category_entropy"]
        max_entropy = np.log(len(transactions_df["category"].unique()))
        category_entropy["rf_category_entropy"] = category_entropy["rf_category_entropy"] / max_entropy
        
        # Spike score - vectorized using groupby
        def calc_spike(group):
            if len(group) > 1:
                amounts = group["amount"].values
                mean_amt = amounts.mean()
                std_amt = amounts.std() + 1
                max_spike = (amounts.max() - mean_amt) / std_amt
                return min(max_spike / 5, 1)
            return 0.0
        
        spike_scores = transactions_df.groupby("user_id").apply(calc_spike, include_groups=False).reset_index()
        spike_scores.columns = ["user_id", "rf_spike_score"]
        
        # Merge features
        df = df.merge(overdraft_counts, on="user_id", how="left")
        df = df.merge(large_txn_ratio, on="user_id", how="left")
        df = df.merge(irregular_time, on="user_id", how="left")
        df = df.merge(category_entropy, on="user_id", how="left")
        df = df.merge(spike_scores, on="user_id", how="left")
        
        # Fill NaN and normalize
        df["rf_overdraft_count"] = df["rf_overdraft_count"].fillna(0).clip(0, 10) / 10
        df["rf_large_txn_ratio"] = df["rf_large_txn_ratio"].fillna(0)
        df["rf_irregular_time"] = df["rf_irregular_time"].fillna(0)
        df["rf_category_entropy"] = df["rf_category_entropy"].fillna(0.5)
        df["rf_spike_score"] = df["rf_spike_score"].fillna(0)
        
        return df
    
    def _normalize_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Normalize all features to 0-1 range."""
        df = features_df.copy()
        
        feature_cols = [col for col in df.columns if col.startswith(("is_", "bs_", "ls_", "rf_"))]
        
        for col in feature_cols:
            if col in df.columns:
                min_val = df[col].min()
                max_val = df[col].max()
                if max_val > min_val:
                    df[col] = (df[col] - min_val) / (max_val - min_val)
                else:
                    df[col] = 0.5
        
        return df
    
    def get_feature_names(self) -> Dict[str, list]:
        """Get feature names by category."""
        return {
            "income_stability": [
                "is_income_regularity", "is_income_variance", "is_income_frequency",
                "is_salary_consistency", "is_income_growth"
            ],
            "expense_discipline": [
                "bs_expense_variance", "bs_essential_ratio", "bs_discretionary_ratio",
                "bs_recurring_rate", "bs_budget_adherence"
            ],
            "liquidity_buffer": [
                "ls_savings_ratio", "ls_avg_balance_ratio", "ls_min_balance_rate",
                "ls_emergency_fund", "ls_balance_volatility"
            ],
            "risk_anomaly": [
                "rf_overdraft_count", "rf_large_txn_ratio", "rf_irregular_time",
                "rf_category_entropy", "rf_spike_score"
            ],
        }


if __name__ == "__main__":
    from fairscore.config import Config
    import pandas as pd
    
    config = Config()
    
    users_df = pd.read_csv(config.data_dir / "users.csv")
    transactions_df = pd.read_csv(config.data_dir / "transactions.csv")
    
    engineer = FeatureEngineer(config)
    features_df = engineer.engineer_features(users_df, transactions_df)
    print(f"Engineered {len(features_df)} user feature vectors")
