"""
Model Trainer for FairScore.
Orchestrates training of all models and generates evaluation plots.
"""

import json
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any
from pathlib import Path
from datetime import datetime
import gc

from fairscore.config import Config
from fairscore.data.generator import DataGenerator
from fairscore.data.features import FeatureEngineer
from fairscore.models.income_stability import IncomeStabilityModel
from fairscore.models.expense_discipline import ExpenseDisciplineModel
from fairscore.models.liquidity_buffer import LiquidityBufferModel
from fairscore.models.risk_anomaly import RiskAnomalyModel


class ModelTrainer:
    """
    Orchestrates training of all FairScore models.
    
    Handles:
    - Data loading or generation
    - Feature engineering
    - Training of all 4 models
    - Evaluation and metrics
    - Generating training plots for presentation
    - Saving models and results
    """
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize trainer."""
        self.config = config or Config()
        self.metrics = {}
    
    def train_all(
        self,
        generate_data: bool = False,
        save_plots: bool = True,
    ) -> Dict[str, Any]:
        """
        Train all models end-to-end.
        
        Args:
            generate_data: If True, generate new synthetic data
            save_plots: If True, save evaluation plots
            
        Returns:
            Dictionary with all training metrics
        """
        print("=" * 60)
        print("FairScore Model Training Pipeline")
        print("=" * 60)
        
        # Step 1: Load or generate data
        if generate_data or not self._data_exists():
            print("\n[1/5] Generating synthetic data...")
            generator = DataGenerator(self.config)
            users_df, transactions_df = generator.generate_all()
        else:
            print("\n[1/5] Loading existing data...")
            users_df = pd.read_csv(self.config.data_dir / "users.csv")
            transactions_df = pd.read_csv(self.config.data_dir / "transactions.csv")
            print(f"  Loaded {len(users_df):,} users, {len(transactions_df):,} transactions")
        
        # Step 2: Feature engineering
        print("\n[2/5] Engineering features...")
        engineer = FeatureEngineer(self.config)
        
        features_path = self.config.data_dir / "features.csv"
        if features_path.exists() and not generate_data:
            features_df = pd.read_csv(features_path)
            print(f"  Loaded existing features: {len(features_df):,} records")
        else:
            features_df = engineer.engineer_features(users_df, transactions_df, save=True)
        
        # Clean up memory
        del transactions_df
        gc.collect()
        
        # Step 3: Train models
        print("\n[3/5] Training models...")
        
        # Train Income Stability Model
        print("\n  Training Income Stability Model (LightGBM)...")
        is_model = IncomeStabilityModel(self.config)
        is_metrics = is_model.train(features_df)
        is_model.save()
        self.metrics["income_stability"] = is_metrics
        print(f"    AUC: {is_metrics['auc_score']:.4f}, Accuracy: {is_metrics['accuracy']:.4f}")
        
        # Train Expense Discipline Model
        print("\n  Training Expense Discipline Model (Random Forest)...")
        bs_model = ExpenseDisciplineModel(self.config)
        bs_metrics = bs_model.train(features_df)
        bs_model.save()
        self.metrics["expense_discipline"] = bs_metrics
        print(f"    AUC: {bs_metrics['auc_score']:.4f}, Accuracy: {bs_metrics['accuracy']:.4f}")
        
        # Train Liquidity Buffer Model
        print("\n  Training Liquidity Buffer Model (Logistic Regression)...")
        ls_model = LiquidityBufferModel(self.config)
        ls_metrics = ls_model.train(features_df)
        ls_model.save()
        self.metrics["liquidity_buffer"] = ls_metrics
        print(f"    AUC: {ls_metrics['auc_score']:.4f}, Accuracy: {ls_metrics['accuracy']:.4f}")
        
        # Train Risk Anomaly Model
        print("\n  Training Risk Anomaly Model (Isolation Forest)...")
        rf_model = RiskAnomalyModel(self.config)
        rf_metrics = rf_model.train(features_df)
        rf_model.save()
        self.metrics["risk_anomaly"] = rf_metrics
        print(f"    Anomaly Rate: {rf_metrics['anomaly_rate']:.2%}")
        
        # Step 4: Generate ensemble scores for validation
        print("\n[4/5] Computing FAIR-SCORES for validation...")
        fair_scores = self._compute_validation_scores(
            features_df, is_model, bs_model, ls_model, rf_model
        )
        self.metrics["score_statistics"] = {
            "mean": float(fair_scores["fair_score"].mean()),
            "std": float(fair_scores["fair_score"].std()),
            "min": int(fair_scores["fair_score"].min()),
            "max": int(fair_scores["fair_score"].max()),
            "median": float(fair_scores["fair_score"].median()),
            "category_distribution": fair_scores["category"].value_counts().to_dict(),
        }
        print(f"    Mean Score: {self.metrics['score_statistics']['mean']:.1f}")
        print(f"    Score Range: {self.metrics['score_statistics']['min']} - {self.metrics['score_statistics']['max']}")
        
        # Step 5: Generate plots
        if save_plots:
            print("\n[5/5] Generating training plots...")
            self._generate_plots(features_df, fair_scores, is_model, bs_model, ls_model, rf_model)
        
        # Save metrics
        self._save_metrics()
        
        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)
        print(f"\nModels saved to: {self.config.models_dir}")
        print(f"Outputs saved to: {self.config.outputs_dir}")
        
        return self.metrics
    
    def _data_exists(self) -> bool:
        """Check if training data exists."""
        users_path = self.config.data_dir / "users.csv"
        transactions_path = self.config.data_dir / "transactions.csv"
        return users_path.exists() and transactions_path.exists()
    
    def _compute_validation_scores(
        self,
        features_df: pd.DataFrame,
        is_model: IncomeStabilityModel,
        bs_model: ExpenseDisciplineModel,
        ls_model: LiquidityBufferModel,
        rf_model: RiskAnomalyModel,
    ) -> pd.DataFrame:
        """Compute FAIR-SCORES for all users."""
        is_scores = is_model.predict(features_df)
        bs_scores = bs_model.predict(features_df)
        ls_scores = ls_model.predict(features_df)
        rf_scores = rf_model.predict(features_df)
        
        # Compute FAIR-SCORE
        weighted_sum = (
            0.35 * is_scores +
            0.30 * bs_scores +
            0.25 * ls_scores +
            0.10 * (1 - rf_scores)
        )
        fair_scores = 300 + 600 * weighted_sum
        fair_scores = np.clip(fair_scores, 300, 900).astype(int)
        
        # Categorize
        def get_category(score):
            if score < 500:
                return "POOR"
            elif score < 650:
                return "FAIR"
            elif score < 750:
                return "GOOD"
            else:
                return "EXCELLENT"
        
        result = pd.DataFrame({
            "user_id": features_df["user_id"],
            "fair_score": fair_scores,
            "is_score": is_scores,
            "bs_score": bs_scores,
            "ls_score": ls_scores,
            "rf_score": rf_scores,
            "category": [get_category(s) for s in fair_scores],
        })
        
        # Save to outputs
        output_path = self.config.get_output_path("fair_scores", "csv")
        result.to_csv(output_path, index=False)
        print(f"    Saved scores to: {output_path}")
        
        return result
    
    def _generate_plots(
        self,
        features_df: pd.DataFrame,
        fair_scores: pd.DataFrame,
        is_model: IncomeStabilityModel,
        bs_model: ExpenseDisciplineModel,
        ls_model: LiquidityBufferModel,
        rf_model: RiskAnomalyModel,
    ):
        """Generate training and evaluation plots."""
        try:
            import matplotlib
            matplotlib.use("Agg")  # Non-interactive backend
            import matplotlib.pyplot as plt
            from matplotlib.gridspec import GridSpec
        except ImportError:
            print("    Matplotlib not available, skipping plots")
            return
        
        # Set style for publication-quality plots
        plt.style.use("seaborn-v0_8-whitegrid")
        plt.rcParams["figure.dpi"] = 150
        plt.rcParams["font.size"] = 10
        
        # 1. Score Distribution Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = {"POOR": "#FF6B35", "FAIR": "#FFB000", "GOOD": "#00CC33", "EXCELLENT": "#00FF41"}
        
        for category in ["POOR", "FAIR", "GOOD", "EXCELLENT"]:
            data = fair_scores[fair_scores["category"] == category]["fair_score"]
            if len(data) > 0:
                ax.hist(data, bins=30, alpha=0.7, label=category, color=colors[category])
        
        ax.set_xlabel("FAIR-SCORE")
        ax.set_ylabel("Count")
        ax.set_title("FAIR-SCORE Distribution by Category")
        ax.legend()
        ax.set_xlim(300, 900)
        
        plot_path = self.config.get_output_path("score_distribution", "png")
        fig.savefig(plot_path, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"    Saved: {plot_path.name}")
        
        # 2. Feature Importance Comparison
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        models = [
            (is_model, "Income Stability (Is)", axes[0, 0]),
            (bs_model, "Expense Discipline (Bs)", axes[0, 1]),
            (ls_model, "Liquidity Buffer (Ls)", axes[1, 0]),
            (rf_model, "Risk Anomaly (Rf)", axes[1, 1]),
        ]
        
        for model, title, ax in models:
            importance = model.get_feature_importance()
            features = list(importance.keys())
            values = list(importance.values())
            
            # Shorten feature names
            short_names = [f.split("_", 1)[1] if "_" in f else f for f in features]
            
            bars = ax.barh(short_names, values, color="#00FF41", alpha=0.8)
            ax.set_xlabel("Importance")
            ax.set_title(title)
            ax.invert_yaxis()
        
        plt.tight_layout()
        plot_path = self.config.get_output_path("feature_importance", "png")
        fig.savefig(plot_path, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"    Saved: {plot_path.name}")
        
        # 3. Model Performance Summary
        fig, ax = plt.subplots(figsize=(10, 6))
        
        model_names = ["Income Stability", "Expense Discipline", "Liquidity Buffer"]
        auc_scores = [
            self.metrics["income_stability"]["auc_score"],
            self.metrics["expense_discipline"]["auc_score"],
            self.metrics["liquidity_buffer"]["auc_score"],
        ]
        accuracy_scores = [
            self.metrics["income_stability"]["accuracy"],
            self.metrics["expense_discipline"]["accuracy"],
            self.metrics["liquidity_buffer"]["accuracy"],
        ]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, auc_scores, width, label="AUC-ROC", color="#00FF41")
        bars2 = ax.bar(x + width/2, accuracy_scores, width, label="Accuracy", color="#FFB000")
        
        ax.set_ylabel("Score")
        ax.set_title("Model Performance Comparison")
        ax.set_xticks(x)
        ax.set_xticklabels(model_names)
        ax.legend()
        ax.set_ylim(0, 1)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f"{height:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=8)
        
        plot_path = self.config.get_output_path("model_performance", "png")
        fig.savefig(plot_path, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"    Saved: {plot_path.name}")
        
        # 4. Score Component Correlation Heatmap
        fig, ax = plt.subplots(figsize=(8, 6))
        
        score_cols = ["is_score", "bs_score", "ls_score", "rf_score", "fair_score"]
        corr = fair_scores[score_cols].corr()
        
        im = ax.imshow(corr, cmap="RdYlGn", aspect="auto", vmin=-1, vmax=1)
        
        ax.set_xticks(np.arange(len(score_cols)))
        ax.set_yticks(np.arange(len(score_cols)))
        labels = ["Is", "Bs", "Ls", "Rf", "FAIR-SCORE"]
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        
        # Add correlation values
        for i in range(len(score_cols)):
            for j in range(len(score_cols)):
                ax.text(j, i, f"{corr.iloc[i, j]:.2f}",
                    ha="center", va="center", color="black", fontsize=10)
        
        ax.set_title("Score Component Correlation Matrix")
        fig.colorbar(im, ax=ax)
        
        plot_path = self.config.get_output_path("score_correlation", "png")
        fig.savefig(plot_path, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"    Saved: {plot_path.name}")
        
        # 5. Score by Occupation Type (if available)
        if "occupation_type" in features_df.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            merged = fair_scores.merge(
                features_df[["user_id", "occupation_type"]], on="user_id"
            )
            
            occupation_groups = merged.groupby("occupation_type")["fair_score"].agg(["mean", "std"]).reset_index()
            occupation_groups = occupation_groups.sort_values("mean", ascending=True)
            
            ax.barh(
                occupation_groups["occupation_type"],
                occupation_groups["mean"],
                xerr=occupation_groups["std"],
                color="#00FF41",
                alpha=0.8,
                capsize=5,
            )
            
            ax.set_xlabel("FAIR-SCORE (Mean +/- Std)")
            ax.set_title("FAIR-SCORE by Occupation Type")
            ax.set_xlim(300, 900)
            
            plot_path = self.config.get_output_path("score_by_occupation", "png")
            fig.savefig(plot_path, bbox_inches="tight", facecolor="white")
            plt.close(fig)
            print(f"    Saved: {plot_path.name}")
    
    def _save_metrics(self):
        """Save training metrics to JSON."""
        metrics_path = self.config.get_output_path("training_metrics", "json")
        
        # Convert numpy types for JSON serialization
        def convert(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj
        
        metrics = convert(self.metrics)
        metrics["training_timestamp"] = datetime.now().isoformat()
        
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        
        print(f"    Saved metrics to: {metrics_path}")


if __name__ == "__main__":
    config = Config()
    trainer = ModelTrainer(config)
    # generate_data=False: use existing data if available, otherwise generate
    trainer.train_all(generate_data=False, save_plots=True)

