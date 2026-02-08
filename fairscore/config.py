"""
Configuration management for FairScore application.
Handles paths, API keys, and model parameters.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """Configuration for ML models."""
    # Income Stability Model (LightGBM)
    is_num_leaves: int = 31
    is_max_depth: int = 10
    is_n_estimators: int = 100
    is_learning_rate: float = 0.05
    
    # Expense Discipline Model (Random Forest)
    bs_n_estimators: int = 100
    bs_max_depth: int = 10
    bs_n_jobs: int = 2
    
    # Liquidity Buffer Model (Logistic Regression)
    ls_max_iter: int = 1000
    ls_C: float = 1.0
    
    # Risk Anomaly Model (Isolation Forest)
    rf_n_estimators: int = 100
    rf_contamination: float = 0.1
    rf_n_jobs: int = 2
    
    # Ensemble weights
    weight_is: float = 0.35
    weight_bs: float = 0.30
    weight_ls: float = 0.25
    weight_rf: float = 0.10


@dataclass
class DataConfig:
    """Configuration for data generation and processing."""
    num_users: int = 10000  # Reduced from 100k for faster processing
    num_transactions_per_user: tuple = (50, 500)
    train_test_split: float = 0.2
    random_seed: int = 42
    batch_size: int = 5000  # For memory-efficient processing


@dataclass
class Config:
    """Main configuration class for FairScore application."""
    
    # Base paths
    base_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    data_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "data")
    models_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "models")
    outputs_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "outputs")
    tests_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "tests")
    
    # Sub-configurations
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    
    # API Configuration
    google_api_key: Optional[str] = field(default_factory=lambda: os.getenv("GOOGLE_API_KEY"))
    
    # Score range
    score_min: int = 300
    score_max: int = 900
    
    def __post_init__(self):
        """Create directories if they don't exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.tests_dir.mkdir(parents=True, exist_ok=True)
    
    @property
    def has_api_key(self) -> bool:
        """Check if Google API key is configured."""
        return self.google_api_key is not None and len(self.google_api_key) > 0
    
    def get_output_path(self, prefix: str, extension: str) -> Path:
        """Generate timestamped output path with UUID."""
        import uuid
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        filename = f"{prefix}_{timestamp}_{unique_id}.{extension}"
        return self.outputs_dir / filename


# Global config instance
config = Config()
