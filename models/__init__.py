"""
FairScore Pre-trained Models Package

This package contains pre-trained machine learning models for credit scoring.
"""
import os
from pathlib import Path

# Get the directory containing the model files
MODELS_DIR = Path(__file__).parent

def get_model_path(model_name: str) -> Path:
    """
    Get the absolute path to a model file.
    
    Args:
        model_name: Name of the model file (e.g., 'income_stability_model.pkl')
        
    Returns:
        Path to the model file
    """
    return MODELS_DIR / model_name

def list_available_models() -> list:
    """
    List all available pre-trained models.
    
    Returns:
        List of model file names
    """
    return [f.name for f in MODELS_DIR.glob("*.pkl")]
