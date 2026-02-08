# FairScore

**AI-Powered Credit Scoring System with Bloomberg Terminal-Style TUI**

FairScore is a fair, explainable, and inclusive credit scoring system that uses behavioral financial data instead of traditional credit bureau metrics. It features a Terminal User Interface (TUI) inspired by Bloomberg Terminal aesthetics.

## Features

- **4 Specialized ML Models**: Income Stability (LightGBM), Expense Discipline (Random Forest), Liquidity Buffer (Logistic Regression), Risk Detection (Isolation Forest)
- **FAIR-SCORE Formula**: Weighted ensemble producing 300-900 score range
- **Explainability**: SHAP-based explanations with waterfall visualizations
- **Dual Input Modes**: PDF parsing (Gemini AI) or manual data entry
- **Bloomberg Terminal UI**: Phosphor green on matte black aesthetic

## Installation

```bash
# Clone or download, then:
pip install -e .
```

## Quick Start

```bash
# Launch TUI
fairscore

# Train models (generates synthetic data first)
python -m fairscore.models.trainer

# Generate synthetic data only
python -c "from fairscore.data import DataGenerator; DataGenerator().generate_all()"
```

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| F1 | Help screen |
| F5 | Refresh display |
| Ctrl+O | Upload PDF |
| Ctrl+M | Manual entry |
| Tab | Navigate fields |
| Q | Quit |

## Score Categories

| Range | Category | Description |
|-------|----------|-------------|
| 750-900 | EXCELLENT | Best creditworthiness |
| 650-749 | GOOD | Above average |
| 500-649 | FAIR | Average |
| 300-499 | POOR | Below average |

## Project Structure

```
fairscore/
  __init__.py       # Package initialization
  app.py            # Main TUI application
  cli.py            # CLI entry points
  config.py         # Configuration management
  theme.py          # Bloomberg Terminal theme
  data/
    generator.py    # Synthetic data generation
    features.py     # Feature engineering
  models/
    income_stability.py    # LightGBM model
    expense_discipline.py  # Random Forest model
    liquidity_buffer.py    # Logistic Regression model
    risk_anomaly.py        # Isolation Forest model
    ensemble.py            # FAIR-SCORE computation
    trainer.py             # Training pipeline
  explainability/
    shap_explainer.py      # SHAP explanations
  parser/
    pdf_parser.py          # Gemini AI PDF parsing
  ui/
    dashboard.py           # Main dashboard
    model_matrix.py        # 4-panel model display
    score_gauge.py         # ASCII score gauge
    input_mode.py          # Input selection/forms
```

## Configuration

Set environment variable for PDF parsing:
```bash
set GOOGLE_API_KEY=your_gemini_api_key
```

## License

MIT License
