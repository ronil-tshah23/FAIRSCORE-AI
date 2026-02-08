"""UI components for FairScore TUI."""

from fairscore.ui.dashboard import DashboardWidget
from fairscore.ui.model_matrix import ModelMatrixWidget
from fairscore.ui.score_gauge import ScoreGaugeWidget
from fairscore.ui.input_mode import InputModeScreen, ManualEntryForm

__all__ = [
    "DashboardWidget",
    "ModelMatrixWidget", 
    "ScoreGaugeWidget",
    "InputModeScreen",
    "ManualEntryForm",
]
