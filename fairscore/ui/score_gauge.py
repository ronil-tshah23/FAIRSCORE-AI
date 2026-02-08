"""
Score Gauge Widget for FairScore TUI.
ASCII visualization of the FAIR-SCORE.
"""

from textual.app import ComposeResult
from textual.widgets import Static
from typing import Optional

from fairscore.theme import PROGRESS, get_score_color, get_score_category


class ScoreGaugeWidget(Static):
    """
    Large ASCII gauge visualization for FAIR-SCORE.
    
    Display:
    +============================================+
    |            FAIR-SCORE: 725                 |
    |  [########################........] 80.4%  |
    |             Category: GOOD                 |
    +============================================+
    """
    
    DEFAULT_CSS = """
    ScoreGaugeWidget {
        height: 9;
        border: double #003311;
        padding: 1;
        text-align: center;
        background: #0A0A0A;
    }
    
    ScoreGaugeWidget.excellent {
        border: double #00FF41;
        color: #00FF41;
    }
    
    ScoreGaugeWidget.good {
        border: double #00CC33;
        color: #00CC33;
    }
    
    ScoreGaugeWidget.fair {
        border: double #FFB000;
        color: #FFB000;
    }
    
    ScoreGaugeWidget.poor {
        border: double #FF6B35;
        color: #FF6B35;
    }
    """
    
    SCORE_MIN = 300
    SCORE_MAX = 900
    BAR_WIDTH = 40
    
    def __init__(self, score_value: Optional[int] = None):
        super().__init__()
        self._score = score_value
    
    def on_mount(self) -> None:
        self._refresh_display()
    
    def set_score(self, score_value: int):
        """Update the displayed score."""
        self._score = score_value
        self._refresh_display()
    
    def clear_score(self):
        """Clear the score display."""
        self._score = None
        self._refresh_display()
    
    def _refresh_display(self):
        """Render the gauge display."""
        if self._score is None:
            text = self._get_empty_content()
            self.remove_class("excellent", "good", "fair", "poor")
        else:
            text = self._get_score_content()
            category = get_score_category(self._score).lower()
            self.remove_class("excellent", "good", "fair", "poor")
            self.add_class(category)
        
        self.update(text)
    
    def _get_empty_content(self) -> str:
        """Get empty state content."""
        lines = [
            "",
            "FAIR-SCORE: ---",
            "",
            f"  {PROGRESS['left_cap']}{'.' * self.BAR_WIDTH}{PROGRESS['right_cap']}  ---%",
            "",
            "Category: AWAITING ANALYSIS",
            "",
        ]
        return "\n".join(lines)
    
    def _get_score_content(self) -> str:
        """Get score gauge content."""
        # Calculate fill percentage
        normalized = (self._score - self.SCORE_MIN) / (self.SCORE_MAX - self.SCORE_MIN)
        normalized = max(0.0, min(1.0, normalized))
        percent = normalized * 100
        
        # Create progress bar
        filled = int(normalized * self.BAR_WIDTH)
        empty = self.BAR_WIDTH - filled
        bar = (
            PROGRESS["left_cap"] +
            PROGRESS["filled"] * filled +
            PROGRESS["empty"] * empty +
            PROGRESS["right_cap"]
        )
        
        category = get_score_category(self._score)
        
        # Build display
        lines = [
            "",
            f"FAIR-SCORE: {self._score}",
            "",
            f"  {bar}  {percent:5.1f}%",
            "",
            f"Category: {category}",
            "",
        ]
        
        return "\n".join(lines)
    
    def get_ascii_art(self) -> str:
        """Get the ASCII art representation for logging/export."""
        if self._score is None:
            return "FAIR-SCORE: Not Available"
        
        normalized = (self._score - self.SCORE_MIN) / (self.SCORE_MAX - self.SCORE_MIN)
        filled = int(normalized * 30)
        empty = 30 - filled
        bar = "#" * filled + "." * empty
        category = get_score_category(self._score)
        
        return f"""
+{'=' * 50}+
|{'FAIR-SCORE RESULT':^50}|
+{'=' * 50}+
|{' ' * 50}|
|  Score: {self._score:>4} / {self.SCORE_MAX}{' ' * 32}|
|  [{bar}] {normalized*100:5.1f}%{' ' * 5}|
|  Category: {category:<38}|
|{' ' * 50}|
+{'=' * 50}+
"""
