"""
Main Dashboard Widget for FairScore TUI.
Implements the Bloomberg Terminal-style layout.
"""

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Static
from typing import Optional

from fairscore.theme import COLORS, create_score_gauge, get_score_category


class DashboardHeader(Static):
    """Top status bar with system info."""
    
    DEFAULT_CSS = """
    DashboardHeader {
        dock: top;
        height: 1;
        background: #003311;
        color: #00FF41;
        text-style: bold;
        padding: 0 1;
    }
    """
    
    def __init__(self, version: str = "1.0.0"):
        super().__init__()
        self._version = version
    
    def on_mount(self) -> None:
        self.update(self._get_status_text())
    
    def _get_status_text(self) -> str:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"FAIRSCORE v{self._version} // SYS_STATUS: OPERATIONAL // {timestamp}"
    
    def update_status(self, status: str = "OPERATIONAL"):
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        text = f"FAIRSCORE v{self._version} // SYS_STATUS: {status} // {timestamp}"
        self.update(text)


class ScoreDisplay(Static):
    """Large FAIR-SCORE display widget."""
    
    DEFAULT_CSS = """
    ScoreDisplay {
        height: 7;
        border: solid #003311;
        padding: 1;
        margin: 1;
        text-align: center;
    }
    
    ScoreDisplay.excellent {
        border: solid #00FF41;
        color: #00FF41;
    }
    
    ScoreDisplay.good {
        border: solid #00CC33;
        color: #00CC33;
    }
    
    ScoreDisplay.fair {
        border: solid #FFB000;
        color: #FFB000;
    }
    
    ScoreDisplay.poor {
        border: solid #FF6B35;
        color: #FF6B35;
    }
    """
    
    def __init__(self, score: Optional[int] = None):
        super().__init__()
        self._score = score
    
    def on_mount(self) -> None:
        self._refresh_display()
    
    def set_score(self, score: int):
        self._score = score
        self._refresh_display()
    
    def _refresh_display(self):
        if self._score is None:
            text = "\n".join([
                "",
                "FAIR-SCORE: ---",
                "",
                "[Awaiting Analysis]",
                "",
            ])
            self.remove_class("excellent", "good", "fair", "poor")
        else:
            category = get_score_category(self._score)
            gauge = create_score_gauge(self._score)
            text = gauge
            
            # Set category class
            self.remove_class("excellent", "good", "fair", "poor")
            self.add_class(category.lower())
        
        self.update(text)


class ModelPanel(Static):
    """Individual model score panel."""
    
    DEFAULT_CSS = """
    ModelPanel {
        border: solid #003311;
        height: 5;
        padding: 0 1;
        margin: 0 1;
    }
    
    ModelPanel .panel-header {
        color: #00FF41;
        text-style: bold;
    }
    
    ModelPanel .panel-value {
        color: #00CC33;
        text-align: center;
    }
    """
    
    def __init__(self, code: str, model_name: str, score: Optional[float] = None):
        super().__init__()
        self.code = code
        self.model_name = model_name
        self._score = score
    
    def on_mount(self) -> None:
        self._refresh_display()
    
    def _refresh_display(self):
        lines = [f"[{self.code}] {self.model_name}"]
        if self._score is not None:
            bar_len = int(self._score * 15)
            bar = "#" * bar_len + "." * (15 - bar_len)
            lines.append(f"[{bar}] {self._score:.0%}")
        else:
            lines.append("[...............] ---%")
        self.update("\n".join(lines))
    
    def set_score(self, score: float):
        self._score = score
        self._refresh_display()


class ComponentBreakdown(Static):
    """SHAP-style component breakdown display."""
    
    DEFAULT_CSS = """
    ComponentBreakdown {
        border: solid #003311;
        padding: 1;
        margin: 1;
        height: auto;
        min-height: 10;
    }
    
    ComponentBreakdown .header {
        color: #00FF41;
        text-style: bold;
        margin-bottom: 1;
    }
    
    ComponentBreakdown .positive {
        color: #00FF41;
    }
    
    ComponentBreakdown .negative {
        color: #FF6B35;
    }
    """
    
    def __init__(self):
        super().__init__()
        self.breakdown_text = "Awaiting analysis..."
    
    def on_mount(self) -> None:
        self.update(self.breakdown_text)
    
    def set_breakdown(self, breakdown_text: str):
        self.breakdown_text = breakdown_text
        self.update(breakdown_text)


class RecommendationsPanel(Static):
    """Recommendations display panel."""
    
    DEFAULT_CSS = """
    RecommendationsPanel {
        border: solid #003311;
        padding: 1;
        margin: 1;
        height: auto;
        min-height: 8;
    }
    
    RecommendationsPanel .header {
        color: #FFB000;
        text-style: bold;
    }
    """
    
    def __init__(self):
        super().__init__()
        self._recommendations = []
    
    def on_mount(self) -> None:
        self._refresh_display()
    
    def set_recommendations(self, recommendations: list):
        self._recommendations = recommendations
        self._refresh_display()
    
    def _refresh_display(self):
        lines = ["RECOMMENDATIONS", "-" * 40]
        
        if not self._recommendations:
            lines.append("No recommendations available.")
        else:
            for i, rec in enumerate(self._recommendations[:5], 1):
                if isinstance(rec, dict):
                    priority = rec.get("priority", "medium").upper()
                    text = rec.get("text", "")
                    impact = rec.get("impact", "")
                    lines.append(f"{i}. [{priority}] {text}")
                    if impact:
                        lines.append(f"   Impact: {impact}")
                else:
                    lines.append(f"{i}. {rec}")
        
        self.update("\n".join(lines))


class DashboardWidget(Container):
    """
    Main dashboard container with Bloomberg Terminal layout.
    
    Layout:
    +------------------------------------------+
    | Status Bar                               |
    +------------------------------------------+
    | Score Display                            |
    +----+----+----+----+---------------------+
    | Is | Bs | Ls | Rf | Component Breakdown |
    +----+----+----+----+---------------------+
    | Recommendations                          |
    +------------------------------------------+
    """
    
    DEFAULT_CSS = """
    DashboardWidget {
        layout: grid;
        grid-size: 2;
        grid-columns: 1fr 1fr;
        grid-rows: auto auto 1fr auto;
        padding: 0;
        background: #0A0A0A;
    }
    
    #score-section {
        column-span: 2;
    }
    
    #models-section {
        height: auto;
        min-height: 6;
    }
    
    #breakdown-section {
        height: auto;
    }
    
    #recommendations-section {
        column-span: 2;
    }
    """
    
    def __init__(self):
        super().__init__()
        self.score_display = ScoreDisplay()
        self.model_panels = {
            "Is": ModelPanel("Is", "Income Stability"),
            "Bs": ModelPanel("Bs", "Expense Discipline"),
            "Ls": ModelPanel("Ls", "Liquidity Buffer"),
            "Rf": ModelPanel("Rf", "Risk Anomaly"),
        }
        self.breakdown = ComponentBreakdown()
        self._recommendations_panel = RecommendationsPanel()
    
    def compose(self) -> ComposeResult:
        with Container(id="score-section"):
            yield self.score_display
        
        with Vertical(id="models-section"):
            yield Static("MODEL MATRIX", classes="section-header")
            with Horizontal():
                yield self.model_panels["Is"]
                yield self.model_panels["Bs"]
            with Horizontal():
                yield self.model_panels["Ls"]
                yield self.model_panels["Rf"]
        
        with Container(id="breakdown-section"):
            yield self.breakdown
        
        with Container(id="recommendations-section"):
            yield self._recommendations_panel
    
    def update_score(self, fair_score: int, is_score: float, bs_score: float,
                     ls_score: float, rf_score: float):
        """Update all score displays."""
        self.score_display.set_score(fair_score)
        self.model_panels["Is"].set_score(is_score)
        self.model_panels["Bs"].set_score(bs_score)
        self.model_panels["Ls"].set_score(ls_score)
        self.model_panels["Rf"].set_score(rf_score)
    
    def update_breakdown(self, breakdown_text: str):
        """Update component breakdown display."""
        self.breakdown.set_breakdown(breakdown_text)
    
    def update_recommendations(self, recommendations: list):
        """Update recommendations display."""
        self._recommendations_panel.set_recommendations(recommendations)
