"""
FairScore TUI Application.
Bloomberg Terminal-inspired credit scoring interface.
"""

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.widgets import Static, Header, Footer, Input, Button
from textual.screen import Screen
from typing import Optional
import asyncio
from pathlib import Path

from fairscore.config import Config
from fairscore.theme import FAIRSCORE_CSS, COLORS
from fairscore.ui.dashboard import DashboardHeader
from fairscore.ui.model_matrix import ModelMatrixWidget
from fairscore.ui.score_gauge import ScoreGaugeWidget
from fairscore.ui.input_mode import InputModeScreen, ManualEntryForm
from fairscore.explainability.shap_explainer import ShapExplainer, compute_manual_features


class HelpScreen(Screen):
    """Help screen with keyboard shortcuts and usage info."""
    
    BINDINGS = [
        Binding("escape", "dismiss", "Close"),
        Binding("q", "dismiss", "Close"),
    ]
    
    DEFAULT_CSS = """
    HelpScreen {
        background: #0A0A0A;
    }
    
    HelpScreen ScrollableContainer {
        background: #0A0A0A;
        padding: 2;
    }
    
    HelpScreen Static {
        color: #00FF41;
    }
    """
    
    def compose(self) -> ComposeResult:
        with ScrollableContainer():
            yield Static("""
+==================================================+
|                 FAIRSCORE HELP                   |
+==================================================+

KEYBOARD SHORTCUTS
------------------
  F1          Show this help screen
  F5          Refresh/recalculate scores
  Ctrl+O      Upload PDF bank statement
  Ctrl+M      Manual data entry
  Tab         Navigate between fields
  Shift+Tab   Navigate backwards
  Enter       Confirm/Submit
  Escape      Cancel/Back
  Q           Quit application

SCORE CATEGORIES
----------------
  EXCELLENT   750-900   Best creditworthiness
  GOOD        650-749   Above average
  FAIR        500-649   Average
  POOR        300-499   Below average

MODEL COMPONENTS
----------------
  [Is] Income Stability
       Evaluates regularity and predictability of income.
       Algorithm: LightGBM
       
  [Bs] Expense Discipline
       Assesses spending patterns and budgeting.
       Algorithm: Random Forest
       
  [Ls] Liquidity & Buffer
       Measures savings strength and emergency fund.
       Algorithm: Logistic Regression
       
  [Rf] Risk & Anomaly Detection
       Identifies irregular or suspicious activity.
       Algorithm: Isolation Forest

FAIR-SCORE FORMULA
------------------
  FAIR-SCORE = 300 + 600 * (0.35*Is + 0.30*Bs + 0.25*Ls + 0.10*(1-Rf))
  
  Score Range: 300 to 900

INPUT MODES
-----------
  PDF Upload:
    - Supported: Bank statements in PDF format
    - Uses Google Gemini AI for parsing
    - Set GOOGLE_API_KEY environment variable
    
  Manual Entry:
    - Enter monthly income, expenses, savings
    - Provide essential expense ratio
    - Report overdraft and irregular transactions

OUTPUTS
-------
  - All results saved to: outputs/
  - Plots in PNG format
  - Scores in CSV format

+==================================================+
|          Press [Escape] or [Q] to close          |
+==================================================+
""")


class ResultScreen(Screen):
    """Screen displaying score results with breakdown."""
    
    BINDINGS = [
        Binding("escape", "dismiss", "Back"),
        Binding("s", "save_result", "Save"),
    ]
    
    DEFAULT_CSS = """
    ResultScreen {
        background: #0A0A0A;
    }
    
    ResultScreen ScrollableContainer {
        background: #0A0A0A;
        padding: 1;
    }
    
    ResultScreen Static {
        color: #00FF41;
    }
    
    ResultScreen .title {
        text-align: center;
        text-style: bold;
        color: #00FF41;
    }
    
    ResultScreen .section-header {
        color: #FFB000;
        text-style: bold;
        margin-top: 1;
    }
    """
    
    def __init__(
        self,
        fair_score: int,
        is_score: float,
        bs_score: float,
        ls_score: float,
        rf_score: float,
        explanation: dict,
    ):
        super().__init__()
        self._fair_score = fair_score
        self._is_score = is_score
        self._bs_score = bs_score
        self._ls_score = ls_score
        self._rf_score = rf_score
        self._explanation = explanation
    
    def compose(self) -> ComposeResult:
        with ScrollableContainer():
            # Title
            yield Static("=" * 60, classes="title")
            yield Static("FAIRSCORE ANALYSIS RESULT", classes="title")
            yield Static("=" * 60, classes="title")
            yield Static("")
            
            # Score gauge
            yield ScoreGaugeWidget(score_value=self._fair_score)
            yield Static("")
            
            # Model scores breakdown
            yield Static("MODEL SCORES", classes="section-header")
            yield Static("-" * 40)
            yield Static(f"  [Is] Income Stability:    {self._is_score:6.1%}")
            yield Static(f"  [Bs] Expense Discipline:  {self._bs_score:6.1%}")
            yield Static(f"  [Ls] Liquidity Buffer:    {self._ls_score:6.1%}")
            yield Static(f"  [Rf] Risk Anomaly:        {self._rf_score:6.1%}")
            yield Static("")
            
            # Waterfall breakdown
            waterfall = self._explanation.get("waterfall", {})
            if waterfall:
                yield Static("SCORE BREAKDOWN", classes="section-header")
                yield Static("-" * 40)
                yield Static(f"  Base Score:               {waterfall.get('base', 300)}")
                for contrib in waterfall.get("contributions", []):
                    sign = "+" if contrib.get("contribution", 0) >= 0 else ""
                    yield Static(f"  {contrib.get('name', '')}: {sign}{contrib.get('contribution', 0):.1f}")
                yield Static(f"  Final Score:              {waterfall.get('final', self._fair_score)}")
                yield Static("")
            
            # Strengths
            strengths = self._explanation.get("strengths", [])
            if strengths:
                yield Static("STRENGTHS", classes="section-header")
                yield Static("-" * 40)
                for s in strengths:
                    yield Static(f"  + {s}")
                yield Static("")
            
            # Weaknesses
            weaknesses = self._explanation.get("weaknesses", [])
            if weaknesses:
                yield Static("AREAS FOR IMPROVEMENT", classes="section-header")
                yield Static("-" * 40)
                for w in weaknesses:
                    yield Static(f"  - {w}")
                yield Static("")
            
            # Recommendations
            recommendations = self._explanation.get("recommendations", [])
            if recommendations:
                yield Static("RECOMMENDATIONS", classes="section-header")
                yield Static("-" * 40)
                for i, rec in enumerate(recommendations[:5], 1):
                    if isinstance(rec, dict):
                        yield Static(f"  {i}. [{rec.get('priority', 'medium').upper()}] {rec.get('text', '')}")
                    else:
                        yield Static(f"  {i}. {rec}")
                yield Static("")
            
            yield Static("")
            yield Static("Press [Escape] to go back, [S] to save results", classes="title")
    
    def action_save_result(self) -> None:
        """Save results to file."""
        config = Config()
        output_path = config.get_output_path("score_result", "txt")
        
        with open(output_path, "w") as f:
            f.write(f"FAIRSCORE Analysis Result\n")
            f.write(f"{'=' * 50}\n\n")
            f.write(f"FAIR-SCORE: {self._fair_score}\n")
            f.write(f"Category: {self._explanation.get('category', 'N/A')}\n\n")
            f.write(f"Component Scores:\n")
            f.write(f"  Income Stability (Is): {self._is_score:.2%}\n")
            f.write(f"  Expense Discipline (Bs): {self._bs_score:.2%}\n")
            f.write(f"  Liquidity Buffer (Ls): {self._ls_score:.2%}\n")
            f.write(f"  Risk Anomaly (Rf): {self._rf_score:.2%}\n")
        
        self.notify(f"Results saved to: {output_path.name}")


class FairScoreApp(App):
    """
    FairScore TUI Application.
    
    Main application implementing the Bloomberg Terminal-style interface
    for credit scoring with AI-powered analysis.
    """
    
    CSS = FAIRSCORE_CSS
    
    TITLE = "FAIRSCORE"
    SUB_TITLE = "AI-Powered Credit Scoring System"
    
    BINDINGS = [
        Binding("f1", "show_help", "Help", show=True),
        Binding("f5", "refresh", "Refresh", show=True),
        Binding("ctrl+o", "open_pdf", "Open PDF", show=True),
        Binding("ctrl+m", "manual_entry", "Manual Entry", show=True),
        Binding("q", "quit", "Quit", show=True),
    ]
    
    DEFAULT_CSS = """
    FairScoreApp {
        background: #0A0A0A;
    }
    
    #main-container {
        background: #0A0A0A;
        padding: 1;
    }
    
    #welcome-section {
        height: auto;
        padding: 1;
    }
    
    #action-buttons {
        height: auto;
        padding: 1;
    }
    
    #action-buttons Button {
        margin: 0 2;
    }
    
    .section-header {
        color: #00FF41;
        text-style: bold;
        text-align: center;
    }
    
    .welcome-text {
        color: #00FF41;
        text-align: center;
    }
    """
    
    def __init__(self):
        super().__init__()
        self.config = Config()
        self.explainer = ShapExplainer(self.config)
        self.current_score: Optional[int] = None
        self.component_scores = {
            "is": None,
            "bs": None,
            "ls": None,
            "rf": None,
        }
    
    def compose(self) -> ComposeResult:
        yield Header()
        
        with ScrollableContainer(id="main-container"):
            # Status header
            yield DashboardHeader(version="1.0.0")
            
            # Welcome message
            yield Static("")
            yield Static("+" + "=" * 58 + "+", classes="welcome-text")
            yield Static("|" + "WELCOME TO FAIRSCORE".center(58) + "|", classes="welcome-text")
            yield Static("|" + "AI-Powered Credit Scoring System".center(58) + "|", classes="welcome-text")
            yield Static("+" + "=" * 58 + "+", classes="welcome-text")
            yield Static("")
            yield Static("Select an input mode to begin:", classes="welcome-text")
            yield Static("")
            
            with Horizontal(id="action-buttons"):
                yield Button("[Ctrl+O] Upload PDF", id="btn-pdf", variant="primary")
                yield Button("[Ctrl+M] Manual Entry", id="btn-manual")
                yield Button("[F1] Help", id="btn-help")
            
            yield Static("")
            yield Static("-" * 60, classes="welcome-text")
            yield Static("")
            
            # Score display (initially empty)
            yield ScoreGaugeWidget()
            yield Static("")
            
            # Model matrix (initially empty)
            yield ModelMatrixWidget()
            yield Static("")
            
            # Instructions
            yield Static("Use Ctrl+M for Manual Entry or Ctrl+O for PDF Upload", classes="welcome-text")
            yield Static("Press F1 for Help, Q to Quit", classes="welcome-text")
        
        yield Footer()
    
    def action_show_help(self) -> None:
        """Show help screen."""
        self.push_screen(HelpScreen())
    
    def action_refresh(self) -> None:
        """Refresh the display."""
        try:
            header = self.query_one(DashboardHeader)
            header.update_status("REFRESHING")
            asyncio.create_task(self._delayed_status_update("OPERATIONAL", 0.5))
        except Exception:
            pass
    
    async def _delayed_status_update(self, status: str, delay: float):
        await asyncio.sleep(delay)
        try:
            header = self.query_one(DashboardHeader)
            header.update_status(status)
        except Exception:
            pass
    
    def action_open_pdf(self) -> None:
        """Open PDF file for parsing."""
        self.notify("PDF parsing requires GOOGLE_API_KEY environment variable")
        
        if not self.config.google_api_key:
            self.notify(
                "Set GOOGLE_API_KEY environment variable to use PDF parsing",
                severity="warning",
            )
            return
        
        self.notify("PDF upload: Enter path in terminal or use Manual Entry")
    
    def action_manual_entry(self) -> None:
        """Open manual entry form."""
        
        def handle_result(result):
            if result is None:
                return
            if isinstance(result, tuple) and result[0] == "manual":
                data = result[1]
                self._process_manual_entry(data)
        
        self.push_screen(InputModeScreen(), handle_result)
    
    def _process_manual_entry(self, data: dict) -> None:
        """Process manual entry data and compute score."""
        # Compute features from manual input
        is_feat, bs_feat, ls_feat, rf_feat = compute_manual_features(
            monthly_income=data.get("monthly_income", 50000),
            income_frequency=data.get("income_frequency", "monthly"),
            monthly_expenses=data.get("monthly_expenses", 35000),
            essential_ratio=data.get("essential_ratio", 0.6),
            avg_balance=data.get("avg_balance", 20000),
            monthly_savings=data.get("monthly_savings", 10000),
            overdraft_count=data.get("overdraft_count", 0),
            irregular_txn_count=data.get("irregular_txn_count", 0),
        )
        
        # Compute simple scores from features
        is_score = sum(is_feat.values()) / len(is_feat)
        bs_score = sum(bs_feat.values()) / len(bs_feat)
        ls_score = sum(ls_feat.values()) / len(ls_feat)
        rf_score = sum(rf_feat.values()) / len(rf_feat)
        
        # Get explanation
        explanation = self.explainer.explain_score_simple(
            is_score, bs_score, ls_score, rf_score,
            is_feat, bs_feat, ls_feat, rf_feat
        )
        
        fair_score = explanation["fair_score"]
        
        # Store current scores
        self.current_score = fair_score
        self.component_scores = {
            "is": is_score,
            "bs": bs_score,
            "ls": ls_score,
            "rf": rf_score,
        }
        
        # Update main display widgets (inside ScrollableContainer)
        try:
            container = self.query_one("#main-container", ScrollableContainer)
            gauge = container.query_one(ScoreGaugeWidget)
            gauge.set_score(fair_score)
            
            matrix = container.query_one(ModelMatrixWidget)
            matrix.update_scores(
                is_score=is_score,
                bs_score=bs_score,
                ls_score=ls_score,
                rf_score=rf_score,
            )
        except Exception as e:
            self.notify(f"Display update: {e}", severity="warning")
        
        # Show detailed result screen
        self.push_screen(ResultScreen(
            fair_score=fair_score,
            is_score=is_score,
            bs_score=bs_score,
            ls_score=ls_score,
            rf_score=rf_score,
            explanation=explanation,
        ))
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-pdf":
            self.action_open_pdf()
        elif event.button.id == "btn-manual":
            self.action_manual_entry()
        elif event.button.id == "btn-help":
            self.action_show_help()


def main():
    """Entry point for the FairScore TUI."""
    app = FairScoreApp()
    app.run()


if __name__ == "__main__":
    main()
