"""
Input Mode Selection and Manual Entry Form for FairScore TUI.
Allows users to either upload PDF or enter financial data manually.
"""

from textual.app import ComposeResult
from textual.containers import Container, Vertical, Horizontal, ScrollableContainer
from textual.widgets import Static, Button, Input, Label, Select, RadioSet, RadioButton
from textual.screen import Screen
from textual.binding import Binding
from typing import Optional, Dict, Any, Callable


class InputModeSelector(Static):
    """Widget for selecting input mode (PDF or Manual)."""
    
    DEFAULT_CSS = """
    InputModeSelector {
        height: auto;
        padding: 2;
        border: solid #003311;
        margin: 1;
    }
    
    InputModeSelector .title {
        color: #00FF41;
        text-style: bold;
        text-align: center;
        margin-bottom: 1;
    }
    
    InputModeSelector Button {
        margin: 1;
        min-width: 30;
    }
    """
    
    def compose(self) -> ComposeResult:
        yield Static("SELECT INPUT MODE", classes="title")
        yield Static("-" * 40)
        yield Static("")
        with Horizontal():
            yield Button("[Ctrl+O] Upload PDF", id="btn-pdf", variant="primary")
            yield Button("[Ctrl+M] Manual Entry", id="btn-manual", variant="default")
        yield Static("")
        yield Static("PDF: Parse bank statement with AI")
        yield Static("Manual: Enter financial data directly")


class ManualEntryForm(ScrollableContainer):
    """
    Scrollable form for manual financial data entry.
    
    Fields match the features expected by the scoring models.
    """
    
    DEFAULT_CSS = """
    ManualEntryForm {
        height: 100%;
        padding: 1;
        background: #0A0A0A;
    }
    
    ManualEntryForm .form-group {
        height: auto;
        margin-bottom: 1;
        padding: 1;
        border: solid #003311;
    }
    
    ManualEntryForm .form-title {
        color: #00FF41;
        text-style: bold;
        margin-bottom: 1;
    }
    
    ManualEntryForm .field-label {
        color: #E0E0E0;
        margin-bottom: 0;
    }
    
    ManualEntryForm Input {
        margin-bottom: 1;
        background: #0A0A0A;
        color: #00FF41;
        border: solid #003311;
    }
    
    ManualEntryForm Input:focus {
        border: solid #00FF41;
    }
    
    ManualEntryForm Select {
        margin-bottom: 1;
        background: #0A0A0A;
        color: #00FF41;
    }
    
    ManualEntryForm Button {
        margin-top: 1;
        width: 100%;
    }
    
    ManualEntryForm .submit-btn {
        background: #003311;
        color: #00FF41;
        border: solid #00FF41;
    }
    """
    
    def __init__(self, on_submit: Optional[Callable] = None):
        super().__init__()
        self.on_submit_callback = on_submit
        self.form_data: Dict[str, Any] = {}
    
    def compose(self) -> ComposeResult:
        # Title
        yield Static("MANUAL FINANCIAL DATA ENTRY", classes="form-title")
        yield Static("=" * 50)
        yield Static("")
        
        # Income Section
        with Vertical(classes="form-group"):
            yield Static("[Is] INCOME INFORMATION", classes="form-title")
            
            yield Label("Monthly Income (INR):", classes="field-label")
            yield Input(placeholder="e.g., 50000", id="monthly_income", type="number")
            
            yield Label("Income Frequency:", classes="field-label")
            yield Select(
                [(text, value) for value, text in [
                    ("monthly", "Monthly (Salaried)"),
                    ("weekly", "Weekly"),
                    ("irregular", "Irregular (Freelance/Gig)"),
                ]],
                id="income_frequency",
                value="monthly",
            )
        
        # Expense Section
        with Vertical(classes="form-group"):
            yield Static("[Bs] EXPENSE INFORMATION", classes="form-title")
            
            yield Label("Monthly Expenses (INR):", classes="field-label")
            yield Input(placeholder="e.g., 35000", id="monthly_expenses", type="number")
            
            yield Label("Essential Expenses Ratio (%):", classes="field-label")
            yield Input(placeholder="e.g., 60 (for 60%)", id="essential_ratio", type="number")
            yield Static("(Rent, utilities, groceries, EMI)", classes="field-label")
        
        # Savings Section
        with Vertical(classes="form-group"):
            yield Static("[Ls] SAVINGS & LIQUIDITY", classes="form-title")
            
            yield Label("Average Account Balance (INR):", classes="field-label")
            yield Input(placeholder="e.g., 25000", id="avg_balance", type="number")
            
            yield Label("Monthly Savings (INR):", classes="field-label")
            yield Input(placeholder="e.g., 10000", id="monthly_savings", type="number")
        
        # Risk Section
        with Vertical(classes="form-group"):
            yield Static("[Rf] RISK INDICATORS", classes="form-title")
            
            yield Label("Overdrafts (last 6 months):", classes="field-label")
            yield Input(placeholder="e.g., 0", id="overdraft_count", type="number", value="0")
            
            yield Label("Large/Irregular Transactions:", classes="field-label")
            yield Input(placeholder="e.g., 2", id="irregular_txn_count", type="number", value="0")
            yield Static("(Transactions > 3x your average)", classes="field-label")
        
        yield Static("")
        yield Button("CALCULATE FAIR-SCORE", id="btn-submit", classes="submit-btn")
        yield Static("")
        yield Static("Press [Tab] to navigate, [Enter] to submit")
    
    def get_form_data(self) -> Dict[str, Any]:
        """Extract and validate form data."""
        def get_float(input_id: str, default: float = 0) -> float:
            try:
                widget = self.query_one(f"#{input_id}", Input)
                return float(widget.value) if widget.value else default
            except Exception:
                return default
        
        def get_str(select_id: str, default: str = "") -> str:
            try:
                widget = self.query_one(f"#{select_id}", Select)
                return str(widget.value) if widget.value else default
            except Exception:
                return default
        
        return {
            "monthly_income": get_float("monthly_income", 50000),
            "income_frequency": get_str("income_frequency", "monthly"),
            "monthly_expenses": get_float("monthly_expenses", 35000),
            "essential_ratio": get_float("essential_ratio", 60) / 100,
            "avg_balance": get_float("avg_balance", 20000),
            "monthly_savings": get_float("monthly_savings", 10000),
            "overdraft_count": int(get_float("overdraft_count", 0)),
            "irregular_txn_count": int(get_float("irregular_txn_count", 0)),
        }
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-submit":
            self.form_data = self.get_form_data()
            if self.on_submit_callback:
                self.on_submit_callback(self.form_data)
            else:
                self.post_message(FormSubmitted(self.form_data))


class FormSubmitted:
    """Message sent when form is submitted."""
    def __init__(self, data: Dict[str, Any]):
        self.data = data


class InputModeScreen(Screen):
    """
    Screen for selecting input mode and entering data.
    Can switch between PDF upload and manual entry.
    """
    
    BINDINGS = [
        Binding("ctrl+o", "open_pdf", "Open PDF"),
        Binding("ctrl+m", "manual_entry", "Manual Entry"),
        Binding("escape", "go_back", "Back"),
    ]
    
    DEFAULT_CSS = """
    InputModeScreen {
        background: #0A0A0A;
    }
    
    InputModeScreen #mode-container {
        height: 100%;
        padding: 2;
    }
    
    InputModeScreen .screen-title {
        text-align: center;
        color: #00FF41;
        text-style: bold;
        margin-bottom: 2;
    }
    """
    
    def __init__(self, on_pdf_selected: Optional[Callable] = None,
                 on_manual_submit: Optional[Callable] = None):
        super().__init__()
        self.on_pdf_selected = on_pdf_selected
        self.on_manual_submit = on_manual_submit
        self.current_mode = "selector"  # "selector", "manual"
    
    def compose(self) -> ComposeResult:
        with ScrollableContainer(id="mode-container"):
            yield Static("FAIRSCORE - DATA INPUT", classes="screen-title")
            yield Static("=" * 50)
            yield InputModeSelector()
            yield ManualEntryForm(on_submit=self._handle_manual_submit)
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-pdf":
            self.action_open_pdf()
        elif event.button.id == "btn-manual":
            self.action_manual_entry()
    
    def action_open_pdf(self) -> None:
        """Handle PDF upload action."""
        if self.on_pdf_selected:
            self.on_pdf_selected()
        self.dismiss("pdf")
    
    def action_manual_entry(self) -> None:
        """Switch to manual entry mode."""
        self.current_mode = "manual"
        # Scroll to form
        form = self.query_one(ManualEntryForm)
        form.scroll_visible()
    
    def action_go_back(self) -> None:
        """Go back/dismiss screen."""
        self.dismiss(None)
    
    def _handle_manual_submit(self, data: Dict[str, Any]) -> None:
        """Handle manual form submission."""
        if self.on_manual_submit:
            self.on_manual_submit(data)
        self.dismiss(("manual", data))
