"""
Model Matrix Widget for FairScore TUI.
Displays the 4 specialized models in a grid layout.
"""

from textual.app import ComposeResult
from textual.containers import Container
from textual.widgets import Static
from typing import Optional, Dict

from fairscore.theme import COLORS, PROGRESS


class ModelCell(Static):
    """Individual model cell in the matrix."""
    
    DEFAULT_CSS = """
    ModelCell {
        border: solid #003311;
        padding: 1;
        margin: 1;
        height: 6;
        width: 100%;
    }
    
    ModelCell:focus {
        border: solid #00FF41;
    }
    
    ModelCell.high-score {
        border: solid #00FF41;
    }
    
    ModelCell.medium-score {
        border: solid #FFB000;
    }
    
    ModelCell.low-score {
        border: solid #FF6B35;
    }
    """
    
    def __init__(
        self,
        code: str,
        model_name: str,
        algorithm: str,
        score: Optional[float] = None,
    ):
        super().__init__()
        self.code = code
        self.model_name = model_name
        self.algorithm = algorithm
        self._score = score
    
    def on_mount(self) -> None:
        self.update(self._get_content())
    
    def _get_content(self) -> str:
        """Generate the cell content string."""
        lines = [
            f"+{'-' * 28}+",
            f"| [{self.code}] {self.model_name:18} |",
            f"| Algorithm: {self.algorithm:15} |",
        ]
        
        if self._score is not None:
            bar_len = int(self._score * 18)
            bar = PROGRESS["filled"] * bar_len + PROGRESS["empty"] * (18 - bar_len)
            lines.append(f"| [{bar}] |")
            lines.append(f"| Score: {self._score:6.1%}             |")
        else:
            lines.append("| [..................] |")
            lines.append("| Score: ---.-%             |")
        
        lines.append(f"+{'-' * 28}+")
        return "\n".join(lines)
    
    def _update_style_classes(self) -> None:
        """Update CSS classes based on score."""
        self.remove_class("high-score", "medium-score", "low-score")
        if self._score is not None:
            if self._score >= 0.7:
                self.add_class("high-score")
            elif self._score >= 0.4:
                self.add_class("medium-score")
            else:
                self.add_class("low-score")
    
    def set_score(self, score: float):
        self._score = score
        self._update_style_classes()
        self.update(self._get_content())


class ModelMatrixWidget(Container):
    """
    2x2 grid displaying all 4 FairScore models.
    
    Layout:
    +----------------+----------------+
    | Income         | Expense        |
    | Stability (Is) | Discipline (Bs)|
    +----------------+----------------+
    | Liquidity      | Risk & Anomaly |
    | Buffer (Ls)    | Detection (Rf) |
    +----------------+----------------+
    """
    
    DEFAULT_CSS = """
    ModelMatrixWidget {
        layout: grid;
        grid-size: 2 2;
        grid-gutter: 1;
        padding: 1;
        height: auto;
        min-height: 14;
        background: #0A0A0A;
    }
    
    ModelMatrixWidget > .title {
        column-span: 2;
        color: #00FF41;
        text-style: bold;
        text-align: center;
        height: 1;
    }
    """
    
    def __init__(self):
        super().__init__()
        self.cells: Dict[str, ModelCell] = {}
    
    def compose(self) -> ComposeResult:
        # Create model cells
        self.cells["Is"] = ModelCell(
            code="Is",
            model_name="Income Stability",
            algorithm="LightGBM",
        )
        self.cells["Bs"] = ModelCell(
            code="Bs",
            model_name="Expense Discipline",
            algorithm="Random Forest",
        )
        self.cells["Ls"] = ModelCell(
            code="Ls",
            model_name="Liquidity Buffer",
            algorithm="Logistic Reg",
        )
        self.cells["Rf"] = ModelCell(
            code="Rf",
            model_name="Risk & Anomaly",
            algorithm="Isolation Forest",
        )
        
        yield self.cells["Is"]
        yield self.cells["Bs"]
        yield self.cells["Ls"]
        yield self.cells["Rf"]
    
    def update_scores(
        self,
        is_score: Optional[float] = None,
        bs_score: Optional[float] = None,
        ls_score: Optional[float] = None,
        rf_score: Optional[float] = None,
    ):
        """Update all model scores."""
        if is_score is not None:
            self.cells["Is"].set_score(is_score)
        if bs_score is not None:
            self.cells["Bs"].set_score(bs_score)
        if ls_score is not None:
            self.cells["Ls"].set_score(ls_score)
        if rf_score is not None:
            # Invert Rf score for display (lower risk = better)
            self.cells["Rf"].set_score(1 - rf_score)
