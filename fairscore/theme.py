"""
Bloomberg Terminal-style theme for FairScore TUI.
Implements phosphor green CRT aesthetic with professional banking look.
"""

# Color Palette - Bloomberg Terminal / IBM 3270 Inspired
COLORS = {
    "background": "#0A0A0A",        # Soft matte black (CRT off-state)
    "primary": "#00FF41",           # Phosphor green (main text/UI)
    "secondary": "#00CC33",         # Dimmer green for less important text
    "accent": "#FFB000",            # Amber for warnings/alerts
    "dimmed": "#003311",            # Very dim green for borders/grids
    "white": "#E0E0E0",             # System white for labels
    "error": "#FF6B35",             # Error state (amber-red)
    "success": "#00FF41",           # Same as primary for success
    "muted": "#505050",             # Muted gray for disabled items
}

# Score category colors
SCORE_COLORS = {
    "poor": "#FF6B35",              # 300-499
    "fair": "#FFB000",              # 500-649
    "good": "#00CC33",              # 650-749
    "excellent": "#00FF41",         # 750-900
}

# ASCII Box Drawing Characters
BOX = {
    "top_left": "+",
    "top_right": "+",
    "bottom_left": "+",
    "bottom_right": "+",
    "horizontal": "-",
    "vertical": "|",
    "cross": "+",
    "t_down": "+",
    "t_up": "+",
    "t_left": "+",
    "t_right": "+",
}

# Progress bar characters
PROGRESS = {
    "filled": "#",
    "empty": ".",
    "left_cap": "[",
    "right_cap": "]",
}

# Textual CSS Theme
TCSS_THEME = '''
/* FairScore Bloomberg Terminal Theme */

Screen {
    background: #0A0A0A;
}

/* Status Bar */
#status-bar {
    dock: top;
    height: 1;
    background: #003311;
    color: #00FF41;
    text-style: bold;
}

/* Main Content Area */
#main-content {
    background: #0A0A0A;
    padding: 1;
}

/* Model Matrix Grid */
.model-panel {
    border: solid #003311;
    background: #0A0A0A;
    padding: 1;
    margin: 1;
}

.model-panel-header {
    color: #00FF41;
    text-style: bold;
    text-align: center;
}

.model-panel-value {
    color: #00CC33;
    text-align: center;
}

/* Score Gauge */
#score-gauge {
    height: 3;
    background: #0A0A0A;
    color: #00FF41;
    text-align: center;
    padding: 1;
}

/* Command Line */
#command-line {
    dock: bottom;
    height: 3;
    background: #003311;
    border-top: solid #00FF41;
    padding: 0 1;
}

#command-input {
    background: #0A0A0A;
    color: #00FF41;
    border: none;
}

/* Buttons */
Button {
    background: #003311;
    color: #00FF41;
    border: solid #00FF41;
    margin: 1;
}

Button:hover {
    background: #00FF41;
    color: #0A0A0A;
}

Button:focus {
    border: double #00FF41;
}

/* Input Fields */
Input {
    background: #0A0A0A;
    color: #00FF41;
    border: solid #003311;
}

Input:focus {
    border: solid #00FF41;
}

/* Labels */
Label {
    color: #E0E0E0;
}

.label-green {
    color: #00FF41;
}

.label-amber {
    color: #FFB000;
}

/* Scrollable Containers */
ScrollableContainer {
    background: #0A0A0A;
    scrollbar-color: #00FF41;
    scrollbar-color-hover: #00CC33;
    scrollbar-background: #003311;
}

/* Data Tables */
DataTable {
    background: #0A0A0A;
}

DataTable > .datatable--header {
    background: #003311;
    color: #00FF41;
    text-style: bold;
}

DataTable > .datatable--cursor {
    background: #00FF41;
    color: #0A0A0A;
}

/* Alerts and Warnings */
.alert-warning {
    background: #1A1000;
    border: solid #FFB000;
    color: #FFB000;
    padding: 1;
}

.alert-error {
    background: #1A0500;
    border: solid #FF6B35;
    color: #FF6B35;
    padding: 1;
}

.alert-success {
    background: #001A00;
    border: solid #00FF41;
    color: #00FF41;
    padding: 1;
}

/* Help Panel */
#help-panel {
    layer: overlay;
    background: #0A0A0A 90%;
    border: solid #00FF41;
    padding: 2;
    margin: 4;
}

/* Loading Indicator */
LoadingIndicator {
    color: #00FF41;
}

/* Tabs */
Tabs {
    background: #003311;
}

Tab {
    background: #0A0A0A;
    color: #00CC33;
}

Tab.-active {
    background: #003311;
    color: #00FF41;
    text-style: bold;
}

/* Footer */
Footer {
    background: #003311;
    color: #00FF41;
}

/* Static text styling */
Static {
    color: #00FF41;
}

/* Header styling */
Header {
    background: #003311;
    color: #00FF41;
}
'''

# Alias for app.py compatibility
FAIRSCORE_CSS = TCSS_THEME


def get_score_color(score: int) -> str:
    """Get the appropriate color for a given FAIR-SCORE value."""
    if score < 500:
        return SCORE_COLORS["poor"]
    elif score < 650:
        return SCORE_COLORS["fair"]
    elif score < 750:
        return SCORE_COLORS["good"]
    else:
        return SCORE_COLORS["excellent"]


def get_score_category(score: int) -> str:
    """Get the category label for a given FAIR-SCORE value."""
    if score < 500:
        return "POOR"
    elif score < 650:
        return "FAIR"
    elif score < 750:
        return "GOOD"
    else:
        return "EXCELLENT"


def create_progress_bar(value: float, width: int = 20, show_percent: bool = True) -> str:
    """
    Create an ASCII progress bar.
    
    Args:
        value: Float between 0 and 1
        width: Total width of the bar (excluding caps)
        show_percent: Whether to show percentage at the end
    
    Returns:
        ASCII progress bar string
    """
    value = max(0.0, min(1.0, value))
    filled = int(value * width)
    empty = width - filled
    
    bar = (
        PROGRESS["left_cap"] +
        PROGRESS["filled"] * filled +
        PROGRESS["empty"] * empty +
        PROGRESS["right_cap"]
    )
    
    if show_percent:
        bar += f" {int(value * 100):3d}%"
    
    return bar


def create_score_gauge(score: int, score_min: int = 300, score_max: int = 900, width: int = 30) -> str:
    """
    Create an ASCII gauge visualization for FAIR-SCORE.
    
    Args:
        score: The FAIR-SCORE value
        score_min: Minimum possible score
        score_max: Maximum possible score
        width: Width of the gauge bar
    
    Returns:
        Multi-line string with gauge visualization
    """
    normalized = (score - score_min) / (score_max - score_min)
    normalized = max(0.0, min(1.0, normalized))
    
    filled = int(normalized * width)
    empty = width - filled
    
    bar = (
        PROGRESS["left_cap"] +
        PROGRESS["filled"] * filled +
        PROGRESS["empty"] * empty +
        PROGRESS["right_cap"]
    )
    
    category = get_score_category(score)
    
    lines = [
        f"FAIR-SCORE: {score}/{score_max}",
        bar,
        f"Category: {category}",
    ]
    
    return "\n".join(lines)
