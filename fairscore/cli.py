"""
CLI entry point for FairScore application.
Launches the TUI when 'fairscore' command is executed.
"""

import sys
from rich.console import Console

console = Console()


def main():
    """Main entry point for the FairScore CLI."""
    try:
        # Import here to avoid circular imports and speed up CLI startup
        from fairscore.app import FairScoreApp
        
        app = FairScoreApp()
        app.run()
        
    except KeyboardInterrupt:
        console.print("\n[dim]FairScore terminated by user.[/dim]")
        sys.exit(0)
    except ImportError as e:
        console.print(f"[red]Error:[/red] Missing dependency - {e}")
        console.print("[dim]Try running: pip install -e .[/dim]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


def train():
    """CLI command to train all models."""
    console.print("[green]Starting model training...[/green]")
    try:
        from fairscore.models.trainer import ModelTrainer
        from fairscore.config import Config
        
        trainer = ModelTrainer(Config())
        trainer.train_all()
        console.print("[green]Training complete![/green]")
    except Exception as e:
        console.print(f"[red]Training failed:[/red] {e}")
        sys.exit(1)


def generate_data():
    """CLI command to generate synthetic training data."""
    console.print("[green]Generating synthetic data...[/green]")
    try:
        from fairscore.data.generator import DataGenerator
        from fairscore.config import Config
        
        generator = DataGenerator(Config())
        generator.generate_all()
        console.print("[green]Data generation complete![/green]")
    except Exception as e:
        console.print(f"[red]Data generation failed:[/red] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
