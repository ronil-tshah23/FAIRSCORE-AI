# Contributing to FairScore

Thank you for your interest in contributing to FairScore! This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)

## Code of Conduct

This project adheres to a [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to the maintainers.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/fairscore.git
   cd fairscore
   ```
3. **Set up the development environment** (see [Development Setup](#development-setup))
4. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## How to Contribute

### Reporting Bugs

- Check if the bug has already been reported in [Issues](https://github.com/fairscore/fairscore/issues)
- If not, create a new issue with:
  - A clear, descriptive title
  - Steps to reproduce the behavior
  - Expected vs actual behavior
  - Your environment (OS, Python version, etc.)

### Suggesting Enhancements

- Open an issue with the `enhancement` label
- Clearly describe the feature and its use case
- Explain why this would be useful to most users

### Submitting Code

1. Ensure your changes align with the project's goals
2. Write tests for new functionality
3. Update documentation as needed
4. Follow the [Coding Standards](#coding-standards)
5. Submit a pull request

## Development Setup

### Prerequisites

- Python 3.10 or higher
- pip package manager

### Installation

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install in development mode with dev dependencies
pip install -e ".[dev]"
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=fairscore --cov-report=html
```

### Training Models

```bash
# Train all models (generates synthetic data first)
python -m fairscore.models.trainer
```

## Pull Request Process

1. **Update the README.md** with details of changes if applicable
2. **Ensure tests pass** and add new tests for new functionality
3. **Update documentation** for any API changes
4. **Keep commits clean** with meaningful commit messages
5. **Request review** from at least one maintainer

### Commit Message Format

Use clear, descriptive commit messages:

```
type(scope): brief description

Detailed explanation if needed.
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

Examples:
- `feat(models): add new credit factor to ensemble`
- `fix(ui): resolve score gauge rendering issue`
- `docs(readme): update installation instructions`

## Coding Standards

### Python Style

- Follow [PEP 8](https://pep8.org/) style guidelines
- Use meaningful variable and function names
- Maximum line length: 100 characters
- Use type hints where appropriate

### Documentation

- Write docstrings for all public functions and classes
- Use Google-style docstrings
- Keep inline comments minimal but meaningful

### Testing

- Write unit tests for new functionality
- Maintain test coverage above 80%
- Use descriptive test names that explain what is being tested

## Project Structure

When adding new features, follow the existing project structure:

```
fairscore/
├── models/         # ML models and training
├── data/           # Data generation and processing
├── explainability/ # SHAP explanations
├── parser/         # PDF parsing
└── ui/             # TUI components
```

## Questions?

Feel free to open an issue for any questions or concerns. We're here to help!

---

Thank you for contributing to FairScore! 🙏
