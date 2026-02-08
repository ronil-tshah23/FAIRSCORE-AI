"""
PDF Parser with Gemini AI for FairScore.
Extracts transaction data from bank statement PDFs using Google Generative AI.
"""

import os
import json
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
from datetime import datetime

from fairscore.config import Config


class PDFParser:
    """
    PDF Parser using Google Gemini AI.
    
    Extracts transaction data from bank statement PDFs and converts
    them into features suitable for FairScore model inference.
    """
    
    # Prompt template for Gemini
    EXTRACTION_PROMPT = '''
Analyze this bank statement and extract the following information in JSON format:

1. Account Summary:
   - account_holder_name: string
   - account_number: string (last 4 digits only)
   - statement_period_start: string (YYYY-MM-DD)
   - statement_period_end: string (YYYY-MM-DD)
   - opening_balance: number
   - closing_balance: number

2. Transaction Summary:
   - total_credits: number (sum of all credit transactions)
   - total_debits: number (sum of all debit transactions)
   - num_transactions: integer

3. Monthly Aggregates (list for each month in statement):
   - month: string (YYYY-MM)
   - income: number (total credits)
   - expenses: number (total debits)
   - balance_end: number

4. Category Breakdown (estimate percentages):
   - salary_income: percentage of credits from salary
   - other_income: percentage of credits from other sources
   - essential_expenses: percentage of debits for rent/utilities/groceries
   - discretionary_expenses: percentage of debits for shopping/entertainment
   - transfers: percentage of debits for transfers

5. Risk Indicators:
   - overdraft_count: integer (number of times balance went negative or very low)
   - large_transactions_count: integer (transactions > 3x average)
   - irregular_time_transactions: integer (transactions at unusual hours if visible)

Return ONLY valid JSON, no explanations. If a value cannot be determined, use null.
'''
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize PDF parser."""
        self.config = config or Config()
        self._client = None
        self._model = None
    
    def _init_client(self):
        """Initialize Gemini client."""
        if self._client is not None:
            return
        
        api_key = self.config.google_api_key
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY environment variable not set. "
                "Please set it with your Gemini API key."
            )
        
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self._model = genai.GenerativeModel("gemini-1.5-flash")
            self._client = genai
        except ImportError:
            raise ImportError(
                "google-generativeai package not installed. "
                "Run: pip install google-generativeai"
            )
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text content from a PDF file.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text content
        """
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError(
                "PyMuPDF not installed. Run: pip install PyMuPDF"
            )
        
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        doc = fitz.open(str(pdf_path))
        text_content = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text_content.append(page.get_text())
        
        doc.close()
        return "\n\n".join(text_content)
    
    def parse_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Parse a bank statement PDF and extract transaction data.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary with extracted transaction data
        """
        self._init_client()
        
        # Extract text from PDF
        text_content = self.extract_text_from_pdf(pdf_path)
        
        # Truncate if too long (Gemini has context limits)
        max_chars = 30000
        if len(text_content) > max_chars:
            text_content = text_content[:max_chars] + "\n[TRUNCATED]"
        
        # Create prompt
        full_prompt = f"{self.EXTRACTION_PROMPT}\n\nBank Statement Content:\n{text_content}"
        
        # Call Gemini
        try:
            response = self._model.generate_content(full_prompt)
            result_text = response.text
            
            # Parse JSON from response
            # Handle potential markdown code blocks
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0]
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0]
            
            parsed_data = json.loads(result_text.strip())
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse Gemini response as JSON: {e}")
        except Exception as e:
            raise ValueError(f"Gemini API error: {e}")
        
        return parsed_data
    
    def extract_features(self, parsed_data: Dict[str, Any]) -> Tuple[Dict, Dict, Dict, Dict]:
        """
        Convert parsed PDF data to model features.
        
        Args:
            parsed_data: Dictionary from parse_pdf
            
        Returns:
            Tuple of (is_features, bs_features, ls_features, rf_features)
        """
        # Extract relevant values with defaults
        account_summary = parsed_data.get("account_summary", {}) or parsed_data.get("1", {})
        txn_summary = parsed_data.get("transaction_summary", {}) or parsed_data.get("2", {})
        monthly_agg = parsed_data.get("monthly_aggregates", []) or parsed_data.get("3", [])
        categories = parsed_data.get("category_breakdown", {}) or parsed_data.get("4", {})
        risk_indicators = parsed_data.get("risk_indicators", {}) or parsed_data.get("5", {})
        
        # Safe getters with defaults
        def safe_get(d, key, default=0):
            if not d:
                return default
            val = d.get(key, default)
            return val if val is not None else default
        
        total_credits = safe_get(txn_summary, "total_credits", 0)
        total_debits = safe_get(txn_summary, "total_debits", 0)
        opening_balance = safe_get(account_summary, "opening_balance", 0)
        closing_balance = safe_get(account_summary, "closing_balance", 0)
        
        # Calculate derived metrics
        num_months = max(len(monthly_agg), 1) if monthly_agg else 1
        avg_monthly_income = total_credits / num_months
        avg_monthly_expense = total_debits / num_months
        
        # Calculate income variance if monthly data available
        if monthly_agg and len(monthly_agg) > 1:
            incomes = [safe_get(m, "income", 0) for m in monthly_agg]
            if incomes:
                import numpy as np
                income_std = np.std(incomes)
                income_mean = np.mean(incomes)
                income_variance = income_std / (income_mean + 1) if income_mean > 0 else 0.5
            else:
                income_variance = 0.5
        else:
            income_variance = 0.3  # Assume moderate stability
        
        # Income Stability features
        is_features = {
            "is_income_regularity": 1 / (1 + income_variance),
            "is_income_variance": min(1, income_variance),
            "is_income_frequency": min(1, num_months / 12),
            "is_salary_consistency": safe_get(categories, "salary_income", 50) / 100,
            "is_income_growth": 0.5,  # Would need historical data
        }
        
        # Expense Discipline features
        essential = safe_get(categories, "essential_expenses", 50) / 100
        discretionary = safe_get(categories, "discretionary_expenses", 30) / 100
        
        expense_to_income = avg_monthly_expense / (avg_monthly_income + 1) if avg_monthly_income > 0 else 1
        
        bs_features = {
            "bs_expense_variance": max(0, 1 - min(1, expense_to_income)),
            "bs_essential_ratio": essential,
            "bs_discretionary_ratio": discretionary,
            "bs_recurring_rate": essential * 0.8,
            "bs_budget_adherence": max(0, 1 - abs(expense_to_income - 0.7)),
        }
        
        # Liquidity Buffer features
        avg_balance = (opening_balance + closing_balance) / 2
        savings = total_credits - total_debits
        savings_ratio = savings / (total_credits + 1) if total_credits > 0 else 0
        
        ls_features = {
            "ls_savings_ratio": max(0, min(1, savings_ratio)),
            "ls_avg_balance_ratio": min(1, avg_balance / (avg_monthly_expense * 3 + 1)),
            "ls_min_balance_rate": 1.0 if closing_balance > 1000 else 0.5,
            "ls_emergency_fund": min(1, avg_balance / (avg_monthly_expense * 3 + 1)),
            "ls_balance_volatility": 0.7,  # Assume moderate without detailed data
        }
        
        # Risk & Anomaly features
        overdraft_count = safe_get(risk_indicators, "overdraft_count", 0)
        large_txn_count = safe_get(risk_indicators, "large_transactions_count", 0)
        irregular_count = safe_get(risk_indicators, "irregular_time_transactions", 0)
        
        rf_features = {
            "rf_overdraft_count": min(1, overdraft_count / 5),
            "rf_large_txn_ratio": min(1, large_txn_count / 10),
            "rf_irregular_time": min(1, irregular_count / 10),
            "rf_category_entropy": 0.5,  # Would need full transaction list
            "rf_spike_score": min(1, large_txn_count / 5),
        }
        
        return is_features, bs_features, ls_features, rf_features
    
    def process_pdf(self, pdf_path: str) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        """
        Complete pipeline: Parse PDF and extract features.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Tuple of (is_features, bs_features, ls_features, rf_features, raw_data)
        """
        parsed_data = self.parse_pdf(pdf_path)
        is_feat, bs_feat, ls_feat, rf_feat = self.extract_features(parsed_data)
        return is_feat, bs_feat, ls_feat, rf_feat, parsed_data
