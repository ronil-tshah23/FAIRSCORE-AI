"""
Synthetic Data Generator for FairScore.
Generates 100,000 realistic financial profiles with transactions
covering various edge cases for model training.
"""

import numpy as np
import pandas as pd
from faker import Faker
from datetime import datetime, timedelta
from typing import Tuple, Optional
from pathlib import Path
import gc

from fairscore.config import Config


class DataGenerator:
    """
    Generates synthetic financial data for FairScore model training.
    
    Creates realistic user profiles and transaction histories that cover:
    - Regular salaried employees
    - Freelancers with irregular income
    - Gig workers
    - Students with stipends
    - Self-employed individuals
    
    Memory-efficient batch processing for 8GB RAM systems.
    """
    
    # Occupation types and their characteristics
    OCCUPATION_PROFILES = {
        "salaried": {
            "income_range": (25000, 200000),
            "income_frequency": "monthly",
            "income_variance": 0.05,
            "weight": 0.40,
        },
        "self_employed": {
            "income_range": (20000, 500000),
            "income_frequency": "irregular",
            "income_variance": 0.30,
            "weight": 0.15,
        },
        "freelancer": {
            "income_range": (15000, 300000),
            "income_frequency": "irregular",
            "income_variance": 0.40,
            "weight": 0.15,
        },
        "gig_worker": {
            "income_range": (10000, 80000),
            "income_frequency": "weekly",
            "income_variance": 0.35,
            "weight": 0.15,
        },
        "student": {
            "income_range": (5000, 30000),
            "income_frequency": "monthly",
            "income_variance": 0.10,
            "weight": 0.15,
        },
    }
    
    # Transaction categories with typical amounts
    TRANSACTION_CATEGORIES = {
        "salary": {"type": "credit", "amount_range": (10000, 500000), "frequency": "monthly"},
        "freelance_payment": {"type": "credit", "amount_range": (5000, 100000), "frequency": "irregular"},
        "rent": {"type": "debit", "amount_range": (5000, 50000), "frequency": "monthly"},
        "utilities": {"type": "debit", "amount_range": (500, 5000), "frequency": "monthly"},
        "groceries": {"type": "debit", "amount_range": (500, 10000), "frequency": "weekly"},
        "shopping": {"type": "debit", "amount_range": (500, 50000), "frequency": "irregular"},
        "emi": {"type": "debit", "amount_range": (2000, 50000), "frequency": "monthly"},
        "entertainment": {"type": "debit", "amount_range": (100, 5000), "frequency": "irregular"},
        "transfer_in": {"type": "credit", "amount_range": (100, 100000), "frequency": "irregular"},
        "transfer_out": {"type": "debit", "amount_range": (100, 100000), "frequency": "irregular"},
        "upi_received": {"type": "credit", "amount_range": (10, 50000), "frequency": "irregular"},
        "upi_sent": {"type": "debit", "amount_range": (10, 50000), "frequency": "irregular"},
        "investment": {"type": "debit", "amount_range": (1000, 50000), "frequency": "monthly"},
        "refund": {"type": "credit", "amount_range": (100, 10000), "frequency": "irregular"},
        "medical": {"type": "debit", "amount_range": (500, 50000), "frequency": "irregular"},
        "education": {"type": "debit", "amount_range": (1000, 100000), "frequency": "irregular"},
    }
    
    CHANNELS = ["bank", "upi", "wallet"]
    CITY_TIERS = ["tier1", "tier2", "tier3"]
    GENDERS = ["male", "female", "other"]
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize the data generator."""
        self.config = config or Config()
        self.fake = Faker("en_IN")  # Indian locale for realistic names
        Faker.seed(self.config.data.random_seed)
        np.random.seed(self.config.data.random_seed)
    
    def generate_all(self, save: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate complete synthetic dataset.
        
        Args:
            save: Whether to save datasets to disk
            
        Returns:
            Tuple of (users_df, transactions_df)
        """
        print(f"Generating {self.config.data.num_users:,} synthetic user profiles...")
        
        users_df = self._generate_users()
        print(f"Generated {len(users_df):,} users")
        
        print("Generating transactions (this may take a while)...")
        transactions_df = self._generate_transactions(users_df)
        print(f"Generated {len(transactions_df):,} transactions")
        
        if save:
            self._save_datasets(users_df, transactions_df)
        
        return users_df, transactions_df
    
    def _generate_users(self) -> pd.DataFrame:
        """Generate synthetic user profiles."""
        users = []
        
        # Calculate number of users per occupation type
        occupation_counts = {
            occ: int(self.config.data.num_users * profile["weight"])
            for occ, profile in self.OCCUPATION_PROFILES.items()
        }
        
        # Adjust for rounding errors
        total = sum(occupation_counts.values())
        if total < self.config.data.num_users:
            occupation_counts["salaried"] += self.config.data.num_users - total
        
        user_id = 0
        for occupation, count in occupation_counts.items():
            profile = self.OCCUPATION_PROFILES[occupation]
            
            for _ in range(count):
                user_id += 1
                
                # Age distribution based on occupation
                if occupation == "student":
                    age = np.random.randint(18, 28)
                elif occupation == "gig_worker":
                    age = np.random.randint(20, 45)
                else:
                    age = np.random.randint(22, 65)
                
                # Income based on occupation and some variation
                base_income = np.random.uniform(*profile["income_range"])
                
                # City tier affects income
                city_tier = np.random.choice(
                    self.CITY_TIERS,
                    p=[0.3, 0.4, 0.3]
                )
                tier_multiplier = {"tier1": 1.3, "tier2": 1.0, "tier3": 0.7}[city_tier]
                annual_income = base_income * 12 * tier_multiplier
                
                users.append({
                    "user_id": f"USR{user_id:08d}",
                    "age": age,
                    "gender": np.random.choice(self.GENDERS, p=[0.48, 0.48, 0.04]),
                    "occupation_type": occupation,
                    "annual_income": round(annual_income, 2),
                    "monthly_income_avg": round(annual_income / 12, 2),
                    "account_tenure_months": np.random.randint(1, 120),
                    "city_tier": city_tier,
                    "has_emi": np.random.random() < 0.35,
                    "has_investments": np.random.random() < 0.40,
                    "income_variance": profile["income_variance"],
                    "income_frequency": profile["income_frequency"],
                })
        
        return pd.DataFrame(users)
    
    def _generate_transactions(self, users_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate synthetic transactions for all users.
        Uses batch processing for memory efficiency.
        """
        all_transactions = []
        batch_size = self.config.data.batch_size
        
        total_users = len(users_df)
        for batch_start in range(0, total_users, batch_size):
            batch_end = min(batch_start + batch_size, total_users)
            batch_users = users_df.iloc[batch_start:batch_end]
            
            batch_transactions = self._generate_transactions_batch(batch_users)
            all_transactions.append(batch_transactions)
            
            # Progress update
            progress = (batch_end / total_users) * 100
            print(f"  Progress: {progress:.1f}% ({batch_end:,}/{total_users:,} users)")
            
            # Garbage collection to free memory
            gc.collect()
        
        return pd.concat(all_transactions, ignore_index=True)
    
    def _generate_transactions_batch(self, users_df: pd.DataFrame) -> pd.DataFrame:
        """Generate transactions for a batch of users."""
        transactions = []
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)  # 1 year of data
        
        for _, user in users_df.iterrows():
            num_transactions = np.random.randint(
                *self.config.data.num_transactions_per_user
            )
            
            user_transactions = self._generate_user_transactions(
                user, num_transactions, start_date, end_date
            )
            transactions.extend(user_transactions)
        
        return pd.DataFrame(transactions)
    
    def _generate_user_transactions(
        self,
        user: pd.Series,
        num_transactions: int,
        start_date: datetime,
        end_date: datetime
    ) -> list:
        """Generate transactions for a single user."""
        transactions = []
        
        monthly_income = user["monthly_income_avg"]
        occupation = user["occupation_type"]
        profile = self.OCCUPATION_PROFILES[occupation]
        
        # Determine spending patterns based on occupation
        monthly_expense_ratio = np.random.uniform(0.5, 0.95)
        monthly_expense = monthly_income * monthly_expense_ratio
        
        # Generate random dates
        date_range = (end_date - start_date).days
        random_dates = [
            start_date + timedelta(days=np.random.randint(0, date_range))
            for _ in range(num_transactions)
        ]
        random_dates.sort()
        
        transaction_id = 0
        running_balance = monthly_income * 2  # Starting balance
        
        for txn_date in random_dates:
            transaction_id += 1
            
            # Determine transaction type based on user profile
            category, amount, txn_type = self._determine_transaction(
                user, txn_date, monthly_income, monthly_expense, occupation
            )
            
            # Update running balance
            if txn_type == "credit":
                running_balance += amount
            else:
                running_balance -= amount
                # Handle overdraft edge case
                if running_balance < 0:
                    running_balance = np.random.uniform(100, 1000)
            
            # Select channel
            channel = self._select_channel(category)
            
            transactions.append({
                "transaction_id": f"{user['user_id']}_TXN{transaction_id:06d}",
                "user_id": user["user_id"],
                "timestamp": txn_date.strftime("%Y-%m-%d %H:%M:%S"),
                "amount": round(amount, 2),
                "type": txn_type,
                "category": category,
                "channel": channel,
                "balance_after": round(max(0, running_balance), 2),
                "is_recurring": category in ["salary", "rent", "utilities", "emi"],
            })
        
        # Add edge cases
        if np.random.random() < 0.1:  # 10% chance of anomalies
            transactions.extend(
                self._generate_anomalies(user, start_date, end_date)
            )
        
        return transactions
    
    def _determine_transaction(
        self,
        user: pd.Series,
        txn_date: datetime,
        monthly_income: float,
        monthly_expense: float,
        occupation: str
    ) -> Tuple[str, float, str]:
        """Determine transaction category, amount, and type."""
        
        # Monthly income transaction
        if txn_date.day in [1, 5, 10, 15] and np.random.random() < 0.3:
            if occupation == "salaried":
                return "salary", monthly_income * np.random.uniform(0.95, 1.05), "credit"
            elif occupation in ["freelancer", "self_employed"]:
                return "freelance_payment", monthly_income * np.random.uniform(0.5, 1.5), "credit"
            elif occupation == "gig_worker":
                return "upi_received", monthly_income / 4 * np.random.uniform(0.8, 1.2), "credit"
        
        # Regular expenses
        expense_categories = [
            ("groceries", 0.25),
            ("shopping", 0.15),
            ("utilities", 0.10),
            ("entertainment", 0.10),
            ("upi_sent", 0.15),
            ("transfer_out", 0.10),
            ("rent", 0.05 if txn_date.day in [1, 2, 3, 4, 5] else 0),
            ("emi", 0.05 if user["has_emi"] and txn_date.day in [5, 10, 15] else 0),
            ("medical", 0.03),
            ("education", 0.02),
        ]
        
        categories, weights = zip(*expense_categories)
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        category = np.random.choice(categories, p=weights)
        cat_info = self.TRANSACTION_CATEGORIES[category]
        
        # Adjust amount based on user's financial profile
        base_amount = np.random.uniform(*cat_info["amount_range"])
        amount_factor = monthly_expense / 50000  # Normalize to typical expense
        amount = base_amount * amount_factor
        amount = max(10, min(amount, monthly_income * 0.5))  # Cap at 50% of monthly income
        
        return category, amount, cat_info["type"]
    
    def _select_channel(self, category: str) -> str:
        """Select appropriate channel for transaction category."""
        if category in ["upi_received", "upi_sent"]:
            return "upi"
        elif category in ["salary", "rent", "emi"]:
            return "bank"
        else:
            return np.random.choice(self.CHANNELS, p=[0.3, 0.5, 0.2])
    
    def _generate_anomalies(
        self,
        user: pd.Series,
        start_date: datetime,
        end_date: datetime
    ) -> list:
        """Generate anomalous transactions for edge case testing."""
        anomalies = []
        monthly_income = user["monthly_income_avg"]
        
        # Large unusual transaction
        if np.random.random() < 0.5:
            anomalies.append({
                "transaction_id": f"{user['user_id']}_ANOM001",
                "user_id": user["user_id"],
                "timestamp": (start_date + timedelta(days=np.random.randint(0, 365))).strftime("%Y-%m-%d %H:%M:%S"),
                "amount": round(monthly_income * np.random.uniform(2, 5), 2),
                "type": "debit",
                "category": "shopping",
                "channel": "bank",
                "balance_after": round(np.random.uniform(1000, 10000), 2),
                "is_recurring": False,
            })
        
        # Late night transactions (potential fraud indicator)
        if np.random.random() < 0.3:
            late_date = start_date + timedelta(days=np.random.randint(0, 365))
            late_date = late_date.replace(hour=np.random.randint(23, 24), minute=np.random.randint(0, 59))
            anomalies.append({
                "transaction_id": f"{user['user_id']}_ANOM002",
                "user_id": user["user_id"],
                "timestamp": late_date.strftime("%Y-%m-%d %H:%M:%S"),
                "amount": round(np.random.uniform(1000, 20000), 2),
                "type": "debit",
                "category": "transfer_out",
                "channel": "upi",
                "balance_after": round(np.random.uniform(500, 5000), 2),
                "is_recurring": False,
            })
        
        return anomalies
    
    def _save_datasets(self, users_df: pd.DataFrame, transactions_df: pd.DataFrame):
        """Save datasets to disk."""
        users_path = self.config.data_dir / "users.csv"
        transactions_path = self.config.data_dir / "transactions.csv"
        
        users_df.to_csv(users_path, index=False)
        print(f"Saved users dataset to: {users_path}")
        
        # Save transactions in chunks to avoid memory issues
        transactions_df.to_csv(transactions_path, index=False)
        print(f"Saved transactions dataset to: {transactions_path}")
        
        # Save summary statistics
        summary = {
            "total_users": len(users_df),
            "total_transactions": len(transactions_df),
            "occupation_distribution": users_df["occupation_type"].value_counts().to_dict(),
            "avg_transactions_per_user": len(transactions_df) / len(users_df),
            "generated_at": datetime.now().isoformat(),
        }
        
        summary_path = self.config.data_dir / "data_summary.json"
        import json
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    # Run standalone for testing
    from fairscore.config import Config
    config = Config()
    generator = DataGenerator(config)
    generator.generate_all()
