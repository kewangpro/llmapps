"""
Portfolio Manager

Handles creation, loading, and saving of stock portfolios.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional

from src.config import Config

logger = logging.getLogger(__name__)

class PortfolioManager:
    """Manages stock portfolios."""

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Config.DATA_DIR / "portfolios"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.default_portfolio_name = "default"
        logger.info(f"Portfolio storage initialized at: {self.storage_path}")

    def get_portfolio_file(self, portfolio_name: str) -> Path:
        return self.storage_path / f"{portfolio_name}.json"

    def save_portfolio(self, portfolio_name: str, symbols: List[str]):
        """Saves a portfolio to a JSON file."""
        portfolio_file = self.get_portfolio_file(portfolio_name)
        try:
            with open(portfolio_file, "w") as f:
                json.dump(symbols, f, indent=4)
            logger.info(f"Successfully saved portfolio: {portfolio_name}")
            logger.debug(f"Saved symbols for {portfolio_name}: {symbols}")
        except Exception as e:
            logger.error(f"Failed to save portfolio {portfolio_name}: {e}")

    def load_portfolio(self, portfolio_name: str) -> List[str]:
        """Loads a portfolio from a JSON file."""
        portfolio_file = self.get_portfolio_file(portfolio_name)
        if not portfolio_file.exists():
            logger.warning(f"Portfolio file not found: {portfolio_name}")
            return []

        try:
            with open(portfolio_file, "r") as f:
                symbols = json.load(f)
            logger.debug(f"Loaded symbols for {portfolio_name}: {symbols}")
            return symbols
        except Exception as e:
            logger.error(f"Failed to load portfolio {portfolio_name}: {e}")
            return []
    
    def list_portfolios(self) -> List[str]:
        """Lists all saved portfolio names."""
        portfolio_files = self.storage_path.glob("*.json")
        return sorted([f.stem for f in portfolio_files])

# Singleton instance
portfolio_manager = PortfolioManager()
