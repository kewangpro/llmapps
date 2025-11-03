"""
Pages module - All application pages
"""

from src.ui.pages.analysis import StockAnalysisApp, create_app
from src.ui.pages.dashboard import DashboardPage
from src.ui.pages.trading import CompactRLPanel, create_compact_rl_panel
from src.ui.pages.models import ModelsPage
from src.ui.pages.portfolio import PortfolioPage

__all__ = [
    'StockAnalysisApp',
    'create_app',
    'DashboardPage',
    'CompactRLPanel',
    'create_compact_rl_panel',
    'ModelsPage',
    'PortfolioPage',
]
