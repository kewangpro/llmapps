"""
Pages module - All application pages
"""

from src.ui.pages.analysis import StockAnalysisApp
from src.ui.pages.dashboard import DashboardPage
from src.ui.pages.rl_training import RLTrainingPanel, create_rl_training_panel
from src.ui.pages.models import ModelsPage
from src.ui.pages.portfolio import PortfolioPage
from src.ui.pages.live_trading import LiveTradingPage, create_live_trading_page
from src.ui.app import create_app

__all__ = [
    'StockAnalysisApp',
    'create_app',
    'DashboardPage',
    'RLTrainingPanel',
    'create_rl_training_panel',
    'ModelsPage',
    'PortfolioPage',
    'LiveTradingPage',
    'create_live_trading_page',
]
