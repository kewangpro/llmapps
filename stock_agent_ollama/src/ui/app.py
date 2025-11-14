"""
Main application layout and configuration.

This module contains the app factory that creates and assembles the entire
Panel application including all pages, sidebar, and navigation.
"""

import panel as pn
import asyncio
import logging
from datetime import datetime

from src.tools.portfolio_manager import portfolio_manager
from src.ui.design_system import Colors

logger = logging.getLogger(__name__)


def create_app():
    """Create and configure the Panel application with professional navigation"""
    from src.ui.pages.rl_training import RLTrainingPanel
    from src.ui.pages.dashboard import DashboardPage
    from src.ui.pages.portfolio import PortfolioPage
    from src.ui.pages.models import ModelsPage
    from src.ui.pages.live_trading import create_live_trading_page
    from src.ui.pages.analysis import StockAnalysisApp
    from src.rl.session_manager import LiveSessionManager
    from src.tools.stock_fetcher import StockFetcher

    # Create session manager
    session_manager = LiveSessionManager()

    # Create watchlist panel for sidebar
    class WatchlistPanel:
        """Watchlist panel that can be refreshed"""
        def __init__(self):
            self.stock_fetcher = StockFetcher()
            self.watchlist_symbols = portfolio_manager.load_portfolio("default")
            self.pane = pn.pane.HTML("Loading watchlist...", sizing_mode="stretch_width")
            pn.state.onload(self.schedule_refresh)

        def schedule_refresh(self):
            """Schedule the refresh to run in the background."""
            pn.state.execute(self.refresh)

        async def refresh(self):
            """Refresh watchlist data"""
            self.watchlist_symbols = portfolio_manager.load_portfolio("default")

            tasks = [asyncio.to_thread(self.stock_fetcher.get_real_time_price, symbol) for symbol in self.watchlist_symbols]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            watchlist_html = f"""
            <div style='background: {Colors.BG_SECONDARY}; border: 1px solid {Colors.BORDER_SUBTLE}; border-radius: 8px; padding: 10px;'>
                <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; border-bottom: 2px solid {Colors.BORDER_SUBTLE}; padding-bottom: 8px;'>
                    <h3 style='margin: 0; font-size: 1rem; color: {Colors.TEXT_PRIMARY};'>⭐ Watchlist</h3>
                    <span style='font-size: 0.7rem; color: {Colors.TEXT_SECONDARY};'>{datetime.now().strftime('%I:%M %p')}</span>
                </div>
                <div style='max-height: 650px; overflow-y: auto;'>
            """

            for i, result in enumerate(results):
                symbol = self.watchlist_symbols[i]
                if isinstance(result, Exception):
                    logger.warning(f"Failed to fetch {symbol} for watchlist: {result}")
                    watchlist_html += f"""
                    <div style='background: {Colors.BG_PRIMARY}; border: 1px solid {Colors.BORDER_SUBTLE}; border-radius: 6px; padding: 10px; margin-bottom: 8px; opacity: 0.5;'>
                        <div style='font-weight: 600; color: {Colors.TEXT_SECONDARY}; font-size: 0.9rem;'>{symbol}</div>
                        <div style='font-size: 0.75rem; color: {Colors.TEXT_MUTED};'>Error</div>
                    </div>
                    """
                else:
                    real_time_data = result
                    price = real_time_data.get('current_price', 0) or 0
                    prev_close = real_time_data.get('previous_close', 0) or 0
                    change = price - prev_close
                    change_pct = (change / prev_close * 100) if prev_close else 0
                    color = Colors.SUCCESS_GREEN if change >= 0 else Colors.DANGER_RED
                    symbol_icon = '▲' if change >= 0 else '▼'

                    watchlist_html += f"""
                    <div style='background: {Colors.BG_PRIMARY};
                                border: 1px solid {Colors.BORDER_SUBTLE};
                                border-radius: 6px;
                                padding: 10px;
                                margin-bottom: 8px;
                                cursor: pointer;
                                transition: all 0.2s;'
                         onmouseover='this.style.background="{Colors.BG_HOVER}"'
                         onmouseout='this.style.background="{Colors.BG_PRIMARY}"'>
                        <div style='display: flex; justify-content: space-between; align-items: center;'>
                            <div style='font-weight: 600; color: {Colors.TEXT_PRIMARY}; font-size: 0.9rem;'>{symbol}</div>
                            <div style='text-align: right;'>
                                <div style='font-family: monospace; font-size: 0.85rem; color: {Colors.TEXT_PRIMARY};'>${price:.2f}</div>
                                <div style='font-size: 0.75rem; color: {color}; font-weight: 600;'>{symbol_icon} {change_pct:+.2f}%</div>
                            </div>
                        </div>
                    </div>
                    """

            watchlist_html += """
                </div>
            </div>
            """

            self.pane.object = watchlist_html

        def get_panel(self):
            """Get the panel component"""
            return self.pane

    # Create watchlist instance
    watchlist_panel = WatchlistPanel()
    watchlist_sidebar = pn.Column(
        watchlist_panel.get_panel(),
        sizing_mode="stretch_width"
    )

    # Get watchlist symbols from portfolio (for portfolio page)
    watchlist_symbols = portfolio_manager.load_portfolio("default")

    # Create all pages
    dashboard_page = DashboardPage(watchlist_panel=watchlist_panel)
    analysis_app = StockAnalysisApp()
    rl_panel = RLTrainingPanel()
    portfolio_page = PortfolioPage(watchlist_symbols=watchlist_symbols, watchlist_panel=watchlist_panel)
    models_page = ModelsPage()
    live_trading_page = create_live_trading_page(session_manager=session_manager)

    # Create professional navigation tabs
    tabs = pn.Tabs(
        ('📊 Dashboard', dashboard_page.get_view()),
        ('📈 Analysis', analysis_app.get_analysis_tab()),
        ('🤖 Training', rl_panel.get_panel()),
        ('🔴 Live Trade', live_trading_page),
        ('📋 Watchlist', portfolio_page.get_view()),
        ('🧠 Models', models_page.get_view()),
        dynamic=True,
        sizing_mode="stretch_width",
        tabs_location='above',
        active=0
    )

    # Main layout with professional styling
    layout = pn.Column(
        tabs,
        sizing_mode="stretch_width",
        max_width=1600,
        margin=(0, 0)
    )

    # Create template with light theme
    template = pn.template.FastListTemplate(
        title="Stock Agent Pro",
        sidebar=[watchlist_sidebar],
        header_background=Colors.ACCENT_PURPLE,
        theme='default',
        main_max_width='1600px',
        theme_toggle=False,
    )
    template.main.append(layout)
    # Return the template instance (do NOT call .servable() here).
    # The caller (pn.serve) should invoke `create_app` per-session so a
    # fresh document and models are created for each connection.
    return template
