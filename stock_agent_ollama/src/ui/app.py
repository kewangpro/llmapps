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
        def __init__(self, session_manager, tabs=None, analysis_app=None):
            self.stock_fetcher = StockFetcher()
            self.session_manager = session_manager
            self.tabs = tabs
            self.analysis_app = analysis_app
            self.watchlist_symbols = portfolio_manager.load_portfolio("default")
            self.pane = pn.Column(
                pn.pane.HTML("Loading watchlist...", sizing_mode="stretch_width"),
                sizing_mode="stretch_width"
            )
            pn.state.onload(self.schedule_refresh)

        def schedule_refresh(self):
            """Schedule the refresh to run in the background."""
            pn.state.execute(self.refresh)

        def handle_stock_click(self, symbol):
            """Handle click on a stock in the watchlist"""
            logger.info(f"Watchlist: Stock {symbol} clicked")
            if self.tabs and self.analysis_app:
                # Switch to Analysis tab (index 1)
                self.tabs.active = 1
                logger.info(f"Watchlist: Switched to Analysis tab, triggering analysis for {symbol}")
                # Trigger analysis for the clicked symbol
                self.analysis_app.analyze_symbol(symbol)
            else:
                logger.warning(f"Watchlist: Cannot handle click - tabs={self.tabs is not None}, analysis_app={self.analysis_app is not None}")

        def _get_positions_for_symbol(self, symbol):
            """Get aggregated position info for a symbol across all sessions"""
            total_shares = 0
            total_value = 0.0

            try:
                # Get all live trading engines
                engines = self.session_manager.get_all_sessions()

                for engine in engines:
                    session = engine.session
                    # Count positions from all sessions (running, paused, or stopped)
                    # Check if this session has a position in this symbol
                    if symbol in session.portfolio.positions:
                        position = session.portfolio.positions[symbol]
                        total_shares += position.shares
                        total_value += position.shares * position.current_price
            except Exception as e:
                logger.warning(f"Error getting positions for {symbol}: {e}")

            return total_shares, total_value

        async def refresh(self):
            """Refresh watchlist data"""
            self.watchlist_symbols = portfolio_manager.load_portfolio("default")

            tasks = [asyncio.to_thread(self.stock_fetcher.get_real_time_price, symbol) for symbol in self.watchlist_symbols]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Create header HTML
            watchlist_header = pn.pane.HTML(f"""
            <div style='display: flex; justify-content: space-between; align-items: center; border-bottom: 2px solid {Colors.BORDER_SUBTLE}; padding-bottom: 8px;'>
                <h3 style='margin: 0; font-size: 1rem; color: {Colors.TEXT_PRIMARY};'>⭐ Watchlist</h3>
                <span style='font-size: 0.7rem; color: {Colors.TEXT_SECONDARY};'>{datetime.now().strftime('%I:%M %p')}</span>
            </div>
            """, sizing_mode="stretch_width")

            # Create stock cards as clickable buttons
            stock_cards = []

            for i, result in enumerate(results):
                symbol = self.watchlist_symbols[i]
                if isinstance(result, Exception):
                    logger.warning(f"Failed to fetch {symbol} for watchlist: {result}")
                    card_html = f"""
                    <div style='background: {Colors.BG_PRIMARY}; border: 1px solid {Colors.BORDER_SUBTLE}; border-radius: 6px; padding: 10px; opacity: 0.5;'>
                        <div style='font-weight: 600; color: {Colors.TEXT_SECONDARY}; font-size: 0.9rem;'>{symbol}</div>
                        <div style='font-size: 0.75rem; color: {Colors.TEXT_MUTED};'>Error</div>
                    </div>
                    """
                    stock_cards.append(pn.pane.HTML(card_html, sizing_mode="stretch_width"))
                else:
                    real_time_data = result
                    price = real_time_data.get('current_price', 0) or 0
                    prev_close = real_time_data.get('previous_close', 0) or 0
                    change = price - prev_close
                    change_pct = (change / prev_close * 100) if prev_close else 0
                    color = Colors.SUCCESS_GREEN if change >= 0 else Colors.DANGER_RED
                    symbol_icon = '▲' if change >= 0 else '▼'

                    # Get position information for this symbol
                    total_shares, total_value = self._get_positions_for_symbol(symbol)

                    # Build position HTML if there are active positions
                    position_html = ""
                    if total_shares > 0:
                        position_html = f"""
                        <div style='font-size: 0.75rem; color: {Colors.ACCENT_PURPLE}; font-weight: 600; margin-top: 4px;'>
                            {total_shares} shares (${total_value:,.0f})
                        </div>
                        """

                    # Create styled card HTML - simple and clean
                    card_html = f"""
                    <div style='background: {Colors.BG_PRIMARY};
                                border: 1px solid #DEE2E6;
                                border-radius: 6px;
                                padding: 10px;
                                margin-bottom: 8px;
                                cursor: pointer;
                                transition: all 0.2s;
                                box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);'
                         onmouseover='this.style.background="{Colors.BG_HOVER}"'
                         onmouseout='this.style.background="{Colors.BG_PRIMARY}"'>
                        <div style='display: flex; justify-content: space-between; align-items: flex-start;'>
                            <div style='flex: 1;'>
                                <div style='font-weight: 600; color: {Colors.TEXT_PRIMARY}; font-size: 0.9rem;'>{symbol}</div>
                                {position_html}
                            </div>
                            <div style='text-align: right;'>
                                <div style='font-family: monospace; font-size: 0.85rem; color: {Colors.TEXT_PRIMARY};'>${price:.2f}</div>
                                <div style='font-size: 0.75rem; color: {color}; font-weight: 600;'>{symbol_icon} {change_pct:+.2f}%</div>
                            </div>
                        </div>
                    </div>
                    """

                    stock_cards.append(pn.pane.HTML(card_html, sizing_mode="stretch_width", margin=0))

            # Update the pane with new content
            self.pane.clear()
            self.pane.extend([
                pn.Column(
                    watchlist_header,
                    *stock_cards,
                    sizing_mode="stretch_width",
                    scroll=True,
                    max_height=650,
                    styles={
                        'background': Colors.BG_SECONDARY,
                        'border': f'1px solid {Colors.BORDER_SUBTLE}',
                        'border-radius': '8px',
                        'padding': '10px'
                    }
                )
            ])

        def get_panel(self):
            """Get the panel component"""
            return self.pane

    # Get watchlist symbols from portfolio (for portfolio page)
    watchlist_symbols = portfolio_manager.load_portfolio("default")

    # Create all pages
    analysis_app = StockAnalysisApp()
    rl_panel = RLTrainingPanel()
    portfolio_page = PortfolioPage(watchlist_symbols=watchlist_symbols, watchlist_panel=None)  # Will set later
    models_page = ModelsPage()
    live_trading_page = create_live_trading_page(session_manager=session_manager)

    # Create professional navigation tabs
    tabs = pn.Tabs(
        ('📊 Dashboard', pn.Column()),  # Will set dashboard later
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

    # Create watchlist instance with tabs and analysis_app references
    watchlist_panel = WatchlistPanel(session_manager, tabs=tabs, analysis_app=analysis_app)
    watchlist_sidebar = pn.Column(
        watchlist_panel.get_panel(),
        sizing_mode="stretch_width"
    )

    # Now create dashboard with watchlist panel
    dashboard_page = DashboardPage(watchlist_panel=watchlist_panel)
    tabs[0] = ('📊 Dashboard', dashboard_page.get_view())

    # Update portfolio page with watchlist panel
    portfolio_page.watchlist_panel = watchlist_panel

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
