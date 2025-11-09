"""
Watchlist Page

Track and monitor stocks with real-time data in a simple table view.
Note: This is a watchlist/stock tracker, not a portfolio manager with positions.
"""

import panel as pn
import param
import logging
import asyncio
import numpy as np
from datetime import datetime, timedelta

from src.tools.portfolio_manager import portfolio_manager
from src.tools.stock_fetcher import StockFetcher
from src.ui.design_system import Colors, HTMLComponents, Typography

logger = logging.getLogger(__name__)

def format_market_cap(mc):
    if mc is None:
        return "N/A"
    if mc >= 1e12:
        return f"${mc/1e12:.2f}T"
    if mc >= 1e9:
        return f"${mc/1e9:.2f}B"
    if mc >= 1e6:
        return f"${mc/1e6:.2f}M"
    return f"${mc:,.0f}"

class PortfolioPage(param.Parameterized):
    """Stock watchlist with multiple views for tracking market data."""

    tracked_stocks = param.List(default=[], doc="List of stock symbols being tracked")
    quick_view_symbols = param.List(default=[], doc="List of symbols in quick view")

    def __init__(self, watchlist_symbols=None, watchlist_panel=None, **params):
        super().__init__(**params)
        self.watchlist_name = "default"
        self.stock_fetcher = StockFetcher()
        self.quick_view_symbols = watchlist_symbols or []
        self.watchlist_panel = watchlist_panel  # Reference to sidebar watchlist panel

        self._load_watchlist()
        self._create_ui()

        # Defer initial load
        pn.state.add_periodic_callback(self._initial_load, period=100, count=1)

    def _load_watchlist(self):
        """Load watchlist from the manager."""
        self.tracked_stocks = portfolio_manager.load_portfolio(self.watchlist_name)

        # If quick view not provided, use tracked stocks
        if not self.quick_view_symbols:
            self.quick_view_symbols = self.tracked_stocks.copy()

    def _save_watchlist(self):
        """Save watchlist using the manager."""
        portfolio_manager.save_portfolio(self.watchlist_name, self.tracked_stocks)
        pn.state.notifications.success(f"Watchlist saved!", duration=2000)

    def _create_ui(self):
        """Create the watchlist UI."""
        # Main container
        self.quick_view_container = pn.Column(sizing_mode="stretch_width")

        # Create main view (no tabs needed)
        self.main_view = self._create_quick_view_content()

    # ========================================================================
    # Watchlist View (Table View)
    # ========================================================================

    def _create_quick_view_content(self) -> pn.Column:
        """Create watchlist with real-time prices in table format"""
        # Add symbol input
        self.quick_view_input = pn.widgets.TextInput(
            placeholder="Enter symbol to watch (e.g., TSLA)",
            sizing_mode="stretch_width"
        )
        self.quick_view_add_btn = pn.widgets.Button(
            name="Add to Watchlist",
            button_type="success",
            icon='plus'
        )
        self.quick_view_add_btn.on_click(self._add_to_quick_view)

        input_row = pn.Row(
            self.quick_view_input,
            self.quick_view_add_btn,
            sizing_mode="stretch_width"
        )

        return pn.Column(
            pn.Column(
                input_row,
                self.quick_view_container,
                styles=dict(
                    background=Colors.BG_SECONDARY,
                    border_radius='8px',
                    padding='20px'
                ),
                sizing_mode="stretch_width"
            ),
            sizing_mode="stretch_width"
        )

    async def _refresh_quick_view(self):
        """Refresh quick view with real-time data"""
        self.quick_view_container.clear()

        if not self.quick_view_symbols:
            self.quick_view_container.append(
                pn.pane.Alert("Quick view is empty. Add symbols above.", alert_type="info")
            )
            return

        # Header
        header_html = f"""
        <div style="
            display: grid;
            grid-template-columns: 140px 130px 130px 200px 150px 1fr;
            gap: 12px;
            padding: 12px;
            background: {Colors.BG_TERTIARY};
            border-radius: 6px;
            font-family: {Typography.FONT_PRIMARY};
            font-weight: 600;
            font-size: {Typography.TEXT_SM};
            color: {Colors.TEXT_SECONDARY};
            margin-bottom: 8px;
        ">
            <div>Symbol</div>
            <div>Price</div>
            <div>Change</div>
            <div>52W Range</div>
            <div>Volume</div>
            <div>Market Cap</div>
        </div>
        """
        self.quick_view_container.append(pn.pane.HTML(header_html, sizing_mode="stretch_width"))

        # Fetch data for all symbols in parallel
        tasks = []
        for symbol in self.quick_view_symbols:
            tasks.append(self._create_quick_view_row(symbol))

        rows = await asyncio.gather(*tasks, return_exceptions=True)

        for row in rows:
            if not isinstance(row, Exception):
                self.quick_view_container.append(row)

    async def _create_quick_view_row(self, symbol: str) -> pn.Row:
        """Create a single watchlist row with real-time data"""
        try:
            data = await asyncio.to_thread(self.stock_fetcher.get_stock_info, symbol)

            price_info = data.get('price_info', {})
            price = price_info.get('current_price')
            prev_close = price_info.get('previous_close')
            volume = price_info.get('volume')
            market_cap = data.get('market_cap')
            week_52_low = price_info.get('fifty_two_week_low')
            week_52_high = price_info.get('fifty_two_week_high')

            if price and prev_close:
                change = price - prev_close
                change_pct = (change / prev_close * 100)
            else:
                change = 0
                change_pct = 0

            change_color = Colors.SUCCESS_GREEN if change >= 0 else Colors.DANGER_RED
            change_symbol = '▲' if change >= 0 else '▼'

            price_str = f"${price:,.2f}" if price else "N/A"
            volume_str = f"{volume:,.0f}" if volume else "N/A"
            mc_str = format_market_cap(market_cap)

            # Format 52-week range
            if week_52_low and week_52_high:
                range_52w = f"${week_52_low:.2f} - ${week_52_high:.2f}"
            else:
                range_52w = "N/A"

            # Row HTML
            row_html = f"""
            <div style="
                display: grid;
                grid-template-columns: 140px 130px 130px 200px 150px 1fr;
                gap: 12px;
                padding: 12px;
                background: {Colors.BG_PRIMARY};
                border: 1px solid {Colors.BORDER_SUBTLE};
                border-radius: 6px;
                margin-bottom: 8px;
                font-family: {Typography.FONT_PRIMARY};
                align-items: center;
            ">
                <div style="font-weight: 600; font-size: {Typography.TEXT_BASE}; color: {Colors.TEXT_PRIMARY};">
                    {symbol}
                </div>
                <div style="font-family: {Typography.FONT_MONO}; font-size: {Typography.TEXT_BASE}; color: {Colors.TEXT_PRIMARY};">
                    {price_str}
                </div>
                <div style="font-family: {Typography.FONT_MONO}; font-size: {Typography.TEXT_SM}; color: {change_color}; font-weight: 600;">
                    {change_symbol} {change_pct:+.2f}%
                </div>
                <div style="font-family: {Typography.FONT_MONO}; font-size: {Typography.TEXT_SM}; color: {Colors.TEXT_SECONDARY};">
                    {range_52w}
                </div>
                <div style="font-size: {Typography.TEXT_SM}; color: {Colors.TEXT_SECONDARY};">
                    {volume_str}
                </div>
                <div style="font-size: {Typography.TEXT_SM}; color: {Colors.TEXT_SECONDARY};">
                    {mc_str}
                </div>
            </div>
            """

            # Action button
            remove_btn = pn.widgets.Button(
                name="×",
                button_type="danger",
                width=30,
                height=30
            )
            remove_btn.on_click(lambda e, s=symbol: self._remove_from_quick_view(s))

            return pn.Row(
                pn.pane.HTML(row_html, sizing_mode="stretch_width"),
                remove_btn,
                sizing_mode="stretch_width"
            )

        except Exception as e:
            logger.error(f"Error creating watchlist row for {symbol}: {e}")
            return pn.pane.HTML(f"<div style='color: {Colors.DANGER_RED}; padding: 8px;'>Error loading {symbol}</div>")

    def _add_to_quick_view(self, event):
        """Add symbol to watchlist"""
        symbol = self.quick_view_input.value.strip().upper()
        if not symbol:
            pn.state.notifications.warning("Please enter a symbol", duration=3000)
            return
        if symbol in self.quick_view_symbols:
            pn.state.notifications.warning(f"{symbol} already in watchlist", duration=3000)
            return

        self.quick_view_symbols.append(symbol)
        if symbol not in self.tracked_stocks:
            self.tracked_stocks.append(symbol)
        self.quick_view_input.value = ""
        self._save_watchlist()

        # Refresh both the page view and sidebar
        asyncio.create_task(self._refresh_quick_view())
        if self.watchlist_panel:
            asyncio.create_task(self.watchlist_panel.refresh())

    def _remove_from_quick_view(self, symbol: str):
        """Remove symbol from watchlist"""
        if symbol in self.quick_view_symbols:
            self.quick_view_symbols.remove(symbol)
        if symbol in self.tracked_stocks:
            self.tracked_stocks.remove(symbol)
        self._save_watchlist()

        # Refresh both the page view and sidebar
        asyncio.create_task(self._refresh_quick_view())
        if self.watchlist_panel:
            asyncio.create_task(self.watchlist_panel.refresh())

    # ========================================================================
    # Main View
    # ========================================================================

    async def _initial_load(self):
        """Initial load of watchlist"""
        await self._refresh_quick_view()

    def get_view(self):
        """Get the watchlist view."""
        return pn.Column(
            self.main_view,
            HTMLComponents.disclaimer(),
            sizing_mode="stretch_width"
        )

    async def refresh(self):
        """Public refresh method for external calls"""
        await self._refresh_quick_view()
