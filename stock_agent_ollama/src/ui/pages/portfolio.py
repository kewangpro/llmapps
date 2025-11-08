"""
Portfolio Management Page

Create and manage portfolios of stocks for training and trading.
Enhanced with tabbed interface: Watchlist, Holdings, Performance.
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
    """Manages a portfolio of stocks with tabbed interface."""

    portfolio = param.List(default=[], doc="List of stock symbols in the portfolio")
    watchlist_symbols = param.List(default=[], doc="List of symbols to watch")

    def __init__(self, watchlist_symbols=None, **params):
        super().__init__(**params)
        self.portfolio_name = "default"
        self.stock_fetcher = StockFetcher()
        self.watchlist_symbols = watchlist_symbols or []

        self._load_portfolio()
        self._create_ui()

        # Defer initial load
        pn.state.add_periodic_callback(self._initial_load, period=100, count=1)

    def _load_portfolio(self):
        """Load portfolio from the manager."""
        self.portfolio = portfolio_manager.load_portfolio(self.portfolio_name)

        # If watchlist not provided, use portfolio as watchlist
        if not self.watchlist_symbols:
            self.watchlist_symbols = self.portfolio.copy()

    def _save_portfolio(self):
        """Save portfolio using the manager."""
        portfolio_manager.save_portfolio(self.portfolio_name, self.portfolio)
        pn.state.notifications.success(f"Portfolio saved!", duration=2000)

    def _create_ui(self):
        """Create the tabbed portfolio UI."""
        # Tab containers
        self.watchlist_container = pn.Column(sizing_mode="stretch_width")
        self.holdings_container = pn.Column(sizing_mode="stretch_width")
        self.performance_container = pn.Column(sizing_mode="stretch_width")

        # Create tabs
        self.tabs = pn.Tabs(
            ("Watchlist", self._create_watchlist_tab()),
            ("Holdings", self._create_holdings_tab()),
            ("Performance", self._create_performance_tab()),
            dynamic=True,
            sizing_mode="stretch_width"
        )

    # ========================================================================
    # Watchlist Tab
    # ========================================================================

    def _create_watchlist_tab(self) -> pn.Column:
        """Create watchlist tab with real-time prices"""
        # Add symbol input
        self.watchlist_input = pn.widgets.TextInput(
            placeholder="Enter symbol to watch (e.g., TSLA)",
            sizing_mode="stretch_width"
        )
        self.watchlist_add_btn = pn.widgets.Button(
            name="Add to Watchlist",
            button_type="success",
            icon='plus'
        )
        self.watchlist_add_btn.on_click(self._add_to_watchlist)

        input_row = pn.Row(
            self.watchlist_input,
            self.watchlist_add_btn,
            sizing_mode="stretch_width"
        )

        return pn.Column(
            HTMLComponents.page_header("Watchlist", "Track stock prices and add to portfolio"),
            pn.Column(
                input_row,
                self.watchlist_container,
                styles=dict(
                    background=Colors.BG_SECONDARY,
                    border_radius='8px',
                    padding='20px'
                ),
                sizing_mode="stretch_width"
            ),
            sizing_mode="stretch_width"
        )

    async def _refresh_watchlist(self):
        """Refresh watchlist with real-time data"""
        self.watchlist_container.clear()

        if not self.watchlist_symbols:
            self.watchlist_container.append(
                pn.pane.Alert("Watchlist is empty. Add symbols above.", alert_type="info")
            )
            return

        # Header
        header_html = f"""
        <div style="
            display: grid;
            grid-template-columns: 1fr 150px 120px 120px 150px 100px;
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
            <div>Volume</div>
            <div>Market Cap</div>
            <div>Actions</div>
        </div>
        """
        self.watchlist_container.append(pn.pane.HTML(header_html, sizing_mode="stretch_width"))

        # Fetch data for all symbols in parallel
        tasks = []
        for symbol in self.watchlist_symbols:
            tasks.append(self._create_watchlist_row(symbol))

        rows = await asyncio.gather(*tasks, return_exceptions=True)

        for row in rows:
            if not isinstance(row, Exception):
                self.watchlist_container.append(row)

    async def _create_watchlist_row(self, symbol: str) -> pn.Row:
        """Create a single watchlist row with real-time data"""
        try:
            data = await asyncio.to_thread(self.stock_fetcher.get_stock_info, symbol)

            price_info = data.get('price_info', {})
            price = price_info.get('current_price')
            prev_close = price_info.get('previous_close')
            volume = price_info.get('volume')
            market_cap = data.get('market_cap')

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

            # Row HTML
            row_html = f"""
            <div style="
                display: grid;
                grid-template-columns: 1fr 150px 120px 120px 150px 100px;
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
                <div style="font-size: {Typography.TEXT_SM}; color: {Colors.TEXT_SECONDARY};">
                    {volume_str}
                </div>
                <div style="font-size: {Typography.TEXT_SM}; color: {Colors.TEXT_SECONDARY};">
                    {mc_str}
                </div>
                <div style="display: flex; gap: 4px;">
                    <!-- Actions handled by buttons below -->
                </div>
            </div>
            """

            # Action buttons
            add_btn = pn.widgets.Button(
                name="Add",
                button_type="primary",
                width=60,
                height=30
            )
            add_btn.on_click(lambda e, s=symbol: self._add_to_portfolio_from_watchlist(s))

            remove_btn = pn.widgets.Button(
                name="×",
                button_type="danger",
                width=30,
                height=30
            )
            remove_btn.on_click(lambda e, s=symbol: self._remove_from_watchlist(s))

            return pn.Row(
                pn.pane.HTML(row_html, sizing_mode="stretch_width"),
                pn.Row(add_btn, remove_btn),
                sizing_mode="stretch_width"
            )

        except Exception as e:
            logger.error(f"Error creating watchlist row for {symbol}: {e}")
            return pn.pane.HTML(f"<div style='color: {Colors.DANGER_RED}; padding: 8px;'>Error loading {symbol}</div>")

    def _add_to_watchlist(self, event):
        """Add symbol to watchlist"""
        symbol = self.watchlist_input.value.strip().upper()
        if not symbol:
            pn.state.notifications.warning("Please enter a symbol", duration=3000)
            return
        if symbol in self.watchlist_symbols:
            pn.state.notifications.warning(f"{symbol} already in watchlist", duration=3000)
            return

        self.watchlist_symbols.append(symbol)
        self.watchlist_input.value = ""
        asyncio.create_task(self._refresh_watchlist())

    def _remove_from_watchlist(self, symbol: str):
        """Remove symbol from watchlist"""
        if symbol in self.watchlist_symbols:
            self.watchlist_symbols.remove(symbol)
            asyncio.create_task(self._refresh_watchlist())

    def _add_to_portfolio_from_watchlist(self, symbol: str):
        """Move symbol from watchlist to portfolio"""
        if symbol not in self.portfolio:
            self.portfolio.append(symbol)
            self._save_portfolio()
            pn.state.notifications.success(f"{symbol} added to portfolio!", duration=2000)
            asyncio.create_task(self._refresh_holdings())

    # ========================================================================
    # Holdings Tab
    # ========================================================================

    def _create_holdings_tab(self) -> pn.Column:
        """Create holdings tab (current portfolio)"""
        # Add stock input
        self.stock_input = pn.widgets.TextInput(
            placeholder="Enter stock symbol (e.g., AAPL)",
            sizing_mode="stretch_width"
        )
        self.add_button = pn.widgets.Button(
            name="Add to Portfolio",
            button_type="primary",
            icon='plus'
        )
        self.add_button.on_click(self._add_stock)

        input_row = pn.Row(
            self.stock_input,
            self.add_button,
            sizing_mode="stretch_width"
        )

        return pn.Column(
            HTMLComponents.page_header("Holdings", "Manage your portfolio stocks"),
            pn.Column(
                input_row,
                self.holdings_container,
                styles=dict(
                    background=Colors.BG_SECONDARY,
                    border_radius='8px',
                    padding='20px'
                ),
                sizing_mode="stretch_width"
            ),
            sizing_mode="stretch_width"
        )

    def _add_stock(self, event):
        """Add a stock to the portfolio."""
        symbol = self.stock_input.value.strip().upper()
        if not symbol:
            pn.state.notifications.warning("Please enter a stock symbol.", duration=3000)
            return
        if symbol in self.portfolio:
            pn.state.notifications.warning(f"'{symbol}' is already in the portfolio.", duration=3000)
            return

        self.portfolio.append(symbol)
        self.stock_input.value = ""
        asyncio.create_task(self._refresh_holdings())
        self._save_portfolio()

    def _remove_stock(self, symbol: str):
        """Remove a stock from the portfolio."""
        if symbol in self.portfolio:
            self.portfolio.remove(symbol)
            asyncio.create_task(self._refresh_holdings())
            self._save_portfolio()

    async def _refresh_holdings(self):
        """Update the display of portfolio holdings"""
        self.holdings_container.clear()

        if not self.portfolio:
            self.holdings_container.append(
                pn.pane.Alert("Portfolio is empty. Add stocks above.", alert_type="info")
            )
            return

        # Create cards for each holding
        for symbol in self.portfolio:
            card = await self._create_holding_card(symbol)
            self.holdings_container.append(card)

    async def _create_holding_card(self, symbol: str) -> pn.Column:
        """Create detailed card for a holding"""
        try:
            data = await asyncio.to_thread(self.stock_fetcher.get_stock_info, symbol)

            price_info = data.get('price_info', {})
            financial_metrics = data.get('financial_metrics', {})

            price = price_info.get('current_price')
            prev_close = price_info.get('previous_close')

            if price and prev_close:
                change = price - prev_close
                change_pct = (change / prev_close * 100)
            else:
                change = 0
                change_pct = 0

            change_color = Colors.SUCCESS_GREEN if change >= 0 else Colors.DANGER_RED
            change_symbol = '▲' if change >= 0 else '▼'

            market_cap = format_market_cap(data.get('market_cap'))
            pe_ratio_val = financial_metrics.get('pe_ratio')
            pe_ratio = f"{pe_ratio_val:.2f}" if pe_ratio_val is not None else "N/A"
            volume_val = price_info.get('volume')
            volume = f"{volume_val:,}" if volume_val is not None else "N/A"
            wk52_low_val = price_info.get('fifty_two_week_low')
            wk52_low = f"${wk52_low_val:.2f}" if wk52_low_val is not None else "N/A"
            wk52_high_val = price_info.get('fifty_two_week_high')
            wk52_high = f"${wk52_high_val:.2f}" if wk52_high_val is not None else "N/A"

            price_str = f"${price:,.2f}" if price is not None else "N/A"

            card_html = f"""
            <div style="
                padding: 20px;
                background: {Colors.BG_PRIMARY};
                border: 1px solid {Colors.BORDER_SUBTLE};
                border-radius: 8px;
                margin-bottom: 12px;
                font-family: {Typography.FONT_PRIMARY};
            ">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
                    <h3 style="margin: 0; font-size: {Typography.TEXT_XL}; color: {Colors.TEXT_PRIMARY};">{symbol}</h3>
                    <div style="text-align: right;">
                        <div style="font-family: {Typography.FONT_MONO}; font-size: {Typography.TEXT_XL}; font-weight: 600;">{price_str}</div>
                        <div style="color: {change_color}; font-weight: 600;">{change_symbol} {change_pct:+.2f}%</div>
                    </div>
                </div>

                <div style="
                    display: grid;
                    grid-template-columns: repeat(4, 1fr);
                    gap: 16px;
                    font-size: {Typography.TEXT_SM};
                    color: {Colors.TEXT_SECONDARY};
                ">
                    <div>
                        <div style="font-weight: 600; color: {Colors.TEXT_PRIMARY};">Market Cap</div>
                        <div>{market_cap}</div>
                    </div>
                    <div>
                        <div style="font-weight: 600; color: {Colors.TEXT_PRIMARY};">P/E Ratio</div>
                        <div>{pe_ratio}</div>
                    </div>
                    <div>
                        <div style="font-weight: 600; color: {Colors.TEXT_PRIMARY};">Volume</div>
                        <div>{volume}</div>
                    </div>
                    <div>
                        <div style="font-weight: 600; color: {Colors.TEXT_PRIMARY};">52-Week Range</div>
                        <div>{wk52_low} - {wk52_high}</div>
                    </div>
                </div>
            </div>
            """

            remove_button = pn.widgets.Button(
                name=f"Remove {symbol}",
                button_type="danger",
                icon='trash',
                width=150
            )
            remove_button.on_click(lambda e, s=symbol: self._remove_stock(s))

            return pn.Column(
                pn.pane.HTML(card_html, sizing_mode="stretch_width"),
                remove_button,
                sizing_mode="stretch_width"
            )

        except Exception as e:
            logger.error(f"Error creating holding card for {symbol}: {e}")
            return pn.pane.HTML(f"<div style='color: {Colors.DANGER_RED}; padding: 20px;'>Error loading {symbol}</div>")

    # ========================================================================
    # Performance Tab
    # ========================================================================

    def _create_performance_tab(self) -> pn.Column:
        """Create performance analytics tab"""
        return pn.Column(
            HTMLComponents.page_header("Performance", "Portfolio analytics and insights"),
            pn.Column(
                self.performance_container,
                styles=dict(
                    background=Colors.BG_SECONDARY,
                    border_radius='8px',
                    padding='20px'
                ),
                sizing_mode="stretch_width"
            ),
            sizing_mode="stretch_width"
        )

    async def _refresh_performance(self):
        """Update performance analytics"""
        self.performance_container.clear()

        if not self.portfolio:
            self.performance_container.append(
                pn.pane.Alert("Add stocks to see performance analytics", alert_type="info")
            )
            return

        try:
            # Aggregate metrics
            total_value = 0
            total_change = 0
            total_prev_value = 0

            for symbol in self.portfolio:
                try:
                    data = await asyncio.to_thread(self.stock_fetcher.get_stock_info, symbol)
                    price_info = data.get('price_info', {})
                    price = price_info.get('current_price', 0)
                    prev_close = price_info.get('previous_close', 0)

                    total_value += price
                    total_prev_value += prev_close
                except:
                    pass

            total_change = total_value - total_prev_value
            total_change_pct = (total_change / total_prev_value * 100) if total_prev_value > 0 else 0

            change_color = Colors.SUCCESS_GREEN if total_change >= 0 else Colors.DANGER_RED

            # Summary card
            summary_html = f"""
            <div style="
                padding: 24px;
                background: linear-gradient(135deg, {Colors.ACCENT_PURPLE} 0%, {Colors.ACCENT_CYAN} 100%);
                border-radius: 12px;
                color: white;
                margin-bottom: 24px;
            ">
                <h2 style="margin: 0 0 16px 0; font-size: {Typography.TEXT_2XL};">Portfolio Summary</h2>
                <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 24px;">
                    <div>
                        <div style="opacity: 0.9; font-size: {Typography.TEXT_SM};">Total Stocks</div>
                        <div style="font-size: {Typography.TEXT_2XL}; font-weight: 600; margin-top: 4px;">{len(self.portfolio)}</div>
                    </div>
                    <div>
                        <div style="opacity: 0.9; font-size: {Typography.TEXT_SM};">Total Value</div>
                        <div style="font-size: {Typography.TEXT_2XL}; font-weight: 600; margin-top: 4px; font-family: {Typography.FONT_MONO};">${total_value:,.2f}</div>
                    </div>
                    <div>
                        <div style="opacity: 0.9; font-size: {Typography.TEXT_SM};">Daily Change</div>
                        <div style="font-size: {Typography.TEXT_2XL}; font-weight: 600; margin-top: 4px; font-family: {Typography.FONT_MONO};">{total_change_pct:+.2f}%</div>
                    </div>
                </div>
            </div>
            """

            self.performance_container.append(pn.pane.HTML(summary_html, sizing_mode="stretch_width"))

            # Top movers
            movers_html = f"""
            <div style="padding: 20px; background: {Colors.BG_PRIMARY}; border: 1px solid {Colors.BORDER_SUBTLE}; border-radius: 8px;">
                <h3 style="margin: 0 0 12px 0; color: {Colors.TEXT_PRIMARY};">Top Movers</h3>
                <p style="color: {Colors.TEXT_SECONDARY}; font-size: {Typography.TEXT_SM};">Coming soon: Best and worst performers</p>
            </div>
            """

            self.performance_container.append(pn.pane.HTML(movers_html, sizing_mode="stretch_width"))

        except Exception as e:
            logger.error(f"Error refreshing performance: {e}")
            self.performance_container.append(
                pn.pane.Alert(f"Error loading performance: {str(e)}", alert_type="danger")
            )

    # ========================================================================
    # Main View
    # ========================================================================

    async def _initial_load(self):
        """Initial load of all tabs"""
        await asyncio.gather(
            self._refresh_watchlist(),
            self._refresh_holdings(),
            self._refresh_performance()
        )

    def get_view(self):
        """Get the portfolio management view."""
        return pn.Column(
            self.tabs,
            HTMLComponents.disclaimer(),
            sizing_mode="stretch_width"
        )

    async def refresh(self):
        """Public refresh method for external calls (e.g., from watchlist)"""
        await self._refresh_watchlist()
