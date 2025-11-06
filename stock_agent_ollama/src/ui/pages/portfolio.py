"""
Portfolio Management Page

Create and manage portfolios of stocks for training and trading.
"""

import panel as pn
import param
import logging
import asyncio
import numpy as np

from src.tools.portfolio_manager import portfolio_manager
from src.tools.stock_fetcher import StockFetcher
from src.ui.design_system import Colors, HTMLComponents

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
    """Manages a portfolio of stocks."""

    portfolio = param.List(default=[], doc="List of stock symbols in the portfolio")

    def __init__(self, watchlist_panel=None, **params):
        super().__init__(**params)
        self.watchlist_panel = watchlist_panel
        self.portfolio_name = "default"  # For now, we manage a single default portfolio
        self.stock_fetcher = StockFetcher()
        self._load_portfolio()
        self._create_ui()
        # Defer the initial metrics loading until the event loop is running
        pn.state.add_periodic_callback(self._update_portfolio_view, period=100, count=1)

    def _load_portfolio(self):
        """Load portfolio from the manager."""
        self.portfolio = portfolio_manager.load_portfolio(self.portfolio_name)

    def _save_portfolio(self):
        """Save portfolio using the manager."""
        portfolio_manager.save_portfolio(self.portfolio_name, self.portfolio)
        pn.state.notifications.success(f"Portfolio '{self.portfolio_name}' saved!", duration=2000)
        if self.watchlist_panel:
            # `watchlist_panel.refresh` is an async coroutine; executing it
            # directly returns a coroutine that would be left un-awaited by
            # `pn.state.execute`. Create a background task so the coroutine
            # is properly scheduled on the event loop.
            pn.state.execute(lambda: asyncio.create_task(self.watchlist_panel.refresh()))

    def _create_ui(self):
        """Create the portfolio management UI."""
        self.stock_input = pn.widgets.TextInput(
            placeholder="Enter stock symbol (e.g., AAPL)",
            sizing_mode="stretch_width"
        )
        self.add_button = pn.widgets.Button(
            name="Add Stock",
            button_type="primary",
            icon='plus'
        )
        self.portfolio_view = pn.Column(sizing_mode="stretch_width")

        self.add_button.on_click(self._add_stock)

    def _add_stock(self, event):
        """Add a stock to the portfolio."""
        symbol = self.stock_input.value.strip().upper()
        if not symbol:
            pn.state.notifications.warning("Please enter a stock symbol.", duration=3000)
            return
        if symbol in self.portfolio:
            pn.state.notifications.warning(f"'{symbol}' is already in the portfolio.", duration=3000)
            return
        
        self.portfolio = self.portfolio + [symbol]
        self.stock_input.value = ""
        self._update_portfolio_view()
        self._save_portfolio()

    def _remove_stock(self, event):
        """Remove a stock from the portfolio."""
        symbol_to_remove = event.obj.name.split(" ")[-1]
        self.portfolio = [s for s in self.portfolio if s != symbol_to_remove]
        self._update_portfolio_view()
        self._save_portfolio()

    async def _fetch_and_update_metrics(self, symbol: str, metrics_pane: pn.pane.HTML):
        """Fetch real-time data and update the metrics pane."""
        try:
            metrics_pane.object = "<div style='font-size: 0.8em; color: #888; padding: 20px;'>Loading...</div>"
            data = await asyncio.to_thread(self.stock_fetcher.get_stock_info, symbol)
            
            price_info = data.get('price_info', {})
            financial_metrics = data.get('financial_metrics', {})

            price = price_info.get('current_price')
            prev_close = price_info.get('previous_close')

            if price is not None and prev_close is not None:
                change = price - prev_close
                change_pct = (change / prev_close * 100) if prev_close != 0 else 0
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
            wk52_low = f"{wk52_low_val:.2f}" if wk52_low_val is not None else "N/A"
            wk52_high_val = price_info.get('fifty_two_week_high')
            wk52_high = f"{wk52_high_val:.2f}" if wk52_high_val is not None else "N/A"

            price_str = f"${price:,.2f}" if price is not None else "N/A"

            html = f"""
            <div style='display: flex; flex-direction: column; width: 100%;'>
                <div style='display: flex; justify-content: space-between; align-items: center;'>
                    <div style='font-family: monospace; font-size: 1.2em; font-weight: 600;'>{price_str}</div>
                    <div style='color: {change_color}; font-weight: 600;'>{change_symbol} {change_pct:+.2f}%</div>
                </div>
                <div style='display: flex; justify-content: space-between; font-size: 0.8em; color: {Colors.TEXT_SECONDARY}; margin-top: 5px;'>
                    <span>MCap: {market_cap}</span>
                    <span>P/E: {pe_ratio}</span>
                    <span>Vol: {volume}</span>
                    <span>52Wk: {wk52_low}-{wk52_high}</span>
                </div>
            </div>
            """
            metrics_pane.object = html
        except Exception as e:
            logger.error(f"Failed to fetch metrics for {symbol}: {e}")
            metrics_pane.object = f"<div style='font-size: 0.8em; color: {Colors.DANGER_RED}; padding: 20px;'>Error fetching data</div>"

    def _update_portfolio_view(self, *events):
        """Update the display of the portfolio."""
        self.portfolio_view.clear()
        if not self.portfolio:
            self.portfolio_view.append(pn.pane.Alert("The portfolio is empty. Add stocks using the input above.", alert_type="info"))
            return

        rows = []
        for symbol in self.portfolio:
            remove_button = pn.widgets.Button(name=f"Remove {symbol}", button_type="danger", icon='trash', width=120)
            remove_button.on_click(self._remove_stock)
            
            metrics_pane = pn.pane.HTML("", sizing_mode="stretch_width")
            
            def make_task(s, p):
                return lambda: asyncio.create_task(self._fetch_and_update_metrics(s, p))
            
            pn.state.execute(make_task(symbol, metrics_pane))

            row = pn.Row(
                pn.pane.Markdown(f"## {symbol}", width=100),
                metrics_pane,
                remove_button,
                sizing_mode="stretch_width",
                align="center",
                styles={"border-bottom": f"1px solid {Colors.BORDER_SUBTLE}", "padding": "10px"}
            )
            rows.append(row)
        self.portfolio_view.extend(rows)

    def get_view(self):
        """Get the portfolio management view."""
        input_section = pn.Row(
            self.stock_input,
            self.add_button,
            sizing_mode="stretch_width"
        )

        return pn.Column(

            pn.Column(
                input_section,
                self.portfolio_view,
                styles=dict(background=Colors.BG_SECONDARY, border_radius='8px', padding='20px'),
                sizing_mode="stretch_width"
            ),
            HTMLComponents.disclaimer(),
            sizing_mode="stretch_width"
        )
