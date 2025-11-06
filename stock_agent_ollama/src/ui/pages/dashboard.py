"""
Dashboard Page - Market Overview
"""

import panel as pn
import param
import logging
import asyncio
from datetime import datetime

from src.ui.design_system import Colors, HTMLComponents
from src.tools.stock_fetcher import StockFetcher

logger = logging.getLogger(__name__)


class DashboardPage(param.Parameterized):
    """Professional dashboard with market overview and quick actions"""

    def __init__(self, watchlist_panel=None, **params):
        super().__init__(**params)
        self.stock_fetcher = StockFetcher()
        self.watchlist_panel = watchlist_panel
        self._create_ui()

    def _create_ui(self):
        """Create dashboard UI components"""
        self.market_overview = pn.pane.HTML("", sizing_mode="stretch_width")
        self.quick_actions_panel = pn.Column(sizing_mode="stretch_width")

        self.refresh_button = pn.widgets.Button(
            name="🔄 Refresh", button_type="primary", width=120
        )
        self.refresh_button.on_click(self._refresh_dashboard)

        pn.state.onload(self._refresh_dashboard)

    def _refresh_dashboard(self, event=None):
        """Refresh dashboard data"""
        try:
            self._load_market_overview()
            self._load_quick_actions()
            # Also refresh watchlist if available
            if self.watchlist_panel:
                # `watchlist_panel.refresh` may be an async coroutine. Use
                # asyncio.create_task inside pn.state.execute so the coroutine
                # is scheduled on the server event loop and not left
                # un-awaited.
                pn.state.execute(lambda: asyncio.create_task(self.watchlist_panel.refresh()))
            pn.state.notifications.success("Dashboard and watchlist refreshed", duration=2000)
        except Exception as e:
            logger.error(f"Dashboard refresh failed: {e}")
            pn.state.notifications.error(f"Failed to refresh dashboard: {str(e)}", duration=5000)

    def _load_market_overview(self):
        """Load market overview with major indices"""
        indices = {
            "^GSPC": "S&P 500", "^IXIC": "NASDAQ", "^DJI": "Dow Jones", "^RUT": "Russell 2000"
        }
        overview_html = f"""
        <div style='background: {Colors.BG_SECONDARY}; border: 1px solid {Colors.BORDER_SUBTLE}; border-left: 4px solid {Colors.ACCENT_PURPLE}; padding: 15px; border-radius: 8px; margin-bottom: 15px; box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);'>
            <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;'>
                <h3 style='margin: 0; font-size: 1.25rem; color: {Colors.TEXT_PRIMARY};'>📊 Markets</h3>
                <span style='font-size: 0.75rem; color: {Colors.TEXT_SECONDARY};'>{datetime.now().strftime('%I:%M %p')}</span>
            </div>
            <div style='display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px;'>
        """
        for symbol, name in indices.items():
            try:
                real_time_data = self.stock_fetcher.get_real_time_price(symbol)
                price = real_time_data.get('current_price', 0) or 0
                prev_close = real_time_data.get('previous_close', 0) or 0
                change = price - prev_close
                change_pct = (change / prev_close * 100) if prev_close else 0
                color = Colors.SUCCESS_GREEN if change >= 0 else Colors.DANGER_RED
                symbol_icon = '▲' if change >= 0 else '▼'
                overview_html += f"""
                <div style='background: {Colors.BG_PRIMARY}; border: 1px solid {Colors.BORDER_SUBTLE}; padding: 12px; border-radius: 8px;'>
                    <div style='font-size: 0.75rem; margin-bottom: 4px; color: {Colors.TEXT_SECONDARY};'>{name}</div>
                    <div style='font-size: 1.25rem; font-weight: 700; margin-bottom: 4px; font-family: monospace; color: {Colors.TEXT_PRIMARY};'>{price:,.2f}</div>
                    <div style='font-size: 0.8rem; color: {color}; font-weight: 600;'>{symbol_icon} {change_pct:+.2f}%</div>
                </div>
                """
            except Exception as e:
                logger.warning(f"Failed to fetch {symbol}: {e}")
                overview_html += f"""
                <div style='background: {Colors.BG_PRIMARY}; border: 1px solid {Colors.BORDER_SUBTLE}; padding: 15px; border-radius: 8px; opacity: 0.6;'>
                    <div style='font-size: 0.875rem; margin-bottom: 5px; color: {Colors.TEXT_SECONDARY};'>{name}</div>
                    <div style='font-size: 1rem; color: {Colors.TEXT_MUTED};'>Loading...</div>
                </div>
                """
        overview_html += "</div></div>"
        self.market_overview.object = overview_html

    def _load_quick_actions(self):
        """Load quick action buttons - Compact version"""
        self.quick_actions_panel.clear()
        header_html = f"""
        <div style='padding: 10px 0; margin-bottom: 10px; border-bottom: 2px solid {Colors.BORDER_SUBTLE};'>
            <h3 style='margin: 0; font-size: 1.25rem; color: {Colors.TEXT_PRIMARY};'>⚡ Quick Actions</h3>
        </div>
        """
        self.quick_actions_panel.append(pn.pane.HTML(header_html))
        actions_html = """
        <div style='display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px;'>
            <div style='background: {Colors.BG_SECONDARY}; border: 1px solid {Colors.BORDER_SUBTLE}; border-radius: 8px; padding: 15px; text-align: center; cursor: pointer; transition: all 0.2s;' onmouseover='this.style.background="{Colors.BG_HOVER}"' onmouseout='this.style.background="{Colors.BG_SECONDARY}"'>
                <div style='font-size: 1.5rem; margin-bottom: 8px;'>🤖</div>
                <div style='font-weight: 600; color: {Colors.TEXT_PRIMARY}; font-size: 0.875rem; margin-bottom: 3px;'>Train LSTM</div>
                <div style='font-size: 0.7rem; color: {Colors.TEXT_SECONDARY};'>AI predictions</div>
            </div>
            <div style='background: {Colors.BG_SECONDARY}; border: 1px solid {Colors.BORDER_SUBTLE}; border-radius: 8px; padding: 15px; text-align: center; cursor: pointer; transition: all 0.2s;' onmouseover='this.style.background="{Colors.BG_HOVER}"' onmouseout='this.style.background="{Colors.BG_SECONDARY}"'>
                <div style='font-size: 1.5rem; margin-bottom: 8px;'>📊</div>
                <div style='font-weight: 600; color: {Colors.TEXT_PRIMARY}; font-size: 0.875rem; margin-bottom: 3px;'>Backtest</div>
                <div style='font-size: 0.7rem; color: {Colors.TEXT_SECONDARY};'>Test strategies</div>
            </div>
                <div style='background: {Colors.BG_SECONDARY}; border: 1px solid {Colors.BORDER_SUBTLE}; border-radius: 8px; padding: 15px; text-align: center; cursor: pointer; transition: all 0.2s;' onmouseover='this.style.background="{Colors.BG_HOVER}"' onmouseout='this.style.background="{Colors.BG_SECONDARY}"'>
                    <div style='font-size: 1.5rem; margin-bottom: 8px;'>🔴</div>
                <div style='font-weight: 600; color: {Colors.TEXT_PRIMARY}; font-size: 0.875rem; margin-bottom: 3px;'>Live Trading</div>
                <div style='font-size: 0.7rem; color: {Colors.TEXT_SECONDARY};'>Real-time</div>
            </div>
            <div style='background: {Colors.BG_SECONDARY}; border: 1px solid {Colors.BORDER_SUBTLE}; border-radius: 8px; padding: 15px; text-align: center; cursor: pointer; transition: all 0.2s;' onmouseover='this.style.background="{Colors.BG_HOVER}"' onmouseout='this.style.background="{Colors.BG_SECONDARY}"'>
                <div style='font-size: 1.5rem; margin-bottom: 8px;'>📈</div>
                <div style='font-weight: 600; color: {Colors.TEXT_PRIMARY}; font-size: 0.875rem; margin-bottom: 3px;'>Report</div>
                <div style='font-size: 0.7rem; color: {Colors.TEXT_SECONDARY};'>AI-powered</div>
            </div>
        </div>
        """.format(Colors=Colors)
        self.quick_actions_panel.append(pn.pane.HTML(actions_html))


    def get_view(self):
        """Get the dashboard view - Wide horizontal layout"""
        left_column = pn.Column(
            self.market_overview,
            sizing_mode="stretch_width"
        )
        right_column = pn.Column(
            self.quick_actions_panel,
            sizing_mode="stretch_width",
        )

        content_row = pn.Row(
            left_column,
            right_column,
            sizing_mode="stretch_width",
            min_height=350
        )

        return pn.Column(
            pn.Row(
                pn.pane.HTML(f"<h1 style='margin: 0; color: {Colors.TEXT_PRIMARY}; font-size: 1.75rem;'>📊 Dashboard</h1>"),
                pn.Spacer(),
                self.refresh_button,
                sizing_mode="stretch_width",
                styles=dict(background=Colors.BG_SECONDARY, border_radius='8px', padding='15px'),
                margin=(0, 0, 15, 0)
            ),
            content_row,
            HTMLComponents.disclaimer(),
            sizing_mode="stretch_width"
        )
