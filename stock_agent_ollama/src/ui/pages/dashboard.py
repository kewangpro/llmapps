"""
Dashboard Page - Market Overview and Watchlist
"""

import panel as pn
import param
import logging
from datetime import datetime

from src.ui.design_system import Colors, HTMLComponents, Styles
from src.tools.stock_fetcher import StockFetcher

logger = logging.getLogger(__name__)


class DashboardPage(param.Parameterized):
    """Professional dashboard with market overview and watchlist"""

    # Watchlist symbols
    watchlist = param.List(default=["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "NVDA", "META", "ORCL"])

    def __init__(self, **params):
        super().__init__(**params)
        self.stock_fetcher = StockFetcher()
        self._create_ui()

    def _create_ui(self):
        """Create dashboard UI components"""
        # Market overview panel
        self.market_overview = pn.pane.HTML("", sizing_mode="stretch_width")

        # Watchlist panel
        self.watchlist_panel = pn.Column(sizing_mode="stretch_width")

        # Quick actions panel
        self.quick_actions_panel = pn.Column(sizing_mode="stretch_width")

        # Featured charts row
        self.featured_charts = pn.Row(sizing_mode="stretch_width")

        # Refresh button
        self.refresh_button = pn.widgets.Button(
            name="🔄 Refresh",
            button_type="primary",
            width=120
        )
        self.refresh_button.on_click(self._refresh_dashboard)

        # Auto-load on init
        self._refresh_dashboard()

    def _refresh_dashboard(self, event=None):
        """Refresh dashboard data"""
        try:
            self._load_market_overview()
            self._load_watchlist()
            self._load_quick_actions()
            self._load_featured_charts()

        except Exception as e:
            logger.error(f"Dashboard refresh failed: {e}")
            pn.state.notifications.error(f"Failed to refresh dashboard: {str(e)}", duration=5000)

    def _load_market_overview(self):
        """Load market overview with major indices"""
        # Use ETFs as proxies for indices to avoid ^ symbol validation issues
        indices = {
            "SPY": "S&P 500",
            "QQQ": "NASDAQ",
            "DIA": "Dow Jones",
            "IWM": "Russell 2000"
        }

        overview_html = f"""
        <div style='background: {Colors.BG_SECONDARY};
                    border: 1px solid {Colors.BORDER_SUBTLE};
                    border-left: 4px solid {Colors.ACCENT_PURPLE};
                    padding: 15px;
                    border-radius: 8px;
                    margin-bottom: 15px;
                    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);'>
            <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;'>
                <h3 style='margin: 0; font-size: 1.25rem; color: {Colors.TEXT_PRIMARY};'>📊 Markets</h3>
                <span style='font-size: 0.75rem; color: {Colors.TEXT_SECONDARY};'>{datetime.now().strftime('%I:%M %p')}</span>
            </div>
            <div style='display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px;'>
        """

        for symbol, name in indices.items():
            try:
                # Fetch current data for index
                real_time_data = self.stock_fetcher.get_real_time_price(symbol)
                price = real_time_data.get('current_price', 0) or 0
                prev_close = real_time_data.get('previous_close', 0) or 0
                change = price - prev_close
                change_pct = (change / prev_close * 100) if prev_close else 0

                color = Colors.SUCCESS_GREEN if change >= 0 else Colors.DANGER_RED
                symbol_icon = '▲' if change >= 0 else '▼'

                overview_html += f"""
                <div style='background: {Colors.BG_PRIMARY};
                            border: 1px solid {Colors.BORDER_SUBTLE};
                            padding: 12px;
                            border-radius: 8px;'>
                    <div style='font-size: 0.75rem; margin-bottom: 4px; color: {Colors.TEXT_SECONDARY};'>{name}</div>
                    <div style='font-size: 1.25rem; font-weight: 700; margin-bottom: 4px; font-family: monospace; color: {Colors.TEXT_PRIMARY};'>
                        {price:,.2f}
                    </div>
                    <div style='font-size: 0.8rem; color: {color}; font-weight: 600;'>
                        {symbol_icon} {change_pct:+.2f}%
                    </div>
                </div>
                """

            except Exception as e:
                logger.warning(f"Failed to fetch {symbol}: {e}")
                overview_html += f"""
                <div style='background: {Colors.BG_PRIMARY};
                            border: 1px solid {Colors.BORDER_SUBTLE};
                            padding: 15px;
                            border-radius: 8px;
                            opacity: 0.6;'>
                    <div style='font-size: 0.875rem; margin-bottom: 5px; color: {Colors.TEXT_SECONDARY};'>{name}</div>
                    <div style='font-size: 1rem; color: {Colors.TEXT_MUTED};'>Loading...</div>
                </div>
                """

        overview_html += """
            </div>
        </div>
        """

        self.market_overview.object = overview_html

    def _load_watchlist(self):
        """Load watchlist with current prices - Compact with max height"""
        self.watchlist_panel.clear()

        # Compact header
        header_html = f"""
        <div style='padding: 10px 0; margin-bottom: 10px; border-bottom: 2px solid {Colors.BORDER_SUBTLE};'>
            <h3 style='margin: 0; font-size: 1.25rem; color: {Colors.TEXT_PRIMARY};'>⭐ Your Watchlist</h3>
        </div>
        """
        self.watchlist_panel.append(pn.pane.HTML(header_html))

        # Compact watchlist table with max-height and scroll
        table_html = f"""
        <div style='background: {Colors.BG_SECONDARY};
                    border: 1px solid {Colors.BORDER_SUBTLE};
                    border-radius: 8px;
                    max-height: 400px;
                    overflow-y: auto;'>
            <table style='width: 100%; border-collapse: collapse;'>
                <thead style='position: sticky; top: 0; z-index: 10;'>
                    <tr style='background: {Colors.BG_TERTIARY};'>
                        <th style='padding: 8px; text-align: left; font-size: 0.7rem; text-transform: uppercase; color: {Colors.TEXT_SECONDARY};'>Symbol</th>
                        <th style='padding: 8px; text-align: right; font-size: 0.7rem; text-transform: uppercase; color: {Colors.TEXT_SECONDARY};'>Price</th>
                        <th style='padding: 8px; text-align: right; font-size: 0.7rem; text-transform: uppercase; color: {Colors.TEXT_SECONDARY};'>Change</th>
                        <th style='padding: 8px; text-align: center; font-size: 0.7rem; text-transform: uppercase; color: {Colors.TEXT_SECONDARY};'>Action</th>
                    </tr>
                </thead>
                <tbody>
        """

        for symbol in self.watchlist:
            try:
                real_time_data = self.stock_fetcher.get_real_time_price(symbol)
                stock_info = self.stock_fetcher.get_stock_info(symbol)

                price = real_time_data.get('current_price', 0) or 0
                prev_close = real_time_data.get('previous_close', 0) or 0
                volume = real_time_data.get('volume', 0) or 0
                name = stock_info.get('name', symbol)

                change = price - prev_close
                change_pct = (change / prev_close * 100) if prev_close else 0
                color = Colors.SUCCESS_GREEN if change >= 0 else Colors.DANGER_RED
                symbol_icon = '▲' if change >= 0 else '▼'

                table_html += f"""
                <tr style='border-top: 1px solid {Colors.BORDER_SUBTLE};'>
                    <td style='padding: 8px; font-weight: 600; color: {Colors.TEXT_PRIMARY}; font-size: 0.875rem;'>{symbol}</td>
                    <td style='padding: 8px; text-align: right; font-family: monospace; color: {Colors.TEXT_PRIMARY}; font-size: 0.875rem;'>${price:.2f}</td>
                    <td style='padding: 8px; text-align: right; color: {color}; font-weight: 600; font-size: 0.8rem;'>
                        {symbol_icon} {change_pct:+.2f}%
                    </td>
                    <td style='padding: 8px; text-align: center;'>
                        <button onclick='window.parent.postMessage({{type: "analyze", symbol: "{symbol}"}}, "*")'
                                style='background: {Colors.ACCENT_PURPLE};
                                       color: white;
                                       border: none;
                                       padding: 4px 10px;
                                       border-radius: 4px;
                                       cursor: pointer;
                                       font-size: 0.7rem;'>
                            Analyze
                        </button>
                    </td>
                </tr>
                """

            except Exception as e:
                logger.warning(f"Failed to fetch {symbol}: {e}")
                table_html += f"""
                <tr style='border-top: 1px solid {Colors.BORDER_SUBTLE};'>
                    <td style='padding: 8px; font-weight: 600; font-size: 0.875rem;'>{symbol}</td>
                    <td colspan='3' style='padding: 8px; color: {Colors.TEXT_MUTED}; font-style: italic; font-size: 0.8rem;'>Loading...</td>
                </tr>
                """

        table_html += """
                </tbody>
            </table>
        </div>
        """

        self.watchlist_panel.append(pn.pane.HTML(table_html))

    def _load_quick_actions(self):
        """Load quick action buttons - Compact version"""
        self.quick_actions_panel.clear()

        header_html = f"""
        <div style='padding: 10px 0; margin-bottom: 10px; border-bottom: 2px solid {Colors.BORDER_SUBTLE};'>
            <h3 style='margin: 0; font-size: 1.25rem; color: {Colors.TEXT_PRIMARY};'>⚡ Quick Actions</h3>
        </div>
        """
        self.quick_actions_panel.append(pn.pane.HTML(header_html))

        actions_html = f"""
        <div style='display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px;'>
            <div style='background: {Colors.BG_SECONDARY};
                        border: 1px solid {Colors.BORDER_SUBTLE};
                        border-radius: 8px;
                        padding: 15px;
                        text-align: center;
                        cursor: pointer;
                        transition: all 0.2s;'
                 onmouseover='this.style.background="{Colors.BG_HOVER}"'
                 onmouseout='this.style.background="{Colors.BG_SECONDARY}"'>
                <div style='font-size: 1.5rem; margin-bottom: 8px;'>🤖</div>
                <div style='font-weight: 600; color: {Colors.TEXT_PRIMARY}; font-size: 0.875rem; margin-bottom: 3px;'>Train LSTM</div>
                <div style='font-size: 0.7rem; color: {Colors.TEXT_SECONDARY};'>AI predictions</div>
            </div>

            <div style='background: {Colors.BG_SECONDARY};
                        border: 1px solid {Colors.BORDER_SUBTLE};
                        border-radius: 8px;
                        padding: 15px;
                        text-align: center;
                        cursor: pointer;
                        transition: all 0.2s;'
                 onmouseover='this.style.background="{Colors.BG_HOVER}"'
                 onmouseout='this.style.background="{Colors.BG_SECONDARY}"'>
                <div style='font-size: 1.5rem; margin-bottom: 8px;'>📊</div>
                <div style='font-weight: 600; color: {Colors.TEXT_PRIMARY}; font-size: 0.875rem; margin-bottom: 3px;'>Backtest</div>
                <div style='font-size: 0.7rem; color: {Colors.TEXT_SECONDARY};'>Test strategies</div>
            </div>

            <div style='background: {Colors.BG_SECONDARY};
                        border: 1px solid {Colors.BORDER_SUBTLE};
                        border-radius: 8px;
                        padding: 15px;
                        text-align: center;
                        cursor: pointer;
                        transition: all 0.2s;'
                 onmouseover='this.style.background="{Colors.BG_HOVER}"'
                 onmouseout='this.style.background="{Colors.BG_SECONDARY}"'>
                <div style='font-size: 1.5rem; margin-bottom: 8px;'>🔍</div>
                <div style='font-weight: 600; color: {Colors.TEXT_PRIMARY}; font-size: 0.875rem; margin-bottom: 3px;'>Compare</div>
                <div style='font-size: 0.7rem; color: {Colors.TEXT_SECONDARY};'>Side-by-side</div>
            </div>

            <div style='background: {Colors.BG_SECONDARY};
                        border: 1px solid {Colors.BORDER_SUBTLE};
                        border-radius: 8px;
                        padding: 15px;
                        text-align: center;
                        cursor: pointer;
                        transition: all 0.2s;'
                 onmouseover='this.style.background="{Colors.BG_HOVER}"'
                 onmouseout='this.style.background="{Colors.BG_SECONDARY}"'>
                <div style='font-size: 1.5rem; margin-bottom: 8px;'>📈</div>
                <div style='font-weight: 600; color: {Colors.TEXT_PRIMARY}; font-size: 0.875rem; margin-bottom: 3px;'>Report</div>
                <div style='font-size: 0.7rem; color: {Colors.TEXT_SECONDARY};'>AI-powered</div>
            </div>
        </div>
        """

        self.quick_actions_panel.append(pn.pane.HTML(actions_html))

    def _load_featured_charts(self):
        """Load mini charts for featured stocks"""
        self.featured_charts.clear()

        header_html = HTMLComponents.section_header("📈 Featured Stocks", "Top stocks from your watchlist")

        self.featured_charts.append(
            pn.Column(
                pn.pane.HTML(header_html),
                pn.pane.HTML(f"""
                    <div style='background: {Colors.BG_SECONDARY};
                                border: 1px solid {Colors.BORDER_SUBTLE};
                                border-radius: 8px;
                                padding: 20px;
                                text-align: center;'>
                        <p style='color: {Colors.TEXT_SECONDARY}; margin: 0;'>
                            📊 Mini charts coming soon!<br/>
                            <span style='font-size: 0.875rem;'>Showing 1-day candlestick charts for watchlist stocks</span>
                        </p>
                    </div>
                """),
                sizing_mode="stretch_width"
            )
        )

    def get_view(self):
        """Get the dashboard view - Wide horizontal layout"""
        # Left column: Market + Watchlist
        left_column = pn.Column(
            self.market_overview,
            self.watchlist_panel,
            sizing_mode="stretch_width",
            max_width=900
        )

        # Right column: Quick Actions + Featured
        right_column = pn.Column(
            self.quick_actions_panel,
            self.featured_charts,
            sizing_mode="stretch_width",
            max_width=600
        )

        # Main layout: Side-by-side with compact header
        return pn.Column(
            pn.Row(
                pn.pane.HTML(f"<h1 style='margin: 0; color: {Colors.TEXT_PRIMARY}; font-size: 1.75rem;'>📊 Dashboard</h1>"),
                pn.Spacer(),
                self.refresh_button,
                sizing_mode="stretch_width",
                styles=dict(background=Colors.BG_SECONDARY, border_radius='8px', padding='15px'),
                margin=(0, 0, 15, 0)
            ),
            pn.Row(
                left_column,
                right_column,
                sizing_mode="stretch_width"
            ),
            HTMLComponents.disclaimer(),
            sizing_mode="stretch_width"
        )
