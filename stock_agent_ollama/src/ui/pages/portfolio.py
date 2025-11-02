"""
Portfolio Page - Holdings and Performance Tracking
"""

import panel as pn
import param
import logging

from src.ui.design_system import Colors, HTMLComponents

logger = logging.getLogger(__name__)


class PortfolioPage(param.Parameterized):
    """Portfolio holdings and performance tracking"""

    def __init__(self, **params):
        super().__init__(**params)
        self._create_ui()

    def _create_ui(self):
        """Create portfolio UI"""
        # Coming soon message
        self.content = pn.pane.HTML(f"""
            <div style='background: {Colors.BG_SECONDARY};
                        border: 1px solid {Colors.BORDER_SUBTLE};
                        border-radius: 10px;
                        padding: 60px;
                        text-align: center;
                        min-height: 400px;
                        display: flex;
                        flex-direction: column;
                        justify-content: center;
                        align-items: center;'>
                <div style='font-size: 4rem; margin-bottom: 20px;'>💼</div>
                <h2 style='color: {Colors.TEXT_PRIMARY}; margin-bottom: 15px;'>Portfolio Management</h2>
                <p style='color: {Colors.TEXT_SECONDARY}; font-size: 1.125rem; margin-bottom: 30px; max-width: 600px;'>
                    Track your holdings, monitor performance, and analyze your portfolio allocation.
                </p>
                <div style='background: {Colors.BG_TERTIARY};
                            border: 1px solid {Colors.BORDER_SUBTLE};
                            border-radius: 8px;
                            padding: 20px;
                            max-width: 500px;'>
                    <h3 style='color: {Colors.TEXT_PRIMARY}; margin-top: 0;'>Coming Soon</h3>
                    <ul style='text-align: left; color: {Colors.TEXT_SECONDARY}; line-height: 1.8;'>
                        <li>Holdings tracking with P&L</li>
                        <li>Portfolio allocation charts</li>
                        <li>Performance vs benchmarks</li>
                        <li>Risk metrics dashboard</li>
                        <li>Transaction history</li>
                    </ul>
                </div>
            </div>
        """, sizing_mode="stretch_width")

    def get_view(self):
        """Get the portfolio view"""
        return pn.Column(
            HTMLComponents.page_header(
                "Portfolio",
                "Track your holdings and performance"
            ),
            self.content,
            HTMLComponents.disclaimer(),
            sizing_mode="stretch_width"
        )
