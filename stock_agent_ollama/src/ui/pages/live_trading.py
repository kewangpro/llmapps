"""
Live Trading Dashboard Page

Real-time paper trading interface using trained RL models.
"""

import panel as pn
import param
from datetime import datetime
from pathlib import Path
import logging
from typing import Optional, Dict, Any

from src.rl.live_trading import (
    LiveTradingEngine,
    LiveTradingConfig,
    TradingStatus,
    TradingAction
)

from src.ui.design_system import Colors, HTMLComponents

logger = logging.getLogger(__name__)

pn.extension(notifications=True)


class LiveTradingPage(pn.viewable.Viewer):
    """Live trading dashboard page"""

    def __init__(self, **params):
        super().__init__(**params)
        self.engine: Optional[LiveTradingEngine] = None
        self.update_callback = None
        self._create_ui()

    def _find_latest_model(self, symbol: str, agent_type: str = None) -> Optional[Dict[str, Any]]:
        """Find the most recently trained model for a symbol and agent type."""
        models_dir = Path("data/models/rl")
        if not models_dir.exists():
            return None

        # Find all model directories for this symbol and agent type
        matching_dirs = []
        if agent_type:
            # Search for specific agent type
            pattern = f"{agent_type.lower()}_{symbol}_*"
            matching_dirs.extend(models_dir.glob(pattern))
        else:
            # Search for all agent types
            for atype in ['ppo', 'a2c']:
                pattern = f"{atype}_{symbol}_*"
                matching_dirs.extend(models_dir.glob(pattern))

        if not matching_dirs:
            return None

        # Sort by modification time (most recent first)
        matching_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)

        # Check if best_model.zip or final_model.zip exists in the most recent directory
        latest_dir = matching_dirs[0]
        model_path = latest_dir / "best_model.zip"

        if not model_path.exists():
            model_path = latest_dir / "final_model.zip"

        if not model_path.exists():
            return None

        # Extract agent type from directory name
        dir_name = latest_dir.name
        found_agent_type = 'ppo' if dir_name.startswith('ppo_') else 'a2c'

        return {
            'path': model_path,
            'agent_type': found_agent_type,
            'directory': latest_dir
        }

    def _create_ui(self):
        """Create the dashboard UI"""

        # Configuration panel
        self.symbol_input = pn.widgets.AutocompleteInput(
            name='',
            value='AAPL',
            options=['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'NVDA', 'META', 'TEAM',
                     'NFLX', 'AMD', 'INTC', 'QCOM', 'CRM', 'ADBE', 'PYPL'],
            placeholder='Enter symbol...',
            case_sensitive=False,
            width=120,
            height=35,
            min_characters=1
        )

        # Algorithm selection (same as training page)
        self.agent_type = pn.widgets.RadioButtonGroup(
            name='',
            options=['PPO', 'A2C'],
            value='PPO',
            button_type='primary',
            button_style='outline'
        )

        # Model status display (will show auto-discovered model)
        self.model_status_pane = pn.pane.HTML(
            "",
            sizing_mode="stretch_width"
        )

        self.capital_input = pn.widgets.FloatInput(
            name='',
            value=10000.0,
            start=1000,
            end=1000000,
            step=1000,
            width=180
        )

        self.max_position_input = pn.widgets.IntInput(
            name='',
            value=100,
            start=1,
            end=1000,
            step=10,
            width=180
        )

        self.stop_loss_input = pn.widgets.FloatInput(
            name='',
            value=5.0,
            start=1.0,
            end=20.0,
            step=0.5,
            width=180
        )

        # Control buttons
        self.start_button = pn.widgets.Button(
            name='▶ Start Trading',
            button_type='success',
            width=150,
            height=40
        )
        self.start_button.on_click(self._start_trading)

        self.stop_button = pn.widgets.Button(
            name='⏹ Stop Trading',
            button_type='danger',
            width=150,
            height=40,
            disabled=True
        )
        self.stop_button.on_click(self._stop_trading)

        self.pause_button = pn.widgets.Button(
            name='⏸ Pause',
            button_type='warning',
            width=150,
            height=40,
            disabled=True
        )

        # Status indicators
        self.status_pane = pn.pane.HTML("", sizing_mode="stretch_width")
        self.portfolio_pane = pn.pane.HTML("", sizing_mode="stretch_width")
        self.positions_pane = pn.pane.HTML("", sizing_mode="stretch_width")
        self.trades_pane = pn.pane.HTML("", sizing_mode="stretch_width")
        self.events_pane = pn.pane.HTML("", sizing_mode="stretch_width")

        # Initial display
        self._update_display()

    def _start_trading(self, event):
        """Start live trading session"""
        try:
            # Validate inputs
            symbol = self.symbol_input.value.strip().upper()
            if not symbol:
                pn.state.notifications.error("Please enter a stock symbol", duration=3000)
                return

            # Get selected agent type
            agent_type = self.agent_type.value

            # Find latest model for this symbol and agent type
            model_info = self._find_latest_model(symbol, agent_type)
            if not model_info:
                pn.state.notifications.error(
                    f"No trained {agent_type} model found for {symbol}. Please train a model first.",
                    duration=5000
                )
                return

            # Update model status display
            model_name = model_info['directory'].name
            self.model_status_pane.object = f"""
            <div style='padding: 10px; background: #D1FAE5; border-radius: 4px; font-size: 12px; color: #065F46; border: 1px solid #A7F3D0;'>
                ✅ Using model: <strong>{model_info['agent_type'].upper()}</strong> - {model_name}
            </div>
            """

            # Create configuration
            config = LiveTradingConfig(
                symbol=symbol,
                agent_path=str(model_info['path']),
                initial_capital=self.capital_input.value,
                max_position_size=self.max_position_input.value,
                stop_loss_pct=self.stop_loss_input.value,
                update_interval=60
            )

            # Create engine
            self.engine = LiveTradingEngine(config)

            # Load agent
            try:
                self.engine.load_agent(config.agent_path)
                pn.state.notifications.success(
                    f"Loaded {model_info['agent_type'].upper()} model for {symbol}",
                    duration=3000
                )
            except Exception as e:
                pn.state.notifications.error(f"Failed to load agent: {str(e)}", duration=5000)
                return

            # Start session
            self.engine.start_session()

            # Update UI
            self.start_button.disabled = True
            self.stop_button.disabled = False
            self.pause_button.disabled = False

            # Populate results container
            self.results_container.clear()
            self.results_container.extend([
                pn.Row(
                    pn.Column(self.status_pane, sizing_mode="stretch_width"),
                    pn.Column(self.portfolio_pane, sizing_mode="stretch_width"),
                    sizing_mode="stretch_width"
                ),
                self.positions_pane,
                pn.Row(
                    pn.Column(self.trades_pane, sizing_mode="stretch_width"),
                    pn.Column(self.events_pane, sizing_mode="stretch_width"),
                    sizing_mode="stretch_width"
                )
            ])

            # Update display to show session data
            self._update_display()

            # Execute first trading cycle immediately
            try:
                result = self.engine.trading_cycle()
                self._update_display()

                if result.get('status') == 'trade_executed':
                    trade = result['trade']
                    pn.state.notifications.success(
                        f"First trade: {trade.action.name} {trade.shares} shares @ ${trade.price:.2f}",
                        duration=4000
                    )
            except Exception as e:
                logger.error(f"Error in first trading cycle: {e}")

            # Start periodic updates
            self.update_callback = pn.state.add_periodic_callback(
                self._trading_update,
                period=config.update_interval * 1000  # Convert to milliseconds
            )

            pn.state.notifications.success(f"Started live trading for {symbol}", duration=3000)

        except Exception as e:
            logger.error(f"Error starting trading: {e}")
            pn.state.notifications.error(f"Failed to start trading: {str(e)}", duration=5000)

    def _stop_trading(self, event):
        """Stop live trading session"""
        if self.engine:
            self.engine.stop_session()


        # Stop updates
        if self.update_callback:
            self.update_callback.stop()
            self.update_callback = None

        # Update UI
        self.start_button.disabled = False
        self.stop_button.disabled = True
        self.pause_button.disabled = True

        # Clear results container
        self.results_container.clear()

        self._update_display()

        pn.state.notifications.info("Trading session stopped", duration=3000)

    def _trading_update(self):
        """Periodic trading cycle update"""
        if self.engine and self.engine._is_running:
            try:
                result = self.engine.trading_cycle()
                self._update_display()

                # Handle different result statuses
                if result.get('status') == 'trade_executed':
                    trade = result['trade']
                    pn.state.notifications.info(
                        f"{trade.action.name}: {trade.shares} shares @ ${trade.price:.2f}",
                        duration=3000
                    )
                elif result.get('status') == 'halted':
                    pn.state.notifications.error(
                        f"Trading halted: {result.get('reason')}",
                        duration=5000
                    )
                    self._stop_trading(None)

            except Exception as e:
                logger.error(f"Error in trading update: {e}")

    def _update_display(self):
        """Update all display panels"""
        self._update_status()
        self._update_portfolio()
        self._update_positions()
        self._update_trades()
        self._update_events()

    def _update_status(self):
        """Update status panel"""
        if not self.engine or not self.engine.session:
            html = f"""
            <div style='background: {Colors.BG_SECONDARY}; padding: 20px; border-radius: 8px; border-left: 4px solid {Colors.BORDER_SUBTLE};'>
                <h3 style='margin-top: 0; color: {Colors.TEXT_PRIMARY};'>Trading Status</h3>
                <p style='color: {Colors.TEXT_SECONDARY}; font-size: 16px;'>
                    <span style='color: {Colors.TEXT_MUTED};'>●</span> Not Running
                </p>
                <p style='color: {Colors.TEXT_MUTED}; font-size: 14px; margin-bottom: 0;'>
                    Configure settings and click "Start Trading" to begin.
                </p>
            </div>
            """
        else:
            session = self.engine.session
            status_color = {
                TradingStatus.RUNNING: Colors.SUCCESS_GREEN,
                TradingStatus.PAUSED: Colors.WARNING_YELLOW,
                TradingStatus.STOPPED: Colors.TEXT_MUTED,
                TradingStatus.HALTED: Colors.DANGER_RED
            }.get(session.status, Colors.TEXT_MUTED)

            elapsed = ""
            if session.start_time:
                duration = datetime.now() - session.start_time
                hours, remainder = divmod(duration.seconds, 3600)
                minutes, seconds = divmod(remainder, 60)
                elapsed = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

            html = f"""
            <div style='background: {Colors.BG_SECONDARY}; padding: 20px; border-radius: 8px; border-left: 4px solid {status_color};'>
                <h3 style='margin-top: 0; color: {Colors.TEXT_PRIMARY};'>Trading Status</h3>
                <div style='display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;'>
                    <div>
                        <p style='color: {Colors.TEXT_MUTED}; font-size: 12px; margin-bottom: 5px;'>Status</p>
                        <p style='color: {status_color}; font-size: 18px; font-weight: 600; margin: 0;'>
                            ● {session.status.value.upper()}
                        </p>
                    </div>
                    <div>
                        <p style='color: {Colors.TEXT_MUTED}; font-size: 12px; margin-bottom: 5px;'>Symbol</p>
                        <p style='color: {Colors.TEXT_PRIMARY}; font-size: 18px; font-weight: 600; margin: 0;'>
                            {self.engine.config.symbol}
                        </p>
                    </div>
                    <div>
                        <p style='color: {Colors.TEXT_MUTED}; font-size: 12px; margin-bottom: 5px;'>Session ID</p>
                        <p style='color: {Colors.TEXT_SECONDARY}; font-size: 14px; margin: 0;'>
                            {session.session_id}
                        </p>
                    </div>
                    <div>
                        <p style='color: {Colors.TEXT_MUTED}; font-size: 12px; margin-bottom: 5px;'>Running Time</p>
                        <p style='color: {Colors.TEXT_SECONDARY}; font-size: 14px; margin: 0;'>
                            {elapsed}
                        </p>
                    </div>
                </div>
            </div>
            """

        self.status_pane.object = html

    def _update_portfolio(self):
        """Update portfolio summary"""
        if not self.engine:
            html = f"""
            <div style='background: {Colors.BG_SECONDARY}; padding: 20px; border-radius: 8px;'>
                <h3 style='margin-top: 0; color: {Colors.TEXT_PRIMARY};'>Portfolio Summary</h3>
                <p style='color: {Colors.TEXT_MUTED};'>No active session</p>
            </div>
            """
        else:
            portfolio = self.engine.portfolio
            pnl_color = Colors.SUCCESS_GREEN if portfolio.total_pnl >= 0 else Colors.DANGER_RED
            pnl_symbol = "▲" if portfolio.total_pnl >= 0 else "▼"

            html = f"""
            <div style='background: {Colors.BG_SECONDARY}; padding: 20px; border-radius: 8px;'>
                <h3 style='margin-top: 0; color: {Colors.TEXT_PRIMARY};'>Portfolio Summary</h3>
                <div style='display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px;'>
                    <div>
                        <p style='color: {Colors.TEXT_MUTED}; font-size: 12px; margin-bottom: 5px;'>Total Value</p>
                        <p style='color: {Colors.TEXT_PRIMARY}; font-size: 24px; font-weight: 600; margin: 0;'>
                            ${portfolio.total_value:,.2f}
                        </p>
                    </div>
                    <div>
                        <p style='color: {Colors.TEXT_MUTED}; font-size: 12px; margin-bottom: 5px;'>Total P&L</p>
                        <p style='color: {pnl_color}; font-size: 24px; font-weight: 600; margin: 0;'>
                            {pnl_symbol} ${abs(portfolio.total_pnl):,.2f} ({portfolio.total_pnl_pct:+.2f}%)
                        </p>
                    </div>
                    <div>
                        <p style='color: {Colors.TEXT_MUTED}; font-size: 12px; margin-bottom: 5px;'>Cash</p>
                        <p style='color: {Colors.TEXT_SECONDARY}; font-size: 18px; margin: 0;'>
                            ${portfolio.cash:,.2f}
                        </p>
                    </div>
                    <div>
                        <p style='color: {Colors.TEXT_MUTED}; font-size: 12px; margin-bottom: 5px;'>Initial Capital</p>
                        <p style='color: {Colors.TEXT_SECONDARY}; font-size: 18px; margin: 0;'>
                            ${portfolio.initial_cash:,.2f}
                        </p>
                    </div>
                </div>
            </div>
            """

        self.portfolio_pane.object = html

    def _update_positions(self):
        """Update current positions"""
        if not self.engine or not self.engine.portfolio.positions:
            html = f"""
            <div style='background: {Colors.BG_SECONDARY}; padding: 20px; border-radius: 8px;'>
                <h3 style='margin-top: 0; color: {Colors.TEXT_PRIMARY};'>Current Positions</h3>
                <p style='color: {Colors.TEXT_MUTED}; text-align: center; padding: 20px;'>No open positions</p>
            </div>
            """
        else:
            rows = ""
            for symbol, pos in self.engine.portfolio.positions.items():
                pnl_color = Colors.SUCCESS_GREEN if pos.unrealized_pnl >= 0 else Colors.DANGER_RED
                pnl_symbol = "▲" if pos.unrealized_pnl >= 0 else "▼"
                pnl_pct = (pos.current_price - pos.avg_entry_price) / pos.avg_entry_price * 100

                rows += f"""
                <tr style='border-bottom: 1px solid {Colors.BORDER_SUBTLE};'>
                    <td style='padding: 12px; color: {Colors.TEXT_PRIMARY}; font-weight: 600;'>{symbol}</td>
                    <td style='padding: 12px; color: {Colors.TEXT_SECONDARY}; text-align: right;'>{pos.shares}</td>
                    <td style='padding: 12px; color: {Colors.TEXT_SECONDARY}; text-align: right;'>${pos.avg_entry_price:.2f}</td>
                    <td style='padding: 12px; color: {Colors.TEXT_SECONDARY}; text-align: right;'>${pos.current_price:.2f}</td>
                    <td style='padding: 12px; color: {pnl_color}; text-align: right; font-weight: 600;'>
                        {pnl_symbol} ${abs(pos.unrealized_pnl):.2f} ({pnl_pct:+.2f}%)
                    </td>
                </tr>
                """

            html = f"""
            <div style='background: {Colors.BG_SECONDARY}; padding: 20px; border-radius: 8px;'>
                <h3 style='margin-top: 0; color: {Colors.TEXT_PRIMARY};'>Current Positions</h3>
                <table style='width: 100%; border-collapse: collapse;'>
                    <thead>
                        <tr style='border-bottom: 2px solid {Colors.BORDER_SUBTLE};'>
                            <th style='padding: 12px; color: {Colors.TEXT_MUTED}; font-size: 12px; text-align: left;'>SYMBOL</th>
                            <th style='padding: 12px; color: {Colors.TEXT_MUTED}; font-size: 12px; text-align: right;'>SHARES</th>
                            <th style='padding: 12px; color: {Colors.TEXT_MUTED}; font-size: 12px; text-align: right;'>AVG ENTRY</th>
                            <th style='padding: 12px; color: {Colors.TEXT_MUTED}; font-size: 12px; text-align: right;'>CURRENT</th>
                            <th style='padding: 12px; color: {Colors.TEXT_MUTED}; font-size: 12px; text-align: right;'>UNREALIZED P&L</th>
                        </tr>
                    </thead>
                    <tbody>
                        {rows}
                    </tbody>
                </table>
            </div>
            """

        self.positions_pane.object = html

    def _update_trades(self):
        """Update recent trades"""
        if not self.engine or not self.engine.portfolio.trades:
            html = f"""
            <div style='background: {Colors.BG_SECONDARY}; padding: 20px; border-radius: 8px;'>
                <h3 style='margin-top: 0; color: {Colors.TEXT_PRIMARY};'>Recent Trades</h3>
                <p style='color: {Colors.TEXT_MUTED}; text-align: center; padding: 20px;'>No trades yet</p>
            </div>
            """
        else:
            rows = ""
            # Show last 10 trades
            for trade in reversed(self.engine.portfolio.trades[-10:]):
                action_color = Colors.SUCCESS_GREEN if trade.action == TradingAction.BUY else Colors.DANGER_RED
                pnl_color = Colors.SUCCESS_GREEN if trade.pnl >= 0 else Colors.DANGER_RED
                time_str = trade.timestamp.strftime('%H:%M:%S')

                rows += f"""
                <tr style='border-bottom: 1px solid {Colors.BORDER_SUBTLE};'>
                    <td style='padding: 10px; color: {Colors.TEXT_MUTED}; font-size: 12px;'>{time_str}</td>
                    <td style='padding: 10px; color: {action_color}; font-weight: 600;'>{trade.action.name}</td>
                    <td style='padding: 10px; color: {Colors.TEXT_PRIMARY};'>{trade.symbol}</td>
                    <td style='padding: 10px; color: {Colors.TEXT_SECONDARY}; text-align: right;'>{trade.shares}</td>
                    <td style='padding: 10px; color: {Colors.TEXT_SECONDARY}; text-align: right;'>${trade.price:.2f}</td>
                    <td style='padding: 10px; color: {pnl_color}; text-align: right; font-weight: 600;'>
                        ${trade.pnl:.2f}
                    </td>
                </tr>
                """

            html = f"""
            <div style='background: {Colors.BG_SECONDARY}; padding: 20px; border-radius: 8px;'>
                <h3 style='margin-top: 0; color: {Colors.TEXT_PRIMARY};'>Recent Trades</h3>
                <div style='max-height: 400px; overflow-y: auto;'>
                    <table style='width: 100%; border-collapse: collapse;'>
                        <thead style='position: sticky; top: 0; background: {Colors.BG_SECONDARY};'>
                            <tr style='border-bottom: 2px solid {Colors.BORDER_SUBTLE};'>
                                <th style='padding: 10px; color: {Colors.TEXT_MUTED}; font-size: 11px; text-align: left;'>TIME</th>
                                <th style='padding: 10px; color: {Colors.TEXT_MUTED}; font-size: 11px; text-align: left;'>ACTION</th>
                                <th style='padding: 10px; color: {Colors.TEXT_MUTED}; font-size: 11px; text-align: left;'>SYMBOL</th>
                                <th style='padding: 10px; color: {Colors.TEXT_MUTED}; font-size: 11px; text-align: right;'>SHARES</th>
                                <th style='padding: 10px; color: {Colors.TEXT_MUTED}; font-size: 11px; text-align: right;'>PRICE</th>
                                <th style='padding: 10px; color: {Colors.TEXT_MUTED}; font-size: 11px; text-align: right;'>P&L</th>
                            </tr>
                        </thead>
                        <tbody>
                            {rows}
                        </tbody>
                    </table>
                </div>
            </div>
            """

        self.trades_pane.object = html

    def _update_events(self):
        """Update event log"""
        if not self.engine or not self.engine.session or not self.engine.session.events:
            html = f"""
            <div style='background: {Colors.BG_SECONDARY}; padding: 20px; border-radius: 8px;'>
                <h3 style='margin-top: 0; color: {Colors.TEXT_PRIMARY};'>Event Log</h3>
                <p style='color: {Colors.TEXT_MUTED}; text-align: center; padding: 20px;'>No events</p>
            </div>
            """
        else:
            rows = ""
            # Show last 20 events
            for event in reversed(self.engine.session.events[-20:]):
                time_str = event['timestamp'].strftime('%H:%M:%S')
                event_type = event['type']

                type_color = {
                    'SESSION_START': Colors.SUCCESS_GREEN,
                    'SESSION_END': Colors.TEXT_MUTED,
                    'TRADE': Colors.INFO_BLUE,
                    'ORDER_REJECTED': Colors.WARNING_YELLOW,
                    'HALT': Colors.DANGER_RED,
                    'ERROR': Colors.DANGER_RED
                }.get(event_type, Colors.TEXT_SECONDARY)

                rows += f"""
                <div style='padding: 8px; border-bottom: 1px solid {Colors.BORDER_SUBTLE};'>
                    <span style='color: {Colors.TEXT_MUTED}; font-size: 11px;'>{time_str}</span>
                    <span style='color: {type_color}; font-weight: 600; margin: 0 10px;'>[{event_type}]</span>
                    <span style='color: {Colors.TEXT_SECONDARY}; font-size: 13px;'>{event['message']}</span>
                </div>
                """

            html = f"""
            <div style='background: {Colors.BG_SECONDARY}; padding: 20px; border-radius: 8px;'>
                <h3 style='margin-top: 0; color: {Colors.TEXT_PRIMARY};'>Event Log</h3>
                <div style='max-height: 300px; overflow-y: auto; font-family: monospace;'>
                    {rows}
                </div>
            </div>
            """

        self.events_pane.object = html

    def __panel__(self):
        """Return the Panel layout"""

        # Configuration section (matching training page style)
        config_section = pn.Column(
            pn.Row(
                pn.Column(
                    pn.pane.HTML("<div style='font-size: 12px; color: #6b7280; margin-bottom: 5px; font-weight: 500;'>Symbol</div>"),
                    self.symbol_input,
                    width=120,
                    margin=(0, 5, 0, 0)
                ),
                pn.Column(
                    pn.pane.HTML("<div style='font-size: 12px; color: #6b7280; margin-bottom: 5px; font-weight: 500;'>Algorithm</div>"),
                    self.agent_type,
                    width=200,
                    margin=(0, 5, 0, 0)
                ),
                pn.Column(
                    pn.pane.HTML("<div style='font-size: 12px; color: #6b7280; margin-bottom: 5px; font-weight: 500;'>Initial Capital ($)</div>"),
                    self.capital_input,
                    width=180,
                    margin=(0, 5, 0, 0)
                ),
                pn.Column(
                    pn.pane.HTML("<div style='font-size: 12px; color: #6b7280; margin-bottom: 5px; font-weight: 500;'>Max Position Size</div>"),
                    self.max_position_input,
                    width=180,
                    margin=(0, 5, 0, 0)
                ),
                pn.Column(
                    pn.pane.HTML("<div style='font-size: 12px; color: #6b7280; margin-bottom: 5px; font-weight: 500;'>Stop Loss (%)</div>"),
                    self.stop_loss_input,
                    width=180
                ),
                align='start'
            ),
            pn.Row(
                self.start_button,
                self.stop_button
            ),
            self.model_status_pane,
            styles=dict(background='#F8F9FA', border_radius='8px', padding='15px'),
            margin=(0, 0, 15, 0)
        )

        # Results panel (always visible, just empty initially like other pages)
        self.results_container = pn.Column(
            sizing_mode="stretch_width",
            min_height=250
        )

        # Main layout - disclaimer at bottom just like training page
        return pn.Column(
            config_section,
            self.results_container,
            pn.pane.HTML(HTMLComponents.disclaimer()),
            sizing_mode="stretch_width"
        )


def create_live_trading_page():
    """Factory function to create live trading page"""
    return LiveTradingPage()
