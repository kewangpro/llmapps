"""
Live Trading Dashboard Page

Real-time paper trading interface using trained RL models.
Enhanced with multi-session support.
"""

import panel as pn
import param
from datetime import datetime
from pathlib import Path
import logging
import json
from typing import Optional, Dict, Any, List

from src.rl.session_manager import LiveSessionManager
from src.rl.live_trading import (
    LiveTradingEngine,
    LiveTradingConfig,
    TradingStatus,
    TradingAction
)

from src.ui.design_system import Colors, HTMLComponents, Typography

logger = logging.getLogger(__name__)

pn.extension(notifications=True)


class LiveTradingPage(pn.viewable.Viewer):
    """Live trading dashboard page with multi-session support"""

    def __init__(self, session_manager: Optional[LiveSessionManager] = None, **params):
        super().__init__(**params)
        self.session_manager = session_manager or LiveSessionManager()
        self.update_callback = None

        self._create_ui()
        # Defer initial render to avoid loading during construction
        pn.state.onload(lambda: self._refresh_view())
        pn.state.location.param.watch(self._on_location_change, 'search')

        # Register cleanup on session destroy
        pn.state.on_session_destroyed(self._cleanup)

    def _cleanup(self, session_context):
        """Cleanup resources when the page is destroyed"""
        if self.update_callback:
            try:
                pn.state.remove_periodic_callback(self.update_callback)
                logger.info("Stopped periodic callback for live trading page")
            except Exception as e:
                logger.error(f"Error stopping periodic callback: {e}")

    def _on_location_change(self, *events):
        """Handle URL query parameter changes."""
        session_id = pn.state.location.query_params.get('session_id')
        if session_id and self.session_manager.active_session_id != session_id:
            if self.session_manager.get_session(session_id):
                self._select_session(session_id)
            else:
                logger.warning(f"Session ID from URL not found: {session_id}")

    def _on_session_row_click(self, event):
        """Handle row click event from the session table."""
        if event.new:
            selected_index = event.new[0]
            if selected_index < len(self.session_manager.get_session_summary()):
                session_id = self.session_manager.get_session_summary()[selected_index]['session_id']
                self._select_session(session_id)

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

            # Also search for LSTM PPO models when looking for PPO
            if agent_type.lower() == 'ppo':
                lstm_pattern = f"lstm_ppo_{symbol}_*"
                matching_dirs.extend(models_dir.glob(lstm_pattern))
        else:
            # Search for all agent types
            for atype in ['ppo', 'a2c', 'dqn']:
                pattern = f"{atype}_{symbol}_*"
                matching_dirs.extend(models_dir.glob(pattern))
            # Also include LSTM PPO models
            lstm_pattern = f"lstm_ppo_{symbol}_*"
            matching_dirs.extend(models_dir.glob(lstm_pattern))

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

        # Extract agent type from directory name (format: ppo_AAPL_timestamp or lstm_ppo_AAPL_timestamp)
        dir_name = latest_dir.name
        if dir_name.startswith('lstm_ppo_'):
            found_agent_type = 'ppo'  # LSTM PPO loads as PPO (RecurrentPPO)
            is_lstm = True
        elif dir_name.startswith('ppo_'):
            found_agent_type = 'ppo'
            is_lstm = False
        elif dir_name.startswith('a2c_'):
            found_agent_type = 'a2c'
            is_lstm = False
        elif dir_name.startswith('dqn_'):
            found_agent_type = 'dqn'
            is_lstm = False
        else:
            found_agent_type = 'unknown'
            is_lstm = False

        return {
            'path': model_path,
            'agent_type': found_agent_type,
            'directory': latest_dir,
            'is_lstm': is_lstm
        }

    def _create_ui(self):
        """Create the dashboard UI"""
        # New session form (collapsible)
        self._create_new_session_form()

        # Content area (will be populated based on view mode)
        self.content_area = pn.Column(sizing_mode="stretch_width")

        # Store references to updateable components
        self.status_card_pane = None
        self.portfolio_card_pane = None
        self.positions_card_pane = None
        self.trades_card_pane = None
        self.events_card_pane = None
        self.dashboard_card_pane = None

        # Start periodic refresh for real-time updates (but don't clear and rebuild everything)
        self.update_callback = pn.state.add_periodic_callback(
            self._update_session_data,
            period=5000  # Refresh every 5 seconds
        )

    def _create_new_session_form(self):
        """Create form for new session creation"""
        # Session configuration inputs
        self.symbol_input = pn.widgets.AutocompleteInput(
            value='AAPL',
            options=['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'NVDA', 'META', 'TEAM',
                     'NFLX', 'AMD', 'INTC', 'QCOM', 'CRM', 'ADBE', 'PYPL'],
            placeholder='Enter symbol...',
            case_sensitive=False,
            width=100,
            min_characters=1
        )

        self.agent_type = pn.widgets.RadioButtonGroup(
            options=['PPO', 'A2C', 'DQN'],
            value='PPO',
            button_type='primary',
            button_style='outline'
        )

        self.capital_input = pn.widgets.FloatInput(
            value=100000.0,
            start=1000,
            end=1000000,
            step=1000,
            width=120
        )

        self.max_position_input = pn.widgets.FloatInput(
            value=40.0,
            start=5.0,
            end=100.0,
            step=5.0,
            width=120
        )

        self.stop_loss_input = pn.widgets.FloatInput(
            value=5.0,
            start=1.0,
            end=20.0,
            step=0.5,
            width=120
        )

        self.allow_extended_hours_input = pn.widgets.Checkbox(
            name='Allow Extended Hours',
            value=False
        )

        # Create button
        self.create_session_btn = pn.widgets.Button(
            name='Create & Start Session',
            button_type='success',
            icon='plus',
            width=200
        )
        self.create_session_btn.on_click(self._create_new_session)

        # Form layout
        self.new_session_form = pn.Column(
            pn.pane.HTML(f"""
                <h3 style="margin: 0 0 16px 0; color: {Colors.TEXT_PRIMARY}; font-family: {Typography.FONT_PRIMARY};">
                    Create New Session
                </h3>
            """),
            pn.Row(
                pn.Column(
                    pn.pane.HTML("<div style='font-size: 12px; color: #6b7280; margin-bottom: 5px; font-weight: 500;'>Symbol</div>"),
                    self.symbol_input,
                ),
                pn.Column(
                    pn.pane.HTML("<div style='font-size: 12px; color: #6b7280; margin-bottom: 5px; font-weight: 500;'>Algorithm</div>"),
                    self.agent_type,
                ),
                pn.Column(
                    pn.pane.HTML("<div style='font-size: 12px; color: #6b7280; margin-bottom: 5px; font-weight: 500;'>Capital ($)</div>"),
                    self.capital_input,
                ),
                pn.Column(
                    pn.pane.HTML("<div style='font-size: 12px; color: #6b7280; margin-bottom: 5px; font-weight: 500;'>Max Position %</div>"),
                    self.max_position_input,
                ),
                pn.Column(
                    pn.pane.HTML("<div style='font-size: 12px; color: #6b7280; margin-bottom: 5px; font-weight: 500;'>Stop Loss (%)</div>"),
                    self.stop_loss_input,
                ),
                pn.Column(
                    self.allow_extended_hours_input,
                    margin=(20, 10, 0, 10)
                ),
                align='end',
                sizing_mode='stretch_width'
            ),
            pn.Row(
                self.create_session_btn,
                margin=(10, 0, 0, 0)
            ),
            styles=dict(
                background=Colors.BG_SECONDARY,
                border_radius='8px',
                padding='20px',
                margin='0 0 20px 0'
            )
        )

    def _create_new_session(self, event):
        """Create and start new trading session"""
        symbol = self.symbol_input.value.strip().upper()
        agent_type = self.agent_type.value

        if not symbol:
            pn.state.notifications.error("Please enter a symbol", duration=3000)
            return

        # Find model
        model_info = self._find_latest_model(symbol, agent_type)
        if not model_info:
            pn.state.notifications.error(
                f"No trained {agent_type} model found for {symbol}",
                duration=5000
            )
            return

        try:
            # Create config
            config = LiveTradingConfig(
                symbol=symbol,
                agent_path=str(model_info['path']),
                initial_capital=self.capital_input.value,
                max_position_size=self.max_position_input.value,
                stop_loss_pct=self.stop_loss_input.value,
                update_interval=60,
                allow_extended_hours=self.allow_extended_hours_input.value
            )

            # Create session via manager (session_id is auto-generated)
            session_id = self.session_manager.create_session(
                config=config,
                strategy_name=f"{symbol} {agent_type}",  # Auto-generated name
            )

            # Start the newly created session
            self.session_manager.start_session(session_id)

            pn.state.notifications.success(
                f"Created and started session: {session_id}",
                duration=3000
            )

            # Set as active
            self.session_manager.set_active_session(session_id)

            # Refresh view
            pn.state.execute(self._refresh_view)

        except Exception as e:
            logger.error(f"Error creating session: {e}")
            pn.state.notifications.error(f"Failed to create session: {str(e)}", duration=5000)

    def _select_session(self, session_id: str):
        """Select a session to view its details"""
        self.session_manager.set_active_session(session_id)
        # Only update session details, don't rebuild dashboard
        self._update_session_details_only()

    def _start_session(self, session_id: str):
        """Handle start session"""
        try:
            success = self.session_manager.start_session(session_id)
            if success:
                pn.state.notifications.success(f"Started session", duration=2000)
                self.session_manager.set_active_session(session_id)
                # Smart update: only refresh dashboard and session details
                self._update_dashboard_and_session()
            else:
                pn.state.notifications.error(f"Failed to start session", duration=3000)
        except Exception as e:
            logger.error(f"Error in start session {session_id}: {e}")
            pn.state.notifications.error(f"Error: {str(e)}", duration=3000)

    def _stop_session(self, session_id: str):
        """Handle stop session"""
        self.session_manager.stop_session(session_id)
        pn.state.notifications.warning(f"Stopped session", duration=2000)
        # Smart update: only refresh dashboard and session details
        self._update_dashboard_and_session()

    def _update_dashboard_and_session(self):
        """Update dashboard table and session details without full page refresh"""
        # Update dashboard card (includes table with updated status/buttons)
        if self.dashboard_card_pane:
            metrics = self.session_manager.get_aggregate_metrics()
            summaries = self.session_manager.get_session_summary()
            new_dashboard = self._create_dashboard_card(metrics, summaries)

            # Replace dashboard card in place
            self.dashboard_card_pane.objects = new_dashboard.objects

        # Update session details in place
        self._update_session_details_only()

    def _update_session_details_only(self):
        """Update only the session details section, keeping dashboard unchanged"""
        active_session = self.session_manager.get_active_session()

        # If we don't have card references yet, need to do initial render
        if not self.status_card_pane:
            # Remove old session details if they exist
            if len(self.content_area.objects) > 2:
                self.content_area.objects = self.content_area.objects[:2]
            # Render session details for first time
            self._render_single_session_view()
            return

        # Update existing cards in place (no DOM removal/re-add, no scroll)
        if not active_session:
            # Clear card references
            self.status_card_pane = None
            self.portfolio_card_pane = None
            self.positions_card_pane = None
            self.trades_card_pane = None
            self.events_card_pane = None
            # Remove session details, show "No Active Session"
            if len(self.content_area.objects) > 2:
                self.content_area.objects = self.content_area.objects[:2]
            self._render_single_session_view()
            return

        # Update each card in place without removing them from DOM
        try:
            self.status_card_pane.object = self._create_status_card(active_session).object
            self.portfolio_card_pane.object = self._create_portfolio_card(active_session).object
            self.positions_card_pane.object = self._create_positions_card(active_session).object
            self.trades_card_pane.object = self._create_trades_card(active_session).object
            self.events_card_pane.object = self._create_events_card(active_session).object
        except Exception as e:
            logger.error(f"Error updating session details: {e}")
            # Fallback to full re-render if update fails
            if len(self.content_area.objects) > 2:
                self.content_area.objects = self.content_area.objects[:2]
            self._render_single_session_view()

    def _refresh_view(self, *events):
        """Update view to show dashboard and single session view"""
        # Clear and refresh content
        self.content_area.clear()

        # Clear all card references since we're doing a full rebuild
        self.dashboard_card_pane = None
        self.status_card_pane = None
        self.portfolio_card_pane = None
        self.positions_card_pane = None
        self.trades_card_pane = None
        self.events_card_pane = None

        # Render dashboard view first
        self._render_dashboard_view()

        # Add a divider
        self.content_area.append(pn.layout.Divider(margin=(24, 0)))

        # Render single session view below
        self._render_single_session_view()

    def _update_session_data(self, *events):
        """Update session data without full re-render to prevent flickering"""
        # Update dashboard metrics if dashboard is rendered
        if self.dashboard_card_pane:
            try:
                metrics = self.session_manager.get_aggregate_metrics()
                summaries = self.session_manager.get_session_summary()

                # Update just the header HTML (aggregate metrics)
                total_pnl = metrics['total_pnl']
                header_html = f"""
                <div style='padding: 16px; background: linear-gradient(135deg, {Colors.ACCENT_PURPLE} 0%, {Colors.ACCENT_CYAN} 100%); border-radius: 12px 12px 0 0; color: white; font-family: {Typography.FONT_PRIMARY};'>
                    <div style='display: grid; grid-template-columns: 0.5fr 0.5fr 1.5fr 1.5fr; gap: 10px;'>
                        <div>
                            <div style='opacity: 0.9; font-size: {Typography.TEXT_SM};'>Total Sessions</div>
                            <div style='font-size: {Typography.TEXT_2XL}; font-weight: 600; margin-top: 4px;'>{metrics['total_sessions']}</div>
                        </div>
                        <div>
                            <div style='opacity: 0.9; font-size: {Typography.TEXT_SM};'>Running</div>
                            <div style='font-size: {Typography.TEXT_2XL}; font-weight: 600; margin-top: 4px;'>{metrics['running_sessions']}</div>
                        </div>
                        <div>
                            <div style='opacity: 0.9; font-size: {Typography.TEXT_SM};'>Total Portfolio Value</div>
                            <div style='font-size: {Typography.TEXT_2XL}; font-weight: 600; margin-top: 4px; font-family: {Typography.FONT_MONO};'>${metrics['total_portfolio_value']:,.2f}</div>
                        </div>
                        <div>
                            <div style='opacity: 0.9; font-size: {Typography.TEXT_SM};'>Aggregate P&L</div>
                            <div style='font-size: {Typography.TEXT_2XL}; font-weight: 600; margin-top: 4px; font-family: {Typography.FONT_MONO};'>${total_pnl:+,.2f} ({metrics['total_pnl_pct']:+.2f}%)</div>
                        </div>
                    </div>
                </div>
                """

                # Update the first object in the dashboard card (the header)
                if hasattr(self.dashboard_card_pane, 'objects') and len(self.dashboard_card_pane.objects) > 0:
                    self.dashboard_card_pane.objects[0].object = header_html
            except Exception as e:
                logger.error(f"Error updating dashboard metrics: {e}")

        # Only update session details if we have an active session
        active_session = self.session_manager.get_active_session()
        if not active_session:
            return

        # Only update if we have card references (meaning single session view is rendered)
        if not self.status_card_pane:
            return

        # Update individual cards in place without rebuilding everything
        try:
            self.status_card_pane.object = self._create_status_card(active_session).object
            self.portfolio_card_pane.object = self._create_portfolio_card(active_session).object
            self.positions_card_pane.object = self._create_positions_card(active_session).object
            self.trades_card_pane.object = self._create_trades_card(active_session).object
            self.events_card_pane.object = self._create_events_card(active_session).object
        except Exception as e:
            logger.error(f"Error updating session data: {e}")

    # ========================================================================
    # Single Session View
    # ========================================================================

    def _render_single_session_view(self):
        """Render detailed view of active session"""
        active_session = self.session_manager.get_active_session()

        if not active_session:
            # No active session - clear card references (but keep dashboard reference)
            self.status_card_pane = None
            self.portfolio_card_pane = None
            self.positions_card_pane = None
            self.trades_card_pane = None
            self.events_card_pane = None

            self.content_area.append(
                pn.Column(
                    HTMLComponents.info_message(
                        "No Active Session",
                        "Create a new session or select one from the table above"
                    ),
                    sizing_mode="stretch_width"
                )
            )
            return

        # Active session exists - show details
        session = active_session.session
        portfolio = active_session.portfolio

        # Create cards and store references for later updates
        self.status_card_pane = self._create_status_card(active_session)
        self.portfolio_card_pane = self._create_portfolio_card(active_session)
        self.positions_card_pane = self._create_positions_card(active_session)
        self.trades_card_pane = self._create_trades_card(active_session)
        self.events_card_pane = self._create_events_card(active_session)

        self.content_area.extend([
            pn.Row(self.status_card_pane, self.portfolio_card_pane, sizing_mode="stretch_width"),
            self.positions_card_pane,
            pn.Row(self.trades_card_pane, self.events_card_pane, sizing_mode="stretch_width")
        ])

    def _create_status_card(self, engine: LiveTradingEngine) -> pn.pane.HTML:
        """Create status card for session"""
        session = engine.session
        status_color = {
            TradingStatus.RUNNING: Colors.SUCCESS_GREEN,
            TradingStatus.PAUSED: "#F59E0B",
            TradingStatus.STOPPED: Colors.TEXT_MUTED,
            TradingStatus.HALTED: Colors.DANGER_RED,
            TradingStatus.IDLE: Colors.TEXT_SECONDARY
        }.get(session.status, Colors.TEXT_MUTED)

        elapsed = ""
        if session.start_time:
            duration = datetime.now() - session.start_time
            hours, remainder = divmod(duration.seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            elapsed = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

        html = f"""
        <div style='background: {Colors.BG_SECONDARY}; padding: 20px; border-radius: 8px; border-left: 4px solid {status_color}; flex: 1;'>
            <h3 style='margin-top: 0; color: {Colors.TEXT_PRIMARY}; font-family: {Typography.FONT_PRIMARY};'>Session Status</h3>
            <div style='display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;'>
                <div>
                    <p style='color: {Colors.TEXT_MUTED}; font-size: 12px; margin-bottom: 5px;'>Status</p>
                    <p style='color: {status_color}; font-size: 18px; font-weight: 600; margin: 0; font-family: {Typography.FONT_PRIMARY};'>
                        ● {session.status.value.upper()}
                    </p>
                </div>
                <div>
                    <p style='color: {Colors.TEXT_MUTED}; font-size: 12px; margin-bottom: 5px;'>Symbol</p>
                    <p style='color: {Colors.TEXT_PRIMARY}; font-size: 18px; font-weight: 600; margin: 0; font-family: {Typography.FONT_PRIMARY};'>
                        {engine.config.symbol}
                    </p>
                </div>
                <div>
                    <p style='color: {Colors.TEXT_MUTED}; font-size: 12px; margin-bottom: 5px;'>Session ID</p>
                    <p style='color: {Colors.TEXT_SECONDARY}; font-size: 14px; margin: 0; font-family: {Typography.FONT_PRIMARY};'>
                        {session.session_id or 'N/A'}
                    </p>
                </div>
                <div>
                    <p style='color: {Colors.TEXT_MUTED}; font-size: 12px; margin-bottom: 5px;'>Running Time</p>
                    <p style='color: {Colors.TEXT_SECONDARY}; font-size: 14px; margin: 0; font-family: {Typography.FONT_MONO};'>
                        {elapsed}
                    </p>
                </div>
            </div>
        </div>
        """

        return pn.pane.HTML(html, sizing_mode="stretch_width")

    def _create_portfolio_card(self, engine: LiveTradingEngine) -> pn.pane.HTML:
        """Create portfolio summary card"""
        # Update position prices with latest market data before rendering
        if engine.portfolio.positions:
            try:
                latest_tick = engine.market_stream.get_latest_tick()
                engine.portfolio.update_valuations(latest_tick)
            except Exception as e:
                logger.warning(f"Failed to update portfolio valuations: {e}")

        portfolio = engine.portfolio
        pnl_color = Colors.SUCCESS_GREEN if portfolio.total_pnl >= 0 else Colors.DANGER_RED
        pnl_symbol = "▲" if portfolio.total_pnl >= 0 else "▼"

        html = f"""
        <div style='background: {Colors.BG_SECONDARY}; padding: 20px; border-radius: 8px; flex: 1;'>
            <h3 style='margin-top: 0; color: {Colors.TEXT_PRIMARY}; font-family: {Typography.FONT_PRIMARY};'>Portfolio</h3>
            <div style='display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px;'>
                <div>
                    <p style='color: {Colors.TEXT_MUTED}; font-size: 12px; margin-bottom: 5px;'>Total Value</p>
                    <p style='color: {Colors.TEXT_PRIMARY}; font-size: 24px; font-weight: 600; margin: 0; font-family: {Typography.FONT_MONO};'>
                        ${portfolio.total_value:,.2f}
                    </p>
                </div>
                <div>
                    <p style='color: {Colors.TEXT_MUTED}; font-size: 12px; margin-bottom: 5px;'>Total P&L</p>
                    <p style='color: {pnl_color}; font-size: 24px; font-weight: 600; margin: 0; font-family: {Typography.FONT_MONO};'>
                        {pnl_symbol} ${abs(portfolio.total_pnl):,.2f} ({portfolio.total_pnl_pct:+.2f}%)
                    </p>
                </div>
                <div>
                    <p style='color: {Colors.TEXT_MUTED}; font-size: 12px; margin-bottom: 5px;'>Cash</p>
                    <p style='color: {Colors.TEXT_SECONDARY}; font-size: 18px; margin: 0; font-family: {Typography.FONT_MONO};'>
                        ${portfolio.cash:,.2f}
                    </p>
                </div>
                <div>
                    <p style='color: {Colors.TEXT_MUTED}; font-size: 12px; margin-bottom: 5px;'>Trades</p>
                    <p style='color: {Colors.TEXT_SECONDARY}; font-size: 18px; margin: 0; font-family: {Typography.FONT_PRIMARY};'>
                        {len(portfolio.trades)}
                    </p>
                </div>
            </div>
        </div>
        """

        return pn.pane.HTML(html, sizing_mode="stretch_width")

    def _create_positions_card(self, engine: LiveTradingEngine) -> pn.pane.HTML:
        """Create positions table card"""
        # Update position prices with latest market data before rendering
        if engine.portfolio.positions:
            try:
                latest_tick = engine.market_stream.get_latest_tick()
                engine.portfolio.update_valuations(latest_tick)
            except Exception as e:
                logger.warning(f"Failed to update position valuations: {e}")

        if not engine.portfolio.positions:
            html = f"""
            <div style='background: {Colors.BG_SECONDARY}; padding: 20px; border-radius: 8px;'>
                <h3 style='margin-top: 0; color: {Colors.TEXT_PRIMARY}; font-family: {Typography.FONT_PRIMARY};'>Positions</h3>
                <p style='color: {Colors.TEXT_MUTED}; text-align: center; padding: 20px;'>No open positions</p>
            </div>
            """
        else:
            rows = ""
            for symbol, pos in engine.portfolio.positions.items():
                pnl_color = Colors.SUCCESS_GREEN if pos.unrealized_pnl >= 0 else Colors.DANGER_RED
                pnl_symbol = "▲" if pos.unrealized_pnl >= 0 else "▼"
                pnl_pct = (pos.current_price - pos.avg_entry_price) / pos.avg_entry_price * 100

                rows += f"""
                <tr style='border-bottom: 1px solid {Colors.BORDER_SUBTLE};'>
                    <td style='padding: 12px; color: {Colors.TEXT_PRIMARY}; font-weight: 600;'>{symbol}</td>
                    <td style='padding: 12px; color: {Colors.TEXT_SECONDARY}; text-align: right;'>{pos.shares}</td>
                    <td style='padding: 12px; color: {Colors.TEXT_SECONDARY}; text-align: right; font-family: {Typography.FONT_MONO};'>${pos.avg_entry_price:.2f}</td>
                    <td style='padding: 12px; color: {Colors.TEXT_SECONDARY}; text-align: right; font-family: {Typography.FONT_MONO};'>${pos.current_price:.2f}</td>
                    <td style='padding: 12px; color: {pnl_color}; text-align: right; font-weight: 600; font-family: {Typography.FONT_MONO};'>
                        {pnl_symbol} ${abs(pos.unrealized_pnl):.2f} ({pnl_pct:+.2f}%)
                    </td>
                </tr>
                """

            html = f"""
            <div style='background: {Colors.BG_SECONDARY}; padding: 20px; border-radius: 8px;'>
                <h3 style='margin-top: 0; color: {Colors.TEXT_PRIMARY}; font-family: {Typography.FONT_PRIMARY};'>Positions</h3>
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

        return pn.pane.HTML(html, sizing_mode="stretch_width")

    def _create_trades_card(self, engine: LiveTradingEngine) -> pn.pane.HTML:
        """Create trades history card"""
        if not engine.portfolio.trades:
            html = f"""
            <div style='background: {Colors.BG_SECONDARY}; padding: 20px; border-radius: 8px; flex: 1;'>
                <h3 style='margin-top: 0; color: {Colors.TEXT_PRIMARY}; font-family: {Typography.FONT_PRIMARY};'>Trades</h3>
                <p style='color: {Colors.TEXT_MUTED}; text-align: center; padding: 20px;'>No trades yet</p>
            </div>
            """
        else:
            rows = ""
            for trade in reversed(engine.portfolio.trades[-10:]):
                action_color = Colors.SUCCESS_GREEN if trade.action in (TradingAction.BUY_SMALL, TradingAction.BUY_LARGE) else Colors.DANGER_RED
                pnl_color = Colors.SUCCESS_GREEN if trade.pnl >= 0 else Colors.DANGER_RED
                time_str = trade.timestamp.strftime('%H:%M:%S')

                rows += f"""
                <tr style='border-bottom: 1px solid {Colors.BORDER_SUBTLE};'>
                    <td style='padding: 10px; color: {Colors.TEXT_MUTED}; font-size: 12px; font-family: {Typography.FONT_MONO};'>{time_str}</td>
                    <td style='padding: 10px; color: {action_color}; font-weight: 600;'>{trade.action.name}</td>
                    <td style='padding: 10px; color: {Colors.TEXT_SECONDARY}; text-align: right;'>{trade.shares}</td>
                    <td style='padding: 10px; color: {Colors.TEXT_SECONDARY}; text-align: right; font-family: {Typography.FONT_MONO};'>${trade.price:.2f}</td>
                    <td style='padding: 10px; color: {pnl_color}; text-align: right; font-weight: 600; font-family: {Typography.FONT_MONO};'>
                        ${trade.pnl:.2f}
                    </td>
                </tr>
                """

            html = f"""
            <div style='background: {Colors.BG_SECONDARY}; padding: 20px; border-radius: 8px; flex: 1;'>
                <h3 style='margin-top: 0; color: {Colors.TEXT_PRIMARY}; font-family: {Typography.FONT_PRIMARY};'>Recent Trades</h3>
                <div style='max-height: 400px; overflow-y: auto;'>
                    <table style='width: 100%; border-collapse: collapse;'>
                        <thead style='position: sticky; top: 0; background: {Colors.BG_SECONDARY};'>
                            <tr style='border-bottom: 2px solid {Colors.BORDER_SUBTLE};'>
                                <th style='padding: 10px; color: {Colors.TEXT_MUTED}; font-size: 11px; text-align: left;'>TIME</th>
                                <th style='padding: 10px; color: {Colors.TEXT_MUTED}; font-size: 11px; text-align: left;'>ACTION</th>
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

        return pn.pane.HTML(html, sizing_mode="stretch_width")

    def _create_events_card(self, engine: LiveTradingEngine) -> pn.pane.HTML:
        """Create event log card"""
        if not engine.session.events:
            html = f"""
            <div style='background: {Colors.BG_SECONDARY}; padding: 20px; border-radius: 8px; flex: 1;'>
                <h3 style='margin-top: 0; color: {Colors.TEXT_PRIMARY}; font-family: {Typography.FONT_PRIMARY};'>Events</h3>
                <p style='color: {Colors.TEXT_MUTED}; text-align: center; padding: 20px;'>No events</p>
            </div>
            """
        else:
            rows = ""
            for event in reversed(engine.session.events[-15:]):
                timestamp = event['timestamp']
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp)
                time_str = timestamp.strftime('%H:%M:%S')
                event_type = event['type']

                type_color = {
                    'SESSION_START': Colors.SUCCESS_GREEN,
                    'SESSION_END': Colors.TEXT_MUTED,
                    'SESSION_PAUSED': "#F59E0B",
                    'SESSION_RESUMED': Colors.SUCCESS_GREEN,
                    'TRADE': Colors.ACCENT_CYAN,
                    'ORDER_REJECTED': "#F59E0B",
                    'HALT': Colors.DANGER_RED,
                    'ERROR': Colors.DANGER_RED
                }.get(event_type, Colors.TEXT_SECONDARY)

                rows += f"""
                <div style='padding: 8px; border-bottom: 1px solid {Colors.BORDER_SUBTLE};'>
                    <span style='color: {Colors.TEXT_MUTED}; font-size: 11px; font-family: {Typography.FONT_MONO};'>{time_str}</span>
                    <span style='color: {type_color}; font-weight: 600; margin: 0 10px;'>[{event_type}]</span>
                    <span style='color: {Colors.TEXT_SECONDARY}; font-size: 13px;'>{event['message']}</span>
                </div>
                """

            html = f"""
            <div style='background: {Colors.BG_SECONDARY}; padding: 20px; border-radius: 8px; flex: 1;'>
                <h3 style='margin-top: 0; color: {Colors.TEXT_PRIMARY}; font-family: {Typography.FONT_PRIMARY};'>Event Log</h3>
                <div style='max-height: 400px; overflow-y: auto; font-family: monospace;'>
                    {rows}
                </div>
            </div>
            """

        return pn.pane.HTML(html, sizing_mode="stretch_width")

    # ========================================================================
    # Dashboard View
    # ========================================================================

    def _render_dashboard_view(self):
        """Render multi-session comparison dashboard"""
        # Aggregate metrics
        metrics = self.session_manager.get_aggregate_metrics()
        summaries = self.session_manager.get_session_summary()

        # Create the combined dashboard card and store reference for updates
        self.dashboard_card_pane = self._create_dashboard_card(metrics, summaries)

        self.content_area.extend([
            self.dashboard_card_pane
        ])

    def _create_dashboard_card(self, metrics: Dict, summaries: List[Dict]) -> pn.Column:
        """Create a single card containing aggregate metrics and the sessions table."""
        # Aggregate metrics header
        total_pnl = metrics['total_pnl']
        header_html = f"""
        <div style='padding: 16px; background: linear-gradient(135deg, {Colors.ACCENT_PURPLE} 0%, {Colors.ACCENT_CYAN} 100%); border-radius: 12px 12px 0 0; color: white; font-family: {Typography.FONT_PRIMARY};'>
            <div style='display: grid; grid-template-columns: 0.5fr 0.5fr 1.5fr 1.5fr; gap: 10px;'>
                <div>
                    <div style='opacity: 0.9; font-size: {Typography.TEXT_SM};'>Total Sessions</div>
                    <div style='font-size: {Typography.TEXT_2XL}; font-weight: 600; margin-top: 4px;'>{metrics['total_sessions']}</div>
                </div>
                <div>
                    <div style='opacity: 0.9; font-size: {Typography.TEXT_SM};'>Running</div>
                    <div style='font-size: {Typography.TEXT_2XL}; font-weight: 600; margin-top: 4px;'>{metrics['running_sessions']}</div>
                </div>
                <div>
                    <div style='opacity: 0.9; font-size: {Typography.TEXT_SM};'>Total Portfolio Value</div>
                    <div style='font-size: {Typography.TEXT_2XL}; font-weight: 600; margin-top: 4px; font-family: {Typography.FONT_MONO};'>${metrics['total_portfolio_value']:,.2f}</div>
                </div>
                <div>
                    <div style='opacity: 0.9; font-size: {Typography.TEXT_SM};'>Aggregate P&L</div>
                    <div style='font-size: {Typography.TEXT_2XL}; font-weight: 600; margin-top: 4px; font-family: {Typography.FONT_MONO};'>${total_pnl:+,.2f} ({metrics['total_pnl_pct']:+.2f}%)</div>
                </div>
            </div>
        </div>
        """

        # Sessions table content
        if not summaries:
            table_content = pn.pane.HTML(f"""
                <div style='padding: 40px; text-align: center; background: {Colors.BG_SECONDARY}; border-radius: 0 0 8px 8px;'>
                    <p style='color: {Colors.TEXT_MUTED}; font-size: {Typography.TEXT_BASE};'>No sessions to display</p>
                </div>
            """, sizing_mode="stretch_width")
        else:
            rows = []
            for summary in summaries:
                session_id = summary['session_id']
                status = summary['status']

                # Extract model name from agent_path if available
                model_name = "N/A"
                if 'agent_path' in summary and summary['agent_path']:
                    agent_path = Path(summary['agent_path'])
                    # Get parent directory name (e.g., "ppo_AAPL_20251107")
                    model_name = agent_path.parent.name
                elif 'model_name' in summary:
                    model_name = summary['model_name']

                view_btn = pn.widgets.Button(name='View', button_type='primary', width=60)
                view_btn.on_click(lambda e, sid=session_id: self._select_session(sid))

                if status == 'running':
                    action_btn = pn.widgets.Button(name='Stop', button_type='danger', width=80)
                    action_btn.on_click(lambda e, sid=session_id: self._stop_session(sid))
                else:
                    action_btn = pn.widgets.Button(name='Start', button_type='success', width=80)
                    action_btn.on_click(lambda e, sid=session_id: self._start_session(sid))

                button_row = pn.Row(view_btn, action_btn, sizing_mode='fixed')

                rows.append(pn.Row(
                    pn.pane.HTML(f"<div>{summary['session_id']}</div>", width=250),
                    pn.pane.HTML(f"<div>{summary['symbol']}</div>", width=80),
                    pn.pane.HTML(f"<div>{model_name}</div>", width=200),
                    pn.pane.HTML(f"<div>{summary['status']}</div>", width=100),
                    button_row,
                    sizing_mode='stretch_width',
                    align='center'
                ))

            header = pn.Row(
                pn.pane.HTML("<b>Session ID</b>", width=250),
                pn.pane.HTML("<b>Symbol</b>", width=80),
                pn.pane.HTML("<b>Model Name</b>", width=200),
                pn.pane.HTML("<b>Status</b>", width=100),
                pn.pane.HTML("<b>Actions</b>", width=180, align='center'),
                sizing_mode='stretch_width'
            )
            
            table_content = pn.Column(header, *rows, sizing_mode="stretch_width")

        return pn.Column(
            pn.pane.HTML(header_html, sizing_mode="stretch_width"),
            table_content,
            styles=dict(background=Colors.BG_SECONDARY, padding='0', border_radius='12px'),
            sizing_mode="stretch_width"
        )

    # ========================================================================
    # Public API for External Integration
    # ========================================================================

    def set_active_session(self, session_id: str):
        """Called when session is selected externally (deprecated - use _select_session)"""
        self._select_session(session_id)

    # ========================================================================
    # Main View
    # ========================================================================

    def __panel__(self):
        """Return the Panel layout"""
        return pn.Column(
            self.new_session_form,
            self.content_area,
            HTMLComponents.disclaimer(),
            sizing_mode="stretch_width"
        )


def create_live_trading_page(session_manager: Optional[LiveSessionManager] = None):
    """Factory function to create live trading page"""
    return LiveTradingPage(session_manager=session_manager)
