"""
Multi-Session Live Trading Manager

Manages multiple concurrent live trading sessions with thread-safe execution.
"""

import threading
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

from .live_trading import (
    LiveTradingEngine,
    LiveTradingConfig,
    TradingStatus,
    TradingSession
)

logger = logging.getLogger(__name__)


class LiveSessionManager:
    """Manages multiple concurrent live trading sessions"""

    def __init__(self, sessions_dir: Optional[Path] = None):
        """
        Initialize session manager

        Args:
            sessions_dir: Directory to store session files (default: data/live_sessions)
        """
        self.sessions: Dict[str, LiveTradingEngine] = {}
        self.active_session_id: Optional[str] = None
        self._session_threads: Dict[str, threading.Thread] = {}
        self._shutdown_events: Dict[str, threading.Event] = {}
        self._lock = threading.Lock()

        # Set sessions directory
        if sessions_dir is None:
            self.sessions_dir = Path("data/live_sessions")
        else:
            self.sessions_dir = Path(sessions_dir)

        self.sessions_dir.mkdir(parents=True, exist_ok=True)

        # Load existing sessions on startup
        self.load_all_sessions()

    # ========================================================================
    # Session Lifecycle
    # ========================================================================

    def create_session(
        self,
        config: LiveTradingConfig,
        strategy_name: str = "",
        description: str = "",
        tags: List[str] = None,
        color: str = "#7C3AED"
    ) -> str:
        """
        Create new trading session

        Args:
            config: Trading configuration
            strategy_name: User-friendly strategy name
            description: Session description
            tags: Tags for filtering/organizing
            color: UI color for session card

        Returns:
            session_id: Created session ID
        """
        with self._lock:
            # Create engine
            engine = LiveTradingEngine(config)

            # Initialize session state
            session = engine.initialize_session_state()
            session_id = session.session_id

            # Add metadata to session
            session.strategy_name = strategy_name or f"{config.symbol} Strategy"
            session.description = description
            session.tags = tags or [config.symbol]
            session.color = color
            session.display_order = len(self.sessions)

            # Store session
            self.sessions[session_id] = engine

            # Set as active if first session
            if self.active_session_id is None:
                self.active_session_id = session_id

            # Save to disk
            self._save_session(session_id)

            logger.info(f"Created session: {session_id}")
            return session_id

    def start_session(self, session_id: str) -> bool:
        """
        Start trading cycle in background thread

        Args:
            session_id: Session to start

        Returns:
            bool: True if started successfully
        """
        with self._lock:
            if session_id not in self.sessions:
                logger.error(f"Session not found: {session_id}")
                return False

            engine = self.sessions[session_id]

            # Check if already running
            if session_id in self._session_threads and self._session_threads[session_id].is_alive():
                logger.warning(f"Session already running: {session_id}")
                return False

            # If agent is not loaded (e.g., stopped session being restarted), load it
            if engine.agent is None:
                try:
                    logger.info(f"Loading agent for session: {session_id}")
                    engine.load_agent(engine.config.agent_path)
                except Exception as e:
                    logger.error(f"Failed to load agent for session {session_id}: {e}")
                    return False

            # Update status
            engine.session.status = TradingStatus.RUNNING
            engine._is_running = True

            # Add event to session log
            if engine.session.start_time:
                # Session was previously started, so this is a resume
                engine.session.add_event("SESSION_RESUMED", f"Resumed trading session")
            else:
                # First time starting this session
                engine.session.add_event("SESSION_START", f"Started trading session for {engine.config.symbol}")

            # Create shutdown event
            self._shutdown_events[session_id] = threading.Event()

            # Start background thread
            thread = threading.Thread(
                target=self._run_session_loop,
                args=(session_id,),
                daemon=True,
                name=f"Session-{session_id}"
            )
            thread.start()
            self._session_threads[session_id] = thread

            self._save_session(session_id)

            logger.info(f"Started session: {session_id}")
            return True

    def pause_session(self, session_id: str) -> bool:
        """
        Pause trading cycle, keep state

        Args:
            session_id: Session to pause

        Returns:
            bool: True if paused successfully
        """
        with self._lock:
            if session_id not in self.sessions:
                logger.error(f"Session not found: {session_id}")
                return False

            engine = self.sessions[session_id]
            engine.session.status = TradingStatus.PAUSED
            engine._is_running = False
            engine.session.add_event("SESSION_PAUSED", "Trading paused by user")

            self._save_session(session_id)

            logger.info(f"Paused session: {session_id}")
            return True

    def resume_session(self, session_id: str) -> bool:
        """
        Resume paused session

        Args:
            session_id: Session to resume

        Returns:
            bool: True if resumed successfully
        """
        with self._lock:
            if session_id not in self.sessions:
                logger.error(f"Session not found: {session_id}")
                return False

            engine = self.sessions[session_id]

            if engine.session.status != TradingStatus.PAUSED:
                logger.warning(f"Session is not paused: {session_id}")
                return False

            engine.session.status = TradingStatus.RUNNING
            engine._is_running = True
            engine.session.add_event("SESSION_RESUMED", "Trading resumed by user")

            self._save_session(session_id)

            logger.info(f"Resumed session: {session_id}")
            return True

    def stop_session(self, session_id: str) -> bool:
        """
        Stop session permanently, save state

        Args:
            session_id: Session to stop

        Returns:
            bool: True if stopped successfully
        """
        with self._lock:
            if session_id not in self.sessions:
                logger.error(f"Session not found: {session_id}")
                return False

            engine = self.sessions[session_id]
            engine.stop_session()

            # Signal thread to shutdown
            if session_id in self._shutdown_events:
                self._shutdown_events[session_id].set()

            # Wait for thread to finish (with timeout)
            if session_id in self._session_threads:
                thread = self._session_threads[session_id]
                thread.join(timeout=5.0)
                if thread.is_alive():
                    logger.warning(f"Session thread did not stop cleanly: {session_id}")

            self._save_session(session_id)

            logger.info(f"Stopped session: {session_id}")
            return True

    def delete_session(self, session_id: str) -> bool:
        """
        Remove session from manager and disk

        Args:
            session_id: Session to delete

        Returns:
            bool: True if deleted successfully
        """
        with self._lock:
            if session_id not in self.sessions:
                logger.error(f"Session not found: {session_id}")
                return False

            # Stop if running
            if self.sessions[session_id].session.status == TradingStatus.RUNNING:
                self.stop_session(session_id)

            # Remove from memory
            del self.sessions[session_id]

            # Clean up thread resources
            if session_id in self._session_threads:
                del self._session_threads[session_id]
            if session_id in self._shutdown_events:
                del self._shutdown_events[session_id]

            # Delete file
            session_file = self.sessions_dir / f"{session_id}.json"
            if session_file.exists():
                session_file.unlink()

            # Update active session if needed
            if self.active_session_id == session_id:
                if self.sessions:
                    self.active_session_id = next(iter(self.sessions.keys()))
                else:
                    self.active_session_id = None

            logger.info(f"Deleted session: {session_id}")
            return True

    # ========================================================================
    # Session Queries
    # ========================================================================

    def get_session(self, session_id: str) -> Optional[LiveTradingEngine]:
        """Get specific session"""
        return self.sessions.get(session_id)

    def get_all_sessions(self) -> List[LiveTradingEngine]:
        """Get all sessions"""
        return list(self.sessions.values())

    def get_active_sessions(self) -> List[LiveTradingEngine]:
        """Get only running/paused sessions"""
        return [
            engine for engine in self.sessions.values()
            if engine.session.status in (TradingStatus.RUNNING, TradingStatus.PAUSED)
        ]

    def get_session_status(self, session_id: str) -> Optional[TradingStatus]:
        """Get current status"""
        engine = self.sessions.get(session_id)
        return engine.session.status if engine else None

    def list_session_ids(self) -> List[str]:
        """Get all session IDs"""
        return list(self.sessions.keys())

    # ========================================================================
    # Active Session Management
    # ========================================================================

    def set_active_session(self, session_id: str) -> bool:
        """
        Set which session is displayed in UI

        Args:
            session_id: Session to make active

        Returns:
            bool: True if set successfully
        """
        if session_id in self.sessions:
            self.active_session_id = session_id
            logger.info(f"Set active session: {session_id}")
            return True
        logger.error(f"Cannot set active session - not found: {session_id}")
        return False

    def get_active_session(self) -> Optional[LiveTradingEngine]:
        """Get currently displayed session"""
        if self.active_session_id:
            return self.sessions.get(self.active_session_id)
        return None

    # ========================================================================
    # Aggregation & Metrics
    # ========================================================================

    def get_aggregate_metrics(self) -> Dict[str, Any]:
        """Calculate total P&L, portfolio value across all sessions"""
        total_value = 0.0
        total_pnl = 0.0
        total_initial = 0.0
        num_running = 0
        num_profitable = 0

        for engine in self.sessions.values():
            portfolio = engine.portfolio
            total_value += portfolio.total_value
            total_pnl += portfolio.total_pnl
            total_initial += portfolio.initial_cash

            if engine.session.status == TradingStatus.RUNNING:
                num_running += 1

            if portfolio.total_pnl > 0:
                num_profitable += 1

        total_pnl_pct = (total_pnl / total_initial * 100) if total_initial > 0 else 0.0

        return {
            "total_sessions": len(self.sessions),
            "running_sessions": num_running,
            "profitable_sessions": num_profitable,
            "total_portfolio_value": total_value,
            "total_pnl": total_pnl,
            "total_pnl_pct": total_pnl_pct,
            "total_initial_capital": total_initial
        }

    def get_session_summary(self) -> List[Dict[str, Any]]:
        """Get summary of all sessions for sidebar display"""
        summaries = []

        for session_id, engine in self.sessions.items():
            session = engine.session
            portfolio = engine.portfolio

            summaries.append({
                "session_id": session_id,
                "strategy_name": getattr(session, 'strategy_name', f"{engine.config.symbol} Strategy"),
                "symbol": engine.config.symbol,
                "status": session.status.value,
                "portfolio_value": portfolio.total_value,
                "pnl": portfolio.total_pnl,
                "pnl_pct": portfolio.total_pnl_pct,
                "color": getattr(session, 'color', '#7C3AED'),
                "num_trades": len(portfolio.trades),
                "start_time": session.start_time.isoformat() if session.start_time else None,
                "display_order": getattr(session, 'display_order', 0),
                "agent_path": engine.config.agent_path
            })

        # Sort by display order
        summaries.sort(key=lambda x: x['display_order'])

        return summaries

    # ========================================================================
    # Persistence
    # ========================================================================

    def save_all_sessions(self):
        """Persist all sessions to disk"""
        for session_id in self.sessions.keys():
            self._save_session(session_id)
        logger.info(f"Saved all {len(self.sessions)} sessions")

    def load_all_sessions(self):
        """Restore all sessions from disk on startup"""
        # Scan directory for all session JSON files
        session_files = list(self.sessions_dir.glob("*.json"))

        if not session_files:
            logger.info("No session files found")
            return

        logger.info(f"Found {len(session_files)} session files, loading...")

        # Load each session
        for session_file in session_files:
            session_id = session_file.stem  # filename without .json
            try:
                self._load_session(session_id)
            except Exception as e:
                logger.error(f"Failed to load session {session_id}: {e}")

        logger.info(f"Loaded {len(self.sessions)} sessions")

    def _save_session(self, session_id: str):
        """Save single session to disk"""
        if session_id not in self.sessions:
            return

        engine = self.sessions[session_id]
        session_file = self.sessions_dir / f"{session_id}.json"
        engine.save_state(session_file)

    def _load_session(self, session_id: str):
        """Load single session from disk"""
        session_file = self.sessions_dir / f"{session_id}.json"

        if not session_file.exists():
            logger.warning(f"Session file not found: {session_file}")
            return

        # Load engine from file
        engine = LiveTradingEngine.load_from_state(session_file)

        self.sessions[session_id] = engine

        # Restart if was running (optional - could require manual restart)
        # if engine.session.status == TradingStatus.RUNNING:
        #     self.start_session(session_id)

    # ========================================================================
    # Background Execution
    # ========================================================================

    def _run_session_loop(self, session_id: str):
        """Background thread for trading cycle"""
        logger.info(f"Session loop started: {session_id}")

        engine = self.sessions.get(session_id)
        if not engine:
            logger.error(f"Session not found in loop: {session_id}")
            return

        shutdown_event = self._shutdown_events.get(session_id)
        update_interval = engine.config.update_interval

        while not shutdown_event.is_set():
            if engine.session.status == TradingStatus.RUNNING:
                try:
                    # Run trading cycle
                    result = engine.trading_cycle()

                    # Auto-save after each cycle
                    self._save_session(session_id)

                    # Check if halted
                    if result.get('status') == 'halted':
                        logger.warning(f"Session halted: {session_id}")
                        break

                except Exception as e:
                    logger.error(f"Error in session loop {session_id}: {e}")
                    engine.session.add_event("ERROR", f"Loop error: {str(e)}")
                    # Continue running unless critical error

            # Sleep until next cycle
            shutdown_event.wait(timeout=update_interval)

        logger.info(f"Session loop ended: {session_id}")

    def shutdown_all(self):
        """Gracefully shutdown all sessions"""
        logger.info("Shutting down all sessions...")

        for session_id in list(self.sessions.keys()):
            self.stop_session(session_id)

        self.save_all_sessions()
        logger.info("All sessions shut down")


# Global singleton instance (optional - can be instantiated in UI)
_session_manager: Optional[LiveSessionManager] = None


def get_session_manager() -> LiveSessionManager:
    """Get global session manager instance"""
    global _session_manager
    if _session_manager is None:
        _session_manager = LiveSessionManager()
    return _session_manager
