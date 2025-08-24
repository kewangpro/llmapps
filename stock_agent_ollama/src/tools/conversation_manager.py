import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import logging
from dataclasses import dataclass, asdict
from pathlib import Path

from src.config import Config
from src.agents.ollama_enhancer import ollama_enhancer

logger = logging.getLogger(__name__)

@dataclass
class ConversationTurn:
    """Represents a single turn in the conversation"""
    timestamp: datetime
    query: str
    response_type: str
    symbols: List[str]
    intent: str
    result: Dict[str, Any]
    educational_content: Optional[str] = None
    user_feedback: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationTurn':
        """Create from dictionary"""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)

class ConversationManager:
    """Manages multi-turn educational conversations about stock analysis"""
    
    def __init__(self):
        self.conversations: Dict[str, List[ConversationTurn]] = {}
        self.max_history = 20  # Keep last 20 turns per session
        self.session_timeout = timedelta(hours=2)  # Sessions expire after 2 hours
        self.persistence_enabled = True
        self.storage_path = Config.DATA_DIR / "conversations"
        
        # Educational progression tracking
        self.learning_topics = {
            'technical_analysis': ['RSI', 'MACD', 'Moving Averages', 'Bollinger Bands', 'Stochastic'],
            'fundamental_analysis': ['P/E Ratio', 'Revenue', 'Earnings', 'Market Cap', 'Debt'],
            'prediction_models': ['LSTM', 'Time Series', 'Machine Learning', 'Regression', 'Ensemble'],
            'risk_management': ['Volatility', 'Beta', 'Sharpe Ratio', 'Diversification', 'Stop Loss'],
            'market_concepts': ['Market Trends', 'Volume', 'Support/Resistance', 'Sector Analysis']
        }
        
        self._ensure_storage_directory()
    
    def _ensure_storage_directory(self):
        """Ensure storage directory exists"""
        if self.persistence_enabled:
            self.storage_path.mkdir(parents=True, exist_ok=True)
    
    def _get_session_file(self, session_id: str) -> Path:
        """Get file path for session storage"""
        return self.storage_path / f"session_{session_id}.json"
    
    def start_session(self, session_id: str) -> Dict[str, Any]:
        """Start a new conversation session"""
        if session_id in self.conversations:
            self._cleanup_old_turns(session_id)
        else:
            self.conversations[session_id] = []
            
        logger.info(f"Started conversation session: {session_id}")
        
        return {
            'session_id': session_id,
            'status': 'started',
            'available_topics': list(self.learning_topics.keys()),
            'welcome_message': (
                "Welcome to your stock analysis learning session! "
                "I can help you understand technical analysis, predictions, and market concepts. "
                "Feel free to ask questions and I'll provide educational explanations."
            )
        }
    
    def end_session(self, session_id: str) -> Dict[str, Any]:
        """End a conversation session"""
        if session_id not in self.conversations:
            return {'status': 'error', 'message': 'Session not found'}
        
        # Save session if persistence is enabled
        if self.persistence_enabled:
            self._save_session(session_id)
        
        # Generate session summary
        summary = self._generate_session_summary(session_id)
        
        # Clean up memory
        del self.conversations[session_id]
        
        logger.info(f"Ended conversation session: {session_id}")
        
        return {
            'session_id': session_id,
            'status': 'ended',
            'summary': summary
        }
    
    def add_turn(
        self,
        session_id: str,
        query: str,
        result: Dict[str, Any],
        educational_content: Optional[str] = None
    ) -> ConversationTurn:
        """Add a new turn to the conversation"""
        if session_id not in self.conversations:
            self.conversations[session_id] = []
        
        # Extract information from result
        symbols = self._extract_symbols_from_result(result)
        intent = self._extract_intent_from_result(result)
        response_type = result.get('type', 'unknown')
        
        turn = ConversationTurn(
            timestamp=datetime.now(),
            query=query,
            response_type=response_type,
            symbols=symbols,
            intent=intent,
            result=result,
            educational_content=educational_content
        )
        
        # Add to conversation history
        self.conversations[session_id].append(turn)
        self._cleanup_old_turns(session_id)
        
        logger.debug(f"Added turn to session {session_id}: {intent} for {symbols}")
        
        return turn
    
    def get_conversation_context(
        self,
        session_id: str,
        last_n_turns: int = 5
    ) -> List[Dict[str, Any]]:
        """Get recent conversation context"""
        if session_id not in self.conversations:
            return []
        
        recent_turns = self.conversations[session_id][-last_n_turns:]
        
        # Convert to simplified format for context
        context = []
        for turn in recent_turns:
            context.append({
                'query': turn.query,
                'intent': turn.intent,
                'symbols': turn.symbols,
                'response_type': turn.response_type,
                'response': turn.educational_content or turn.result.get('message', '')[:200],
                'timestamp': turn.timestamp.isoformat()
            })
        
        return context
    
    async def generate_contextual_followup(
        self,
        session_id: str,
        current_query: str,
        current_result: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Generate contextual follow-up suggestions and educational content"""
        context = self.get_conversation_context(session_id, last_n_turns=3)
        
        if not context:
            return None
        
        try:
            # Generate contextual response using Ollama
            contextual_response = await ollama_enhancer.generate_contextual_response(
                current_query, context, current_result
            )
            
            if contextual_response:
                # Analyze learning progression
                learning_progress = self._analyze_learning_progression(session_id)
                
                # Generate personalized suggestions
                suggestions = self._generate_progressive_suggestions(
                    context, current_result, learning_progress
                )
                
                return {
                    'contextual_response': contextual_response,
                    'learning_progress': learning_progress,
                    'suggestions': suggestions,
                    'next_topics': self._suggest_next_topics(session_id)
                }
        
        except Exception as e:
            logger.error(f"Failed to generate contextual follow-up: {e}")
        
        return None
    
    def _analyze_learning_progression(self, session_id: str) -> Dict[str, Any]:
        """Analyze user's learning progression across topics"""
        if session_id not in self.conversations:
            return {'topics_covered': [], 'complexity_level': 'beginner'}
        
        turns = self.conversations[session_id]
        
        # Track topics covered
        topics_covered = set()
        complexity_indicators = []
        
        for turn in turns:
            # Categorize the query/intent into learning topics
            for topic, concepts in self.learning_topics.items():
                if any(concept.lower() in turn.query.lower() for concept in concepts):
                    topics_covered.add(topic)
                    
            # Assess complexity based on query sophistication
            if any(word in turn.query.lower() for word in ['compare', 'analyze', 'predict', 'correlation']):
                complexity_indicators.append('intermediate')
            elif any(word in turn.query.lower() for word in ['explain', 'what is', 'how']):
                complexity_indicators.append('beginner')
            elif any(word in turn.query.lower() for word in ['optimize', 'strategy', 'portfolio']):
                complexity_indicators.append('advanced')
        
        # Determine overall complexity level
        if not complexity_indicators:
            complexity_level = 'beginner'
        else:
            advanced_count = complexity_indicators.count('advanced')
            intermediate_count = complexity_indicators.count('intermediate')
            
            if advanced_count > len(complexity_indicators) * 0.3:
                complexity_level = 'advanced'
            elif intermediate_count > len(complexity_indicators) * 0.4:
                complexity_level = 'intermediate'
            else:
                complexity_level = 'beginner'
        
        return {
            'topics_covered': list(topics_covered),
            'complexity_level': complexity_level,
            'total_turns': len(turns),
            'session_duration': (turns[-1].timestamp - turns[0].timestamp).total_seconds() / 60 if turns else 0
        }
    
    def _generate_progressive_suggestions(
        self,
        context: List[Dict[str, Any]],
        current_result: Dict[str, Any],
        learning_progress: Dict[str, Any]
    ) -> List[str]:
        """Generate learning suggestions based on progression"""
        suggestions = []
        topics_covered = set(learning_progress.get('topics_covered', []))
        complexity_level = learning_progress.get('complexity_level', 'beginner')
        
        # Get recent symbols for personalized suggestions
        recent_symbols = set()
        for turn in context:
            recent_symbols.update(turn.get('symbols', []))
        
        symbol = list(recent_symbols)[0] if recent_symbols else 'AAPL'
        
        # Generate suggestions based on complexity level
        if complexity_level == 'beginner':
            if 'technical_analysis' not in topics_covered:
                suggestions.append(f"Learn about basic technical indicators for {symbol}")
            if 'market_concepts' not in topics_covered:
                suggestions.append("Understand market trends and how to read them")
            suggestions.append("Explore the difference between technical and fundamental analysis")
        
        elif complexity_level == 'intermediate':
            if 'prediction_models' not in topics_covered:
                suggestions.append(f"Learn how machine learning predicts {symbol}'s price")
            if 'risk_management' not in topics_covered:
                suggestions.append("Understand risk management and portfolio diversification")
            suggestions.append(f"Compare {symbol} with sector peers")
        
        else:  # advanced
            suggestions.append("Explore advanced trading strategies and their risks")
            suggestions.append("Learn about quantitative analysis methods")
            suggestions.append("Understand market efficiency and behavioral finance")
        
        return suggestions[:4]  # Return top 4
    
    def _suggest_next_topics(self, session_id: str) -> List[str]:
        """Suggest next learning topics based on conversation history"""
        learning_progress = self._analyze_learning_progression(session_id)
        covered_topics = set(learning_progress.get('topics_covered', []))
        
        # Suggest uncovered topics
        available_topics = []
        for topic in self.learning_topics.keys():
            if topic not in covered_topics:
                available_topics.append(topic.replace('_', ' ').title())
        
        return available_topics[:3]  # Return top 3 suggestions
    
    def _extract_symbols_from_result(self, result: Dict[str, Any]) -> List[str]:
        """Extract stock symbols from result"""
        symbols = []
        
        # Direct symbol field
        if 'symbol' in result:
            symbols.append(result['symbol'])
        
        # Multiple symbols field
        if 'symbols' in result:
            symbols.extend(result['symbols'])
        
        return list(set(symbols))  # Remove duplicates
    
    def _extract_intent_from_result(self, result: Dict[str, Any]) -> str:
        """Extract intent from result type"""
        result_type = result.get('type', '')
        
        type_to_intent = {
            'stock_analysis': 'analyze',
            'prediction': 'predict',
            'comparison': 'compare',
            'price_info': 'price',
            'educational': 'explain',
            'conversational': 'discuss'
        }
        
        return type_to_intent.get(result_type, 'general')
    
    def _cleanup_old_turns(self, session_id: str):
        """Remove old turns to maintain memory efficiency"""
        if session_id in self.conversations:
            conversation = self.conversations[session_id]
            
            # Remove turns older than session timeout
            cutoff_time = datetime.now() - self.session_timeout
            conversation[:] = [turn for turn in conversation if turn.timestamp > cutoff_time]
            
            # Keep only the most recent turns
            if len(conversation) > self.max_history:
                conversation[:] = conversation[-self.max_history:]
    
    def _generate_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Generate a summary of the conversation session"""
        if session_id not in self.conversations:
            return {}
        
        turns = self.conversations[session_id]
        if not turns:
            return {'message': 'No conversation history found'}
        
        # Basic statistics
        symbols_discussed = set()
        intents_used = set()
        
        for turn in turns:
            symbols_discussed.update(turn.symbols)
            intents_used.add(turn.intent)
        
        learning_progress = self._analyze_learning_progression(session_id)
        
        return {
            'total_turns': len(turns),
            'duration_minutes': (turns[-1].timestamp - turns[0].timestamp).total_seconds() / 60,
            'symbols_discussed': list(symbols_discussed),
            'topics_explored': list(intents_used),
            'learning_progress': learning_progress,
            'recommendation': self._get_learning_recommendation(learning_progress)
        }
    
    def _get_learning_recommendation(self, learning_progress: Dict[str, Any]) -> str:
        """Get personalized learning recommendation"""
        complexity_level = learning_progress.get('complexity_level', 'beginner')
        topics_covered = learning_progress.get('topics_covered', [])
        
        if complexity_level == 'beginner' and len(topics_covered) < 2:
            return "Continue exploring basic concepts like technical indicators and market trends."
        elif complexity_level == 'intermediate' and 'prediction_models' not in topics_covered:
            return "Consider learning about machine learning in finance and prediction models."
        elif complexity_level == 'advanced':
            return "Explore advanced topics like algorithmic trading and quantitative analysis."
        else:
            return "Great progress! Keep practicing with real examples and different stocks."
    
    def _save_session(self, session_id: str):
        """Save session to persistent storage"""
        if session_id not in self.conversations:
            return
        
        try:
            session_data = {
                'session_id': session_id,
                'timestamp': datetime.now().isoformat(),
                'turns': [turn.to_dict() for turn in self.conversations[session_id]]
            }
            
            session_file = self._get_session_file(session_id)
            with open(session_file, 'w') as f:
                json.dump(session_data, f, indent=2, default=str)
                
            logger.debug(f"Saved session {session_id} to {session_file}")
            
        except Exception as e:
            logger.error(f"Failed to save session {session_id}: {e}")
    
    def load_session(self, session_id: str) -> bool:
        """Load session from persistent storage"""
        if not self.persistence_enabled:
            return False
        
        try:
            session_file = self._get_session_file(session_id)
            if not session_file.exists():
                return False
            
            with open(session_file, 'r') as f:
                session_data = json.load(f)
            
            # Reconstruct conversation turns
            turns = []
            for turn_data in session_data.get('turns', []):
                turns.append(ConversationTurn.from_dict(turn_data))
            
            self.conversations[session_id] = turns
            logger.info(f"Loaded session {session_id} with {len(turns)} turns")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load session {session_id}: {e}")
            return False
    
    def get_active_sessions(self) -> List[str]:
        """Get list of currently active session IDs"""
        return list(self.conversations.keys())
    
    def cleanup_expired_sessions(self):
        """Clean up expired sessions"""
        cutoff_time = datetime.now() - self.session_timeout
        expired_sessions = []
        
        for session_id, turns in self.conversations.items():
            if not turns or turns[-1].timestamp < cutoff_time:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            if self.persistence_enabled:
                self._save_session(session_id)
            del self.conversations[session_id]
            logger.info(f"Cleaned up expired session: {session_id}")

# Singleton instance
conversation_manager = ConversationManager()