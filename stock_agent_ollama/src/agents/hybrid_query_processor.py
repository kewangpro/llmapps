import asyncio
from typing import Dict, Any, List, Optional, Tuple
import logging
from datetime import datetime, timedelta

from src.agents.query_processor import QueryProcessor
from src.agents.ollama_enhancer import ollama_enhancer
from src.config import Config

logger = logging.getLogger(__name__)

class HybridQueryProcessor(QueryProcessor):
    """Enhanced query processor that combines Ollama NLU with regex fallback"""
    
    def __init__(self):
        super().__init__()
        self.ollama_enhancer = ollama_enhancer
        self.use_ollama = Config.OLLAMA_ENABLED
        self.fallback_to_regex = Config.OLLAMA_FALLBACK_TO_REGEX
        
        # Enhanced educational context tracking
        self.educational_keywords = {
            'explain', 'what is', 'how does', 'why', 'learn', 'teach', 
            'understand', 'meaning', 'definition', 'help me', 'show me how'
        }
        
        # Context-aware query types
        self.query_enhancements = {
            'educational': ['explain', 'learn', 'understand', 'teach'],
            'comparative': ['vs', 'versus', 'compare', 'difference', 'better'],
            'predictive': ['will', 'future', 'forecast', 'predict', 'expect'],
            'analytical': ['analyze', 'analysis', 'review', 'assess', 'evaluate']
        }
    
    async def process_query(
        self,
        query: str,
        force_retrain: bool = False,
        progress_callback: Any = None,
        training_complete_callback: Any = None,
        prediction_callback: Any = None,
        conversation_context: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Enhanced query processing with Ollama integration and regex fallback

        Args:
            query: Natural language query string
            force_retrain: If True, force retrain LSTM models for predictions
            progress_callback: Optional callback for LSTM training progress
            training_complete_callback: Optional callback when training completes
            prediction_callback: Optional callback for LSTM prediction progress
            conversation_context: Optional conversation history for context
        """
        try:
            query_lower = query.lower().strip()
            conversation_context = conversation_context or []

            # Store these for use in handlers
            self._force_retrain = force_retrain
            self._progress_callback = progress_callback
            self._training_complete_callback = training_complete_callback
            self._prediction_callback = prediction_callback

            # First attempt: Use Ollama for natural language understanding
            ollama_result = None
            if self.use_ollama:
                logger.info("Attempting Ollama query understanding")
                ollama_result = await self.ollama_enhancer.understand_query(query)

                if ollama_result and ollama_result.get('confidence', 0) > 0.7:
                    logger.info(f"Ollama understanding successful with confidence: {ollama_result['confidence']}")
                    return await self._process_ollama_result(query, ollama_result, conversation_context)
                else:
                    logger.info(f"Ollama understanding failed or low confidence: {ollama_result}")

            # Fallback: Use original regex-based processing
            if self.fallback_to_regex:
                logger.info("Falling back to regex-based query processing")
                return await self._process_with_regex_fallback(query, conversation_context)
            else:
                return self._handle_processing_failure(query)

        except Exception as e:
            logger.error(f"Hybrid query processing failed: {e}")
            return {
                'type': 'error',
                'message': f"Sorry, I encountered an error processing your query: {str(e)}",
                'query': query,
                'suggestions': self._get_fallback_suggestions()
            }
    
    async def _process_ollama_result(
        self, 
        original_query: str, 
        ollama_result: Dict[str, Any], 
        conversation_context: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Process query using Ollama understanding result"""
        intent = ollama_result.get('intent')
        symbols = ollama_result.get('symbols', [])
        time_period = ollama_result.get('time_period', '1y')
        needs_explanation = ollama_result.get('needs_explanation', False)
        
        # Create entities dict for existing methods
        entities = {
            'symbols': symbols,
            'period': time_period
        }
        
        # Route to appropriate handler based on intent
        if intent == 'analyze':
            result = await self._handle_analyze_query(
                entities,
                prediction_callback=getattr(self, '_prediction_callback', None)
            )
        elif intent == 'predict':
            result = await self._handle_predict_query(
                entities,
                force_retrain=getattr(self, '_force_retrain', False),
                progress_callback=getattr(self, '_progress_callback', None),
                training_complete_callback=getattr(self, '_training_complete_callback', None),
                prediction_callback=getattr(self, '_prediction_callback', None)
            )
        elif intent == 'compare':
            result = await self._handle_compare_query(entities)
        elif intent == 'price':
            result = await self._handle_price_query(entities)
        elif intent == 'explain':
            result = await self._handle_educational_query(original_query, entities, conversation_context)
        else:
            result = await self._handle_general_enhanced_query(original_query, conversation_context)
        
        # Enhance result with educational content if needed
        if needs_explanation and result.get('type') != 'error':
            result = await self._add_educational_enhancements(result, original_query, conversation_context)
        
        return result
    
    async def _process_with_regex_fallback(
        self,
        query: str,
        conversation_context: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Process query using original regex patterns with enhancements"""
        # Use original regex processing with stored parameters
        result = await super().process_query(
            query,
            force_retrain=getattr(self, '_force_retrain', False),
            progress_callback=getattr(self, '_progress_callback', None),
            training_complete_callback=getattr(self, '_training_complete_callback', None),
            prediction_callback=getattr(self, '_prediction_callback', None)
        )

        # Add educational enhancements if the query suggests learning intent
        if self._is_educational_query(query):
            result = await self._add_educational_enhancements(result, query, conversation_context)

        return result
    
    def _is_educational_query(self, query: str) -> bool:
        """Determine if query has educational intent"""
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in self.educational_keywords)
    
    async def _handle_educational_query(
        self,
        query: str,
        entities: Dict[str, Any],
        conversation_context: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Handle educational/explanatory queries"""
        symbols = entities.get('symbols', [])
        
        if not symbols:
            return {
                'type': 'educational',
                'message': "I can explain various stock analysis concepts. What would you like to learn about?",
                'suggestions': [
                    "Explain technical indicators like RSI and MACD",
                    "How do stock predictions work?",
                    "What is fundamental vs technical analysis?",
                    "Explain volatility and risk in stocks"
                ],
                'query': query
            }
        
        # Get basic analysis first
        try:
            symbol = symbols[0]
            
            # Fetch stock data for educational context
            stock_data = self.stock_fetcher.fetch_stock_data(symbol, '3m')
            technical_analysis = self.technical_analysis.analyze_trends(stock_data)
            stock_info = self.stock_fetcher.get_stock_info(symbol)
            
            # Generate educational explanation
            explanation = await self.ollama_enhancer.explain_technical_analysis(
                symbol, technical_analysis, {'query': query, 'context': 'educational'}
            )
            
            return {
                'type': 'educational',
                'symbol': symbol,
                'explanation': explanation,
                'technical_data': technical_analysis,
                'stock_info': stock_info,
                'chart_data': stock_data.tail(30),  # Recent data for context
                'suggestions': self.ollama_enhancer.get_educational_suggestions('explain', symbols),
                'query': query
            }
            
        except Exception as e:
            logger.error(f"Educational query processing failed: {e}")
            return {
                'type': 'error',
                'message': f"Sorry, I couldn't process the educational query for {symbols[0] if symbols else 'the requested symbol'}: {str(e)}",
                'query': query
            }
    
    async def _handle_general_enhanced_query(
        self,
        query: str,
        conversation_context: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Handle general queries with conversational enhancement"""
        # Try to generate contextual response using conversation history
        if conversation_context and self.use_ollama:
            contextual_response = await self.ollama_enhancer.generate_contextual_response(
                query, conversation_context, {}
            )
            
            if contextual_response:
                return {
                    'type': 'conversational',
                    'message': contextual_response,
                    'query': query,
                    'suggestions': self._generate_contextual_suggestions(query, conversation_context)
                }
        
        # Fall back to original general handling
        return self._handle_general_query(query)
    
    async def _add_educational_enhancements(
        self,
        result: Dict[str, Any],
        original_query: str,
        conversation_context: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Add educational explanations to existing results"""
        if result.get('type') == 'error':
            return result
        
        try:
            # Add educational explanations based on result type
            if result.get('type') == 'stock_analysis' and result.get('technical_analysis'):
                explanation = await self.ollama_enhancer.explain_technical_analysis(
                    result['symbol'],
                    result['technical_analysis'],
                    {'query': original_query, 'analysis_result': True}
                )
                if explanation:
                    result['educational_explanation'] = explanation
            
            elif result.get('type') == 'prediction' and result.get('predictions'):
                interpretation = await self.ollama_enhancer.interpret_prediction(
                    result['symbol'],
                    result['predictions'],
                    result.get('model_metrics', {})
                )
                if interpretation:
                    result['prediction_interpretation'] = interpretation
            
            # Add educational suggestions
            symbols = [result.get('symbol')] if result.get('symbol') else []
            intent = self._infer_intent_from_result(result)
            result['educational_suggestions'] = self.ollama_enhancer.get_educational_suggestions(intent, symbols)
            
        except Exception as e:
            logger.warning(f"Failed to add educational enhancements: {e}")
        
        return result
    
    def _infer_intent_from_result(self, result: Dict[str, Any]) -> str:
        """Infer intent from result type"""
        result_type = result.get('type', '')
        
        type_to_intent = {
            'stock_analysis': 'analyze',
            'prediction': 'predict',
            'comparison': 'compare',
            'price_info': 'price',
            'educational': 'explain'
        }
        
        return type_to_intent.get(result_type, 'analyze')
    
    def _generate_contextual_suggestions(
        self,
        query: str,
        conversation_context: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate contextual suggestions based on conversation history"""
        suggestions = []
        
        # Analyze recent conversation for patterns
        recent_symbols = set()
        recent_intents = set()
        
        for turn in conversation_context[-3:]:
            if 'symbols' in turn:
                recent_symbols.update(turn['symbols'])
            if 'intent' in turn:
                recent_intents.add(turn['intent'])
        
        # Generate relevant suggestions
        if recent_symbols:
            symbol = list(recent_symbols)[0]
            suggestions.extend([
                f"Compare {symbol} with similar stocks",
                f"Explain {symbol}'s technical indicators",
                f"Show {symbol}'s prediction analysis"
            ])
        
        suggestions.extend([
            "Learn about different analysis methods",
            "Understand risk management in investing",
            "Explore market trends and patterns"
        ])
        
        return suggestions[:4]  # Return top 4
    
    def _handle_processing_failure(self, query: str) -> Dict[str, Any]:
        """Handle cases where both Ollama and regex processing fail"""
        return {
            'type': 'processing_failure',
            'message': "I'm having trouble understanding your query. Could you rephrase it or be more specific?",
            'suggestions': [
                "Try: 'Analyze AAPL stock'",
                "Try: 'Predict GOOGL price'", 
                "Try: 'Compare AAPL vs MSFT'",
                "Try: 'Explain RSI indicator'"
            ],
            'query': query,
            'help': "You can ask about stock analysis, predictions, comparisons, or request explanations of financial concepts."
        }
    
    def _get_fallback_suggestions(self) -> List[str]:
        """Get fallback suggestions when processing fails"""
        return [
            "Ask for stock analysis: 'Analyze AAPL'",
            "Request predictions: 'Predict Tesla stock'",
            "Compare stocks: 'AAPL vs GOOGL'",
            "Learn concepts: 'Explain technical analysis'"
        ]
    
    async def get_ollama_health_status(self) -> Dict[str, Any]:
        """Check Ollama service health for diagnostics"""
        if not self.use_ollama:
            return {
                'enabled': False,
                'available': False,
                'message': 'Ollama integration is disabled'
            }
        
        is_healthy = await self.ollama_enhancer.health_check()
        return {
            'enabled': True,
            'available': is_healthy,
            'model': Config.OLLAMA_MODEL,
            'base_url': Config.OLLAMA_BASE_URL,
            'message': 'Ollama is available' if is_healthy else 'Ollama is not responding'
        }
    
    async def close(self):
        """Clean up resources"""
        await self.ollama_enhancer.close()