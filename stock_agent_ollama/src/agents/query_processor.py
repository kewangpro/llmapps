import re
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import logging

from src.tools.stock_fetcher import StockFetcher
from src.tools.lstm_predictor import LSTMPredictor
from src.tools.visualizer import Visualizer
from src.tools.technical_analysis import TechnicalAnalysis
from src.config import Config

logger = logging.getLogger(__name__)

class QueryProcessor:
    """Process natural language queries about stocks using pattern matching
    
    Note: This is the legacy query processor. For enhanced functionality with Ollama integration,
    use HybridQueryProcessor from src.agents.hybrid_query_processor
    """
    
    def __init__(self):
        self.stock_fetcher = StockFetcher()
        self.lstm_predictor = LSTMPredictor()
        self.visualizer = Visualizer()
        self.technical_analysis = TechnicalAnalysis()
        
        # Migration helper for backwards compatibility
        self.is_hybrid_available = self._check_hybrid_availability()
        
        # Common stock symbols for pattern matching
        self.popular_symbols = self.stock_fetcher.get_available_symbols()
        
        # Query patterns
        self.query_patterns = {
            'analyze': [
                r'analyze\s+(\w+)',
                r'analysis\s+of\s+(\w+)',
                r'show\s+me\s+(\w+)',
                r'(\w+)\s+analysis',
                r'look\s+at\s+(\w+)'
            ],
            'predict': [
                r'predict\s+(\w+)',
                r'forecast\s+(\w+)',
                r'(\w+)\s+prediction',
                r'what\s+will\s+(\w+)\s+be',
                r'future\s+price\s+of\s+(\w+)'
            ],
            'compare': [
                r'compare\s+(\w+)\s+(?:and|vs|with)\s+(\w+)',
                r'(\w+)\s+vs\s+(\w+)',
                r'(\w+)\s+and\s+(\w+)\s+comparison'
            ],
            'price': [
                r'(?:price|cost)\s+of\s+(\w+)',
                r'(\w+)\s+(?:current\s+)?price',
                r'how\s+much\s+is\s+(\w+)'
            ]
        }
        
        # Time period patterns
        self.time_patterns = {
            'last week': 7,
            'last month': 30,
            'last 3 months': 90,
            'last 6 months': 180,
            'last year': 365,
            'ytd': None,  # Year to date
            '1w': 7,
            '1m': 30,
            '3m': 90,
            '6m': 180,
            '1y': 365,
            '2y': 730
        }
    
    async def process_query(self, query: str, force_retrain: bool = False, progress_callback: Any = None, training_complete_callback: Any = None, prediction_callback: Any = None) -> Dict[str, Any]:
        """Process natural language query and return structured response

        Args:
            query: Natural language query string
            force_retrain: If True, force retrain LSTM models for predictions
            progress_callback: Optional callback for LSTM training progress
            training_complete_callback: Optional callback when training completes
            prediction_callback: Optional callback for LSTM prediction progress
        """
        try:
            query_lower = query.lower().strip()

            # Extract intent and entities
            intent, entities = self._extract_intent_and_entities(query_lower)

            if intent == 'analyze':
                return await self._handle_analyze_query(entities, prediction_callback=prediction_callback)
            elif intent == 'predict':
                return await self._handle_predict_query(entities, force_retrain=force_retrain, progress_callback=progress_callback, training_complete_callback=training_complete_callback, prediction_callback=prediction_callback)
            elif intent == 'compare':
                return await self._handle_compare_query(entities)
            elif intent == 'price':
                return await self._handle_price_query(entities)
            else:
                return self._handle_general_query(query)
                
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            return {
                'type': 'error',
                'message': f"Sorry, I encountered an error processing your query: {str(e)}",
                'query': query
            }
    
    def _extract_intent_and_entities(self, query: str) -> Tuple[str, dict]:
        """Extract intent and entities from query using pattern matching"""
        entities = {}

        # Try to match query patterns
        for intent, patterns in self.query_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, query, re.IGNORECASE)
                if match:
                    # Extract stock symbols
                    symbols = []
                    for group in match.groups():
                        if group:
                            # First check if it's in our known symbols list
                            if group.upper() in self.popular_symbols:
                                symbols.append(group.upper())
                            else:
                                # Try to match company name to symbol
                                potential_symbol = self._find_symbol_by_name(group)
                                if potential_symbol:
                                    symbols.append(potential_symbol)
                                else:
                                    # Accept any valid-looking ticker symbol (alphanumeric, 1-5 chars typically)
                                    # This allows symbols not in the predefined list (e.g., FDIG, SNOW, etc.)
                                    symbols.append(group.upper())

                    if symbols:
                        entities['symbols'] = symbols

                        # Extract time period
                        time_period = self._extract_time_period(query)
                        if time_period:
                            entities['period'] = time_period

                        return intent, entities
        
        # If no specific pattern matched, try to extract symbols anyway
        symbols = self._extract_symbols_from_text(query)
        if symbols:
            entities['symbols'] = symbols
            time_period = self._extract_time_period(query)
            if time_period:
                entities['period'] = time_period
            return 'analyze', entities  # Default to analyze
        
        return 'general', entities
    
    def _extract_symbols_from_text(self, text: str) -> List[str]:
        """Extract stock symbols from text"""
        symbols = []
        words = text.split()

        for word in words:
            word_upper = word.upper().strip('.,!?')
            # Check if it's in known symbols
            if word_upper in self.popular_symbols:
                symbols.append(word_upper)
            else:
                # Try to find by company name
                potential_symbol = self._find_symbol_by_name(word)
                if potential_symbol:
                    symbols.append(potential_symbol)
                elif len(word_upper) >= 1 and len(word_upper) <= 5 and word_upper.isalpha():
                    # Accept any 1-5 letter alphabetic word as a potential ticker symbol
                    # This catches symbols not in the predefined list (e.g., FDIG)
                    symbols.append(word_upper)

        return list(set(symbols))  # Remove duplicates
    
    def _find_symbol_by_name(self, name: str) -> Optional[str]:
        """Find stock symbol by company name (simplified mapping)"""
        name_to_symbol = {
            'apple': 'AAPL',
            'google': 'GOOGL',
            'alphabet': 'GOOGL',
            'microsoft': 'MSFT',
            'amazon': 'AMZN',
            'tesla': 'TSLA',
            'meta': 'META',
            'facebook': 'META',
            'nvidia': 'NVDA',
            'netflix': 'NFLX',
            'amd': 'AMD',
            'intel': 'INTC',
            'disney': 'DIS',
            'boeing': 'BA',
            'ford': 'F',
            'jp morgan': 'JPM',
            'jpmorgan': 'JPM',
            'goldman sachs': 'GS',
            'bitcoin': 'BTC-USD',
            'ethereum': 'ETH-USD'
        }
        
        name_lower = name.lower()
        return name_to_symbol.get(name_lower)
    
    def _extract_time_period(self, query: str) -> Optional[str]:
        """Extract time period from query"""
        for period_text, days in self.time_patterns.items():
            if period_text in query:
                if days is None:  # YTD
                    return 'ytd'
                elif days <= 7:
                    return '5d'
                elif days <= 30:
                    return '1mo'
                elif days <= 90:
                    return '3mo'
                elif days <= 180:
                    return '6mo'
                elif days <= 365:
                    return '1y'
                else:
                    return '2y'
        
        return '1y'  # Default period
    
    async def _handle_analyze_query(self, entities: Dict[str, Any], prediction_callback: Any = None) -> Dict[str, Any]:
        """Handle stock analysis queries

        Args:
            entities: Extracted entities from query
            prediction_callback: Optional callback for prediction progress updates
        """
        symbols = entities.get('symbols', [])
        if not symbols:
            return {
                'type': 'error',
                'message': 'Please specify a stock symbol to analyze (e.g., AAPL, GOOGL, MSFT)'
            }

        symbol = symbols[0]  # Analyze first symbol
        period = entities.get('period', '1y')
        
        try:
            logger.info(f"[{symbol}] Starting analysis: fetching stock data.")
            stock_data = self.stock_fetcher.fetch_stock_data(symbol, period)
            logger.info(f"[{symbol}] Stock data fetched. Rows: {len(stock_data)}")

            current_data = self.stock_fetcher.get_real_time_price(symbol)
            stock_info = self.stock_fetcher.get_stock_info(symbol)
            logger.info(f"[{symbol}] Current price and info fetched.")
            
            logger.info(f"[{symbol}] Performing technical analysis.")
            technical_analysis = self.technical_analysis.analyze_trends(stock_data)
            trading_signals = self.technical_analysis.generate_trading_signals(technical_analysis)
            logger.info(f"[{symbol}] Technical analysis complete.")
            
            logger.info(f"[{symbol}] Generating analysis text.")
            analysis_text = self._generate_analysis_text(
                symbol, stock_info, technical_analysis, trading_signals
            )
            logger.info(f"[{symbol}] Analysis text generated.")

            predictions = None
            logger.info(f"[{symbol}] Checking if LSTM model is trained.")
            if self.lstm_predictor.is_model_trained(symbol):
                logger.info(f"[{symbol}] LSTM model is trained. Attempting prediction.")
                try:
                    predictions = self.lstm_predictor.predict(symbol, stock_data, prediction_callback=prediction_callback)
                    logger.info(f"[{symbol}] LSTM prediction complete.")
                except Exception as e:
                    logger.warning(f"[{symbol}] Prediction failed: {e}")
            else:
                logger.info(f"[{symbol}] LSTM model not trained. Skipping prediction.")
            
            logger.info(f"[{symbol}] Preparing final analysis result.")
            return {
                'type': 'stock_analysis',
                'symbol': symbol,
                'stock_info': stock_info,
                'current_data': current_data,
                'chart_data': stock_data,
                'technical_analysis': technical_analysis,
                'trading_signals': trading_signals,
                'predictions': predictions,
                'analysis': analysis_text
            }
            
        except Exception as e:
            logger.error(f"Analysis failed for {symbol}: {e}")
            return {
                'type': 'error',
                'message': f"Failed to analyze {symbol}: {str(e)}"
            }
    
    async def _handle_predict_query(self, entities: Dict[str, Any], force_retrain: bool = False, progress_callback: Any = None, training_complete_callback: Any = None, prediction_callback: Any = None) -> Dict[str, Any]:
        """Handle stock prediction queries

        Args:
            entities: Extracted entities from query
            force_retrain: If True, retrain model even if it exists
            progress_callback: Optional callback for training progress
            training_complete_callback: Optional callback when training completes
            prediction_callback: Optional callback for prediction progress
        """
        symbols = entities.get('symbols', [])
        logger.debug(f"Prediction request - entities: {entities}, force_retrain: {force_retrain}")

        if not symbols:
            logger.warning("No symbols found in prediction request")
            return {
                'type': 'error',
                'message': 'Please specify a stock symbol to predict (e.g., AAPL, GOOGL, MSFT)'
            }

        symbol = symbols[0]
        logger.info(f"Processing prediction request for symbol: {symbol}, force_retrain: {force_retrain}")
        
        try:
            # Check if model needs training
            needs_training = force_retrain or not self.lstm_predictor.is_model_trained(symbol)
            training_metrics = None

            if needs_training:
                # Train/retrain model with available data
                action = "Retraining" if force_retrain else "Training new"
                logger.info(f"{action} model for {symbol}")
                stock_data = self.stock_fetcher.fetch_stock_data(symbol, '2y')

                if len(stock_data) < 120:  # Need minimum data
                    return {
                        'type': 'error',
                        'message': f"Insufficient data to train prediction model for {symbol}. Need at least 120 days of data."
                    }

                try:
                    training_metrics = self.lstm_predictor.train_ensemble(
                        stock_data,
                        symbol,
                        progress_callback=progress_callback
                    )
                    logger.info(f"Model trained for {symbol} with RMSE: {training_metrics['rmse']:.4f}")

                    # Show training history immediately after training completes
                    if training_complete_callback:
                        model_info = self.lstm_predictor.get_model_info(symbol)
                        training_complete_callback({
                            'symbol': symbol,
                            'model_info': model_info,
                            'training_metrics': training_metrics
                        })

                except Exception as e:
                    return {
                        'type': 'error',
                        'message': f"Failed to train prediction model for {symbol}: {str(e)}"
                    }

            # Generate predictions
            logger.debug(f"Fetching stock data for predictions: {symbol}")
            stock_data = self.stock_fetcher.fetch_stock_data(symbol, '1y')
            logger.debug(f"Stock data fetched: {len(stock_data)} rows")

            logger.debug(f"Generating predictions for {symbol}")
            predictions = self.lstm_predictor.predict(symbol, stock_data, prediction_callback=prediction_callback)
            logger.info(f"Predictions generated for {symbol}: {len(predictions.get('predictions', []))} days")
            
            # Get current data for context
            logger.debug(f"Fetching current data for {symbol}")
            current_data = self.stock_fetcher.get_real_time_price(symbol)

            # Get model info for training history
            model_info = self.lstm_predictor.get_model_info(symbol)

            result = {
                'type': 'prediction',
                'symbol': symbol,
                'predictions': predictions,
                'current_data': current_data,
                'chart_data': stock_data.tail(60),  # Show recent data for context
                'training_metrics': training_metrics,  # Include training metrics if just trained
                'model_info': model_info,  # Include model metadata with training history
                'was_retrained': needs_training,  # True if model was just trained
                'message': f"Generated {predictions['prediction_period_days']}-day price prediction for {symbol}"
            }
            logger.info(f"Prediction result prepared for {symbol}")
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed for {symbol}: {e}")
            return {
                'type': 'error',
                'message': f"Failed to generate predictions for {symbol}: {str(e)}"
            }
    
    async def _handle_compare_query(self, entities: Dict[str, Any]) -> Dict[str, Any]:
        """Handle stock comparison queries"""
        symbols = entities.get('symbols', [])
        if len(symbols) < 2:
            return {
                'type': 'error',
                'message': 'Please specify at least two stock symbols to compare (e.g., AAPL vs GOOGL)'
            }
        
        period = entities.get('period', '1y')
        
        try:
            # Fetch data for all symbols
            comparison_data = {}
            for symbol in symbols[:5]:  # Limit to 5 symbols for performance
                try:
                    data = self.stock_fetcher.fetch_stock_data(symbol, period)
                    comparison_data[symbol] = data
                except Exception as e:
                    logger.warning(f"Failed to fetch data for {symbol}: {e}")
            
            if not comparison_data:
                return {
                    'type': 'error',
                    'message': 'Failed to fetch data for any of the specified symbols'
                }
            
            return {
                'type': 'comparison',
                'symbols': list(comparison_data.keys()),
                'comparison_data': comparison_data,
                'message': f"Comparison of {', '.join(comparison_data.keys())} over {period}"
            }
            
        except Exception as e:
            logger.error(f"Comparison failed: {e}")
            return {
                'type': 'error',
                'message': f"Failed to compare stocks: {str(e)}"
            }
    
    async def _handle_price_query(self, entities: Dict[str, Any]) -> Dict[str, Any]:
        """Handle current price queries"""
        symbols = entities.get('symbols', [])
        if not symbols:
            return {
                'type': 'error',
                'message': 'Please specify a stock symbol to get price (e.g., AAPL price)'
            }
        
        symbol = symbols[0]
        
        try:
            current_data = self.stock_fetcher.get_real_time_price(symbol)
            stock_info = self.stock_fetcher.get_stock_info(symbol)
            
            return {
                'type': 'price_info',
                'symbol': symbol,
                'current_data': current_data,
                'stock_info': stock_info,
                'message': f"Current price information for {symbol}"
            }
            
        except Exception as e:
            logger.error(f"Price query failed for {symbol}: {e}")
            return {
                'type': 'error',
                'message': f"Failed to get price for {symbol}: {str(e)}"
            }
    
    def _handle_general_query(self, query: str) -> Dict[str, Any]:
        """Handle general queries with helpful suggestions"""
        return {
            'type': 'general',
            'message': "I can help you analyze stocks, generate predictions, and compare different stocks.",
            'suggestions': [
                "Try: 'Analyze AAPL for the last 6 months'",
                "Try: 'Predict GOOGL'",
                "Try: 'Compare AAPL vs MSFT'",
                "Try: 'What's the current price of TSLA?'"
            ],
            'query': query
        }
    
    def _generate_analysis_text(
        self, 
        symbol: str, 
        stock_info: Dict[str, Any], 
        technical_analysis: Dict[str, Any], 
        trading_signals: Dict[str, Any]
    ) -> str:
        """Generate human-readable analysis text"""
        
        company_name = stock_info.get('name', symbol)
        trend = technical_analysis.get('overall_trend', 'Unknown')
        signal = trading_signals.get('primary_signal', 'HOLD')
        confidence = trading_signals.get('confidence', 0)
        
        latest_values = technical_analysis.get('latest_values', {})
        price = latest_values.get('price', 0)
        rsi = latest_values.get('rsi', 0)
        
        analysis = f"Analysis for {company_name} ({symbol}):\n\n"
        analysis += f"Current trend: {trend}\n"
        analysis += f"Recommended action: {signal} (Confidence: {confidence:.1f}%)\n\n"
        
        if rsi:
            if rsi > 70:
                analysis += f"RSI is {rsi:.1f}, indicating overbought conditions. "
            elif rsi < 30:
                analysis += f"RSI is {rsi:.1f}, indicating oversold conditions. "
            else:
                analysis += f"RSI is {rsi:.1f}, indicating neutral momentum. "
        
        # Add risk factors
        risk_factors = trading_signals.get('risk_factors', [])
        if risk_factors:
            analysis += f"\n\nRisk factors to consider:\n"
            for factor in risk_factors:
                analysis += f"• {factor}\n"
        
        # Add recommendations
        recommendations = trading_signals.get('recommendations', [])
        if recommendations:
            analysis += f"\nRecommendations:\n"
            for rec in recommendations:
                analysis += f"• {rec}\n"
        
        analysis += "\n⚠️ This analysis is for educational purposes only and should not be considered as financial advice."
        
        return analysis
    
    def _check_hybrid_availability(self) -> bool:
        """Check if HybridQueryProcessor is available for migration"""
        try:
            from src.agents.hybrid_query_processor import HybridQueryProcessor
            return True
        except ImportError:
            return False
    
    def get_hybrid_processor(self):
        """Get HybridQueryProcessor instance for enhanced functionality"""
        if not self.is_hybrid_available:
            logger.warning("HybridQueryProcessor not available. Using legacy QueryProcessor.")
            return None
        
        try:
            from src.agents.hybrid_query_processor import HybridQueryProcessor
            return HybridQueryProcessor()
        except Exception as e:
            logger.error(f"Failed to create HybridQueryProcessor: {e}")
            return None
    
    @classmethod
    def create_enhanced_processor(cls):
        """Factory method to create the best available processor"""
        try:
            from src.agents.hybrid_query_processor import HybridQueryProcessor
            logger.info("Creating HybridQueryProcessor with Ollama integration")
            return HybridQueryProcessor()
        except ImportError:
            logger.info("HybridQueryProcessor not available, using legacy QueryProcessor")
            return cls()
        except Exception as e:
            logger.warning(f"Failed to create HybridQueryProcessor: {e}, falling back to legacy")
            return cls()
    
    def suggest_hybrid_upgrade(self) -> Dict[str, Any]:
        """Suggest upgrading to hybrid processor for enhanced features"""
        if not self.is_hybrid_available:
            return {
                'upgrade_available': False,
                'message': 'HybridQueryProcessor not available'
            }
        
        return {
            'upgrade_available': True,
            'message': 'Enhanced query processing with Ollama integration is available',
            'benefits': [
                'Natural language understanding with AI',
                'Educational explanations for technical concepts',
                'Multi-turn conversation support',
                'Contextual response generation',
                'Fallback to regex patterns for reliability'
            ],
            'usage': 'Use QueryProcessor.create_enhanced_processor() or import HybridQueryProcessor directly'
        }