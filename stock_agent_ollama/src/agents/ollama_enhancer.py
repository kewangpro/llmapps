import json
import asyncio
import aiohttp
from typing import Dict, Any, List, Optional, Tuple
import logging
from datetime import datetime, timedelta

from src.config import Config

logger = logging.getLogger(__name__)

class OllamaEnhancer:
    """Enhanced query processing and educational content generation using Ollama"""
    
    def __init__(self):
        self.base_url = Config.OLLAMA_BASE_URL
        self.model = Config.OLLAMA_MODEL
        self.timeout = Config.OLLAMA_TIMEOUT
        self.enabled = Config.OLLAMA_ENABLED
        self.session = None
        
        # Educational prompts for different contexts
        self.prompts = {
            'query_understanding': """You are a stock analysis assistant. Parse this natural language query and extract:
1. Intent (analyze, predict, compare, price, explain)
2. Stock symbols mentioned (convert company names to symbols)
3. Time period requested
4. Specific analysis type requested

Query: "{query}"

Respond with JSON only:
{{
    "intent": "analyze|predict|compare|price|explain",
    "symbols": ["AAPL", "GOOGL"],
    "time_period": "1y|6m|3m|1m|5d",
    "analysis_type": "technical|fundamental|trend|prediction",
    "confidence": 0.85,
    "needs_explanation": true
}}""",
            
            'technical_explanation': """You are an expert financial educator. Explain these technical indicators in simple terms for educational purposes:

Technical Analysis Results:
{technical_data}

Stock: {symbol}
Current Context: {context}

Provide a clear, educational explanation covering:
1. What each indicator means
2. Current values and interpretation
3. What this suggests about the stock's trend
4. Educational insights for learning
5. Risk considerations

End with: "This analysis is for educational purposes only and should not be considered financial advice."
""",
            
            'prediction_interpretation': """You are a financial education expert. Help explain this LSTM prediction model output:

Prediction Data:
{prediction_data}

Stock: {symbol}
Model Metrics: {model_metrics}

Explain in educational terms:
1. How LSTM models work for stock prediction
2. What these predictions suggest
3. Model reliability and limitations
4. Why machine learning predictions have uncertainty
5. How to interpret confidence intervals

Emphasize educational value and limitations. End with disclaimer about not being financial advice.""",
            
            'conversation_context': """You are a knowledgeable stock analysis tutor. The user is asking about: "{current_query}"

Previous conversation context:
{conversation_history}

Current analysis results:
{analysis_results}

Provide an educational response that:
1. Directly answers their question
2. Builds on previous discussion
3. Explains concepts clearly
4. Suggests follow-up learning opportunities
5. Maintains educational focus

Keep responses conversational but informative. Always include educational disclaimers."""
        }
    
    async def _ensure_session(self):
        """Ensure aiohttp session is available"""
        if self.session is None:
            connector = aiohttp.TCPConnector(limit=10)
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout
            )
    
    async def close(self):
        """Close aiohttp session"""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def _call_ollama(self, prompt: str, temperature: float = 0.7) -> Optional[str]:
        """Call Ollama API with error handling and fallback"""
        if not self.enabled:
            logger.debug("Ollama integration disabled")
            return None
            
        try:
            await self._ensure_session()
            
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "top_p": 0.9,
                    "num_predict": 500
                }
            }
            
            async with self.session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get("response", "").strip()
                else:
                    logger.warning(f"Ollama API returned status {response.status}")
                    return None
                    
        except asyncio.TimeoutError:
            logger.warning("Ollama request timed out")
            return None
        except aiohttp.ClientError as e:
            logger.warning(f"Ollama client error: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error calling Ollama: {e}")
            return None
    
    async def understand_query(self, query: str) -> Optional[Dict[str, Any]]:
        """Use Ollama to understand natural language queries"""
        prompt = self.prompts['query_understanding'].format(query=query)
        
        response = await self._call_ollama(prompt, temperature=0.3)
        if not response:
            return None
            
        try:
            # Handle potential markdown code block wrapping (common with gemma models)
            json_text = response
            if response.startswith('```json'):
                # Extract JSON from markdown code block
                start = response.find('{')
                end = response.rfind('}') + 1
                if start != -1 and end != 0:
                    json_text = response[start:end]
            
            # Try to parse JSON response
            result = json.loads(json_text)
            
            # Validate required fields
            required_fields = ['intent', 'symbols', 'confidence']
            if all(field in result for field in required_fields):
                return result
            else:
                logger.warning(f"Ollama response missing required fields: {response}")
                return None
                
        except json.JSONDecodeError:
            logger.warning(f"Could not parse Ollama JSON response: {response}")
            return None
    
    async def explain_technical_analysis(
        self, 
        symbol: str, 
        technical_data: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> Optional[str]:
        """Generate educational explanations for technical analysis"""
        context_str = json.dumps(context or {}, indent=2)
        technical_str = json.dumps(technical_data, indent=2, default=str)
        
        prompt = self.prompts['technical_explanation'].format(
            symbol=symbol,
            technical_data=technical_str,
            context=context_str
        )
        
        explanation = await self._call_ollama(prompt, temperature=0.5)
        return explanation
    
    async def interpret_prediction(
        self,
        symbol: str,
        prediction_data: Dict[str, Any],
        model_metrics: Dict[str, Any] = None
    ) -> Optional[str]:
        """Generate educational interpretations of LSTM predictions"""
        prediction_str = json.dumps(prediction_data, indent=2, default=str)
        metrics_str = json.dumps(model_metrics or {}, indent=2, default=str)
        
        prompt = self.prompts['prediction_interpretation'].format(
            symbol=symbol,
            prediction_data=prediction_str,
            model_metrics=metrics_str
        )
        
        interpretation = await self._call_ollama(prompt, temperature=0.5)
        return interpretation
    
    async def generate_contextual_response(
        self,
        current_query: str,
        conversation_history: List[Dict[str, Any]],
        analysis_results: Dict[str, Any]
    ) -> Optional[str]:
        """Generate contextual responses for multi-turn conversations"""
        # Format conversation history
        history_str = ""
        for i, turn in enumerate(conversation_history[-3:]):  # Last 3 turns
            history_str += f"Turn {i+1}:\n"
            history_str += f"  Query: {turn.get('query', '')}\n"
            history_str += f"  Response: {turn.get('response', '')[:200]}...\n\n"
        
        results_str = json.dumps(analysis_results, indent=2, default=str)
        
        prompt = self.prompts['conversation_context'].format(
            current_query=current_query,
            conversation_history=history_str,
            analysis_results=results_str
        )
        
        response = await self._call_ollama(prompt, temperature=0.6)
        return response
    
    async def health_check(self) -> bool:
        """Check if Ollama is available and responsive"""
        if not self.enabled:
            return False
            
        try:
            await self._ensure_session()
            
            async with self.session.get(f"{self.base_url}/api/tags") as response:
                if response.status == 200:
                    data = await response.json()
                    models = [model.get('name', '') for model in data.get('models', [])]
                    return self.model in models or any(self.model in model for model in models)
                return False
                
        except Exception as e:
            logger.debug(f"Ollama health check failed: {e}")
            return False
    
    def get_educational_suggestions(self, intent: str, symbols: List[str]) -> List[str]:
        """Generate educational follow-up suggestions"""
        suggestions = []
        
        base_suggestions = {
            'analyze': [
                f"Learn about technical indicators for {symbols[0] if symbols else 'stocks'}",
                "Understand what RSI and MACD signals mean",
                "Explore trend analysis concepts"
            ],
            'predict': [
                "Learn how LSTM neural networks work for predictions",
                "Understand machine learning limitations in finance",
                "Explore different prediction models"
            ],
            'compare': [
                "Learn about relative performance analysis",
                "Understand correlation between stocks",
                "Explore sector comparison techniques"
            ]
        }
        
        suggestions.extend(base_suggestions.get(intent, [
            "Learn about fundamental vs technical analysis",
            "Understand risk management principles",
            "Explore portfolio diversification concepts"
        ]))
        
        return suggestions[:3]  # Return top 3 suggestions

# Singleton instance
ollama_enhancer = OllamaEnhancer()