from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain.memory import ConversationBufferMemory
from langchain_core.callbacks import BaseCallbackHandler
from typing import List, Dict, Any
import logging

from stock_fetcher import StockFetcher
from lstm_predictor import LSTMPredictor
from visualizer import StockVisualizer

logger = logging.getLogger(__name__)


class AgentLoggingCallback(BaseCallbackHandler):
    """Custom callback to log agent thoughts and actions"""
    
    def on_agent_action(self, action, **kwargs):
        logger.info(f"🤖 AGENT ACTION:")
        logger.info(f"   Tool: {action.tool}")
        logger.info(f"   Input: {action.tool_input}")
        if hasattr(action, 'log') and action.log:
            # Extract the thought/reasoning part from the log
            log_lines = action.log.split('\n')
            for line in log_lines:
                line = line.strip()
                if line and not line.startswith('Action') and not line.startswith('Observation'):
                    # Show the agent's thinking process
                    if line.startswith('Thought:'):
                        logger.info(f"   💭 {line}")
                    elif 'think' in line.lower() or 'need' in line.lower() or 'should' in line.lower():
                        logger.info(f"   💭 REASONING: {line}")
                    else:
                        logger.info(f"   💭 {line}")
    
    def on_agent_finish(self, finish, **kwargs):
        logger.info(f"🎯 AGENT FINISH:")
        output = finish.return_values.get('output', '')
        logger.info(f"   Final Answer: {output[:300]}{'...' if len(output) > 300 else ''}")
    
    def on_tool_start(self, serialized, input_str, **kwargs):
        tool_name = serialized.get("name", "Unknown")
        logger.info(f"🔧 TOOL START: {tool_name}")
    
    def on_tool_end(self, output, **kwargs):
        logger.info(f"✅ TOOL COMPLETED")
    
    def on_tool_error(self, error, **kwargs):
        logger.error(f"❌ TOOL ERROR: {str(error)}")
    
    def on_text(self, text, **kwargs):
        # This captures the agent's reasoning text
        if text.strip() and not text.startswith('Invoking:'):
            # Clean up the text and show reasoning
            clean_text = text.strip()
            if clean_text and len(clean_text) > 3:  # Avoid logging very short text
                # Highlight different types of reasoning
                if clean_text.startswith('Thought:'):
                    logger.info(f"💭 {clean_text}")
                elif clean_text.startswith('I need to') or clean_text.startswith('I should'):
                    logger.info(f"🧠 PLANNING: {clean_text}")
                elif 'analyze' in clean_text.lower() or 'fetch' in clean_text.lower():
                    logger.info(f"📊 STRATEGY: {clean_text}")
                else:
                    logger.info(f"💭 REASONING: {clean_text}")
    
    def on_agent_start(self, serialized, inputs, **kwargs):
        logger.info(f"🚀 AGENT STARTING: Analyzing query...")
        query = inputs.get('input', 'Unknown query')[:100]
        logger.info(f"   Query: {query}{'...' if len(str(inputs.get('input', ''))) > 100 else ''}")
    
    def on_llm_start(self, serialized, prompts, **kwargs):
        logger.info(f"🤖 LLM THINKING: Processing agent reasoning...")
    
    def on_llm_end(self, response, **kwargs):
        pass  # Skip LLM end logging to reduce noise


class StockAnalysisAgent:
    def __init__(self, model_name: str = "gemma3:latest"):
        logger.info(f"Initializing StockAnalysisAgent with model: {model_name}")
        
        # Initialize Ollama LLM
        self.llm = Ollama(
            model=model_name, 
            temperature=0.3,  # Slightly higher for better format following
            verbose=False,
            timeout=60  # Add timeout to prevent hanging
        )
        
        # Initialize tools
        self.tools = [
            StockFetcher(),
            LSTMPredictor(),
            StockVisualizer()
        ]
        
        # Create agent prompt
        self.prompt = PromptTemplate.from_template(
            """You are a stock market analysis expert with access to powerful tools for fetching data, 
            making predictions, and creating visualizations. Your role is to help users analyze stocks 
            and provide insights about trends and future predictions.

            Available tools:
            {tools}

            Use the following format EXACTLY:

            Question: the input question you must answer
            Thought: you should always think about what to do
            Action: the action to take, should be one of [{tool_names}]
            Action Input: the input to the action
            Observation: the result of the action
            ... (this Thought/Action/Action Input/Observation can repeat N times)
            Thought: I now know the final answer
            Final Answer: the final answer to the original input question

            IMPORTANT: 
            - Always follow Thought with Action and Action Input
            - Never provide Final Answer until you have used tools to gather information
            - Use JSON format for Action Input like: {{"symbol": "AAPL", "period": "2y"}}
            - Be precise with the format - each line must start with the exact keywords

            Question: {input}
            Thought: {agent_scratchpad}
            """
        )
        
        # Create memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Create agent
        try:
            self.agent = create_react_agent(
                llm=self.llm,
                tools=self.tools,
                prompt=self.prompt
            )
            logger.info("ReAct agent created successfully")
        except Exception as e:
            logger.error(f"Failed to create ReAct agent: {e}")
            raise
        
        # Create callback handler for detailed logging
        self.callback_handler = AgentLoggingCallback()
        
        # Create agent executor
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,  # Enable to capture agent reasoning
            handle_parsing_errors="Check your output and make sure it conforms to the format instructions!",
            max_iterations=5,  # Reduce to prevent infinite loops
            return_intermediate_steps=False,  # Set to False to avoid the multiple keys issue
            callbacks=[self.callback_handler]
        )
        logger.info("StockAnalysisAgent initialization completed")
    
    def analyze_stock(self, query: str) -> Dict[str, Any]:
        """
        Main method to analyze stocks based on user query
        """
        logger.info(f"Agent analyzing query: '{query[:100]}{'...' if len(query) > 100 else ''}'")
        
        try:
            # Execute the agent with error handling
            try:
                result = self.agent_executor.invoke({"input": query})
            except Exception as invoke_error:
                logger.warning(f"Invoke method failed: {invoke_error}")
                # Try alternative execution method
                try:
                    result = self.agent_executor.run(query)
                    # If run() returns a string directly, wrap it
                    if isinstance(result, str):
                        result = {"output": result}
                except Exception as run_error:
                    logger.error(f"Both invoke and run methods failed: {run_error}")
                    raise invoke_error
            
            # Debug: Log the result structure
            logger.info(f"Agent result type: {type(result)}")
            if isinstance(result, dict):
                logger.info(f"Agent result keys: {list(result.keys())}")
            
            # Handle different result formats
            if isinstance(result, dict):
                if "output" in result:
                    response = result["output"]
                elif "result" in result:
                    response = result["result"]
                else:
                    # If result is a dict but no expected key, take the first string value
                    for key, value in result.items():
                        if isinstance(value, str) and value.strip():
                            response = value
                            logger.info(f"Using key '{key}' as response")
                            break
                    else:
                        response = str(result)
                        logger.warning("No string value found in result, converting entire dict to string")
            else:
                # If result is not a dict, convert to string
                response = str(result)
                logger.info("Result is not a dict, converting to string")
            
            response_length = len(response)
            logger.info(f"✨ Agent analysis completed. Response length: {response_length} characters")
            
            return {
                "success": True,
                "response": response,
                "intermediate_steps": []
            }
            
        except Exception as e:
            logger.error(f"Agent analysis failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "response": f"I encountered an error while analyzing the stock: {str(e)}"
            }
    
    def get_conversation_history(self) -> List[Dict]:
        """Get the conversation history"""
        return self.memory.chat_memory.messages


class StockWorkflow:
    """Complete workflow for stock analysis"""
    
    def __init__(self, agent: StockAnalysisAgent):
        self.agent = agent
        self.stock_fetcher = StockFetcher()
        self.lstm_predictor = LSTMPredictor()
        self.visualizer = StockVisualizer()
    
    def complete_analysis_with_progress(self, symbol: str, update_progress_callback, start_step: int, total_steps: int, period: str = "2y") -> Dict[str, Any]:
        """
        Perform complete stock analysis workflow with progress updates
        """
        logger.info(f"Starting complete analysis workflow for {symbol} with period {period}")
        
        current_step = start_step
        
        try:
            # Step 1: Fetch stock data
            update_progress_callback(current_step + 1, total_steps, f"Fetching {symbol} historical data...")
            stock_data = self.stock_fetcher._run(symbol=symbol, period=period)
            
            if "error" in stock_data:
                logger.error(f"Stock data fetch failed: {stock_data['error']}")
                return {"error": stock_data["error"]}
            
            logger.info(f"Fetched {stock_data['total_records']} records for {symbol}")
            
            # Step 2: Generate predictions
            update_progress_callback(current_step + 2, total_steps, f"Training LSTM model for {symbol}...")
            prediction_result = self.lstm_predictor._run(
                data=stock_data["data"],
                symbol=symbol,
                prediction_days=30
            )
            
            if "error" in prediction_result:
                logger.error(f"LSTM prediction failed: {prediction_result['error']}")
                return {"error": prediction_result["error"]}
            
            logger.info("LSTM predictions generated successfully")
            
            # Step 3: Create visualizations
            update_progress_callback(current_step + 3, total_steps, f"Creating {symbol} charts and visualizations...")
            visualization_result = self.visualizer._run(
                historical_data=stock_data["data"],
                predictions=prediction_result["predictions"],
                future_dates=prediction_result["future_dates"],
                symbol=symbol,
                trend_analysis=prediction_result["trend_analysis"]
            )
            
            if "error" in visualization_result:
                logger.error(f"Visualization creation failed: {visualization_result['error']}")
                return {"error": visualization_result["error"]}
            
            logger.info("Visualizations created successfully")
            
            # Step 4: Generate AI analysis
            update_progress_callback(current_step + 4, total_steps, f"Generating AI analysis for {symbol}...")
            analysis_prompt = f"""Based on the following stock analysis data for {symbol}:

**Company Information:**
- Name: {stock_data.get('company_name', 'Unknown')}
- Current Price: ${stock_data.get('current_price', 'N/A')}
- Market Cap: {stock_data.get('market_cap', 'N/A')}
- P/E Ratio: {stock_data.get('pe_ratio', 'N/A')}
- Historical Data: {stock_data.get('total_records', 0)} records

**LSTM Prediction Results:**
- Trend Direction: {prediction_result['trend_analysis']['direction']}
- Current Price: ${prediction_result['trend_analysis']['current_price']:.2f}
- Predicted 30-day Price: ${prediction_result['trend_analysis']['predicted_price_30d']:.2f}
- Expected Change: {prediction_result['trend_analysis']['percentage_change']:.2f}%
- Price Change: ${prediction_result['trend_analysis']['price_change']:.2f}
- Model Confidence: {prediction_result['prediction_confidence']}

Please provide a comprehensive stock analysis including:
1. **Current Market Position**: Evaluate the stock's current valuation and market standing
2. **Trend Analysis**: Interpret the predicted trend and what it means for investors
3. **Risk Assessment**: Analyze potential risks and volatility factors
4. **Investment Insights**: Provide educational insights (not financial advice)

Keep the analysis professional, informative, and include relevant context about the company and market conditions."""

            try:
                ai_analysis = self.agent.llm.invoke(analysis_prompt)
                logger.info("AI analysis generated successfully using direct LLM call")
            except Exception as llm_error:
                logger.warning(f"Direct LLM analysis failed: {llm_error}")
                ai_analysis = f"""**Stock Analysis for {symbol} ({stock_data.get('company_name', 'Unknown')})**

**Current Status:**
- Current Price: ${prediction_result['trend_analysis']['current_price']:.2f}
- Market Position: Based on {stock_data.get('total_records', 0)} days of data

**LSTM Predictions:**
- 30-day Forecast: ${prediction_result['trend_analysis']['predicted_price_30d']:.2f}
- Trend Direction: {prediction_result['trend_analysis']['direction']}
- Expected Change: {prediction_result['trend_analysis']['percentage_change']:.2f}%
- Model Confidence: {prediction_result['prediction_confidence']}

**Key Insights:**
The LSTM model predicts a {prediction_result['trend_analysis']['direction'].lower()} trend with {prediction_result['prediction_confidence'].lower()} confidence. The forecasted price movement represents a {abs(prediction_result['trend_analysis']['percentage_change']):.1f}% change over the next 30 days.

*Note: This analysis is for educational purposes only and should not be considered financial advice.*"""
            
            result = {
                "success": True,
                "stock_data": stock_data,
                "predictions": prediction_result,
                "visualizations": visualization_result,
                "ai_analysis": ai_analysis,
                "summary": {
                    "symbol": symbol,
                    "company_name": stock_data.get('company_name', 'Unknown'),
                    "current_price": stock_data.get('current_price', 0),
                    "predicted_price": prediction_result['trend_analysis']['predicted_price_30d'],
                    "trend_direction": prediction_result['trend_analysis']['direction'],
                    "confidence": prediction_result['prediction_confidence'],
                    "percentage_change": prediction_result['trend_analysis']['percentage_change']
                }
            }
            
            logger.info(f"Complete analysis workflow finished successfully for {symbol}")
            return result
            
        except Exception as e:
            logger.error(f"Workflow error for {symbol}: {str(e)}")
            return {"error": f"Workflow error: {str(e)}"}

    def complete_analysis(self, symbol: str, period: str = "2y") -> Dict[str, Any]:
        """
        Perform complete stock analysis workflow
        """
        logger.info(f"Starting complete analysis workflow for {symbol} with period {period}")
        
        try:
            # Step 1: Fetch stock data
            stock_data = self.stock_fetcher._run(symbol=symbol, period=period)
            
            if "error" in stock_data:
                logger.error(f"Stock data fetch failed: {stock_data['error']}")
                return {"error": stock_data["error"]}
            
            logger.info(f"Fetched {stock_data['total_records']} records for {symbol}")
            
            # Step 2: Generate predictions
            prediction_result = self.lstm_predictor._run(
                data=stock_data["data"],
                symbol=symbol,
                prediction_days=30
            )
            
            if "error" in prediction_result:
                logger.error(f"LSTM prediction failed: {prediction_result['error']}")
                return {"error": prediction_result["error"]}
            
            logger.info("LSTM predictions generated successfully")
            
            # Step 3: Create visualizations
            visualization_result = self.visualizer._run(
                historical_data=stock_data["data"],
                predictions=prediction_result["predictions"],
                future_dates=prediction_result["future_dates"],
                symbol=symbol,
                trend_analysis=prediction_result["trend_analysis"]
            )
            
            if "error" in visualization_result:
                logger.error(f"Visualization creation failed: {visualization_result['error']}")
                return {"error": visualization_result["error"]}
            
            logger.info("Visualizations created successfully")
            
            # Step 4: Generate AI analysis using Ollama directly (no tools needed)
            analysis_prompt = f"""Based on the following stock analysis data for {symbol}:

**Company Information:**
- Name: {stock_data.get('company_name', 'Unknown')}
- Current Price: ${stock_data.get('current_price', 'N/A')}
- Market Cap: {stock_data.get('market_cap', 'N/A')}
- P/E Ratio: {stock_data.get('pe_ratio', 'N/A')}
- Historical Data: {stock_data.get('total_records', 0)} records

**LSTM Prediction Results:**
- Trend Direction: {prediction_result['trend_analysis']['direction']}
- Current Price: ${prediction_result['trend_analysis']['current_price']:.2f}
- Predicted 30-day Price: ${prediction_result['trend_analysis']['predicted_price_30d']:.2f}
- Expected Change: {prediction_result['trend_analysis']['percentage_change']:.2f}%
- Price Change: ${prediction_result['trend_analysis']['price_change']:.2f}
- Model Confidence: {prediction_result['prediction_confidence']}

Please provide a comprehensive stock analysis including:
1. **Current Market Position**: Evaluate the stock's current valuation and market standing
2. **Trend Analysis**: Interpret the predicted trend and what it means for investors
3. **Risk Assessment**: Analyze potential risks and volatility factors
4. **Investment Insights**: Provide educational insights (not financial advice)

Keep the analysis professional, informative, and include relevant context about the company and market conditions."""

            try:
                ai_analysis = self.agent.llm.invoke(analysis_prompt)
                logger.info("AI analysis generated successfully using direct LLM call")
            except Exception as llm_error:
                logger.warning(f"Direct LLM analysis failed: {llm_error}")
                ai_analysis = f"""**Stock Analysis for {symbol} ({stock_data.get('company_name', 'Unknown')})**

**Current Status:**
- Current Price: ${prediction_result['trend_analysis']['current_price']:.2f}
- Market Position: Based on {stock_data.get('total_records', 0)} days of data

**LSTM Predictions:**
- 30-day Forecast: ${prediction_result['trend_analysis']['predicted_price_30d']:.2f}
- Trend Direction: {prediction_result['trend_analysis']['direction']}
- Expected Change: {prediction_result['trend_analysis']['percentage_change']:.2f}%
- Model Confidence: {prediction_result['prediction_confidence']}

**Key Insights:**
The LSTM model predicts a {prediction_result['trend_analysis']['direction'].lower()} trend with {prediction_result['prediction_confidence'].lower()} confidence. The forecasted price movement represents a {abs(prediction_result['trend_analysis']['percentage_change']):.1f}% change over the next 30 days.

*Note: This analysis is for educational purposes only and should not be considered financial advice.*"""
            
            result = {
                "success": True,
                "stock_data": stock_data,
                "predictions": prediction_result,
                "visualizations": visualization_result,
                "ai_analysis": ai_analysis,
                "summary": {
                    "symbol": symbol,
                    "company_name": stock_data.get('company_name', 'Unknown'),
                    "current_price": stock_data.get('current_price', 0),
                    "predicted_price": prediction_result['trend_analysis']['predicted_price_30d'],
                    "trend_direction": prediction_result['trend_analysis']['direction'],
                    "confidence": prediction_result['prediction_confidence'],
                    "percentage_change": prediction_result['trend_analysis']['percentage_change']
                }
            }
            
            logger.info(f"Complete analysis workflow finished successfully for {symbol}")
            return result
            
        except Exception as e:
            logger.error(f"Workflow error for {symbol}: {str(e)}")
            return {"error": f"Workflow error: {str(e)}"}


# Pre-defined query templates for common analysis types
QUERY_TEMPLATES = {
    "basic_analysis": "Analyze the stock {symbol} and provide insights on its current performance and future outlook.",
    "trend_analysis": "Focus on the trending patterns of {symbol} stock and predict short-term movements.",
    "risk_assessment": "Evaluate the risk profile of investing in {symbol} stock based on historical data and predictions.",
    "comparison": "Compare {symbol1} and {symbol2} stocks and recommend which might be a better investment.",
    "sector_analysis": "Analyze {symbol} in the context of its sector performance and market conditions."
}