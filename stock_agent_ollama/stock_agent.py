from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain.memory import ConversationBufferMemory
from langchain_core.callbacks import BaseCallbackHandler
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from typing import List, Dict, Any
import logging
import re

from stock_fetcher import StockFetcher
from lstm_predictor import LSTMPredictor
from visualizer import StockVisualizer
from general_assistant import GeneralAssistant

logger = logging.getLogger(__name__)


class CleanOutputParser(ReActSingleInputOutputParser):
    """Custom parser to clean markdown formatting from LLM output"""
    
    def __init__(self):
        super().__init__()
        # Use a class variable to store fallback symbol and period
        CleanOutputParser._fallback_symbol = "AAPL"
        CleanOutputParser._fallback_period = "2y"
    
    def set_fallback_symbol(self, symbol: str):
        """Set the fallback symbol to use when parsing fails"""
        CleanOutputParser._fallback_symbol = symbol if symbol else "AAPL"
    
    def set_fallback_period(self, period: str):
        """Set the fallback period to use when parsing fails"""
        CleanOutputParser._fallback_period = period if period else "2y"
    
    @property
    def fallback_symbol(self):
        return getattr(CleanOutputParser, '_fallback_symbol', 'AAPL')
    
    @property
    def fallback_period(self):
        return getattr(CleanOutputParser, '_fallback_period', '2y')
    
    def parse(self, text: str):
        logger.info(f"Parsing LLM output: {text[:200]}...")
        
        # Clean any markdown formatting
        cleaned_text = re.sub(r'```+.*?```+', '', text, flags=re.DOTALL)
        cleaned_text = re.sub(r'`+', '', cleaned_text)
        
        # First try standard parsing which handles Final Answer
        try:
            result = super().parse(cleaned_text)
            logger.info("Standard parsing successful")
            return result
        except Exception as e:
            logger.warning(f"Standard parsing failed: {e}")
            
            # Check if this is a Final Answer (agent is done)
            if "Final Answer:" in cleaned_text:
                logger.info("Found Final Answer in text")
                from langchain.schema import AgentFinish
                final_answer = cleaned_text.split("Final Answer:")[-1].strip()
                return AgentFinish(
                    return_values={"output": final_answer},
                    log=cleaned_text
                )
            
            # Look for Action and Action Input patterns more flexibly
            action_pattern = r'Action:\s*([^\n]+)'
            input_pattern = r'Action Input:\s*(\{[^}]*\})'
            
            action_match = re.search(action_pattern, cleaned_text, re.IGNORECASE)
            input_match = re.search(input_pattern, cleaned_text)
            
            if action_match and input_match:
                from langchain.schema import AgentAction
                tool_name = action_match.group(1).strip()
                tool_input = input_match.group(1)
                logger.info(f"Parsed action: {tool_name}, input: {tool_input}")
                return AgentAction(
                    tool=tool_name,
                    tool_input=tool_input,
                    log=cleaned_text
                )
            
            # If no clear action found, try to infer based on content and fallback
            if any(phrase in cleaned_text.lower() for phrase in ["fetch", "get data", "historical"]):
                logger.warning("Inferring stock_fetcher action from text content")
                from langchain.schema import AgentAction
                return AgentAction(
                    tool="stock_fetcher",
                    tool_input=f'{{"symbol": "{self.fallback_symbol}", "period": "{self.fallback_period}"}}',
                    log=f"Inferred action from content: {cleaned_text[:100]}"
                )
            
            # Last resort: force a stock_fetcher action
            logger.warning("Using fallback stock_fetcher action")
            from langchain.schema import AgentAction
            return AgentAction(
                tool="stock_fetcher",
                tool_input=f'{{"symbol": "{self.fallback_symbol}", "period": "{self.fallback_period}"}}',
                log=f"Fallback action due to parse failure: {str(e)}"
            )


class AgentLoggingCallback(BaseCallbackHandler):
    """Custom callback to log agent thoughts and actions"""
    
    def __init__(self, progress_callback=None, period="2y"):
        self.progress_callback = progress_callback
        self.period = period
        self.original_query = ""  # Store original query for workflow detection
        # Convert period to readable format
        period_text = {"5y": "5 years", "2y": "2 years", "1y": "1 year", "6mo": "6 months", "3mo": "3 months", "1mo": "1 month"}.get(period, f"{period}")
        
        # Tool step numbers for logical sequencing
        self.tool_step_numbers = {
            'stock_fetcher': 1,
            'lstm_predictor': 2, 
            'stock_visualizer': 3,
            'general_assistant': 1
        }
        
        self.step_mapping = {
            'stock_fetcher': f'📊 Fetching {period_text} of historical data...',
            'lstm_predictor': '🧠 Training LSTM ensemble models & predicting...',
            'stock_visualizer': '📈 Creating interactive charts & visualizations...',
            'general_assistant': '🤖 Providing general information and assistance...'
        }
        self.tools_used = []
        self.current_step = 0
    
    def reset_progress(self):
        """Reset progress tracking for a new analysis"""
        self.tools_used = []
        self.current_step = 0
    
    def update_period(self, period):
        """Update the period and regenerate step mapping"""
        self.period = period
        period_text = {"5y": "5 years", "2y": "2 years", "1y": "1 year", "6mo": "6 months", "3mo": "3 months", "1mo": "1 month"}.get(period, f"{period}")
        
        # Tool step numbers remain the same
        self.tool_step_numbers = {
            'stock_fetcher': 1,
            'lstm_predictor': 2, 
            'stock_visualizer': 3,
            'general_assistant': 1
        }
        
        self.step_mapping = {
            'stock_fetcher': f'📊 Fetching {period_text} of historical data...',
            'lstm_predictor': '🧠 Training LSTM ensemble models & predicting...',
            'stock_visualizer': '📈 Creating interactive charts & visualizations...',
            'general_assistant': '🤖 Providing general information and assistance...'
        }
    
    def on_agent_action(self, action, **kwargs):
        print(f"\n🤖 AGENT ACTION:")
        print(f"   Tool: {action.tool}")
        print(f"   Input: {action.tool_input}")
        logger.info(f"🤖 AGENT ACTION:")
        logger.info(f"   Tool: {action.tool}")
        logger.info(f"   Input: {action.tool_input}")
        
        # Track tools used dynamically
        if action.tool not in self.tools_used:
            self.tools_used.append(action.tool)
        
        # Current step is always the position of this tool in the sequence
        self.current_step = len(self.tools_used)
        
        # Update progress if callback provided
        if self.progress_callback and action.tool in self.step_mapping:
            progress_text = self.step_mapping[action.tool]
            
            # Make progress text dynamic for stock_fetcher based on actual period used
            if action.tool == 'stock_fetcher' and hasattr(action, 'tool_input'):
                try:
                    import json
                    # Parse the tool input to extract period
                    if isinstance(action.tool_input, str):
                        tool_input = json.loads(action.tool_input)
                    else:
                        tool_input = action.tool_input
                    
                    if 'period' in tool_input:
                        period = tool_input['period']
                        period_text = {"5y": "5 years", "2y": "2 years", "1y": "1 year", "6mo": "6 months", "3mo": "3 months", "1mo": "1 month"}.get(period, f"{period}")
                        progress_text = f'📊 Fetching {period_text} of historical data...'
                except Exception as e:
                    logger.debug(f"Could not parse tool input for dynamic progress: {e}")
                    # Fall back to default progress text
                    pass
            # Use logical step number based on tool type, not execution order
            current_step = self.tool_step_numbers.get(action.tool, self.current_step)
            
            # Dynamic total steps estimation based on workflow type:
            # Once a general_assistant tool is used, lock into general mode (1/1)
            if 'general_assistant' in self.tools_used:
                # General questions are single-step - stay in this mode
                estimated_total_steps = 1
                current_step = 1  # Always step 1 for general questions
            elif action.tool == 'general_assistant':
                # First time using general_assistant
                estimated_total_steps = 1
                current_step = 1
            else:
                # Stock analysis workflow: detect workflow type early
                # Check if this is a risk assessment from the original query
                if any(keyword in self.original_query.lower() for keyword in ['risk', 'risk profile', 'risk assessment', 'evaluate risk']):
                    # Risk assessment detected - use 2-step workflow
                    estimated_total_steps = 2
                    if action.tool == 'stock_visualizer':
                        current_step = 2
                elif (action.tool == 'stock_visualizer' and 
                      'stock_fetcher' in self.tools_used and 
                      'lstm_predictor' not in self.tools_used):
                    # Risk assessment workflow: fetcher → visualizer (2 steps)
                    estimated_total_steps = 2
                    current_step = 2
                elif 'lstm_predictor' in self.tools_used or action.tool == 'lstm_predictor':
                    # Full analysis workflow: fetcher → lstm_predictor → visualizer (3 steps)
                    estimated_total_steps = 3
                else:
                    # Default to 3 steps for stock analysis
                    estimated_total_steps = 3
            
            logger.info(f"Calling progress callback: step {current_step}/{estimated_total_steps}, tool: {action.tool}")
            try:
                self.progress_callback(current_step, estimated_total_steps, progress_text)
                logger.info(f"Progress callback completed successfully")
            except Exception as e:
                logger.error(f"Progress callback failed: {e}")
        else:
            logger.info(f"Progress callback not called: callback={self.progress_callback is not None}, tool_in_mapping={action.tool in self.step_mapping}, tool={action.tool}")
        if hasattr(action, 'log') and action.log:
            # Extract the thought/reasoning part from the log
            log_lines = action.log.split('\n')
            for line in log_lines:
                line = line.strip()
                if line and not line.startswith('Action') and not line.startswith('Observation'):
                    # Show the agent's thinking process
                    if line.startswith('Thought:'):
                        print(f"   💭 {line}")
                        logger.info(f"   💭 {line}")
                    elif 'think' in line.lower() or 'need' in line.lower() or 'should' in line.lower():
                        print(f"   💭 REASONING: {line}")
                        logger.info(f"   💭 REASONING: {line}")
                    else:
                        print(f"   💭 {line}")
                        logger.info(f"   💭 {line}")
    
    def on_agent_finish(self, finish, **kwargs):
        print(f"\n🎯 AGENT FINISH:")
        output = finish.return_values.get('output', '')
        print(f"   Final Answer: {output[:300]}{'...' if len(output) > 300 else ''}")
        logger.info(f"🎯 AGENT FINISH:")
        logger.info(f"   Final Answer: {output[:300]}{'...' if len(output) > 300 else ''}")
    
    def on_tool_start(self, serialized, input_str, **kwargs):
        tool_name = serialized.get("name", "Unknown")
        logger.info(f"🔧 TOOL START: {tool_name}")
    
    def on_tool_end(self, output, **kwargs):
        logger.info(f"✅ TOOL COMPLETED - Agent will now process tool output...")
        # Log the tool output to see what the agent received
        if isinstance(output, dict) and 'tool_used' in output:
            logger.info(f"Tool output received: {output.get('tool_used')} - success: {output.get('success')}")
    
    def on_tool_error(self, error, **kwargs):
        logger.error(f"❌ TOOL ERROR: {str(error)}")
    
    def on_text(self, text, **kwargs):
        # This captures the agent's reasoning text
        if text.strip() and not text.startswith('Invoking:'):
            # Clean up the text and show reasoning
            clean_text = text.strip()
            if clean_text and len(clean_text) > 3:  # Avoid logging very short text
                # Add timestamp for better tracking of delays
                from datetime import datetime
                timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                
                # Highlight different types of reasoning
                if clean_text.startswith('Thought:'):
                    print(f"💭 {clean_text}")
                    logger.info(f"💭 [{timestamp}] {clean_text}")
                elif clean_text.startswith('Final Answer:'):
                    logger.info(f"🎯 [{timestamp}] Agent starting Final Answer generation...")
                    print(f"🎯 FINAL ANSWER: {clean_text[:100]}...")
                    logger.info(f"🎯 FINAL ANSWER: {clean_text[:100]}...")
                elif clean_text.startswith('I need to') or clean_text.startswith('I should'):
                    print(f"🧠 PLANNING: {clean_text}")
                    logger.info(f"🧠 [{timestamp}] PLANNING: {clean_text}")
                elif 'analyze' in clean_text.lower() or 'fetch' in clean_text.lower():
                    print(f"📊 STRATEGY: {clean_text}")
                    logger.info(f"📊 [{timestamp}] STRATEGY: {clean_text}")
                else:
                    print(f"💭 REASONING: {clean_text}")
                    logger.info(f"💭 [{timestamp}] REASONING: {clean_text}")
    
    def on_agent_start(self, serialized, inputs, **kwargs):
        print(f"\n🚀 AGENT STARTING: Analyzing query...")
        query = inputs.get('input', 'Unknown query')[:100]
        print(f"   Query: {query}{'...' if len(str(inputs.get('input', ''))) > 100 else ''}")
        logger.info(f"🚀 AGENT STARTING: Analyzing query...")
        logger.info(f"   Query: {query}{'...' if len(str(inputs.get('input', ''))) > 100 else ''}")
    
    def on_llm_start(self, serialized, prompts, **kwargs):
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"🤖 LLM THINKING: Processing agent reasoning...")
        logger.info(f"🤖 [{timestamp}] LLM START: Processing agent reasoning...")
    
    def on_llm_end(self, response, **kwargs):
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        logger.info(f"🤖 [{timestamp}] LLM COMPLETED: Agent reasoning finished")


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
            StockVisualizer(),
            GeneralAssistant()
        ]
        
        # Create agent prompt that lets the agent reason about the user's question
        self.prompt = PromptTemplate.from_template(
            """You are a financial analysis agent. Plan and execute actions to answer the user's question.

            Available tools:
            {tool_names}

            {tools}

            DECISION TREE: First determine the question type:
            - If user asks about "top stocks", "most valuable", "best performing", company information, financial concepts, or market knowledge → use ONLY general_assistant
            - If user provides specific stock symbols (AAPL, GOOGL, etc.) and wants analysis/prediction/charts → use stock analysis workflow
            
            IMPORTANT: Choose the appropriate workflow based on the user's question:

            FOR SPECIFIC STOCK ANALYSIS (questions with specific stock symbols like AAPL, GOOGL, TSLA):
            
            RISK ASSESSMENT queries (contains "risk", "evaluate risk", "risk profile"):
            1. Use stock_fetcher to get historical data
            2. Skip lstm_predictor (risk assessment focuses on historical patterns, not predictions)
            3. Use stock_visualizer to create charts and visualizations
            Examples: "Evaluate risk profile of TEAM", "Risk assessment for AAPL"
            
            PREDICTION/ANALYSIS queries (contains "predict", "forecast", "analyze", "trends"):
            1. ALWAYS start with stock_fetcher to get historical data (REQUIRED FIRST)
            2. ONLY after stock_fetcher succeeds, use lstm_predictor to generate forecasts
            3. ONLY after lstm_predictor succeeds, use stock_visualizer to create charts
            Examples: "Analyze AAPL stock", "Predict GOOGL price", "Show TSLA trends"
            
            CRITICAL: You MUST use stock_fetcher FIRST before any other tool. NEVER use lstm_predictor or stock_visualizer without data from stock_fetcher.

            FOR GENERAL QUESTIONS (market knowledge, concepts, company info, strategies, "top stocks", "most valuable stocks"):
            - Use ONLY general_assistant tool - NO OTHER TOOLS
            - IMMEDIATELY use the general_assistant response as your Final Answer
            - Do NOT add additional reasoning or thoughts after general_assistant
            - Do NOT use stock_fetcher, lstm_predictor, or stock_visualizer for general questions
            - Examples: "Which stock is most valuable?", "What are the top 10 most valuable US stocks?", "What is the stock symbol for Nvidia?", "What company does TSLA represent?", "What is compound interest?", "How do bonds work?", "Best performing sector?", "How to calculate stock rating?"
            
            CRITICAL RULE: For general questions, use ONLY general_assistant then IMMEDIATELY output Final Answer.
            Do NOT think or reason after general_assistant responds - just use its response directly.
            The general_assistant provides complete answers that need no additional processing.
            
            IMPORTANT: If any tool fails or times out, proceed to Final Answer with the available information.
            Do NOT keep retrying failed tools - provide analysis with existing data.

            Use this format:
            
            FOR STOCK ANALYSIS:
            Thought: I need to analyze [SYMBOL] stock. I must start with stock_fetcher to get data first.
            Action: stock_fetcher
            Action Input: {{"symbol": "SYMBOL", "period": "2y"}}
            Observation: [Data fetched successfully]
            Thought: Now I have the data, I can proceed with [lstm_predictor for predictions OR stock_visualizer for risk assessment]
            Action: [Next appropriate tool based on query type]
            Action Input: {{"symbol": "SYMBOL", "other_params": "as_needed"}}
            Observation: [Tool result will appear here]
            ... (continue with remaining tools in sequence)
            Thought: I now have enough information to answer
            Final Answer: [Your comprehensive response to the user]
            
            FOR GENERAL QUESTIONS:
            Thought: This is a general question, I'll use general_assistant
            Action: general_assistant
            Action Input: {{"question": "user's question"}}
            Observation: [Tool result will appear here]
            Final Answer: [Use the general_assistant response directly without modification]

            Begin!

            Question: {input}
            {agent_scratchpad}"""
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
        
        # Create callback handler for detailed logging (progress callback set later)
        self.callback_handler = AgentLoggingCallback()
        
        # Create custom output parser for handling markdown formatting
        self.output_parser = CleanOutputParser()
        
        # Create agent executor
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,  # Enable to capture agent reasoning
            handle_parsing_errors=self._handle_parsing_errors,  # Use custom error handler
            max_iterations=4,  # Reduced to prevent infinite loops
            return_intermediate_steps=False,  # Set to False to avoid the multiple keys issue
            callbacks=[self.callback_handler],
            early_stopping_method="force"  # Stop when max iterations reached
        )
        logger.info("StockAnalysisAgent initialization completed")
    
    def set_progress_callback(self, callback):
        """Set progress callback for the agent"""
        self.callback_handler.progress_callback = callback
        # Also pass the callback to the LSTM predictor tool
        for tool in self.tools:
            if hasattr(tool, 'name') and tool.name == 'lstm_predictor':
                tool.set_progress_callback(callback)
    
    def _handle_parsing_errors(self, error) -> str:
        """Custom error handler that tries to clean and reparse LLM output"""
        error_msg = str(error)
        logger.warning(f"Parsing error encountered: {error_msg}")
        
        # Try to extract the original text that failed to parse
        if hasattr(error, 'llm_output'):
            original_text = error.llm_output
        else:
            # Extract from error message if possible
            original_text = error_msg
        
        try:
            # Use our custom parser to clean the text
            cleaned_result = self.output_parser.parse(original_text)
            logger.info("Successfully cleaned and reparsed LLM output")
            
            # If we got a valid action, execute it
            if hasattr(cleaned_result, 'tool') and hasattr(cleaned_result, 'tool_input'):
                return f"Action: {cleaned_result.tool}\nAction Input: {cleaned_result.tool_input}"
            else:
                return "I'll analyze the stock data systematically using my tools."
        except Exception as parse_error:
            logger.error(f"Custom parsing also failed: {parse_error}")
            return f"Action: stock_fetcher\nAction Input: {{\"symbol\": \"{self.output_parser.fallback_symbol}\", \"period\": \"{self.output_parser.fallback_period}\"}}"
    
    def analyze_stock(self, query: str) -> Dict[str, Any]:
        """
        Main method to analyze stocks based on user query
        """
        logger.info(f"Agent analyzing query: '{query[:100]}{'...' if len(query) > 100 else ''}'")
        
        # Store original query in callback handler for workflow detection
        self.callback_handler.original_query = query
        
        # Reset progress tracking for new analysis
        self.callback_handler.reset_progress()
        
        # Extract symbol and period from query and set as fallback for parser
        from utils import extract_stock_symbols
        symbols = extract_stock_symbols(query)
        if symbols:
            self.output_parser.set_fallback_symbol(symbols[0])
            logger.info(f"Set fallback symbol to: {symbols[0]}")
        else:
            # Fallback to basic pattern if app extraction fails
            symbol_match = re.search(r'\b([A-Z]{3,5})\b', query.upper())
            if symbol_match:
                self.output_parser.set_fallback_symbol(symbol_match.group(1))
                logger.info(f"Set fallback symbol to: {symbol_match.group(1)}")
        
        # Extract period from query
        period_match = re.search(r'using (\d+y)', query.lower())
        if period_match:
            period = period_match.group(1)
            self.output_parser.set_fallback_period(period)
            logger.info(f"Set fallback period to: {period}")
        else:
            self.output_parser.set_fallback_period("2y")  # Default fallback
        
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




# Pre-defined query templates for common analysis types
QUERY_TEMPLATES = {
    "basic_analysis": "Analyze the stock {symbol} and provide insights on its current performance and future outlook.",
    "trend_analysis": "Focus on the trending patterns of {symbol} stock and predict short-term movements.",
    "risk_assessment": "Evaluate the risk profile of investing in {symbol} stock based on historical data and predictions.",
    "comparison": "Compare {symbol1} and {symbol2} stocks and recommend which might be a better investment.",
    "sector_analysis": "Analyze {symbol} in the context of its sector performance and market conditions."
}