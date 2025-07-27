import streamlit as st
import sys
import os
import logging
from datetime import datetime

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from stock_agent import StockAnalysisAgent, QUERY_TEMPLATES
from visualizer import display_charts
from config import get_config
from utils import extract_stock_symbols, extract_stock_symbol

# Configure logging
from config import setup_logging, get_logger

# Setup logging first, before any other imports that might create loggers
setup_logging()

logger = get_logger(__name__)

# Page configuration
config = get_config()
st.set_page_config(
    page_title=config["streamlit"]["page_title"],
    page_icon=config["streamlit"]["page_icon"],
    layout=config["streamlit"]["layout"]
)

# Custom CSS for metric cards
st.markdown("""
<style>
    .metric-card {
        background: var(--background-color);
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        border: 1px solid var(--secondary-background-color);
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant", 
                "content": "👋 Hello! I'm your AI Stock Analysis Assistant. I can help you analyze stocks, predict trends, and create visualizations.\n\n**What I can do:**\n- 📊 Fetch real-time stock data\n- 🧠 Generate LSTM price predictions\n- 📈 Create interactive charts\n- 💡 Provide AI-powered analysis\n\n**Try asking:**\n- \"Analyze AAPL stock\"\n- \"Predict TSLA trends for 30 days\"\n- \"Compare GOOGL vs MSFT\"\n\nWhat stock would you like to analyze?"
            }
        ]
    
    if "agent" not in st.session_state:
        try:
            st.session_state.agent = StockAnalysisAgent()
            logger.info("StockAnalysisAgent created successfully")
        except Exception as e:
            logger.error(f"Failed to create StockAnalysisAgent: {e}")
            raise
    
    
    if "analysis_results" not in st.session_state:
        st.session_state.analysis_results = {}
    
    if "processing" not in st.session_state:
        st.session_state.processing = False
    
    if "progress_data" not in st.session_state:
        st.session_state.progress_data = {
            "step": 0,
            "total": 0,
            "message": "",
            "training_info": None
        }

def display_chat_message(message: dict):
    """Display a chat message using Streamlit's chat message component"""
    if message["role"] == "user":
        with st.chat_message("user"):
            st.write(message["content"])
    else:
        with st.chat_message("assistant"):
            st.write(message["content"])


def process_user_query_with_progress(prompt: str, update_progress_callback) -> str:
    """Process a user query with progress updates"""
    logger.info(f"Processing user query with progress: '{prompt}'")
    
    # Extract stock symbols for analysis (handles both single and comparison queries)
    symbols = extract_stock_symbols(prompt)  
    symbol = symbols[0] if symbols else None
    
    analysis_keywords = ['analyze', 'analysis', 'predict', 'forecast', 'trend', 'future', 'compare', 'comparison', 'vs', 'versus']
    has_analysis_keyword = any(keyword in prompt.lower() for keyword in analysis_keywords)
    
    # Check if this is a comparison query
    is_comparison = len(symbols) > 1 or any(word in prompt.lower() for word in ['compare', 'vs', 'versus'])
    
    if symbols and has_analysis_keyword and is_comparison:
        # Handle comparison of multiple stocks
        logger.info(f"Processing comparison analysis for symbols: {symbols}")
        
        total_steps = len(symbols) * 3 + 1  # 3 steps per stock + final comparison
        current_step = 0
        
        comparison_results = {}
        comparison_responses = []
        
        for i, sym in enumerate(symbols[:2]):  # Limit to 2 stocks for comparison
            current_step += 1
            update_progress_callback(current_step, total_steps, f"Fetching {sym} stock data...")
            
            logger.info(f"🔧 Running analysis for {sym}")
            try:
                # Use agent for comparison analysis
                current_step += 1
                update_progress_callback(current_step, total_steps, f"Agent analyzing {sym}...")
                
                agent_query = f"Analyze {sym} stock with historical data, LSTM predictions, and visualizations"
                agent_results = st.session_state.agent.analyze_stock(agent_query)
                
                current_step += 1  
                update_progress_callback(current_step, total_steps, f"Processing {sym} results...")
                
                if agent_results.get("success"):
                    # Store agent response for comparison
                    comparison_results[sym] = {"ai_analysis": agent_results["response"], "success": True}
                    comparison_responses.append(f"""**{sym}**: Agent analysis completed""")
                else:
                    comparison_responses.append(f"**{sym}**: Analysis failed")
            except Exception as e:
                comparison_responses.append(f"**{sym}**: Error - {str(e)}")
                logger.error(f"Comparison analysis failed for {sym}: {e}")
        
        current_step += 1
        update_progress_callback(current_step, total_steps, "Generating comparison analysis...")
        
        # Create comparison response
        if len(comparison_results) >= 2:
            sym1, sym2 = list(comparison_results.keys())[:2]
            r1, r2 = comparison_results[sym1], comparison_results[sym2]
            
            response = f"""📊 **Stock Comparison: {sym1} vs {sym2}**

{comparison_responses[0]}
{comparison_responses[1]}

**Agent Analysis Results:**
Both stocks have been analyzed using AI agent orchestration with LSTM predictions and technical analysis.

📈 Detailed analysis for each stock is available below."""
            
            # Store both stocks' results for comparison visualization
            st.session_state.analysis_results = {
                "comparison_mode": True,
                "stocks": comparison_results,
                "symbols": [sym1, sym2]
            }
        else:
            response = f"❌ **Comparison Failed**\n\nCould only analyze {len(comparison_results)} of {len(symbols)} requested stocks."
            
    elif symbol and has_analysis_keyword:
        logger.info(f"Starting agent-only analysis for {symbol}")
        
        total_steps = 3
        current_step = 0
        
        # Agent-only orchestration
        logger.info(f"🤖 Using AI agent to orchestrate analysis of {symbol}")
        print(f"\n🤖 AI AGENT ORCHESTRATING ANALYSIS FOR {symbol}")
        print("="*60)
        print("🧠 Agent will decide the sequence and call tools as needed...")
        print("="*60)
        
        # Agent will handle all progress updates during execution
        
        try:
            print(f"\n🎯 AGENT STARTING ANALYSIS")
            print("="*50)
            print(f"🧠 Agent will reason about: '{prompt}'")
            print("="*50)
            
            # Set progress callback for the agent
            st.session_state.agent.set_progress_callback(update_progress_callback)
            
            # Get the current sidebar period selection (if available)
            period_from_sidebar = getattr(st.session_state, 'current_period', None)
            if period_from_sidebar:
                st.session_state.agent.output_parser.set_fallback_period(period_from_sidebar)
                st.session_state.agent.callback_handler.update_period(period_from_sidebar)
                logger.info(f"Using sidebar period selection: {period_from_sidebar}")
            
            # Let agent reason about the user's actual question and decide what to do
            agent_result = st.session_state.agent.analyze_stock(prompt)
            
            if agent_result.get("success"):
                agent_response = agent_result["response"]
                logger.info("✅ Agent orchestration completed successfully")
                print(f"\n✅ AGENT ORCHESTRATION COMPLETED")
                print("="*50)
                
                # Update progress to completion
                update_progress_callback(total_steps, total_steps, f"✅ {symbol} analysis complete!")
                
                # Get visualization results from data store if available
                from data_store import get_data_store
                data_store = get_data_store()
                
                # Get prediction data for summary metrics (try both key formats)
                prediction_key = f"{symbol}_PREDICTIONS"
                predictions_data = data_store.get_stock_data(prediction_key)
                if not predictions_data:
                    prediction_key = f"{symbol}_predictions"
                    predictions_data = data_store.get_stock_data(prediction_key)
                
                # Try to get visualization data (try both key formats)
                viz_key = f"{symbol}_VISUALIZATIONS"
                viz_data = data_store.get_stock_data(viz_key)
                if not viz_data:
                    viz_key = f"{symbol}_visualizations"
                    viz_data = data_store.get_stock_data(viz_key)
                
                # Build summary from prediction data
                summary = {
                    "symbol": symbol,
                    "analysis_type": "Agent Orchestrated",
                    "confidence": "Agent Determined"
                }
                
                if predictions_data and 'trend_analysis' in predictions_data:
                    trend = predictions_data['trend_analysis']
                    summary.update({
                        "current_price": trend.get('current_price', 0),
                        "predicted_price": trend.get('predicted_price_30d', 0),
                        "percentage_change": trend.get('percentage_change', 0),
                        "confidence": predictions_data.get('prediction_confidence', 'Unknown')
                    })
                
                # Store results for display
                st.session_state.analysis_results = {
                    "success": True,
                    "agent_orchestrated": True,
                    "ai_analysis": agent_response,
                    "summary": summary,
                    "visualizations": viz_data  # Include visualization data
                }
                
                # Create response for chat
                response = f"✅ **{symbol} Analysis Complete**\n\n{agent_response}"
                
            else:
                logger.error(f"❌ Agent orchestration failed: {agent_result.get('error', 'Unknown error')}")
                print(f"\n❌ AGENT ORCHESTRATION FAILED")
                
                error_msg = agent_result.get('error', 'Unknown error')
                
                # Provide helpful error message
                if "No data found" in error_msg:
                    response = f"""❌ **Unable to analyze {symbol}**

I couldn't find data for this symbol. This could be because:

1. **Invalid Symbol**: '{symbol}' might not be a valid stock ticker
2. **Market Closed**: The symbol might be delisted or not actively traded  
3. **Spelling**: Please check if the symbol is spelled correctly

**Try these popular symbols instead:**
- AAPL (Apple), GOOGL (Google), MSFT (Microsoft)
- TSLA (Tesla), AMZN (Amazon), NVDA (NVIDIA)"""
                else:
                    response = f"❌ **Analysis Error for {symbol}**\n\n{error_msg}"
                        
        except Exception as agent_error:
            logger.error(f"Agent exception: {agent_error}")
            response = f"❌ **Analysis Error for {symbol}**\n\nI encountered an error while analyzing the stock: {str(agent_error)}"
    elif has_analysis_keyword and not symbol:
        # User wants analysis but we couldn't extract a symbol
        response = """I can see you want a stock analysis, but I couldn't identify a specific stock symbol in your query.

**Please specify a stock symbol like:**
- "Analyze AAPL stock"
- "Predict TSLA trends" 
- "Forecast GOOGL performance"

**Popular symbols you can try:**
- AAPL (Apple), GOOGL (Google), MSFT (Microsoft)
- TSLA (Tesla), AMZN (Amazon), NVDA (NVIDIA)
- META (Meta/Facebook), NFLX (Netflix)

Or ask me general questions about the stock market!"""
        logger.info("Analysis requested but no valid symbol found")
    else:
        logger.info(f"Using direct LLM for general query (symbol={symbol}, has_keywords={has_analysis_keyword})")
        
        # For general queries, use direct LLM instead of complex ReAct agent
        try:
            logger.info("Using direct LLM for general stock questions...")
            # Use the LLM directly for simpler queries
            llm_prompt = f"""You are a knowledgeable stock market analyst. Answer the following question about stocks, trading, or financial markets in a helpful and educational way:

Question: {prompt}

Provide a clear, informative response. If the question is about a specific stock, mention that for detailed analysis with charts and predictions, the user should ask for a comprehensive analysis."""
            
            response = st.session_state.agent.llm.invoke(llm_prompt)
            logger.info("Direct LLM response completed successfully")
            
            # Clear previous analysis results for non-analysis queries
            st.session_state.analysis_results = {}
            logger.debug("Cleared previous analysis results")
                
        except Exception as e:
            response = f"I encountered an error while processing your request: {str(e)}"
            logger.error(f"Direct LLM exception: {e}")
    
    return response

def process_user_query(prompt: str) -> str:
    """Process a user query and return the response"""
    logger.info(f"Processing user query: '{prompt}'")
    
    # Extract stock symbols for analysis (handles both single and comparison queries)
    symbols = extract_stock_symbols(prompt)  
    symbol = symbols[0] if symbols else None
    
    analysis_keywords = ['analyze', 'analysis', 'predict', 'forecast', 'trend', 'future', 'compare', 'comparison', 'vs', 'versus']
    has_analysis_keyword = any(keyword in prompt.lower() for keyword in analysis_keywords)
    
    # Check if this is a comparison query
    is_comparison = len(symbols) > 1 or any(word in prompt.lower() for word in ['compare', 'vs', 'versus'])
    
    if symbols and has_analysis_keyword and is_comparison:
        # Handle comparison of multiple stocks
        logger.info(f"Processing comparison analysis for symbols: {symbols}")
        
        comparison_results = {}
        comparison_responses = []
        
        for sym in symbols[:2]:  # Limit to 2 stocks for comparison
            logger.info(f"🔧 Running analysis for {sym}")
            try:
                # Use agent to analyze each stock in the comparison
                agent_query = f"Analyze {sym} stock performance, trends, and predictions"
                agent_results = st.session_state.agent.analyze_stock(agent_query)
                if agent_results.get("success"):
                    comparison_results[sym] = {"ai_analysis": agent_results["response"], "success": True}
                    comparison_responses.append(f"""**{sym}**: Agent analysis completed""")
                else:
                    comparison_responses.append(f"**{sym}**: Analysis failed")
            except Exception as e:
                comparison_responses.append(f"**{sym}**: Error - {str(e)}")
                logger.error(f"Comparison analysis failed for {sym}: {e}")
        
        # Create comparison response
        if len(comparison_results) >= 2:
            sym1, sym2 = list(comparison_results.keys())[:2]
            r1, r2 = comparison_results[sym1], comparison_results[sym2]
            
            response = f"""📊 **Stock Comparison: {sym1} vs {sym2}**

{comparison_responses[0]}
{comparison_responses[1]}

**Agent Analysis Results:**
Both stocks have been analyzed using AI agent orchestration with LSTM predictions and technical analysis.

📈 Detailed analysis for each stock is available below."""
            
            # Store both stocks' results for comparison visualization
            st.session_state.analysis_results = {
                "comparison_mode": True,
                "stocks": comparison_results,
                "symbols": [sym1, sym2]
            }
        else:
            response = f"❌ **Comparison Failed**\n\nCould only analyze {len(comparison_results)} of {len(symbols)} requested stocks."
            
    elif symbol and has_analysis_keyword:
        logger.info(f"Starting agent-only analysis for {symbol}")
        
        # Agent-only orchestration
        logger.info(f"🤖 Using AI agent to orchestrate analysis of {symbol}")
        
        try:
            # Get the current sidebar period selection (if available)
            period_from_sidebar = getattr(st.session_state, 'current_period', None)
            if period_from_sidebar:
                st.session_state.agent.output_parser.set_fallback_period(period_from_sidebar)
                st.session_state.agent.callback_handler.update_period(period_from_sidebar)
                logger.info(f"Using sidebar period selection: {period_from_sidebar}")
            
            # Let agent reason about the user's actual question and decide what to do
            agent_result = st.session_state.agent.analyze_stock(prompt)
            
            if agent_result.get("success"):
                agent_response = agent_result["response"]
                logger.info("✅ Agent orchestration completed successfully")
                
                # Store results for display
                st.session_state.analysis_results = {
                    "success": True,
                    "agent_orchestrated": True,
                    "ai_analysis": agent_response,
                    "summary": {
                        "symbol": symbol,
                        "analysis_type": "Agent Orchestrated",
                        "confidence": "Agent Determined"
                    }
                }
                
                # Create response for chat
                response = f"✅ **{symbol} Analysis Complete**\n\n{agent_response}"
                
            else:
                logger.error(f"❌ Agent orchestration failed: {agent_result.get('error', 'Unknown error')}")
                
                error_msg = agent_result.get('error', 'Unknown error')
                
                # Provide helpful error message
                if "No data found" in error_msg:
                    response = f"""❌ **Unable to analyze {symbol}**

I couldn't find data for this symbol. This could be because:

1. **Invalid Symbol**: '{symbol}' might not be a valid stock ticker
2. **Market Closed**: The symbol might be delisted or not actively traded  
3. **Spelling**: Please check if the symbol is spelled correctly

**Try these popular symbols instead:**
- AAPL (Apple), GOOGL (Google), MSFT (Microsoft)
- TSLA (Tesla), AMZN (Amazon), NVDA (NVIDIA)"""
                else:
                    response = f"❌ **Analysis Error for {symbol}**\n\n{error_msg}"
                        
        except Exception as agent_error:
            logger.error(f"Agent exception: {agent_error}")
            response = f"❌ **Analysis Error for {symbol}**\n\nI encountered an error while analyzing the stock: {str(agent_error)}"
    elif has_analysis_keyword and not symbol:
        # User wants analysis but we couldn't extract a symbol
        response = """I can see you want a stock analysis, but I couldn't identify a specific stock symbol in your query.

**Please specify a stock symbol like:**
- "Analyze AAPL stock"
- "Predict TSLA trends" 
- "Forecast GOOGL performance"

**Popular symbols you can try:**
- AAPL (Apple), GOOGL (Google), MSFT (Microsoft)
- TSLA (Tesla), AMZN (Amazon), NVDA (NVIDIA)
- META (Meta/Facebook), NFLX (Netflix)

Or ask me general questions about the stock market!"""
        logger.info("Analysis requested but no valid symbol found")
    else:
        logger.info(f"Using direct LLM for general query (symbol={symbol}, has_keywords={has_analysis_keyword})")
        
        # For general queries, use direct LLM instead of complex ReAct agent
        try:
            logger.info("Using direct LLM for general stock questions...")
            # Use the LLM directly for simpler queries
            llm_prompt = f"""You are a knowledgeable stock market analyst. Answer the following question about stocks, trading, or financial markets in a helpful and educational way:

Question: {prompt}

Provide a clear, informative response. If the question is about a specific stock, mention that for detailed analysis with charts and predictions, the user should ask for a comprehensive analysis."""
            
            response = st.session_state.agent.llm.invoke(llm_prompt)
            logger.info("Direct LLM response completed successfully")
            
            # Clear previous analysis results for non-analysis queries
            st.session_state.analysis_results = {}
            logger.debug("Cleared previous analysis results")
                
        except Exception as e:
            response = f"I encountered an error while processing your request: {str(e)}"
            logger.error(f"Direct LLM exception: {e}")
    
    return response

def main():
    logger.info("Starting Stock Analysis AI application")
    
    # Initialize session state
    initialize_session_state()
    
    # App header
    st.title("📈 Stock Analysis AI")
    st.markdown("*Powered by Ollama (Gemma3) + LangChain + LSTM Predictions*")
    
    # Sidebar
    with st.sidebar:
        # Quick analysis options
        st.subheader("Quick Analysis")
        quick_symbol = st.text_input("Stock Symbol", placeholder="e.g., AAPL, GOOGL, TSLA")
        
        # Analysis parameters
        col1, col2 = st.columns(2)
        with col1:
            past_data_period = st.selectbox("Past Data", ["2y", "5y"], index=0)
        with col2:
            future_prediction = st.selectbox("Prediction", ["30 days", "1 week"], index=0)
        
        # Store the selected period in session state for agent access
        st.session_state.current_period = past_data_period
        
        if st.button("🔍 Quick Analysis", use_container_width=True):
            if quick_symbol:
                logger.info(f"Quick analysis requested for symbol: {quick_symbol}")
                # Convert future prediction to days
                prediction_days = 30 if future_prediction == "30 days" else 7
                query = f"Analyze {quick_symbol.upper()} stock using {past_data_period} of historical data and predict {prediction_days} days ahead"
                st.session_state.messages.append({"role": "user", "content": query})
                st.rerun()
            else:
                st.warning("Please enter a stock symbol first")
        
        st.divider()
        
        # Analysis templates
        st.subheader("📊 Analysis Templates")
        template_options = {
            "Basic Analysis": "basic_analysis",
            "Trend Analysis": "trend_analysis", 
            "Risk Assessment": "risk_assessment"
        }
        
        selected_template = st.selectbox("Choose template:", list(template_options.keys()))
        template_symbol = st.text_input("Symbol for template", placeholder="e.g., AAPL")
        
        if st.button("📈 Use Template", use_container_width=True):
            if template_symbol:
                logger.info(f"Template analysis requested: {selected_template} for {template_symbol}")
                template_key = template_options[selected_template]
                query = QUERY_TEMPLATES[template_key].format(symbol=template_symbol.upper())
                st.session_state.messages.append({"role": "user", "content": query})
                st.rerun()
            else:
                st.warning("Please enter a stock symbol first")
        
    
    # Check if there are unprocessed messages (from button clicks)
    if len(st.session_state.messages) > 0:
        last_message = st.session_state.messages[-1]
        if last_message["role"] == "user" and not st.session_state.processing:
            # There's an unprocessed user message, process it
            message_content = last_message["content"]
            symbols = extract_stock_symbols(message_content)
            analysis_keywords = ['analyze', 'analysis', 'predict', 'forecast', 'trend', 'future', 'compare', 'comparison', 'vs', 'versus']
            has_analysis_keyword = any(keyword in message_content.lower() for keyword in analysis_keywords)
            valid_symbol = symbols and has_analysis_keyword and len(symbols[0]) >= 2 and len(symbols[0]) <= 5 and symbols[0].isalpha()
            symbol_display = f" for {symbols[0]}" if valid_symbol else ""
            
            # Set processing flag
            st.session_state.processing = True
            
            # Add immediate "thinking" message with detailed progress
            thinking_message = f"""🤔 **Analyzing{symbol_display}...**

**Expected Steps:**
1. 📊 Fetch historical stock data
2. 🧠 Train LSTM ensemble models (this takes 3-5 minutes)
3. 📈 Create interactive visualizations
4. 💡 Generate AI analysis

⏳ **Status:** Starting analysis...

*Please wait while the AI agent works...*"""
            st.session_state.messages.append({"role": "assistant", "content": thinking_message})
            
            # Show the thinking message immediately
            st.rerun()
            
        elif st.session_state.processing and len(st.session_state.messages) > 1:
            # Continue processing the last user message
            message_content = None
            for msg in reversed(st.session_state.messages):
                if msg["role"] == "user":
                    message_content = msg["content"]
                    break
            
            if message_content:
                # Create dedicated progress containers BEFORE processing starts
                progress_container = st.container()
                
                with progress_container:
                    st.markdown("### 🔄 Analysis in Progress")
                    
                    # Create progress elements that can be updated
                    progress_bar = st.progress(0, text="Initializing...")
                    status_text = st.empty()
                    step_details = st.empty()
                    
                    # Training progress (shown when LSTM starts)
                    training_container = st.container()
                    training_progress = None
                    training_status = None
                
                def update_progress_ui(step, total, message):
                    """Update the progress UI elements"""
                    nonlocal training_progress, training_status
                    
                    # Determine step name and details
                    if step == 1:
                        step_name = "📊 Fetching Historical Data"
                        progress_text = "Downloading stock data from yfinance..."
                    elif step == 2:
                        step_name = "🧠 LSTM Neural Network Training"
                        progress_text = "Training ensemble models for price prediction..."
                    elif step == 3:
                        step_name = "📈 Creating Visualizations"
                        progress_text = "Generating interactive charts and analysis..."
                    else:
                        step_name = "🔄 Processing"
                        progress_text = "Working on your analysis..."
                    
                    # Update main progress bar with detailed info
                    progress_value = step / total if total > 0 else 0
                    progress_bar.progress(progress_value, text=f"{step_name} ({step}/{total})")
                    
                    # Update status with more context
                    status_text.markdown(f"**{step_name}**")
                    
                    # Show detailed current activity
                    step_details.markdown(f"💡 {progress_text}")
                    
                    # Show training progress when LSTM training starts
                    if "Training" in message and training_progress is None:
                        with training_container:
                            st.markdown("**🧠 LSTM Ensemble Training Details:**")
                            training_progress = st.progress(0, text="Initializing neural networks...")
                            training_status = st.empty()
                    
                    # Update training progress with detailed info
                    if training_progress and ("Model" in message or "Epoch" in message):
                        # Parse training details
                        model_info = ""
                        loss_info = ""
                        
                        if "Model" in message:
                            model_part = message.split("Model")[1].split("•")[0].strip()
                            model_info = f"🔬 **Ensemble Model {model_part}**"
                        
                        if "Loss:" in message and "Val Loss:" in message:
                            loss_part = message.split("Loss:")[1].split("•")[0].strip()
                            val_loss_part = message.split("Val Loss:")[1].strip()
                            loss_info = f"📉 Training Loss: {loss_part} | Validation Loss: {val_loss_part}"
                        
                        training_status.markdown(f"{model_info}\n\n{loss_info}")
                        
                        # Update epoch progress bar
                        if "Epoch" in message and "/" in message:
                            try:
                                parts = message.split("Epoch")[-1].strip()
                                if "/" in parts:
                                    current = int(parts.split("/")[0].strip())
                                    total_epochs = int(parts.split("/")[1].split("•")[0].strip())
                                    epoch_progress = current / total_epochs
                                    
                                    # Calculate overall training progress (across all 3 models)
                                    if "Model 1/3" in message:
                                        overall_progress = epoch_progress / 3
                                    elif "Model 2/3" in message:
                                        overall_progress = (1 + epoch_progress) / 3
                                    elif "Model 3/3" in message:
                                        overall_progress = (2 + epoch_progress) / 3
                                    else:
                                        overall_progress = epoch_progress
                                    
                                    training_progress.progress(overall_progress, text=f"Epoch {current}/{total_epochs} - Overall Training: {overall_progress*100:.1f}%")
                            except:
                                pass
                
                # Process the query with real progress updates
                response = process_user_query_with_progress(message_content, update_progress_ui)
                
                # Clear progress displays and show completion
                progress_container.empty()
                st.success("✅ Analysis completed successfully!")
                
                # Replace thinking message with actual response
                st.session_state.messages[-1] = {"role": "assistant", "content": response}
                st.session_state.processing = False
                logger.info("Analysis completed and response added")
                
                # Rerun to update the display
                st.rerun()
    
    # Main chat interface
    chat_container = st.container()
    
    with chat_container:
        # Display chat history
        for message in st.session_state.messages:
            display_chat_message(message)
        
        # Display analysis results if available
        if st.session_state.analysis_results:
            st.divider()
            
            results = st.session_state.analysis_results
            
            # Put analysis results and charts in an expander
            with st.expander("📊 Analysis Results & Charts", expanded=True):
                
                # Check if this is comparison mode
                if results.get("comparison_mode"):
                    # Display comparison results for both stocks
                    stocks = results["stocks"]
                    symbols = results["symbols"]
                
                    # Create tabs for each stock
                    tab1, tab2 = st.tabs([f"📈 {symbols[0]}", f"📈 {symbols[1]}"])
                    
                    for i, (tab, symbol) in enumerate(zip([tab1, tab2], symbols)):
                        with tab:
                            stock_results = stocks[symbol]
                            
                            # Display summary metrics for this stock
                            if "summary" in stock_results:
                                summary = stock_results["summary"]
                                col1, col2, col3, col4 = st.columns(4)
                                
                                with col1:
                                    st.metric("Current Price", f"${summary.get('current_price', 0):.2f}")
                                with col2:
                                    st.metric("Predicted Price", f"${summary.get('predicted_price', 0):.2f}")
                                with col3:
                                    change = summary.get('percentage_change', 0)
                                    st.metric("Expected Change", f"{change:.2f}%", delta=f"{change:.2f}%")
                                with col4:
                                    st.metric("Confidence", summary.get('confidence', 'Unknown'))
                            
                            # Display visualizations for this stock
                            if "visualizations" in stock_results and "chart_objects" in stock_results["visualizations"]:
                                st.subheader(f"📈 {symbol} Price Charts")
                                display_charts(stock_results["visualizations"])
                            
                else:
                    # Single stock mode (existing logic)
                    # Display summary metrics
                    if "summary" in results:
                        summary = results["summary"]
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Current Price", f"${summary.get('current_price', 0):.2f}")
                        with col2:
                            st.metric("Predicted Price", f"${summary.get('predicted_price', 0):.2f}")
                        with col3:
                            change = summary.get('percentage_change', 0)
                            st.metric("Expected Change", f"{change:.2f}%", delta=f"{change:.2f}%")
                        with col4:
                            st.metric("Confidence", summary.get('confidence', 'Unknown'))
                    
                    # Display visualizations
                    if "visualizations" in results and results["visualizations"] and "chart_objects" in results["visualizations"]:
                        st.subheader("📈 Price Charts")
                        display_charts(results["visualizations"])
    
    # Chat input
    if prompt := st.chat_input("Ask about any stock (e.g., 'Analyze AAPL stock' or 'Compare TSLA vs NVDA')"):
        logger.info(f"User query received: '{prompt[:50]}{'...' if len(prompt) > 50 else ''}'")
        
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Rerun to trigger processing
        st.rerun()
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.8em;'>
        Built with Streamlit • LangChain • Ollama • LSTM • Made for educational purposes only
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()