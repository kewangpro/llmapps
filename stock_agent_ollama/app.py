import streamlit as st
import sys
import os
import logging
from datetime import datetime

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from stock_agent import StockAnalysisAgent, StockWorkflow, QUERY_TEMPLATES
from visualizer import display_charts
from config import get_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('stock_analysis.log')
    ]
)

# Set specific loggers to reduce noise
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("yfinance").setLevel(logging.WARNING)
logging.getLogger("peewee").setLevel(logging.WARNING)
logging.getLogger("absl").setLevel(logging.WARNING)
logging.getLogger("tensorflow").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

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
    
    if "workflow" not in st.session_state:
        try:
            st.session_state.workflow = StockWorkflow(st.session_state.agent)
            logger.info("StockWorkflow created successfully")
        except Exception as e:
            logger.error(f"Failed to create StockWorkflow: {e}")
            raise
    
    if "analysis_results" not in st.session_state:
        st.session_state.analysis_results = {}

def display_chat_message(message: dict):
    """Display a chat message using Streamlit's chat message component"""
    if message["role"] == "user":
        with st.chat_message("user"):
            st.write(message["content"])
    else:
        with st.chat_message("assistant"):
            st.write(message["content"])

def extract_stock_symbols(query: str) -> list:
    """Extract all stock symbols from user query, including comparisons"""    
    import re
    
    symbols = []
    
    # Check for comparison patterns first (vs, versus, compared to)
    comparison_patterns = [
        r'COMPARE\s+([A-Z]{2,5})\s+VS\s+([A-Z]{2,5})',
        r'([A-Z]{2,5})\s+VS\s+([A-Z]{2,5})',
        r'COMPARE\s+([A-Z]{2,5})\s+VERSUS\s+([A-Z]{2,5})',
        r'([A-Z]{2,5})\s+VERSUS\s+([A-Z]{2,5})',
        r'COMPARE\s+([A-Z]{2,5})\s+AND\s+([A-Z]{2,5})',
    ]
    
    for pattern in comparison_patterns:
        matches = re.findall(pattern, query.upper())
        for match in matches:
            if isinstance(match, tuple):
                symbols.extend([s for s in match if len(s) >= 2 and len(s) <= 5 and s.isalpha()])
            else:
                if len(match) >= 2 and len(match) <= 5 and match.isalpha():
                    symbols.append(match)
    
    # If we found comparison symbols, return them
    if symbols:
        return list(set(symbols))  # Remove duplicates
    
    # Single symbol patterns
    symbol_patterns = [
        r'(?:investing in|analyze|analysis of|forecast for|predict|evaluate)\s+([A-Z]{2,5})(?:\s+stock|\b)',
        r'\b([A-Z]{2,5})\s+(?:stock|analysis|company|share|price|trends|predictions)',
        r'(?:symbol|ticker)\s+([A-Z]{2,5})\b',
        r'\b([A-Z]{2,5})\s+(?:based on|with|data|information)\b',
        # Match well-known symbols
        r'\b(AAPL|GOOGL|MSFT|TSLA|AMZN|NVDA|META|NFLX|CRM|ORCL|IBM|INTC|AMD|QCOM|ADBE|PYPL|NDAQ|COST|AVGO|TXN|HON|UNP|V|MA|JPM|BAC|WFC|GS|MS|C|AXP|BRK|JNJ|PFE|UNH|ABBV|MRK|LLY|TMO|DHR|GILD|BIIB|VRTX|REGN|CELG|XOM|CVX|COP|SLB|HAL|EOG|PXD|MPC|VLO|PSX|KMI|OKE|ENB|TRP|WMB|AMGN|MMM|CAT|DE|BA|LMT|RTX|NOC|GD|LHX|SPGI|MCO|BLK|SCHW|TFC|USB|PNC|COF|CME|ICE|CBOE)\b'
    ]
    
    # Try each pattern for single symbols
    for pattern in symbol_patterns:
        match = re.search(pattern, query.upper())
        if match:
            symbol = match.group(1)
            if len(symbol) >= 2 and len(symbol) <= 5 and symbol.isalpha():
                return [symbol]
    
    # Fallback: look for any potential symbols
    exclude_words = {
        'THE', 'AND', 'OR', 'OF', 'TO', 'FOR', 'WITH', 'IN', 'ON', 'AT', 'A', 'AN',
        'STOCK', 'STOCKS', 'ANALYSIS', 'ANALYZE', 'PRICE', 'TREND', 'PREDICT', 'FORECAST',
        'COMPANY', 'MARKET', 'TRADING', 'INVESTMENT', 'SHARE', 'SHARES', 'DATA', 'CHART',
        'FUTURE', 'PAST', 'CURRENT', 'TODAY', 'TOMORROW', 'WEEK', 'MONTH', 'YEAR', 'DAY',
        'BUY', 'SELL', 'HOLD', 'PERFORMANCE', 'VALUE', 'GROWTH', 'EARNINGS', 'REVENUE',
        'COMPREHENSIVE', 'PROVIDE', 'GIVE', 'SHOW', 'TELL', 'WHAT', 'HOW', 'WHEN', 'WHERE',
        'INCLUDING', 'TRENDS', 'PREDICTIONS', 'INSIGHTS', 'RECOMMENDATION', 'ADVICE',
        'RISK', 'RISKS', 'PROFILE', 'ASSESSMENT', 'EVALUATE', 'EVALUATION', 'BASED', 
        'HISTORICAL', 'INVESTING', 'INVESTOR', 'INVESTORS', 'COMPARE', 'COMPARISON'
    }
    
    words = query.upper().split()
    for word in words:
        clean_word = ''.join(c for c in word if c.isalpha())
        if (len(clean_word) >= 2 and len(clean_word) <= 5 and 
            clean_word.isalpha() and 
            clean_word not in exclude_words):
            return [clean_word]
    
    return []

def extract_stock_symbol(query: str) -> str:
    """Extract single stock symbol from user query (for backward compatibility)"""
    symbols = extract_stock_symbols(query)
    return symbols[0] if symbols else None

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
                # Create a custom workflow call with progress updates
                current_step += 1
                update_progress_callback(current_step, total_steps, f"Training LSTM model for {sym}...")
                
                workflow_results = st.session_state.workflow.complete_analysis(sym)
                
                current_step += 1  
                update_progress_callback(current_step, total_steps, f"Generating {sym} predictions and charts...")
                
                if workflow_results.get("success"):
                    comparison_results[sym] = workflow_results
                    summary = workflow_results["summary"]
                    comparison_responses.append(f"""**{sym}**: ${summary.get('current_price', 0):.2f} → ${summary.get('predicted_price', 0):.2f} ({summary.get('percentage_change', 0):+.2f}%)""")
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

**Performance Comparison:**
- **{sym1}**: {r1['summary']['trend_direction']} trend ({r1['summary']['confidence']} confidence)
- **{sym2}**: {r2['summary']['trend_direction']} trend ({r2['summary']['confidence']} confidence)

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
        logger.info(f"Starting comprehensive analysis for {symbol}")
        
        total_steps = 5
        current_step = 0
        
        # Try agent first (to show agent thoughts), then fallback to workflow
        agent_response = None
        try:
            # Skip agent analysis for now due to ReAct parsing issues
            # The workflow provides comprehensive analysis with LSTM predictions
            logger.info("Skipping agent analysis - using workflow for reliable results")
            
        except Exception as agent_error:
            logger.warning(f"⚠️ Agent setup failed: {agent_error}")
        
        # Always run workflow for charts and data (essential functionality)
        current_step += 1
        update_progress_callback(current_step, total_steps, f"Fetching {symbol} historical data...")
        
        logger.info(f"🔧 Running workflow analysis for {symbol}")
        try:
            workflow_results = st.session_state.workflow.complete_analysis_with_progress(symbol, update_progress_callback, current_step, total_steps)
            
            if workflow_results.get("success"):
                st.session_state.analysis_results = workflow_results
                workflow_response = workflow_results["ai_analysis"]
                logger.info("✅ Workflow analysis completed successfully")
                
                # Create a concise chat response, detailed analysis goes in expander
                summary = workflow_results["summary"]
                response = f"""✅ **Analysis Complete for {symbol}**

📊 **Key Metrics:**
- Current Price: ${summary.get('current_price', 0):.2f}
- 30-Day Prediction: ${summary.get('predicted_price', 0):.2f}
- Expected Change: {summary.get('percentage_change', 0):.2f}%
- Trend: {summary.get('trend_direction', 'Unknown')} ({summary.get('confidence', 'Unknown')} confidence)

📈 Interactive charts and detailed AI analysis are available below."""
                
                logger.info("Using concise summary response")
                    
            else:
                error_msg = workflow_results.get('error', 'Unknown error')
                logger.error(f"Workflow failed: {error_msg}")
                
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
                        
        except Exception as workflow_error:
            logger.error(f"Workflow exception: {workflow_error}")
            
            response = f"❌ **Analysis Error for {symbol}**\n\nI encountered an error while analyzing the stock: {str(workflow_error)}"
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
                workflow_results = st.session_state.workflow.complete_analysis(sym)
                if workflow_results.get("success"):
                    comparison_results[sym] = workflow_results
                    summary = workflow_results["summary"]
                    comparison_responses.append(f"""**{sym}**: ${summary.get('current_price', 0):.2f} → ${summary.get('predicted_price', 0):.2f} ({summary.get('percentage_change', 0):+.2f}%)""")
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

**Performance Comparison:**
- **{sym1}**: {r1['summary']['trend_direction']} trend ({r1['summary']['confidence']} confidence)
- **{sym2}**: {r2['summary']['trend_direction']} trend ({r2['summary']['confidence']} confidence)

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
        logger.info(f"Starting comprehensive analysis for {symbol}")
        
        # Try agent first (to show agent thoughts), then fallback to workflow
        agent_response = None
        logger.info(f"🚀 Attempting agent analysis for {symbol} (agent thoughts will be visible)")
        
        # Create a simple query that's more likely to work with ReAct format
        simple_query = f"What can you tell me about {symbol} stock?"
        
        try:
            # Skip agent analysis for now due to ReAct parsing issues
            # The workflow provides comprehensive analysis with LSTM predictions
            logger.info("Skipping agent analysis - using workflow for reliable results")
            
        except Exception as agent_error:
            logger.warning(f"⚠️ Agent setup failed: {agent_error}")
        
        # Always run workflow for charts and data (essential functionality)
        logger.info(f"🔧 Running workflow analysis for {symbol}")
        try:
            workflow_results = st.session_state.workflow.complete_analysis(symbol)
            
            if workflow_results.get("success"):
                st.session_state.analysis_results = workflow_results
                workflow_response = workflow_results["ai_analysis"]
                logger.info("✅ Workflow analysis completed successfully")
                
                # Create a concise chat response, detailed analysis goes in expander
                summary = workflow_results["summary"]
                response = f"""✅ **Analysis Complete for {symbol}**

📊 **Key Metrics:**
- Current Price: ${summary.get('current_price', 0):.2f}
- 30-Day Prediction: ${summary.get('predicted_price', 0):.2f}
- Expected Change: {summary.get('percentage_change', 0):.2f}%
- Trend: {summary.get('trend_direction', 'Unknown')} ({summary.get('confidence', 'Unknown')} confidence)

📈 Interactive charts and detailed AI analysis are available below."""
                
                logger.info("Using concise summary response")
                    
            else:
                error_msg = workflow_results.get('error', 'Unknown error')
                logger.error(f"Workflow failed: {error_msg}")
                
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
                        
        except Exception as workflow_error:
            logger.error(f"Workflow exception: {workflow_error}")
            
            response = f"❌ **Analysis Error for {symbol}**\n\nI encountered an error while analyzing the stock: {str(workflow_error)}"
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
        st.header("🛠️ Analysis Tools")
        
        # Quick analysis options
        st.subheader("Quick Analysis")
        quick_symbol = st.text_input("Stock Symbol", placeholder="e.g., AAPL, GOOGL, TSLA")
        
        if st.button("🔍 Quick Analysis", use_container_width=True):
            if quick_symbol:
                logger.info(f"Quick analysis requested for symbol: {quick_symbol}")
                query = f"Analyze {quick_symbol.upper()} stock with trends and predictions"
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
        if last_message["role"] == "user":
            # There's an unprocessed user message, process it
            # Create a progress container
            progress_container = st.empty()
            
            with progress_container.container():
                st.info("🤔 **Analyzing stock data and generating predictions...**")
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Update progress during processing
                def update_progress(step, total_steps, message):
                    progress = step / total_steps
                    progress_bar.progress(progress)
                    status_text.text(f"Step {step}/{total_steps}: {message}")
                
                # Process the query with progress updates
                response = process_user_query_with_progress(last_message["content"], update_progress)
            
            # Clear the progress display
            progress_container.empty()
            
            # Add assistant response
            st.session_state.messages.append({"role": "assistant", "content": response})
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
            st.subheader("📊 Analysis Results")
            
            results = st.session_state.analysis_results
            
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
                        
                        # Display AI insights for this stock
                        if "ai_analysis" in stock_results:
                            with st.expander(f"🤖 {symbol} AI Analysis", expanded=False):
                                st.markdown(stock_results["ai_analysis"])
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
                if "visualizations" in results and "chart_objects" in results["visualizations"]:
                    st.subheader("📈 Price Charts")
                    display_charts(results["visualizations"])
                
                # Display AI insights
                if "ai_analysis" in results:
                    with st.expander("🤖 AI Analysis", expanded=False):
                        st.markdown(results["ai_analysis"])
    
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