"""
Stock Analysis Agent - Specialized agent for stock market analysis
"""

import json
import logging
import re
from typing import Dict, Any
from datetime import datetime
from .base_agent import BaseAgent

logger = logging.getLogger("StockAnalysisAgent")


class StockAnalysisAgent(BaseAgent):
    """Specialized agent for stock market analysis using Yahoo Finance"""

    def execute(self, query: str) -> Dict[str, Any]:
        """Execute stock analysis with intelligent parameter extraction"""
        try:
            logger.info(f"📈 StockAnalysisAgent analyzing: '{query}'")

            # Extract parameters from query using LLM
            prompt = f"""Extract stock analysis parameters from: "{query}"

IMPORTANT: Look carefully at the query for ANY stock symbol or company name mentioned.

Analyze the query and determine:
1. Stock symbol(s) mentioned - extract the EXACT symbol from the query
2. Time period for analysis (if mentioned: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
3. Type of analysis requested (basic, technical, comprehensive)

Guidelines:
- Extract the ACTUAL stock symbol mentioned in the query (e.g., if query says "TEAM", use "TEAM", not "AAPL")
- If no period is specified, use "1y" as default
- If no analysis type is specified, use "comprehensive" as default
- Look for company names and convert to symbols if possible (e.g., "Apple" -> "AAPL", "Tesla" -> "TSLA")

Common stock symbols:
- Apple: AAPL
- Microsoft: MSFT
- Google/Alphabet: GOOGL
- Amazon: AMZN
- Tesla: TSLA
- Meta/Facebook: META
- Netflix: NFLX
- Nvidia: NVDA
- AMD: AMD
- Intel: INTC

Respond with JSON only using the format:
{{"symbol": "EXTRACTED_SYMBOL", "period": "1y", "analysis_type": "comprehensive"}}

Replace EXTRACTED_SYMBOL with the actual symbol from the query. If no clear stock symbol is found, use "SPY"."""

            response = self.llm.call(prompt)
            logger.info(f"📈 Parameter extraction: {response.strip()}")

            try:
                # Clean up the response to extract JSON
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    params = json.loads(json_match.group())
                else:
                    raise json.JSONDecodeError("No JSON found", response, 0)
            except json.JSONDecodeError:
                # Fallback parameters with intelligent defaults
                symbol = self._extract_symbol_fallback(query)
                params = {
                    "symbol": symbol,
                    "period": self._extract_period_fallback(query),
                    "analysis_type": self._extract_analysis_type_fallback(query)
                }

            # Validate and clean parameters
            params = self._validate_parameters(params)

            # Execute stock analysis tool (data retrieval and metrics calculation only)
            tool_result = self._execute_tool_script("stock_analysis", params)

            # Generate LLM analysis of the tool data
            if tool_result.get("success"):
                # Use LLM to analyze the stock data returned by the tool
                llm_analysis = self._analyze_stock_data_with_llm(tool_result, query)

                # Format tool_data as CSV for downstream agents
                formatted_tool_data = self._format_tool_data_as_csv(tool_result)

                result = {
                    "tool_data": formatted_tool_data,  # CSV formatted data for chaining
                    "llm_analysis": llm_analysis  # LLM insights
                }
            else:
                result = tool_result

            return {
                "tool": "stock_analysis",
                "parameters": params,
                "result": result,
                "success": True,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"📈 StockAnalysisAgent error: {str(e)}")
            return {
                "tool": "stock_analysis",
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def _extract_symbol_fallback(self, query: str) -> str:
        """Extract stock symbol using pattern matching as fallback"""
        # Common patterns for stock symbols
        symbol_patterns = [
            r'\b([A-Z]{1,5})\b',  # 1-5 uppercase letters
            r'\$([A-Z]{1,5})\b',  # Symbol with $ prefix
        ]

        # Company name mappings
        company_mappings = {
            'apple': 'AAPL',
            'microsoft': 'MSFT',
            'google': 'GOOGL',
            'alphabet': 'GOOGL',
            'amazon': 'AMZN',
            'tesla': 'TSLA',
            'meta': 'META',
            'facebook': 'META',
            'netflix': 'NFLX',
            'nvidia': 'NVDA',
            'amd': 'AMD',
            'intel': 'INTC',
            'disney': 'DIS',
            'walmart': 'WMT',
            'coca cola': 'KO',
            'pepsi': 'PEP',
            'mastercard': 'MA',
            'visa': 'V',
            'boeing': 'BA',
            'caterpillar': 'CAT'
        }

        query_lower = query.lower()

        # FIRST check for symbol patterns (prioritize explicit symbols like "TEAM")
        for pattern in symbol_patterns:
            matches = re.findall(pattern, query.upper())
            if matches:
                # Filter out common words that aren't stock symbols
                excluded = {'AND', 'THE', 'FOR', 'WITH', 'STOCK', 'PRICE', 'DATA', 'PREDICT', 'FUTURE', 'TRENDS', 'ANALYSIS'}
                valid_symbols = [m for m in matches if m not in excluded and len(m) <= 5]
                if valid_symbols:
                    logger.info(f"📈 Fallback extracted symbol: {valid_symbols[0]} from query: {query}")
                    return valid_symbols[0]

        # THEN check for company names
        for company, symbol in company_mappings.items():
            if company in query_lower:
                logger.info(f"📈 Fallback mapped company '{company}' to symbol: {symbol}")
                return symbol

        logger.info(f"📈 Fallback: No symbol found, using SPY default")
        return "SPY"  # Default to S&P 500 ETF

    def _extract_period_fallback(self, query: str) -> str:
        """Extract time period using pattern matching"""
        query_lower = query.lower()

        if any(word in query_lower for word in ['day', 'daily', 'today']):
            return "1d"
        elif any(word in query_lower for word in ['week', 'weekly']):
            return "5d"
        elif any(word in query_lower for word in ['month', 'monthly', '1 month']):
            return "1mo"
        elif any(word in query_lower for word in ['quarter', '3 month', 'quarterly']):
            return "3mo"
        elif any(word in query_lower for word in ['6 month', 'half year']):
            return "6mo"
        elif any(word in query_lower for word in ['year', 'yearly', '12 month', 'annual']):
            return "1y"
        elif any(word in query_lower for word in ['2 year', 'two year']):
            return "2y"
        elif any(word in query_lower for word in ['5 year', 'five year']):
            return "5y"
        elif any(word in query_lower for word in ['decade', '10 year']):
            return "10y"
        elif 'ytd' in query_lower or 'year to date' in query_lower:
            return "ytd"
        elif any(word in query_lower for word in ['max', 'maximum', 'all time', 'historical']):
            return "max"

        return "1y"  # Default

    def _extract_analysis_type_fallback(self, query: str) -> str:
        """Extract analysis type using pattern matching"""
        query_lower = query.lower()

        if any(word in query_lower for word in ['basic', 'simple', 'quick', 'overview']):
            return "basic"
        elif any(word in query_lower for word in ['technical', 'rsi', 'moving average', 'bollinger', 'indicators']):
            return "technical"
        elif any(word in query_lower for word in ['comprehensive', 'detailed', 'full', 'complete', 'thorough']):
            return "comprehensive"

        return "comprehensive"  # Default

    def _validate_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clean parameters"""
        # Validate symbol
        symbol = params.get("symbol", "SPY").upper().strip()
        if not symbol or len(symbol) > 10:
            symbol = "SPY"

        # Validate period
        valid_periods = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]
        period = params.get("period", "1y")
        if period not in valid_periods:
            period = "1y"

        # Validate analysis type
        valid_types = ["basic", "technical", "comprehensive"]
        analysis_type = params.get("analysis_type", "comprehensive")
        if analysis_type not in valid_types:
            analysis_type = "comprehensive"

        return {
            "symbol": symbol,
            "period": period,
            "analysis_type": analysis_type
        }

    def _analyze_stock_data_with_llm(self, tool_result: Dict[str, Any], original_query: str) -> str:
        """Use LLM to analyze the stock data returned by the tool"""
        try:
            # Extract key information from tool result
            symbol = tool_result.get("symbol", "Unknown")
            company_name = tool_result.get("company_name", symbol)

            # Create a comprehensive prompt for LLM analysis
            analysis_prompt = f"""Analyze this stock data for {company_name} ({symbol}) and provide insights for the user query: "{original_query}"

Stock Data:
{json.dumps(tool_result, indent=2)}

Please provide a comprehensive analysis including:
1. Current price performance and trends
2. Key technical indicators and what they suggest
3. Risk assessment based on volatility and other metrics
4. Investment insights and recommendations
5. Any notable patterns or concerns

Format your response in a clear, professional manner that would be helpful for an investor. Use specific numbers from the data and explain what they mean."""

            # Get LLM analysis
            llm_response = self.llm.call(analysis_prompt)
            logger.info(f"📊 Generated LLM analysis for {symbol}")

            return llm_response.strip()

        except Exception as e:
            logger.error(f"📈 Failed to generate LLM analysis: {str(e)}")
            return f"Unable to generate analysis at this time: {str(e)}"

    def _format_tool_data_as_csv(self, tool_result: Dict[str, Any]) -> str:
        """Format tool result as CSV with multivariate features for enhanced forecasting

        Exports Priority 1 essential features:
        - Volume: Predicts volatility spikes
        - RSI: Momentum indicator
        - Volatility: Market uncertainty measure
        - High/Low: Price range and volatility
        """
        try:
            historical_data = tool_result.get("historical_data", {})
            if not historical_data:
                return "No historical data available"

            dates = historical_data.get("dates", [])
            prices = historical_data.get("prices", [])
            opens = historical_data.get("opens", [])
            highs = historical_data.get("highs", [])
            lows = historical_data.get("lows", [])
            volumes = historical_data.get("volumes", [])
            ma_20 = historical_data.get("ma_20", [])
            ma_50 = historical_data.get("ma_50", [])

            if not dates or not prices:
                return "Insufficient data for CSV export"

            # Calculate RSI time series (14-period)
            rsi_values = self._calculate_rsi_series(prices)

            # Calculate rolling volatility time series (20-period annualized)
            volatility_values = self._calculate_volatility_series(prices)

            # Format as wide CSV with all features for multivariate forecasting
            csv_data = "date,close,open,high,low,volume,rsi,volatility,ma_20,ma_50\n"

            for i in range(len(dates)):
                date = dates[i]
                close = prices[i] if i < len(prices) else ""
                open_price = opens[i] if i < len(opens) else ""
                high = highs[i] if i < len(highs) else ""
                low = lows[i] if i < len(lows) else ""
                volume = volumes[i] if i < len(volumes) else ""
                rsi = f"{rsi_values[i]:.2f}" if i < len(rsi_values) and rsi_values[i] is not None else ""
                volatility = f"{volatility_values[i]:.4f}" if i < len(volatility_values) and volatility_values[i] is not None else ""
                ma20 = f"{ma_20[i]:.2f}" if i < len(ma_20) and ma_20[i] is not None else ""
                ma50 = f"{ma_50[i]:.2f}" if i < len(ma_50) and ma_50[i] is not None else ""

                csv_data += f"{date},{close},{open_price},{high},{low},{volume},{rsi},{volatility},{ma20},{ma50}\n"

            feature_count = sum([
                1,  # close (always present)
                1 if opens else 0,
                1 if highs else 0,
                1 if lows else 0,
                1 if volumes else 0,
                1,  # rsi (calculated)
                1,  # volatility (calculated)
                1 if ma_20 else 0,
                1 if ma_50 else 0
            ])

            logger.info(f"📊 Formatted multivariate stock data: {len(dates)} dates × {feature_count} features")
            logger.info(f"📊 Features: close, open, high, low, volume, rsi, volatility, ma_20, ma_50")
            return csv_data

        except Exception as e:
            logger.error(f"📈 Failed to format tool data as CSV: {str(e)}")
            return f"Error formatting data: {str(e)}"

    def _calculate_rsi_series(self, prices: list, period: int = 14) -> list:
        """Calculate RSI (Relative Strength Index) time series

        RSI = 100 - (100 / (1 + RS))
        where RS = Average Gain / Average Loss over period
        """
        try:
            if len(prices) < period + 1:
                return [None] * len(prices)

            rsi_values = [None] * period  # First 'period' values are None

            # Calculate price changes
            deltas = []
            for i in range(1, len(prices)):
                delta = prices[i] - prices[i-1]
                deltas.append(delta)

            # Separate gains and losses
            gains = [max(d, 0) for d in deltas]
            losses = [abs(min(d, 0)) for d in deltas]

            # Calculate initial average gain/loss
            avg_gain = sum(gains[:period]) / period
            avg_loss = sum(losses[:period]) / period

            # Calculate first RSI value
            if avg_loss == 0:
                rsi_values.append(100)
            else:
                rs = avg_gain / avg_loss
                rsi_values.append(100 - (100 / (1 + rs)))

            # Calculate subsequent RSI values using smoothed averages
            for i in range(period, len(deltas)):
                avg_gain = (avg_gain * (period - 1) + gains[i]) / period
                avg_loss = (avg_loss * (period - 1) + losses[i]) / period

                if avg_loss == 0:
                    rsi_values.append(100)
                else:
                    rs = avg_gain / avg_loss
                    rsi_values.append(100 - (100 / (1 + rs)))

            return rsi_values

        except Exception as e:
            logger.error(f"📈 RSI calculation error: {str(e)}")
            return [None] * len(prices)

    def _calculate_volatility_series(self, prices: list, window: int = 20) -> list:
        """Calculate rolling volatility (annualized standard deviation of returns)

        Volatility = StdDev(returns) * sqrt(252)
        where returns = (price[t] - price[t-1]) / price[t-1]
        """
        try:
            if len(prices) < window + 1:
                return [None] * len(prices)

            volatility_values = [None] * window  # First 'window' values are None

            # Calculate returns
            returns = []
            for i in range(1, len(prices)):
                if prices[i-1] != 0:
                    ret = (prices[i] - prices[i-1]) / prices[i-1]
                    returns.append(ret)
                else:
                    returns.append(0)

            # Calculate rolling volatility
            for i in range(window - 1, len(returns)):
                window_returns = returns[i - window + 1:i + 1]

                # Calculate standard deviation
                mean_return = sum(window_returns) / len(window_returns)
                variance = sum((r - mean_return) ** 2 for r in window_returns) / len(window_returns)
                std_dev = variance ** 0.5

                # Annualize (252 trading days per year)
                annualized_volatility = std_dev * (252 ** 0.5)
                volatility_values.append(annualized_volatility)

            return volatility_values

        except Exception as e:
            logger.error(f"📈 Volatility calculation error: {str(e)}")
            return [None] * len(prices)

