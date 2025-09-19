"""
Stock Analysis Agent - Specialized agent for stock market analysis
"""

import json
import logging
import re
from typing import Dict, Any
from datetime import datetime
from .base_agent import BaseAgent

logger = logging.getLogger("MultiAgentSystem")


class StockAnalysisAgent(BaseAgent):
    """Specialized agent for stock market analysis using Yahoo Finance"""

    def execute(self, query: str) -> Dict[str, Any]:
        """Execute stock analysis with intelligent parameter extraction"""
        try:
            logger.info(f"📈 StockAnalysisAgent analyzing: '{query}'")

            # Extract parameters from query using LLM
            prompt = f"""Extract stock analysis parameters from: "{query}"

Analyze the query and determine:
1. Stock symbol(s) mentioned (e.g., AAPL, GOOGL, TSLA, MSFT, etc.)
2. Time period for analysis (if mentioned: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
3. Type of analysis requested (basic, technical, comprehensive)

Guidelines:
- If no period is specified, use "1y" as default
- If no analysis type is specified, use "comprehensive" as default
- Extract the first/main stock symbol mentioned
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

Respond with JSON only:
{{"symbol": "AAPL", "period": "1y", "analysis_type": "comprehensive"}}

If no clear stock symbol is found, use "SPY" (S&P 500 ETF) as default."""

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

            # Execute stock analysis tool
            result = self._execute_tool_script("stock_analysis", params)

            # Enhance result with LLM-generated insights if successful
            if result.get("success"):
                enhanced_result = self._generate_insights(result, query)
                result.update(enhanced_result)

            return {
                "agent": "StockAnalysisAgent",
                "tool": "stock_analysis",
                "parameters": params,
                "result": result,
                "success": True,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"📈 StockAnalysisAgent error: {str(e)}")
            return {
                "agent": "StockAnalysisAgent",
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

        # First check for company names
        for company, symbol in company_mappings.items():
            if company in query_lower:
                return symbol

        # Then check for symbol patterns
        for pattern in symbol_patterns:
            matches = re.findall(pattern, query.upper())
            if matches:
                # Filter out common words that aren't stock symbols
                excluded = {'AND', 'THE', 'FOR', 'WITH', 'STOCK', 'PRICE', 'DATA'}
                valid_symbols = [m for m in matches if m not in excluded and len(m) <= 5]
                if valid_symbols:
                    return valid_symbols[0]

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

    def _generate_insights(self, result: Dict[str, Any], original_query: str) -> Dict[str, Any]:
        """Generate beautifully formatted insights from the analysis results"""
        try:
            # Extract key metrics for formatting
            symbol = result.get("symbol", "Unknown")
            company_name = result.get("company_name", symbol)

            # Build beautiful formatted response
            formatted_response = self._create_beautiful_stock_report(result, original_query)

            return {
                "ai_insights": {
                    "analysis": formatted_response,
                    "generated_at": datetime.now().isoformat()
                }
            }

        except Exception as e:
            logger.error(f"📈 Failed to generate insights: {str(e)}")
            return {
                "ai_insights": {
                    "analysis": "Unable to generate AI insights at this time.",
                    "error": str(e)
                }
            }

    def _create_beautiful_stock_report(self, result: Dict[str, Any], original_query: str) -> str:
        """Create a beautifully formatted stock analysis report"""
        symbol = result.get("symbol", "Unknown")
        company_name = result.get("company_name", symbol)

        # Start with header
        report = f"📈 STOCK ANALYSIS: {company_name} ({symbol})\n{'='*50}\n\n"

        # Price overview section
        if "basic_metrics" in result:
            metrics = result["basic_metrics"]
            current_price = metrics.get('current_price', 'N/A')
            price_change = metrics.get('price_change', 0)
            price_change_pct = metrics.get('price_change_percentage', 0)

            # Choose emoji based on performance
            trend_emoji = "🟢" if price_change_pct > 0 else "🔴" if price_change_pct < 0 else "⚪"
            arrow = "↗️" if price_change_pct > 0 else "↘️" if price_change_pct < 0 else "➡️"

            report += f"💰 CURRENT PRICE OVERVIEW\n"
            report += f"{trend_emoji} ${current_price} {arrow} {price_change_pct:+.2f}% (${price_change:+.2f})\n\n"

            # Market stats in a clean format
            period_high = metrics.get('period_high', 'N/A')
            period_low = metrics.get('period_low', 'N/A')
            avg_volume = metrics.get('average_volume', 0)

            report += f"Period Range: ${period_low} - ${period_high}\n"
            report += f"Avg Volume: {avg_volume:,} shares\n"

            # Market info if available
            market_cap = metrics.get('market_cap')
            if market_cap:
                report += f"Market Cap: ${market_cap:,.0f}\n"
            sector = metrics.get('sector')
            if sector:
                report += f"Sector: {sector}\n"
            report += "\n"

        # Technical analysis section
        if "technical_indicators" in result:
            tech = result["technical_indicators"]
            report += f"📊 TECHNICAL ANALYSIS\n"

            # Moving averages
            ma_data = tech.get("moving_averages", {})
            if any(ma_data.values()):
                report += f"Moving Averages:\n"
                if ma_data.get("ma_10"):
                    report += f"  • 10-day: ${ma_data['ma_10']:.2f}\n"
                if ma_data.get("ma_20"):
                    report += f"  • 20-day: ${ma_data['ma_20']:.2f}\n"
                if ma_data.get("ma_50"):
                    report += f"  • 50-day: ${ma_data['ma_50']:.2f}\n"
                report += "\n"

            # RSI
            momentum = tech.get("momentum", {})
            rsi = momentum.get("rsi")
            if rsi:
                rsi_signal = momentum.get("rsi_signal", "neutral")
                rsi_emoji = "🔴" if rsi_signal == "overbought" else "🟢" if rsi_signal == "oversold" else "🟡"
                report += f"RSI: {rsi_emoji} {rsi:.1f} ({rsi_signal})\n\n"

            # Trading signals
            signals = tech.get("trading_signals", {})
            if signals.get("signals"):
                sentiment = signals.get("overall_sentiment", "neutral")
                sentiment_emoji = "🔵" if sentiment == "bullish" else "🟠" if sentiment == "bearish" else "⚪"
                report += f"Trading Signals: {sentiment_emoji} {sentiment.title()}\n"
                for signal in signals["signals"][:3]:  # Show top 3 signals
                    report += f"  • {signal}\n"
                report += "\n"

        # Risk and performance section
        if "comprehensive_metrics" in result:
            comp = result["comprehensive_metrics"]
            risk_metrics = comp.get("risk_metrics", {})

            report += f"⚠️  RISK ASSESSMENT\n"
            volatility = risk_metrics.get("volatility")
            risk_level = risk_metrics.get("risk_level", "unknown")

            risk_emoji = "🔴" if risk_level == "high" else "🟡" if risk_level == "medium" else "🟢"

            if volatility:
                report += f"{risk_emoji} Risk Level: {risk_level.title()} ({volatility:.1f}% volatility)\n"

            max_drawdown = risk_metrics.get("max_drawdown")
            if max_drawdown is not None:
                report += f"Max Drawdown: {max_drawdown:.1f}%\n"

            sharpe_ratio = risk_metrics.get("sharpe_ratio")
            if sharpe_ratio is not None:
                report += f"Sharpe Ratio: {sharpe_ratio:.2f}\n"

            report += "\n"

            # Investment recommendation
            investment_summary = comp.get("investment_summary", {})
            recommendation = investment_summary.get("recommendation")
            if recommendation:
                rec_emoji = "🟢" if "BUY" in recommendation else "🔴" if "SELL" in recommendation else "🟡"
                report += f"🎯 INVESTMENT RECOMMENDATION\n"
                report += f"{rec_emoji} {recommendation}\n\n"

                key_insight = investment_summary.get("key_insight")
                if key_insight:
                    report += f"💡 Key Insight: {key_insight}\n\n"

        # Generate AI commentary
        try:
            # Create context for AI analysis
            context = self._extract_key_metrics_for_ai(result)

            ai_prompt = f"""Based on this stock analysis for {company_name} ({symbol}), provide 2-3 brief, actionable insights for investors. Focus on:
1. What the data tells us about the stock's current situation
2. Key factors investors should consider
3. Any notable opportunities or risks

Data: {context}

Keep it concise and practical - 2-3 sentences total."""

            ai_insights = self.llm.call(ai_prompt)

            report += f"🤖 AI ANALYSIS\n"
            report += f"{ai_insights.strip()}\n\n"

        except Exception as e:
            logger.debug(f"Could not generate AI insights: {e}")

        # Footer
        report += f"{'='*50}\nAnalysis period: {result.get('period', 'N/A')} | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"

        return report

    def _extract_key_metrics_for_ai(self, result: Dict[str, Any]) -> str:
        """Extract key metrics for AI analysis"""
        metrics = []

        if "basic_metrics" in result:
            basic = result["basic_metrics"]
            metrics.append(f"Price: ${basic.get('current_price', 'N/A')}")
            metrics.append(f"Change: {basic.get('price_change_percentage', 0):.2f}%")

        if "performance_summary" in result:
            perf = result["performance_summary"]
            metrics.append(f"Trend: {perf.get('trend', 'N/A')}")
            metrics.append(f"Volatility: {perf.get('volatility', 'N/A')}")

        if "comprehensive_metrics" in result:
            comp = result["comprehensive_metrics"]
            risk = comp.get("risk_metrics", {})
            if risk.get("risk_level"):
                metrics.append(f"Risk: {risk['risk_level']}")

            volume_analysis = comp.get("volume_analysis", {})
            if volume_analysis.get("volume_trend"):
                metrics.append(f"Volume: {volume_analysis['volume_trend']}")

        return "; ".join(metrics)