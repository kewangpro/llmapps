"""
Utility functions for stock analysis
"""
import re


def extract_stock_symbols(query: str) -> list:
    """Extract all stock symbols from user query, including comparisons"""    
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
        r'\b(AAPL|GOOGL|MSFT|TSLA|AMZN|NVDA|META|NFLX|CRM|ORCL|IBM|INTC|AMD|QCOM|ADBE|PYPL|NDAQ|COST|AVGO|TXN|HON|UNP|V|MA|JPM|BAC|WFC|GS|MS|C|AXP|BRK|JNJ|PFE|UNH|ABBV|MRK|LLY|TMO|DHR|GILD|BIIB|VRTX|REGN|CELG|XOM|CVX|COP|SLB|HAL|EOG|PXD|MPC|VLO|PSX|KMI|OKE|ENB|TRP|WMB|AMGN|MMM|CAT|DE|BA|LMT|RTX|NOC|GD|LHX|SPGI|MCO|BLK|SCHW|TFC|USB|PNC|COF|CME|ICE|CBOE|TEAM)\b'
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