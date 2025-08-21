import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class TechnicalAnalysis:
    """Technical analysis indicators and signals"""
    
    @staticmethod
    def calculate_sma(data: pd.Series, window: int) -> pd.Series:
        """Calculate Simple Moving Average"""
        return data.rolling(window=window).mean()
    
    @staticmethod
    def calculate_ema(data: pd.Series, window: int) -> pd.Series:
        """Calculate Exponential Moving Average"""
        return data.ewm(span=window).mean()
    
    @staticmethod
    def calculate_rsi(data: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def calculate_macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """Calculate MACD indicators"""
        ema_fast = TechnicalAnalysis.calculate_ema(data, fast)
        ema_slow = TechnicalAnalysis.calculate_ema(data, slow)
        
        macd = ema_fast - ema_slow
        macd_signal = TechnicalAnalysis.calculate_ema(macd, signal)
        macd_histogram = macd - macd_signal
        
        return {
            'macd': macd,
            'macd_signal': macd_signal,
            'macd_histogram': macd_histogram
        }
    
    @staticmethod
    def calculate_bollinger_bands(data: pd.Series, window: int = 20, num_std: float = 2) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = TechnicalAnalysis.calculate_sma(data, window)
        std = data.rolling(window=window).std()
        
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        
        return {
            'bb_upper': upper_band,
            'bb_middle': sma,
            'bb_lower': lower_band
        }
    
    @staticmethod
    def calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, 
                           k_window: int = 14, d_window: int = 3) -> Dict[str, pd.Series]:
        """Calculate Stochastic Oscillator"""
        lowest_low = low.rolling(window=k_window).min()
        highest_high = high.rolling(window=k_window).max()
        
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_window).mean()
        
        return {
            'stoch_k': k_percent,
            'stoch_d': d_percent
        }
    
    @staticmethod
    def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        
        true_range = pd.DataFrame({
            'hl': high_low,
            'hc': high_close,
            'lc': low_close
        }).max(axis=1)
        
        return true_range.rolling(window=window).mean()
    
    @staticmethod
    def analyze_trends(data: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive trend analysis"""
        close_prices = data['Close']
        
        # Calculate various indicators
        sma_20 = TechnicalAnalysis.calculate_sma(close_prices, 20)
        sma_50 = TechnicalAnalysis.calculate_sma(close_prices, 50)
        ema_12 = TechnicalAnalysis.calculate_ema(close_prices, 12)
        rsi = TechnicalAnalysis.calculate_rsi(close_prices)
        macd_indicators = TechnicalAnalysis.calculate_macd(close_prices)
        bb_indicators = TechnicalAnalysis.calculate_bollinger_bands(close_prices)
        
        # Get latest values
        latest_price = close_prices.iloc[-1]
        latest_sma_20 = sma_20.iloc[-1]
        latest_sma_50 = sma_50.iloc[-1]
        latest_rsi = rsi.iloc[-1]
        latest_macd = macd_indicators['macd'].iloc[-1]
        latest_macd_signal = macd_indicators['macd_signal'].iloc[-1]
        
        # Trend signals
        signals = {
            'price_above_sma20': latest_price > latest_sma_20,
            'price_above_sma50': latest_price > latest_sma_50,
            'sma20_above_sma50': latest_sma_20 > latest_sma_50,
            'rsi_oversold': latest_rsi < 30,
            'rsi_overbought': latest_rsi > 70,
            'macd_bullish': latest_macd > latest_macd_signal,
            'price_near_bb_upper': latest_price > bb_indicators['bb_upper'].iloc[-1],
            'price_near_bb_lower': latest_price < bb_indicators['bb_lower'].iloc[-1]
        }
        
        # Overall trend assessment
        bullish_signals = sum([
            signals['price_above_sma20'],
            signals['price_above_sma50'],
            signals['sma20_above_sma50'],
            signals['macd_bullish'],
            not signals['rsi_overbought']
        ])
        
        bearish_signals = sum([
            not signals['price_above_sma20'],
            not signals['price_above_sma50'],
            not signals['sma20_above_sma50'],
            not signals['macd_bullish'],
            signals['rsi_overbought']
        ])
        
        if bullish_signals >= 3:
            overall_trend = "Bullish"
        elif bearish_signals >= 3:
            overall_trend = "Bearish"
        else:
            overall_trend = "Neutral"
        
        # Support and resistance levels (simplified)
        recent_highs = data['High'].tail(20).nlargest(3).mean()
        recent_lows = data['Low'].tail(20).nsmallest(3).mean()
        
        return {
            'indicators': {
                'sma_20': sma_20,
                'sma_50': sma_50,
                'ema_12': ema_12,
                'rsi': rsi,
                **macd_indicators,
                **bb_indicators
            },
            'signals': signals,
            'overall_trend': overall_trend,
            'trend_strength': max(bullish_signals, bearish_signals),
            'support_level': recent_lows,
            'resistance_level': recent_highs,
            'latest_values': {
                'price': latest_price,
                'sma_20': latest_sma_20,
                'sma_50': latest_sma_50,
                'rsi': latest_rsi,
                'macd': latest_macd,
                'macd_signal': latest_macd_signal
            }
        }
    
    @staticmethod
    def generate_trading_signals(analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate actionable trading signals based on analysis"""
        signals = analysis['signals']
        latest_values = analysis['latest_values']
        
        # Signal strength calculation
        buy_signals = 0
        sell_signals = 0
        
        # Moving average signals
        if signals['price_above_sma20'] and signals['sma20_above_sma50']:
            buy_signals += 2
        elif not signals['price_above_sma20'] and not signals['sma20_above_sma50']:
            sell_signals += 2
        
        # RSI signals
        if signals['rsi_oversold']:
            buy_signals += 1
        elif signals['rsi_overbought']:
            sell_signals += 1
        
        # MACD signals
        if signals['macd_bullish']:
            buy_signals += 1
        else:
            sell_signals += 1
        
        # Bollinger Bands signals
        if signals['price_near_bb_lower']:
            buy_signals += 1
        elif signals['price_near_bb_upper']:
            sell_signals += 1
        
        # Determine primary signal
        if buy_signals > sell_signals + 1:
            primary_signal = "BUY"
            confidence = min(buy_signals / 5 * 100, 100)
        elif sell_signals > buy_signals + 1:
            primary_signal = "SELL"
            confidence = min(sell_signals / 5 * 100, 100)
        else:
            primary_signal = "HOLD"
            confidence = 50
        
        # Risk assessment
        risk_factors = []
        if latest_values['rsi'] > 80:
            risk_factors.append("Extremely overbought conditions")
        elif latest_values['rsi'] < 20:
            risk_factors.append("Extremely oversold conditions")
        
        if signals['price_near_bb_upper']:
            risk_factors.append("Price near upper Bollinger Band")
        elif signals['price_near_bb_lower']:
            risk_factors.append("Price near lower Bollinger Band")
        
        return {
            'primary_signal': primary_signal,
            'confidence': confidence,
            'buy_signals_count': buy_signals,
            'sell_signals_count': sell_signals,
            'risk_factors': risk_factors,
            'recommendations': TechnicalAnalysis._generate_recommendations(
                primary_signal, confidence, risk_factors, analysis
            )
        }
    
    @staticmethod
    def _generate_recommendations(signal: str, confidence: float, risk_factors: list, analysis: Dict) -> list:
        """Generate specific recommendations based on analysis"""
        recommendations = []
        
        if signal == "BUY":
            recommendations.append(f"Consider buying with {confidence:.1f}% confidence")
            if analysis['support_level']:
                recommendations.append(f"Support level around ${analysis['support_level']:.2f}")
            if risk_factors:
                recommendations.append("Monitor risk factors closely")
        elif signal == "SELL":
            recommendations.append(f"Consider selling with {confidence:.1f}% confidence")
            if analysis['resistance_level']:
                recommendations.append(f"Resistance level around ${analysis['resistance_level']:.2f}")
        else:
            recommendations.append("Hold current position and monitor for clearer signals")
            recommendations.append("Wait for trend confirmation before making major moves")
        
        # Add general risk management advice
        recommendations.append("Always use proper position sizing and stop-losses")
        recommendations.append("Consider overall market conditions and news events")
        
        return recommendations