"""
Simple in-memory data store for sharing data between tools without exposing it to the agent
"""
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class DataStore:
    """Simple in-memory store for stock data"""
    
    def __init__(self):
        self._data = {}
        
    def store_stock_data(self, symbol: str, data: Dict[str, Any]) -> None:
        """Store stock data for a symbol"""
        self._data[symbol.upper()] = data
        logger.info(f"Stored data for {symbol.upper()}: {len(data.get('data', []))} records")
        
    def get_stock_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Retrieve stock data for a symbol"""
        data = self._data.get(symbol.upper())
        if data:
            logger.info(f"Retrieved data for {symbol.upper()}: {len(data.get('data', []))} records")
        else:
            logger.warning(f"No data found for {symbol.upper()}")
        return data
        
    def get_stock_historical_data(self, symbol: str) -> Optional[list]:
        """Get just the historical data records for a symbol"""
        data = self.get_stock_data(symbol)
        return data.get('data', []) if data else None
        
    def clear_symbol(self, symbol: str) -> None:
        """Clear data for a symbol"""
        if symbol.upper() in self._data:
            del self._data[symbol.upper()]
            logger.info(f"Cleared data for {symbol.upper()}")
            
    def clear_all(self) -> None:
        """Clear all stored data"""
        self._data.clear()
        logger.info("Cleared all stored data")

# Global instance
_data_store = DataStore()

def get_data_store() -> DataStore:
    """Get the global data store instance"""
    return _data_store