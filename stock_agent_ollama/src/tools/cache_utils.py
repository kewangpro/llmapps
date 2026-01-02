import json
import time
from pathlib import Path
from typing import Any, Optional
import hashlib
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class SafeJSONEncoder(json.JSONEncoder):
    """JSON encoder that can handle numpy arrays and pandas DataFrames"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return {'__numpy_array__': obj.tolist()}
        elif isinstance(obj, pd.DataFrame):
            # Use to_json with split orientation to preserve MultiIndex and types
            return {'__pandas_json_split__': obj.to_json(orient='split', date_format='iso')}
        elif isinstance(obj, pd.Series):
            return {'__pandas_series__': obj.to_dict(), '__pandas_index__': list(obj.index)}
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif hasattr(obj, '__dict__'):
            # For custom objects, serialize their dict representation
            return {'__custom_object__': obj.__class__.__name__, '__data__': obj.__dict__}
        return super().default(obj)

def safe_json_decode(dct):
    """JSON decoder that can reconstruct numpy arrays and pandas DataFrames"""
    if '__numpy_array__' in dct:
        return np.array(dct['__numpy_array__'])
    elif '__pandas_json_split__' in dct:
        from io import StringIO
        # Use StringIO to avoid potential path interpretation issues
        df = pd.read_json(StringIO(dct['__pandas_json_split__']), orient='split')
        
        # Attempt to restore MultiIndex if columns are tuples (common with yfinance)
        if not df.empty and len(df.columns) > 0:
            try:
                # Check if first column is a tuple (or list masquerading as tuple in some contexts)
                first_col = df.columns[0]
                if isinstance(first_col, (tuple, list)):
                    df.columns = pd.MultiIndex.from_tuples(df.columns)
            except Exception:
                # If restoration fails, keep as is
                pass
                
        return df
    elif '__pandas_dataframe__' in dct:
        df = pd.DataFrame(dct['__pandas_dataframe__'])
        if '__pandas_index__' in dct:
            df.index = dct['__pandas_index__']
        return df
    elif '__pandas_series__' in dct:
        series = pd.Series(dct['__pandas_series__'])
        if '__pandas_index__' in dct:
            series.index = dct['__pandas_index__']
        return series
    return dct

class FileCache:
    """Simple file-based cache with TTL support"""
    
    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_cache_path(self, key: str) -> Path:
        """Generate cache file path from key"""
        # Create hash of key to handle special characters
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.cache"
    
    def _get_metadata_path(self, key: str) -> Path:
        """Generate metadata file path from key"""
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.meta"
    
    def set(self, key: str, value: Any, ttl: int = 3600) -> None:
        """Store value in cache with TTL"""
        try:
            cache_path = self._get_cache_path(key)
            meta_path = self._get_metadata_path(key)
            
            # Store data using safe JSON serialization
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(value, f, cls=SafeJSONEncoder, ensure_ascii=False, indent=2)
            
            # Store metadata
            metadata = {
                'key': key,
                'timestamp': time.time(),
                'ttl': ttl,
                'expires_at': time.time() + ttl,
                'format': 'json'  # Mark as JSON format for future compatibility
            }
            
            with open(meta_path, 'w') as f:
                json.dump(metadata, f)
                
            logger.debug(f"Cached data for key: {key}")
            
        except (TypeError, ValueError) as e:
            logger.warning(f"Data not JSON serializable for key {key}: {e}. Skipping cache.")
        except Exception as e:
            logger.error(f"Failed to cache data for key {key}: {e}")
    
    def get(self, key: str) -> Optional[Any]:
        """Retrieve value from cache if not expired"""
        try:
            cache_path = self._get_cache_path(key)
            meta_path = self._get_metadata_path(key)
            
            # Check if files exist
            if not cache_path.exists() or not meta_path.exists():
                return None
            
            # Check metadata and TTL
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
            
            if time.time() > metadata['expires_at']:
                # Cache expired, clean up
                self._cleanup_key(key)
                logger.debug(f"Cache expired for key: {key}")
                return None
            
            # Check format and load accordingly
            cache_format = metadata.get('format', 'pickle')  # Default to pickle for old files
            
            if cache_format == 'json':
                # Load JSON data (secure)
                with open(cache_path, 'r', encoding='utf-8') as f:
                    value = json.load(f, object_hook=safe_json_decode)
            else:
                # Old pickle format - for security, we'll remove these files
                logger.warning(f"Found old pickle cache file for key {key}. Removing for security.")
                self._cleanup_key(key)
                return None
                
            logger.debug(f"Cache hit for key: {key}")
            return value
            
        except Exception as e:
            logger.error(f"Failed to retrieve cached data for key {key}: {e}")
            # Clean up potentially corrupted cache files
            try:
                self._cleanup_key(key)
            except:
                pass
            return None
    
    def _cleanup_key(self, key: str) -> None:
        """Remove cache and metadata files for a key"""
        try:
            cache_path = self._get_cache_path(key)
            meta_path = self._get_metadata_path(key)
            
            cache_path.unlink(missing_ok=True)
            meta_path.unlink(missing_ok=True)
            
        except Exception as e:
            logger.error(f"Failed to cleanup cache for key {key}: {e}")
    
    def clear(self) -> None:
        """Clear all cached data"""
        try:
            for file_path in self.cache_dir.glob("*.cache"):
                file_path.unlink()
            for file_path in self.cache_dir.glob("*.meta"):
                file_path.unlink()
            logger.info("Cache cleared")
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
    
    def cleanup_expired(self) -> int:
        """Remove all expired cache entries, return count of removed items"""
        removed_count = 0
        try:
            for meta_file in self.cache_dir.glob("*.meta"):
                try:
                    with open(meta_file, 'r') as f:
                        metadata = json.load(f)
                    
                    if time.time() > metadata['expires_at']:
                        self._cleanup_key(metadata['key'])
                        removed_count += 1
                        
                except Exception as e:
                    logger.warning(f"Failed to process metadata file {meta_file}: {e}")
                    # Remove corrupted metadata file
                    meta_file.unlink(missing_ok=True)
                    
            if removed_count > 0:
                logger.info(f"Cleaned up {removed_count} expired cache entries")
                
        except Exception as e:
            logger.error(f"Failed to cleanup expired cache: {e}")
            
        return removed_count
    
    def remove_pickle_cache_files(self) -> int:
        """Remove all old pickle cache files for security, return count of removed items"""
        removed_count = 0
        try:
            for meta_file in self.cache_dir.glob("*.meta"):
                try:
                    with open(meta_file, 'r') as f:
                        metadata = json.load(f)
                    
                    # Check if this is an old pickle format cache
                    cache_format = metadata.get('format', 'pickle')  # Default to pickle for old files
                    if cache_format != 'json':
                        key = metadata.get('key', '')
                        logger.info(f"Removing old pickle cache for security: {key}")
                        self._cleanup_key(key)
                        removed_count += 1
                        
                except Exception as e:
                    logger.warning(f"Failed to process metadata file {meta_file}: {e}")
                    # Remove corrupted metadata file
                    try:
                        meta_file.unlink(missing_ok=True)
                    except:
                        pass
                    
            if removed_count > 0:
                logger.info(f"Removed {removed_count} old pickle cache files for security")
                
        except Exception as e:
            logger.error(f"Failed to remove pickle cache files: {e}")
            
        return removed_count
