import json
import pickle
import time
from pathlib import Path
from typing import Any, Optional
import hashlib
import logging

logger = logging.getLogger(__name__)

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
            
            # Store data
            with open(cache_path, 'wb') as f:
                pickle.dump(value, f)
            
            # Store metadata
            metadata = {
                'key': key,
                'timestamp': time.time(),
                'ttl': ttl,
                'expires_at': time.time() + ttl
            }
            
            with open(meta_path, 'w') as f:
                json.dump(metadata, f)
                
            logger.debug(f"Cached data for key: {key}")
            
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
            
            # Load and return data
            with open(cache_path, 'rb') as f:
                value = pickle.load(f)
                
            logger.debug(f"Cache hit for key: {key}")
            return value
            
        except Exception as e:
            logger.error(f"Failed to retrieve cached data for key {key}: {e}")
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