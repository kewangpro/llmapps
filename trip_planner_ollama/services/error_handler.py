import logging
import asyncio
import functools
from typing import Any, Callable, Optional, Type, Union, Dict
from datetime import datetime, timedelta
import traceback
from config import get_config

logger = logging.getLogger(__name__)

class APIError(Exception):
    """Base exception for API-related errors."""
    def __init__(self, message: str, error_code: Optional[str] = None, status_code: Optional[int] = None):
        super().__init__(message)
        self.error_code = error_code
        self.status_code = status_code
        self.timestamp = datetime.utcnow()

class GoogleSearchError(APIError):
    """Google Search API specific errors."""
    pass

class RouteOptimizationError(Exception):
    """Route optimization specific errors."""
    pass

class RateLimitError(APIError):
    """Rate limiting specific errors."""
    def __init__(self, message: str, retry_after: Optional[int] = None):
        super().__init__(message, error_code="RATE_LIMIT_EXCEEDED")
        self.retry_after = retry_after or 60

class ConfigurationError(Exception):
    """Configuration related errors."""
    pass

class CircuitBreaker:
    """Circuit breaker pattern implementation for API calls."""
    
    def __init__(self, 
                 failure_threshold: int = 5, 
                 reset_timeout: int = 60,
                 expected_exception: Type[Exception] = Exception):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        if self.state == 'OPEN':
            if self._should_attempt_reset():
                self.state = 'HALF_OPEN'
            else:
                raise APIError("Circuit breaker is OPEN - service temporarily unavailable")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        
        except self.expected_exception as e:
            self._on_failure()
            raise
    
    async def call_async(self, func: Callable, *args, **kwargs) -> Any:
        """Execute async function with circuit breaker protection."""
        if self.state == 'OPEN':
            if self._should_attempt_reset():
                self.state = 'HALF_OPEN'
            else:
                raise APIError("Circuit breaker is OPEN - service temporarily unavailable")
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        
        except self.expected_exception as e:
            self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt to reset."""
        return (
            self.last_failure_time and
            datetime.utcnow() - self.last_failure_time >= timedelta(seconds=self.reset_timeout)
        )
    
    def _on_success(self):
        """Handle successful call."""
        self.failure_count = 0
        self.state = 'CLOSED'
    
    def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()
        
        if self.failure_count >= self.failure_threshold:
            self.state = 'OPEN'

class RetryManager:
    """Manages retry logic with exponential backoff."""
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 60.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
    
    async def retry_async(self, 
                         func: Callable, 
                         *args, 
                         exceptions: tuple = (Exception,),
                         **kwargs) -> Any:
        """Retry an async function with exponential backoff."""
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                if attempt > 0:
                    delay = min(self.base_delay * (2 ** (attempt - 1)), self.max_delay)
                    logger.info(f"Retrying after {delay:.2f}s (attempt {attempt + 1}/{self.max_retries + 1})")
                    await asyncio.sleep(delay)
                
                return await func(*args, **kwargs)
            
            except exceptions as e:
                last_exception = e
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                
                if attempt == self.max_retries:
                    logger.error(f"All {self.max_retries + 1} attempts failed")
                    break
        
        raise last_exception
    
    def retry_sync(self, 
                   func: Callable, 
                   *args, 
                   exceptions: tuple = (Exception,),
                   **kwargs) -> Any:
        """Retry a synchronous function with exponential backoff."""
        import time
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                if attempt > 0:
                    delay = min(self.base_delay * (2 ** (attempt - 1)), self.max_delay)
                    logger.info(f"Retrying after {delay:.2f}s (attempt {attempt + 1}/{self.max_retries + 1})")
                    time.sleep(delay)
                
                return func(*args, **kwargs)
            
            except exceptions as e:
                last_exception = e
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                
                if attempt == self.max_retries:
                    logger.error(f"All {self.max_retries + 1} attempts failed")
                    break
        
        raise last_exception

class ErrorHandler:
    """Comprehensive error handling and recovery system."""
    
    def __init__(self):
        self.config = get_config()
        self.circuit_breakers = {}
        self.retry_manager = RetryManager(
            max_retries=self.config.max_retries,
            base_delay=self.config.retry_delay
        )
        self.error_stats = {
            'total_errors': 0,
            'api_errors': 0,
            'circuit_breaker_trips': 0,
            'successful_retries': 0,
            'fallback_activations': 0
        }
    
    def get_circuit_breaker(self, service_name: str) -> CircuitBreaker:
        """Get or create circuit breaker for a service."""
        if service_name not in self.circuit_breakers:
            self.circuit_breakers[service_name] = CircuitBreaker(
                failure_threshold=5,
                reset_timeout=60,
                expected_exception=APIError
            )
        return self.circuit_breakers[service_name]
    
    async def handle_api_call(self, 
                             service_name: str, 
                             func: Callable, 
                             fallback_func: Optional[Callable] = None,
                             *args, **kwargs) -> Any:
        """Handle API call with circuit breaker, retry, and fallback."""
        circuit_breaker = self.get_circuit_breaker(service_name)
        
        try:
            # Try main API call with circuit breaker protection
            result = await circuit_breaker.call_async(
                self.retry_manager.retry_async,
                func,
                *args,
                exceptions=(APIError, ConnectionError, TimeoutError),
                **kwargs
            )
            return result
            
        except Exception as e:
            self.error_stats['total_errors'] += 1
            self.error_stats['api_errors'] += 1
            
            logger.error(f"API call failed for {service_name}: {e}")
            
            # Try fallback if available
            if fallback_func and self.config.enable_fallback:
                try:
                    logger.info(f"Attempting fallback for {service_name}")
                    self.error_stats['fallback_activations'] += 1
                    
                    if asyncio.iscoroutinefunction(fallback_func):
                        return await fallback_func(*args, **kwargs)
                    else:
                        return fallback_func(*args, **kwargs)
                        
                except Exception as fallback_error:
                    logger.error(f"Fallback also failed for {service_name}: {fallback_error}")
                    raise APIError(f"Both primary and fallback failed for {service_name}")
            
            # Re-raise original exception if no fallback
            raise
    
    def handle_configuration_error(self, service_name: str, required_config: list) -> None:
        """Handle configuration validation errors."""
        missing_config = []
        
        for config_key in required_config:
            if not hasattr(self.config, config_key) or not getattr(self.config, config_key):
                missing_config.append(config_key)
        
        if missing_config:
            error_msg = f"Missing configuration for {service_name}: {missing_config}"
            logger.error(error_msg)
            raise ConfigurationError(error_msg)
    
    def validate_request_data(self, data: Dict[str, Any], required_fields: list) -> None:
        """Validate request data and raise appropriate errors."""
        missing_fields = []
        invalid_fields = []
        
        for field in required_fields:
            if field not in data:
                missing_fields.append(field)
            elif data[field] is None or data[field] == '':
                invalid_fields.append(field)
        
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")
        
        if invalid_fields:
            raise ValueError(f"Invalid values for fields: {invalid_fields}")
    
    def log_error_details(self, error: Exception, context: Dict[str, Any] = None) -> None:
        """Log detailed error information."""
        context = context or {}
        
        error_details = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'timestamp': datetime.utcnow().isoformat(),
            'context': context,
            'traceback': traceback.format_exc()
        }
        
        logger.error(f"Error details: {error_details}")
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics."""
        return {
            **self.error_stats,
            'circuit_breaker_states': {
                service: breaker.state 
                for service, breaker in self.circuit_breakers.items()
            }
        }
    
    def reset_error_stats(self) -> None:
        """Reset error statistics."""
        self.error_stats = {
            'total_errors': 0,
            'api_errors': 0,
            'circuit_breaker_trips': 0,
            'successful_retries': 0,
            'fallback_activations': 0
        }

def with_error_handling(service_name: str, fallback_func: Optional[Callable] = None):
    """Decorator for automatic error handling."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            error_handler = ErrorHandler()
            return await error_handler.handle_api_call(
                service_name, func, fallback_func, *args, **kwargs
            )
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            error_handler = ErrorHandler()
            # For sync functions, we'll handle errors more simply
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_handler.log_error_details(e, {'service': service_name})
                if fallback_func:
                    try:
                        return fallback_func(*args, **kwargs)
                    except Exception as fallback_error:
                        error_handler.log_error_details(fallback_error, {'service': f"{service_name}_fallback"})
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator

def handle_rate_limit(retry_after: int = None):
    """Handle rate limiting with appropriate delays."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except RateLimitError as e:
                delay = e.retry_after or retry_after or 60
                logger.warning(f"Rate limit hit, waiting {delay} seconds")
                await asyncio.sleep(delay)
                # Retry once after rate limit delay
                return await func(*args, **kwargs)
        
        return wrapper
    return decorator

# Global error handler instance
global_error_handler = ErrorHandler()

# Convenience functions
async def safe_api_call(service_name: str, 
                       func: Callable, 
                       fallback_func: Optional[Callable] = None,
                       *args, **kwargs) -> Any:
    """Safely execute an API call with comprehensive error handling."""
    return await global_error_handler.handle_api_call(
        service_name, func, fallback_func, *args, **kwargs
    )

def validate_config(service_name: str, required_config: list) -> None:
    """Validate configuration for a service."""
    global_error_handler.handle_configuration_error(service_name, required_config)

def log_error(error: Exception, context: Dict[str, Any] = None) -> None:
    """Log error with detailed information."""
    global_error_handler.log_error_details(error, context)