# utils/rate_limiter.py
import asyncio
import time
from typing import Optional


class RateLimiter:
    """Rate limiter for API requests."""
    
    def __init__(self, requests_per_minute: int):
        """
        Initialize rate limiter.
        
        Args:
            requests_per_minute: Maximum requests per minute
        """
        self.rate = requests_per_minute
        self.remaining = requests_per_minute
        self.window_start = time.time()
        self.lock = asyncio.Lock()
    
    async def acquire(self, timeout: Optional[float] = None) -> bool:
        """
        Acquire a token respecting the rate limit.
        
        Args:
            timeout: Maximum time to wait for a token
            
        Returns:
            True if a token was acquired, False if timed out
        """
        start_time = time.time()
        
        while timeout is None or time.time() - start_time < timeout:
            async with self.lock:
                current_time = time.time()
                # If a minute has passed, reset the counter
                if current_time - self.window_start >= 60:
                    self.remaining = self.rate
                    self.window_start = current_time
                
                # If we have remaining tokens, use one
                if self.remaining > 0:
                    self.remaining -= 1
                    return True
                
                # Calculate time until next token is available
                wait_time = 60 - (current_time - self.window_start)
            
            # Wait until next token might be available
            if timeout is not None and time.time() + wait_time - start_time > timeout:
                return False
            
            await asyncio.sleep(wait_time)
        
        return False