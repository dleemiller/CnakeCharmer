# connectors/api_client.py
import asyncio
import logging
import time
from typing import Dict, Optional, Any

import aiohttp

from core.models import ApiResponse
from core.exceptions import ApiConnectionError, ApiResponseError, RateLimitError


class CnakeCharmerClient:
    """Client for interacting with the CnakeCharmer API."""
    
    def __init__(self, base_url: str, api_key: Optional[str] = None, 
                 timeout: int = 30, max_retries: int = 3, retry_delay: int = 5):
        """
        Initialize API client.
        
        Args:
            base_url: Base URL of the API
            api_key: API key for authentication
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries
            retry_delay: Delay between retries in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.logger = logging.getLogger(__name__)
        
        # Use session for connection pooling
        self.session = None
    
    async def _ensure_session(self):
        """Ensure aiohttp session is created."""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(
                headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
            )
    
    async def close(self):
        """Close the aiohttp session."""
        if self.session and not self.session.closed:
            await self.session.close()
    
    async def _make_request(self, method: str, endpoint: str, 
                           data: Optional[Dict[str, Any]] = None, 
                           params: Optional[Dict[str, Any]] = None) -> ApiResponse:
        """
        Make a request to the API with retry logic.
        
        Args:
            method: HTTP method
            endpoint: API endpoint
            data: Request payload
            params: Query parameters
            
        Returns:
            ApiResponse object
        """
        await self._ensure_session()
        
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        retries = 0
        
        while retries <= self.max_retries:
            try:
                self.logger.debug(f"Making {method} request to {url}")
                
                async with self.session.request(
                    method=method,
                    url=url,
                    json=data,
                    params=params,
                    timeout=self.timeout
                ) as response:
                    response_json = await response.json()
                    
                    if response.status == 429:
                        retry_after = int(response.headers.get('Retry-After', self.retry_delay))
                        self.logger.warning(f"Rate limited, retrying after {retry_after} seconds")
                        await asyncio.sleep(retry_after)
                        retries += 1
                        continue
                    
                    if 200 <= response.status < 300:
                        return ApiResponse(
                            success=True,
                            request_id=response_json.get('request_id'),
                            data=response_json
                        )
                    else:
                        error_message = response_json.get('error', f"API returned status code {response.status}")
                        return ApiResponse(
                            success=False,
                            error=error_message
                        )
            
            except asyncio.TimeoutError:
                self.logger.warning(f"Request timed out, retry {retries + 1}/{self.max_retries}")
            except aiohttp.ClientError as e:
                self.logger.warning(f"Request failed: {e}, retry {retries + 1}/{self.max_retries}")
            
            retries += 1
            await asyncio.sleep(self.retry_delay)
        
        raise ApiConnectionError(f"Failed to connect to API after {self.max_retries} retries")
    
    async def generate_python_from_cython(self, cython_code: str, 
                                          metadata: Optional[Dict[str, Any]] = None) -> ApiResponse:
        """
        Request Python code generation from Cython.
        
        Args:
            cython_code: Cython code to convert
            metadata: Additional metadata
            
        Returns:
            ApiResponse object
        """
        endpoint = "generate"
        data = {
            "source_language": "cython",
            "target_languages": ["python"],
            "source_code": cython_code,
            "metadata": metadata or {}
        }
        
        return await self._make_request("POST", endpoint, data)
    
    async def get_generation_status(self, request_id: str) -> ApiResponse:
        """
        Check status of a code generation request.
        
        Args:
            request_id: ID of the request
            
        Returns:
            ApiResponse object
        """
        endpoint = f"status/{request_id}"
        return await self._make_request("GET", endpoint)
    
    async def get_generation_result(self, request_id: str) -> ApiResponse:
        """
        Get the result of a completed generation request.
        
        Args:
            request_id: ID of the request
            
        Returns:
            ApiResponse object
        """
        endpoint = f"results/{request_id}"
        return await self._make_request("GET", endpoint)