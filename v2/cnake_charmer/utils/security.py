# utils/security.py
import secrets
import hmac
import hashlib
import logging
from typing import Dict, List, Optional, Any


class SecurityManager:
    """Manager for security-related functionality."""
    
    def __init__(self, api_keys: List[str] = None):
        """
        Initialize security manager.
        
        Args:
            api_keys: List of valid API keys
        """
        self.api_keys = set(api_keys or [])
        self.logger = logging.getLogger(__name__)
    
    def validate_api_key(self, api_key: str) -> bool:
        """
        Validate an API key.
        
        Args:
            api_key: API key to validate
            
        Returns:
            True if the API key is valid, False otherwise
        """
        return not self.api_keys or api_key in self.api_keys
    
    def generate_api_key(self) -> str:
        """
        Generate a new API key.
        
        Returns:
            New API key
        """
        api_key = secrets.token_hex(32)
        self.api_keys.add(api_key)
        return api_key
    
    def revoke_api_key(self, api_key: str) -> bool:
        """
        Revoke an API key.
        
        Args:
            api_key: API key to revoke
            
        Returns:
            True if the API key was revoked, False if it didn't exist
        """
        if api_key in self.api_keys:
            self.api_keys.remove(api_key)
            return True
        return False
    
    def hash_password(self, password: str) -> str:
        """
        Hash a password.
        
        Args:
            password: Password to hash
            
        Returns:
            Hashed password
        """
        salt = secrets.token_hex(16)
        hash_obj = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
        hash_str = hash_obj.hex()
        return f"{salt}${hash_str}"
    
    def verify_password(self, password: str, hashed_password: str) -> bool:
        """
        Verify a password against a hash.
        
        Args:
            password: Password to verify
            hashed_password: Hashed password
            
        Returns:
            True if the password matches the hash, False otherwise
        """
        salt, hash_str = hashed_password.split('$')
        hash_obj = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
        return hash_obj.hex() == hash_str