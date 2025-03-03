# utils/config.py
import os
import json
from typing import Dict, Any, Optional


class Config:
    """Configuration manager."""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_file: Path to configuration file (optional)
        """
        # Default configuration
        self.config = {
            "api": {
                "host": "0.0.0.0",
                "port": 8000,
                "workers": 4,
                "timeout": 60
            },
            "database": {
                "url": "postgresql://user:password@localhost/cnake_charmer"
            },
            "celery": {
                "broker_url": "redis://localhost:6379/0",
                "result_backend": "redis://localhost:6379/0",
                "task_serializer": "json",
                "accept_content": ["json"],
                "result_serializer": "json",
                "enable_utc": True
            },
            "logging": {
                "level": "INFO",
                "file": None
            },
            "security": {
                "api_key_header": "X-API-Key",
                "api_keys": []
            }
        }
        
        # Load from file if provided
        if config_file and os.path.exists(config_file):
            self._load_from_file(config_file)
        
        # Override with environment variables
        self._load_from_env()
    
    def _load_from_file(self, config_file: str):
        """
        Load configuration from file.
        
        Args:
            config_file: Path to configuration file
        """
        try:
            with open(config_file, 'r') as f:
                file_config = json.load(f)
                self._deep_update(self.config, file_config)
        except Exception as e:
            print(f"Error loading config file: {e}")
    
    def _load_from_env(self):
        """Load configuration from environment variables."""
        # API configuration
        if os.environ.get("API_HOST"):
            self.config["api"]["host"] = os.environ.get("API_HOST")
        
        if os.environ.get("API_PORT"):
            self.config["api"]["port"] = int(os.environ.get("API_PORT"))
        
        if os.environ.get("API_WORKERS"):
            self.config["api"]["workers"] = int(os.environ.get("API_WORKERS"))
        
        if os.environ.get("API_TIMEOUT"):
            self.config["api"]["timeout"] = int(os.environ.get("API_TIMEOUT"))
        
        # Database configuration
        if os.environ.get("DATABASE_URL"):
            self.config["database"]["url"] = os.environ.get("DATABASE_URL")
        
        # Celery configuration
        if os.environ.get("CELERY_BROKER_URL"):
            self.config["celery"]["broker_url"] = os.environ.get("CELERY_BROKER_URL")
        
        if os.environ.get("CELERY_RESULT_BACKEND"):
            self.config["celery"]["result_backend"] = os.environ.get("CELERY_RESULT_BACKEND")
        
        # Logging configuration
        if os.environ.get("LOG_LEVEL"):
            self.config["logging"]["level"] = os.environ.get("LOG_LEVEL")
        
        if os.environ.get("LOG_FILE"):
            self.config["logging"]["file"] = os.environ.get("LOG_FILE")
        
        # Security configuration
        if os.environ.get("API_KEY_HEADER"):
            self.config["security"]["api_key_header"] = os.environ.get("API_KEY_HEADER")
        
        if os.environ.get("API_KEYS"):
            self.config["security"]["api_keys"] = os.environ.get("API_KEYS").split(",")
    
    def _deep_update(self, d: Dict[str, Any], u: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively update a dictionary with values from another dictionary.
        
        Args:
            d: Dictionary to update
            u: Dictionary with updates
            
        Returns:
            Updated dictionary
        """
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                d[k] = self._deep_update(d[k], v)
            else:
                d[k] = v
        return d
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            key: Configuration key (dot-separated)
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key.split(".")
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value