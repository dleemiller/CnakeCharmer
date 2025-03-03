# core/config.py
import json
import os
from dataclasses import dataclass
from typing import Dict, Any, Optional

from core.exceptions import ConfigurationError


@dataclass
class ApiConfig:
    base_url: str
    api_key: Optional[str] = None
    timeout: int = 30
    max_retries: int = 3
    retry_delay: int = 5


@dataclass
class DuckDBConfig:
    db_path: str
    table_name: str = "cython_entries"
    result_table: str = "python_entries"
    batch_table: str = "batch_jobs"
    error_table: str = "processing_errors"


@dataclass
class ProcessingConfig:
    batch_size: int = 50
    max_concurrent: int = 5
    requests_per_minute: int = 60


@dataclass
class LoggingConfig:
    level: str = "INFO"
    file: Optional[str] = None


@dataclass
class AppConfig:
    api: ApiConfig
    duckdb: DuckDBConfig
    processing: ProcessingConfig
    logging: LoggingConfig


def load_config(config_path: Optional[str] = None) -> AppConfig:
    """
    Load application configuration from file, environment variables, or defaults.
    
    Args:
        config_path: Path to config file (optional)
        
    Returns:
        AppConfig instance
    """
    # Default configuration
    config_data = {
        "api": {
            "base_url": "http://localhost:8000/api",
            "api_key": None,
            "timeout": 30,
            "max_retries": 3,
            "retry_delay": 5
        },
        "duckdb": {
            "db_path": "./cython_code.duckdb",
            "table_name": "cython_entries",
            "result_table": "python_entries",
            "batch_table": "batch_jobs",
            "error_table": "processing_errors"
        },
        "processing": {
            "batch_size": 50,
            "max_concurrent": 5,
            "requests_per_minute": 60
        },
        "logging": {
            "level": "INFO",
            "file": None
        }
    }
    
    # Load from file if provided
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                file_config = json.load(f)
                deep_update(config_data, file_config)
        except Exception as e:
            raise ConfigurationError(f"Error loading config file: {e}")
    
    # Override with environment variables
    if os.environ.get('CNAKE_API_URL'):
        config_data['api']['base_url'] = os.environ.get('CNAKE_API_URL')
    
    if os.environ.get('CNAKE_API_KEY'):
        config_data['api']['api_key'] = os.environ.get('CNAKE_API_KEY')
    
    if os.environ.get('CNAKE_DUCKDB_PATH'):
        config_data['duckdb']['db_path'] = os.environ.get('CNAKE_DUCKDB_PATH')
    
    if os.environ.get('CNAKE_BATCH_SIZE'):
        config_data['processing']['batch_size'] = int(os.environ.get('CNAKE_BATCH_SIZE'))
    
    # Create config objects
    api_config = ApiConfig(**config_data['api'])
    duckdb_config = DuckDBConfig(**config_data['duckdb'])
    processing_config = ProcessingConfig(**config_data['processing'])
    logging_config = LoggingConfig(**config_data['logging'])
    
    return AppConfig(
        api=api_config,
        duckdb=duckdb_config,
        processing=processing_config,
        logging=logging_config
    )


def deep_update(d: Dict[str, Any], u: Dict[str, Any]) -> Dict[str, Any]:
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
            d[k] = deep_update(d[k], v)
        else:
            d[k] = v
    return d  