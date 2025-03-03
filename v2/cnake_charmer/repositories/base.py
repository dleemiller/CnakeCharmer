# repositories/base.py
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, TypeVar, Generic

T = TypeVar('T')


class Repository(Generic[T], ABC):
    """Abstract base class for repositories."""
    
    @abstractmethod
    def create(self, entity: T) -> str:
        """Create a new entity and return its ID."""
        pass
    
    @abstractmethod
    def get_by_id(self, id: str) -> Optional[T]:
        """Get an entity by its ID."""
        pass
    
    @abstractmethod
    def update(self, entity: T) -> bool:
        """Update an existing entity."""
        pass
    
    @abstractmethod
    def delete(self, id: str) -> bool:
        """Delete an entity by its ID."""
        pass