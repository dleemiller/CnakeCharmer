# tests/conftest.py
"""
Test fixtures for the CnakeCharmer system.
"""
import os
import tempfile
import pytest
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

from core.enums import LanguageType
from repositories.postgres import PostgresRepository
from repositories.request_repo import CodeRequestRepository
from repositories.code_repo import GeneratedCodeRepository
from services.code_generator import CodeGeneratorService
from builders.python_builder import PythonBuilder
from builders.cython_builder import CythonBuilder
from analyzers.static_analyzer import StaticCodeAnalyzer
from equivalency.checker import SimpleEquivalencyChecker
from utils.db_init import initialize_database


@pytest.fixture(scope="session")
def test_db():
    """Create a temporary test database."""
    # Create a unique database name
    db_name = f"cnake_test_{os.urandom(4).hex()}"
    
    # Connect to PostgreSQL
    conn = psycopg2.connect(
        host=os.environ.get("TEST_DB_HOST", "localhost"),
        user=os.environ.get("TEST_DB_USER", "postgres"),
        password=os.environ.get("TEST_DB_PASSWORD", "postgres")
    )
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    
    # Create the test database
    with conn.cursor() as cur:
        cur.execute(f"CREATE DATABASE {db_name}")
    
    conn.close()
    
    # Construct the connection URL for the test database
    db_url = f"postgresql://{os.environ.get('TEST_DB_USER', 'postgres')}:{os.environ.get('TEST_DB_PASSWORD', 'postgres')}@{os.environ.get('TEST_DB_HOST', 'localhost')}/{db_name}"
    
    # Initialize the schema
    initialize_database(db_url)
    
    yield db_url
    
    # Clean up - drop the test database
    conn = psycopg2.connect(
        host=os.environ.get("TEST_DB_HOST", "localhost"),
        user=os.environ.get("TEST_DB_USER", "postgres"),
        password=os.environ.get("TEST_DB_PASSWORD", "postgres")
    )
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    
    with conn.cursor() as cur:
        cur.execute(f"DROP DATABASE {db_name}")
    
    conn.close()


@pytest.fixture
def request_repo(test_db):
    """Create a test request repository."""
    repo = CodeRequestRepository(test_db)
    yield repo
    repo.close()


@pytest.fixture
def code_repo(test_db):
    """Create a test code repository."""
    repo = GeneratedCodeRepository(test_db)
    yield repo
    repo.close()


@pytest.fixture
def python_builder():
    """Create a Python builder."""
    return PythonBuilder()


@pytest.fixture
def cython_builder():
    """Create a Cython builder."""
    return CythonBuilder()


@pytest.fixture
def builders(python_builder, cython_builder):
    """Create a dictionary of builders."""
    return {
        LanguageType.PYTHON: python_builder,
        LanguageType.CYTHON: cython_builder
    }


@pytest.fixture
def static_analyzer():
    """Create a static code analyzer."""
    return StaticCodeAnalyzer()


@pytest.fixture
def equivalency_checker(builders):
    """Create an equivalency checker."""
    return SimpleEquivalencyChecker(builders)


@pytest.fixture
def code_generator(request_repo, code_repo):
    """Create a code generator service."""
    return CodeGeneratorService(request_repo, code_repo)