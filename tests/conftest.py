"""Pytest configuration and fixtures."""

import os
import pytest
from unittest.mock import MagicMock
from dotenv import load_dotenv
from app.models import WorkflowOptions


# Load test environment variables BEFORE any imports
test_env_path = os.path.join(os.path.dirname(__file__), ".env.test")
if os.path.exists(test_env_path):
    load_dotenv(test_env_path, override=True)


@pytest.fixture
def mock_logger():
    """Create a mock logger for testing."""
    logger = MagicMock()
    logger.info = MagicMock()
    logger.debug = MagicMock()
    logger.warning = MagicMock()
    logger.error = MagicMock()
    return logger


@pytest.fixture
def workflow_options():
    """Create WorkflowOptions with default settings for testing."""
    return WorkflowOptions(
        max_retrieval_iterations=3,
        enable_query_rewriting=True,
        enable_reflection=True,
    )
