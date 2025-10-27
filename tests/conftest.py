import pytest
import json
from pathlib import Path

import sys
import os
# This will let us import files and modules located in the parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config 

@pytest.fixture(autouse=True)
def override_default_config(monkeypatch):
    """
    This fixture automatically runs for every test.
    It uses monkeypatch to change the "DEFAULT_CONFIG_PATH" variable
    """
    monkeypatch.setattr(
        config,
        "DEFAULT_CONFIG_PATH",
        Path(__file__).parent / 'configuration.json' # gives /tests/configuration.json
    )