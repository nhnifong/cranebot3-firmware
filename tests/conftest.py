import pytest
import json
from pathlib import Path

import nf_robot.common.config_loader 

@pytest.fixture(autouse=True)
def override_default_config(monkeypatch):
    """
    This fixture automatically runs for every test.
    It uses monkeypatch to change the "DEFAULT_CONFIG_PATH" variable
    """
    monkeypatch.setattr(
        nf_robot.common.config_loader,
        "DEFAULT_CONFIG_PATH",
        Path(__file__).parent / 'configuration.json' # gives /tests/configuration.json
    )