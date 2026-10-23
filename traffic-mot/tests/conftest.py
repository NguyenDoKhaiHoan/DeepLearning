"""Real checkpoint tests are explicit, never silently replaced with mocks."""
import pytest


def pytest_addoption(parser):
    parser.addoption("--run-model", action="store_true", help="Load verified local checkpoint and run CPU inference")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-model"):
        for item in items:
            if "real_model" in item.keywords:
                item.add_marker(pytest.mark.skip(reason="Use --run-model for real checkpoint tests"))
