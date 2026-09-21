import pytest


def pytest_addoption(parser):
    parser.addoption("--run-slow", action="store_true", default=False, help="run tests that load the real Laya model")


def pytest_collection_modifyitems(config, items):
    # Skip (visible, with reason) rather than deselect, so running the file directly explains itself.
    if config.getoption("--run-slow"):
        return
    skip = pytest.mark.skip(reason="loads the real Laya model; pass --run-slow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)
