import pytest


@pytest.fixture(scope="module")
def use_cuda():
    return False
