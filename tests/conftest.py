import pytest

from matchsense.sim import load_default_teams, simulate

SEED = 7


@pytest.fixture(scope="session")
def match():
    """One full 90-minute match shared by every test module."""
    home, away = load_default_teams()
    return simulate(home, away, seed=SEED)
