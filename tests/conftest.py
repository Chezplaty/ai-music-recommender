from pathlib import Path

import pytest

from recommender import load_catalog

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@pytest.fixture(scope="session")
def catalog():
    return load_catalog(str(DATA_DIR / "catalog.csv"))
