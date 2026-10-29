import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[3]
FIXTURES = ROOT / "fixtures"


def wire_fixtures():
    return sorted((FIXTURES / "wire").glob("*/*.json"))


def type_fixtures():
    return sorted((FIXTURES / "types").glob("*.json"))


def load(path):
    return json.loads(path.read_text())


@pytest.fixture
def fixtures_dir():
    return FIXTURES
