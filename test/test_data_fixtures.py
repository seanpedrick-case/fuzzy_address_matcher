from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SEARCH_PATH = REPO_ROOT / "example_data" / "search_addresses_london.csv"
REFERENCE_PATH = REPO_ROOT / "example_data" / "reference_addresses_london.csv"


def test_example_fixture_files_exist() -> None:
    assert SEARCH_PATH.exists(), f"Missing fixture: {SEARCH_PATH}"
    assert REFERENCE_PATH.exists(), f"Missing fixture: {REFERENCE_PATH}"


def test_example_fixture_schema_and_sizes() -> None:
    search_df = pd.read_csv(SEARCH_PATH)
    reference_df = pd.read_csv(REFERENCE_PATH)

    assert list(search_df.columns) == ["address_line_1", "address_line_2", "postcode"]
    assert list(reference_df.columns) == [
        "addr1",
        "addr2",
        "addr3",
        "addr4",
        "postcode",
    ]
    assert len(search_df) == 10
    assert len(reference_df) == 20
