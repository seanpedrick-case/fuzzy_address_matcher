from pathlib import Path

import pandas as pd
import pytest

from tools.matcher_funcs import fuzzy_address_match

REPO_ROOT = Path(__file__).resolve().parents[1]
SEARCH_PATH = REPO_ROOT / "example_data" / "search_addresses_london.csv"
REFERENCE_PATH = REPO_ROOT / "example_data" / "reference_addresses_london.csv"


@pytest.mark.integration
def test_integration_fuzzy_address_match_pipeline() -> None:
    search_df = pd.read_csv(SEARCH_PATH)
    reference_df = pd.read_csv(REFERENCE_PATH)

    msg, output_files, estimate_time, _ = fuzzy_address_match(
        search_df=search_df,
        ref_df=reference_df,
        in_colnames=["address_line_1", "address_line_2", "postcode"],
        in_refcol=["addr1", "addr2", "addr3", "addr4", "postcode"],
    )

    assert isinstance(msg, str)
    assert output_files is not None
    assert len(output_files) >= 1
    assert estimate_time >= 0.0
