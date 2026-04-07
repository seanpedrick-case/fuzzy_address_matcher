import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

pd = pytest.importorskip("pandas")
pytest.importorskip("gradio")
pytest.importorskip("recordlinkage")
pytest.importorskip("rapidfuzz")

from tools import matcher_funcs

SEARCH_FIXTURE = REPO_ROOT / "example_data" / "search_addresses_london.csv"
REF_FIXTURE = REPO_ROOT / "example_data" / "reference_addresses_london.csv"

SEARCH_COLS = ["address_line_1", "address_line_2", "postcode"]
REF_COLS = ["addr1", "addr2", "addr3", "addr4", "postcode"]


def _assert_matcher_result_shape(result):
    assert isinstance(result, tuple)
    assert len(result) in (3, 4)
    assert isinstance(result[0], str)
    assert isinstance(result[1], list)
    assert isinstance(result[2], (int, float))
    if len(result) == 4:
        assert isinstance(result[3], str)


def test_fixture_files_exist_and_have_expected_shape():
    assert SEARCH_FIXTURE.exists()
    assert REF_FIXTURE.exists()

    search_df = pd.read_csv(SEARCH_FIXTURE)
    ref_df = pd.read_csv(REF_FIXTURE)

    assert list(search_df.columns) == SEARCH_COLS
    assert list(ref_df.columns) == REF_COLS
    assert len(search_df) == 10
    assert len(ref_df) == 20


def test_fuzzy_address_match_with_file_paths():
    result = matcher_funcs.fuzzy_address_match(
        in_file=str(SEARCH_FIXTURE),
        in_ref=str(REF_FIXTURE),
        in_colnames=SEARCH_COLS,
        in_refcol=REF_COLS,
        run_batches_in_parallel=False,
    )

    _assert_matcher_result_shape(result)


def test_fuzzy_address_match_with_dataframes():
    search_df = pd.read_csv(SEARCH_FIXTURE)
    ref_df = pd.read_csv(REF_FIXTURE).rename(columns={"postcode": "Postcode"})

    result = matcher_funcs.fuzzy_address_match(
        search_df=search_df,
        ref_df=ref_df,
        in_colnames=SEARCH_COLS,
        in_refcol=["addr1", "addr2", "addr3", "addr4", "Postcode"],
        run_batches_in_parallel=False,
    )

    _assert_matcher_result_shape(result)
