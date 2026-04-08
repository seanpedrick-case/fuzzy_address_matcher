import os
from pathlib import Path

import pandas as pd

from fuzzy_address_matcher.preparation import (
    check_no_number_addresses,
    prepare_ref_address,
    prepare_search_address,
    remove_non_postal,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SEARCH_PATH = REPO_ROOT / "example_data" / "search_addresses_london.csv"
REFERENCE_PATH = REPO_ROOT / "example_data" / "reference_addresses_london.csv"


def _set_backend(backend: str):
    previous = os.environ.get("PREPARATION_BACKEND")
    os.environ["PREPARATION_BACKEND"] = backend
    return previous


def _restore_backend(previous: str | None) -> None:
    if previous is None:
        os.environ.pop("PREPARATION_BACKEND", None)
    else:
        os.environ["PREPARATION_BACKEND"] = previous


def _noop_progress(*_args, **_kwargs):
    return None


def test_prepare_search_address_parity() -> None:
    search_df = pd.read_csv(SEARCH_PATH)
    address_cols = ["address_line_1", "address_line_2", "postcode"]
    postcode_col = ["postcode"]
    key_col = "index"

    previous = _set_backend("pandas")
    try:
        pandas_out = prepare_search_address(
            search_df.copy(),
            address_cols,
            postcode_col,
            key_col,
            progress=_noop_progress,
        )
    finally:
        _restore_backend(previous)

    previous = _set_backend("polars")
    try:
        polars_out = prepare_search_address(
            search_df.copy(),
            address_cols,
            postcode_col,
            key_col,
            progress=_noop_progress,
        )
    finally:
        _restore_backend(previous)

    for col in ["index", "full_address", "postcode"]:
        pd.testing.assert_series_equal(
            pandas_out[col].reset_index(drop=True),
            polars_out[col].reset_index(drop=True),
            check_names=False,
            check_dtype=False,
        )


def test_non_postal_and_no_number_parity() -> None:
    df = pd.DataFrame(
        {
            "full_address": [
                "flat 1 alpha road",
                "garage 2 alpha road",
                "visitor bay alpha",
                "room seven alpha road",
            ]
        }
    )

    previous = _set_backend("pandas")
    try:
        pandas_out = check_no_number_addresses(
            remove_non_postal(df.copy(), "full_address"), "full_address"
        )
    finally:
        _restore_backend(previous)

    previous = _set_backend("polars")
    try:
        polars_out = check_no_number_addresses(
            remove_non_postal(df.copy(), "full_address"), "full_address"
        )
    finally:
        _restore_backend(previous)

    pd.testing.assert_series_equal(
        pandas_out["Excluded from search"].reset_index(drop=True),
        polars_out["Excluded from search"].reset_index(drop=True),
        check_names=False,
        check_dtype=False,
    )


def test_prepare_ref_address_parity() -> None:
    ref_df = pd.read_csv(REFERENCE_PATH).rename(columns={"postcode": "Postcode"})
    ref_cols = ["addr1", "addr2", "addr3", "addr4", "Postcode"]

    previous = _set_backend("pandas")
    try:
        pandas_out = prepare_ref_address(
            ref_df.copy(),
            ref_cols,
            new_join_col=[],
            progress=_noop_progress,
        )
    finally:
        _restore_backend(previous)

    previous = _set_backend("polars")
    try:
        polars_out = prepare_ref_address(
            ref_df.copy(),
            ref_cols,
            new_join_col=[],
            progress=_noop_progress,
        )
    finally:
        _restore_backend(previous)

    for col in ["fulladdress", "Postcode", "Street", "ref_index"]:
        pd.testing.assert_series_equal(
            pandas_out[col].reset_index(drop=True),
            polars_out[col].reset_index(drop=True),
            check_names=False,
            check_dtype=False,
        )
