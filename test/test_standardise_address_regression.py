import os

import pandas as pd

from tools.standardise import (
    extract_letter_one_number_address,
    remove_flat_one_number_address,
    remove_non_housing,
    replace_floor_flat,
    standardise_address,
)


def _set_backend(backend: str):
    previous = os.environ.get("STANDARDISE_BACKEND")
    os.environ["STANDARDISE_BACKEND"] = backend
    return previous


def _restore_backend(previous: str | None) -> None:
    if previous is None:
        os.environ.pop("STANDARDISE_BACKEND", None)
    else:
        os.environ["STANDARDISE_BACKEND"] = previous


def test_floor_and_flat_helpers_regression_outputs() -> None:
    df = pd.DataFrame(
        {
            "addr": [
                "flat 12 oak road",
                "room 7 alpha house",
                "2b sycamore road",
                "flat ground floor 10 high street",
                "garage 12 any road",
            ]
        }
    )

    one_number = remove_flat_one_number_address(df.copy(), "addr").tolist()
    assert one_number == [
        " 12 oak road",
        " 7 alpha house",
        "2b sycamore road",
        " ground floor 10 high street",
        "garage 12 any road",
    ]

    letter_after_number = extract_letter_one_number_address(df.copy(), "addr").tolist()
    assert letter_after_number == [
        "flat 12 oak road",
        "room 7 alpha house",
        "flat b 2  sycamore road",
        "flat ground floor 10 high street",
        "garage 12 any road",
    ]

    floor_replaced = replace_floor_flat(df.copy(), "addr").tolist()
    assert floor_replaced == [
        "flat 12 oak road",
        "room 7 alpha house",
        "flat b 2  sycamore road",
        "flat a   10 high street",
        "garage 12 any road",
    ]


def test_remove_non_housing_regression_output() -> None:
    df = pd.DataFrame(
        {"addr": ["flat 12 oak road", "garage 1 lane", "visitor bay 9", "room 7 alpha"]}
    )
    filtered = remove_non_housing(df, "addr")
    assert filtered["addr"].tolist() == ["flat 12 oak road", "room 7 alpha"]


def test_standardise_address_regression_outputs() -> None:
    df = pd.DataFrame(
        {
            "address": [
                "Flat 2B Sycamore Road SE5 4HB",
                "Flat ground floor 10 High Street London SE1 1AA",
                "Room 7 Alpha House SW1A 1AA",
            ]
        }
    )

    out = standardise_address(df, "address", "std", standardise=True, out_london=False)

    assert out["std"].tolist() == [
        "flat 2b sycamore road",
        "flat a   10 high street",
        "flat 7 alpha house",
    ]
    assert pd.isna(out.loc[0, "property_number"])
    assert out.loc[1, "property_number"] == "10"
    assert pd.isna(out.loc[2, "property_number"])
    assert out["flat_number"].tolist() == ["2b", "a", "7"]
    assert out["room_number"].isna().all()
    assert pd.isna(out.loc[0, "house_court_name"])
    assert pd.isna(out.loc[1, "house_court_name"])
    assert out.loc[2, "house_court_name"] == "alpha"


def test_standardise_address_polars_parity_regression() -> None:
    df = pd.DataFrame(
        {
            "address": [
                "Flat 2B Sycamore Road SE5 4HB",
                "Flat ground floor 10 High Street London SE1 1AA",
                "Room 7 Alpha House SW1A 1AA",
                "Apartment 12, 5 First Floor Place SW2 1AA",
                "Basement Flat 3 River Court SE10 1AA",
            ]
        }
    )
    previous = _set_backend("pandas")
    try:
        pandas_out = standardise_address(
            df.copy(), "address", "std", standardise=True, out_london=False
        )
    finally:
        _restore_backend(previous)

    previous = _set_backend("polars")
    try:
        polars_out = standardise_address(
            df.copy(), "address", "std", standardise=True, out_london=False
        )
    finally:
        _restore_backend(previous)

    cols_to_compare = [
        "std",
        "property_number",
        "flat_number",
        "room_number",
        "house_court_name",
        "block_number",
        "unit_number",
    ]
    for col in cols_to_compare:
        pd.testing.assert_series_equal(
            pandas_out[col].reset_index(drop=True),
            polars_out[col].reset_index(drop=True),
            check_names=False,
            check_dtype=False,
        )


def test_extract_flat_and_other_no_polars_schema_parity() -> None:
    df = pd.DataFrame(
        {
            "addr": [
                "flat 2b sycamore road",
                "flat a 10 high street",
                "flat 7 alpha house",
                "apartment 3 river road",
            ]
        }
    )
    previous = _set_backend("pandas")
    try:
        pandas_out = standardise_address(
            df.copy(), "addr", "std", standardise=False, out_london=True
        )
    finally:
        _restore_backend(previous)

    previous = _set_backend("polars")
    try:
        polars_out = standardise_address(
            df.copy(), "addr", "std", standardise=False, out_london=True
        )
    finally:
        _restore_backend(previous)

    expected_cols = [
        "prop_number",
        "flat_number",
        "apart_number",
        "first_sec_number",
        "first_letter_flat_number",
        "first_letter_no_more_numbers",
    ]
    for col in expected_cols:
        assert col in polars_out.columns
        pd.testing.assert_series_equal(
            pandas_out[col].reset_index(drop=True),
            polars_out[col].reset_index(drop=True),
            check_names=False,
            check_dtype=False,
        )
