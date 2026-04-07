import pandas as pd

from tools.standardise import (
    extract_letter_one_number_address,
    remove_flat_one_number_address,
    remove_non_housing,
    replace_floor_flat,
    standardise_address,
)


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
