from __future__ import annotations

import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_DATA_DIR = REPO_ROOT / "example_data"
SEARCH_CSV = EXAMPLE_DATA_DIR / "search_addresses_london.csv"
REFERENCE_CSV = EXAMPLE_DATA_DIR / "reference_addresses_london.csv"


def _build_search_rows() -> list[dict[str, str]]:
    return [
        {
            "address_line_1": "Flat 1 12 Brixton Road",
            "address_line_2": "London",
            "postcode": "SW9 6BU",
        },
        {
            "address_line_1": "221B Baker Street",
            "address_line_2": "London",
            "postcode": "NW1 6XE",
        },
        {
            "address_line_1": "10 Downing Street",
            "address_line_2": "London",
            "postcode": "SW1A 2AA",
        },
        {
            "address_line_1": "1 Waterloo Road",
            "address_line_2": "London",
            "postcode": "SE1 8TY",
        },
        {
            "address_line_1": "9 Clapham Common North Side",
            "address_line_2": "London",
            "postcode": "SW4 0QW",
        },
        {
            "address_line_1": "3 Coldharbour Lane",
            "address_line_2": "London",
            "postcode": "SW9 8LH",
        },
        {
            "address_line_1": "17 Acre Lane",
            "address_line_2": "London",
            "postcode": "SW2 5SP",
        },
        {
            "address_line_1": "65 Camberwell Church Street",
            "address_line_2": "London",
            "postcode": "SE5 8TR",
        },
        {
            "address_line_1": "88 Fictional Road",
            "address_line_2": "London",
            "postcode": "SE1 1AA",
        },
        {
            "address_line_1": "19 Imaginary Parade",
            "address_line_2": "London",
            "postcode": "SW8 9ZZ",
        },
    ]


def _build_reference_rows() -> list[dict[str, str]]:
    return [
        {
            "addr1": "Flat 1",
            "addr2": "12 Brixton Road",
            "addr3": "London",
            "addr4": "",
            "postcode": "SW9 6BU",
        },
        {
            "addr1": "",
            "addr2": "221B Baker Street",
            "addr3": "London",
            "addr4": "",
            "postcode": "NW1 6XE",
        },
        {
            "addr1": "",
            "addr2": "10 Downing Street",
            "addr3": "London",
            "addr4": "City of Westminster",
            "postcode": "SW1A 2AA",
        },
        {
            "addr1": "",
            "addr2": "1 Waterloo Road",
            "addr3": "London",
            "addr4": "Lambeth",
            "postcode": "SE1 8TY",
        },
        {
            "addr1": "",
            "addr2": "9 Clapham Common North Side",
            "addr3": "London",
            "addr4": "Clapham",
            "postcode": "SW4 0QW",
        },
        {
            "addr1": "",
            "addr2": "3 Coldharbour Lane",
            "addr3": "London",
            "addr4": "Brixton",
            "postcode": "SW9 8LH",
        },
        {
            "addr1": "",
            "addr2": "17 Acre Lane",
            "addr3": "London",
            "addr4": "Brixton",
            "postcode": "SW2 5SP",
        },
        {
            "addr1": "",
            "addr2": "65 Camberwell Church Street",
            "addr3": "London",
            "addr4": "Camberwell",
            "postcode": "SE5 8TR",
        },
        {
            "addr1": "",
            "addr2": "5 Vauxhall Bridge Road",
            "addr3": "London",
            "addr4": "",
            "postcode": "SW1V 2RE",
        },
        {
            "addr1": "",
            "addr2": "24 Kennington Road",
            "addr3": "London",
            "addr4": "",
            "postcode": "SE1 7BL",
        },
        {
            "addr1": "",
            "addr2": "41 Atlantic Road",
            "addr3": "London",
            "addr4": "Brixton",
            "postcode": "SW9 8JL",
        },
        {
            "addr1": "",
            "addr2": "3 Electric Avenue",
            "addr3": "London",
            "addr4": "Brixton",
            "postcode": "SW9 8JX",
        },
        {
            "addr1": "",
            "addr2": "48 Denmark Hill",
            "addr3": "London",
            "addr4": "Camberwell",
            "postcode": "SE5 8RS",
        },
        {
            "addr1": "",
            "addr2": "30 Streatham High Road",
            "addr3": "London",
            "addr4": "Streatham",
            "postcode": "SW16 1DB",
        },
        {
            "addr1": "",
            "addr2": "52 Tulse Hill",
            "addr3": "London",
            "addr4": "",
            "postcode": "SW2 2QA",
        },
        {
            "addr1": "",
            "addr2": "120 Norwood Road",
            "addr3": "London",
            "addr4": "Herne Hill",
            "postcode": "SE24 9AF",
        },
        {
            "addr1": "",
            "addr2": "44 Landor Road",
            "addr3": "London",
            "addr4": "Stockwell",
            "postcode": "SW9 9PH",
        },
        {
            "addr1": "",
            "addr2": "18 Brixton Hill",
            "addr3": "London",
            "addr4": "",
            "postcode": "SW2 1QA",
        },
        {
            "addr1": "",
            "addr2": "101 Clapham Park Road",
            "addr3": "London",
            "addr4": "Clapham",
            "postcode": "SW4 7EW",
        },
        {
            "addr1": "",
            "addr2": "9 Herne Hill",
            "addr3": "London",
            "addr4": "",
            "postcode": "SE24 9NE",
        },
    ]


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    EXAMPLE_DATA_DIR.mkdir(parents=True, exist_ok=True)

    search_rows = _build_search_rows()
    reference_rows = _build_reference_rows()

    _write_csv(SEARCH_CSV, search_rows)
    _write_csv(REFERENCE_CSV, reference_rows)

    print(f"Wrote search fixture: {SEARCH_CSV}")
    print(f"Wrote reference fixture: {REFERENCE_CSV}")
    print(f"search rows={len(search_rows)}, reference rows={len(reference_rows)}")


if __name__ == "__main__":
    main()
