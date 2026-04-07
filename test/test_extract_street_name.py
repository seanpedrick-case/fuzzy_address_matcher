from tools.preparation import extract_street_name


def test_extract_street_name():
    assert extract_street_name("1 Ash Park Road SE54 3HB") == "ASH PARK ROAD"
    assert extract_street_name("Flat 14 1 Ash Park Road SE54 3HB") == "ASH PARK ROAD"
    assert extract_street_name("123 Main Blvd") == "MAIN BLVD"
    assert extract_street_name("456 Maple AvEnUe") == "MAPLE AVENUE"
    assert extract_street_name("789 Oak Street") == "OAK STREET"

    # Additional test cases
    assert extract_street_name("42 Elm Drive") == "ELM DRIVE"
    assert extract_street_name("15 Willow Ln") == "WILLOW LN"
    assert extract_street_name("789 Maple Terrace") == "MAPLE TERRACE"
    assert extract_street_name("10 Oak Cove") == "OAK COVE"
    assert extract_street_name("675 Pine Circle") == "PINE CIRCLE"

    # Apartment prefixes are ignored and street should still be extracted.
    assert extract_street_name("Apartment 5, 27 Park Avenue") == "PARK AVENUE"

    # Test with only street number
    assert extract_street_name("1234") == ""

    # Test with empty address
    assert extract_street_name("") == ""
