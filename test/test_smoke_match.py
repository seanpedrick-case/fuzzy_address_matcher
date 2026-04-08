import warnings
from pathlib import Path

import pandas as pd
import pytest
from pandas.errors import SettingWithCopyWarning

from tools.matcher_funcs import fuzzy_address_match

# Example fixtures are bundled with the package under tools/example_data/.
TOOLS_ROOT = Path(__file__).resolve().parents[1] / "tools"
SEARCH_PATH = TOOLS_ROOT / "example_data" / "search_addresses_london.csv"
REFERENCE_PATH = TOOLS_ROOT / "example_data" / "reference_addresses_london.csv"


@pytest.mark.smoke
def test_smoke_fuzzy_address_match_with_dataframes() -> None:
    search_df = pd.read_csv(SEARCH_PATH)
    reference_df = pd.read_csv(REFERENCE_PATH)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        msg, output_files, estimate_time, _ = fuzzy_address_match(
            search_df=search_df,
            ref_df=reference_df,
            in_colnames=["address_line_1", "address_line_2", "postcode"],
            in_refcol=["addr1", "addr2", "addr3", "addr4", "postcode"],
        )

    assert isinstance(msg, str)
    assert output_files is not None
    assert len(output_files) >= 1
    assert isinstance(estimate_time, float)

    # Regression check: the warning fixes should keep these modules clean.
    pandas_warnings = []
    for w in caught:
        filename = getattr(w, "filename", "") or ""
        _fn = filename.replace("\\", "/")
        _is_target_module = (
            "/tools/fuzzy_match.py" in _fn or "/tools/matcher_funcs.py" in _fn
        )
        if _is_target_module and (
            issubclass(w.category, FutureWarning)
            or issubclass(w.category, SettingWithCopyWarning)
        ):
            pandas_warnings.append(w)

    assert not pandas_warnings, "Unexpected pandas warnings:\n" + "\n".join(
        f"- {w.category.__name__}: {w.message} ({w.filename}:{w.lineno})"
        for w in pandas_warnings
    )
