import warnings
from pathlib import Path

import pandas as pd
import pytest
from pandas.errors import SettingWithCopyWarning

from fuzzy_address_matcher.matcher_funcs import fuzzy_address_match


def _resolve_example_file(file_name: str) -> Path:
    """
    Resolve example fixture paths across common runtime contexts:
    - installed package data under `fuzzy_address_matcher/example_data/`
    - repo root `example_data/`
    """
    repo_root = Path(__file__).resolve().parents[1]
    candidates = [
        repo_root / "fuzzy_address_matcher" / "example_data" / file_name,
        repo_root / "example_data" / file_name,
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


SEARCH_PATH = _resolve_example_file("search_addresses_london.csv")
REFERENCE_PATH = _resolve_example_file("reference_addresses_london.csv")


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
            # Avoid cross-test contamination from cached standardisation files under the
            # default output folder (which may contain artifacts from earlier local runs).
            output_folder=".tmp_smoke_test_output",
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
            "/fuzzy_address_matcher/fuzzy_match.py" in _fn
            or "/fuzzy_address_matcher/matcher_funcs.py" in _fn
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


@pytest.mark.smoke
def test_street_only_outputs_do_not_duplicate_rows() -> None:
    search_df = pd.read_csv(SEARCH_PATH)
    reference_df = pd.read_csv(REFERENCE_PATH)

    msg, output_files, estimate_time, _ = fuzzy_address_match(
        search_df=search_df,
        ref_df=reference_df,
        in_colnames=["address_line_1", "address_line_2", "postcode"],
        in_refcol=["addr1", "addr2", "addr3", "addr4", "postcode"],
        use_postcode_blocker=False,
        # Isolate artifacts for this regression check.
        output_folder=".tmp_street_only_output",
    )

    assert isinstance(msg, str)
    assert isinstance(estimate_time, float)
    assert output_files

    results_candidates = [p for p in output_files if "results_" in str(p)]
    assert results_candidates, "Expected a results CSV in output_files"
    results_df = pd.read_csv(results_candidates[0])

    # Street-only mode should still return one row per input record.
    assert len(results_df) == len(search_df)
    assert "index" in results_df.columns
    assert not results_df["index"].duplicated().any()
