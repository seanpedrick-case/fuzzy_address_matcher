"""Regression tests for postcode blocker / duplicate-column handling."""

from types import SimpleNamespace

import pandas as pd

from fuzzy_address_matcher.fuzzy_match import add_fuzzy_block_sequence_col
from fuzzy_address_matcher.matcher_funcs import (
    _column_has_usable_values,
    _normalize_join_key_strings,
    _postcode_batch_covered_search_keys_normalized,
    _resolve_column_series,
    _slice_frame_by_normalized_keys,
    _street_overflow_unbatched_search_enabled,
    _strip_runtime_fuzzy_cols_from_stand_cache,
    _uncovered_search_key_values_for_street_overflow,
    create_batch_ranges,
)


def test_column_has_usable_values_duplicate_label_first_column_nonempty():
    left = pd.DataFrame({"postcode_search": ["nw16hr", ""]})
    right = pd.DataFrame({"postcode_search": ["", ""]})
    df = pd.concat([left, right], axis=1)
    assert df.columns.tolist().count("postcode_search") == 2
    assert _column_has_usable_values(df, "postcode_search") is True
    ser = _resolve_column_series(df, "postcode_search")
    assert ser is not None
    assert ser.iloc[0] == "nw16hr"


def test_column_has_usable_values_duplicate_label_all_empty():
    left = pd.DataFrame({"postcode_search": ["", ""]})
    right = pd.DataFrame({"postcode_search": ["", ""]})
    df = pd.concat([left, right], axis=1)
    assert _column_has_usable_values(df, "postcode_search") is False


def test_add_fuzzy_block_sequence_col_duplicate_postcode_search_labels():
    base = pd.DataFrame({"postcode_search": ["p1", "p1", "p2"], "idx": [0, 1, 2]})
    extra = pd.DataFrame({"postcode_search": ["p1", "p1", "p2"]})
    df = pd.concat([base, extra], axis=1)
    out = add_fuzzy_block_sequence_col(df, "postcode_search")
    o = out.sort_index()
    assert [int(o.loc[i, "_fuzzy_block_seq"]) for i in (0, 1, 2)] == [0, 1, 0]


def test_slice_frame_by_normalized_keys_matches_key_column_not_positional_index():
    """Parquet reload uses RangeIndex; batch keys are original join labels."""
    df = pd.DataFrame(
        {
            "index": ["5000", "5001", "99999"],
            "postcode_search": ["a", "b", "c"],
        }
    )
    out = _slice_frame_by_normalized_keys(df, "index", [5000, 99999])
    assert len(out) == 2
    assert set(out["index"].tolist()) == {"5000", "99999"}


def test_create_batch_ranges_uses_join_column_not_dataframe_index():
    """Batches must list the same ids as ``search_df_key_field`` / ``ref_index``."""
    df = pd.DataFrame(
        {
            "postcode": ["AB1 2CD", "AB1 2CD"],
            "index": ["rowA", "rowB"],
        },
        index=[0, 1],
    )
    ref_df = pd.DataFrame(
        {
            "Postcode": ["AB1 2CD"],
            "ref_index": [99],
        },
        index=[0],
    )
    out = create_batch_ranges(
        df,
        ref_df,
        5000,
        5000,
        "postcode",
        "Postcode",
        search_df_key_field="index",
        ref_key_field="ref_index",
    )
    assert out["search_range"].iloc[0] == ["rowA", "rowB"]
    assert out["ref_range"].iloc[0] == [99]


def test_slice_frame_by_normalized_keys_large_labels_not_in_rangeindex():
    df = pd.DataFrame(
        {"index": ["35000", "35001"], "x": [1, 2]},
    )
    search_range = [35000]
    out_old_index = df.loc[df.index.isin(search_range)]
    assert len(out_old_index) == 0
    out = _slice_frame_by_normalized_keys(df, "index", search_range)
    assert len(out) == 1
    assert out.iloc[0]["index"] == "35000"


def test_normalize_join_key_strings_int_float_string_align():
    left = pd.Series([6199, 6199.0, "6199.0"], dtype=object)
    right = pd.Series([6199], dtype="Int64")
    a = set(_normalize_join_key_strings(left).tolist())
    b = set(_normalize_join_key_strings(right).tolist())
    assert a == {"6199"}
    assert b == {"6199"}


def test_postcode_batch_covered_keys_and_uncovered_search_values():
    range_df = pd.DataFrame({"search_range": [["k1"], ["k3"]]})
    cov = _postcode_batch_covered_search_keys_normalized(range_df)
    assert cov == {"k1", "k3"}
    matcher = SimpleNamespace(
        search_df_key_field="index",
        search_df_cleaned=pd.DataFrame({"index": ["k1", "k2", "k3"]}),
    )
    unc = _uncovered_search_key_values_for_street_overflow(matcher, cov)
    assert set(unc) == {"k2"}


def test_create_batch_ranges_omits_search_only_postcode_from_covered_keys():
    """Search row whose truncated postcode is not in ref never enters a postcode batch."""
    search = pd.DataFrame(
        {
            "postcode": ["AB1 2CD", "ZZ9 9ZZ"],
            "index": ["in_ref_pc", "search_only_pc"],
        }
    )
    ref = pd.DataFrame(
        {
            "Postcode": ["AB1 2CD"],
            "ref_index": [0],
        }
    )
    range_df = create_batch_ranges(
        search.copy(),
        ref.copy(),
        batch_size=1,
        ref_batch_size=1,
        search_postcode_col="postcode",
        ref_postcode_col="Postcode",
        search_df_key_field="index",
        ref_key_field="ref_index",
    )
    cov = _postcode_batch_covered_search_keys_normalized(range_df)
    assert "search_only_pc" not in cov
    assert "in_ref_pc" in cov
    matcher = SimpleNamespace(
        search_df_key_field="index",
        search_df_cleaned=search,
    )
    unc = _uncovered_search_key_values_for_street_overflow(matcher, cov)
    assert unc == ["search_only_pc"]


def test_street_overflow_unbatched_search_env_toggle(monkeypatch):
    monkeypatch.delenv("STREET_OVERFLOW_UNBATCHED_SEARCH", raising=False)
    assert _street_overflow_unbatched_search_enabled() is True
    monkeypatch.setenv("STREET_OVERFLOW_UNBATCHED_SEARCH", "0")
    assert _street_overflow_unbatched_search_enabled() is False


def test_strip_runtime_fuzzy_cols_from_stand_cache():
    df = pd.DataFrame({"a": [1], "_fuzzy_block_seq": [3]})
    out = _strip_runtime_fuzzy_cols_from_stand_cache(df)
    assert "_fuzzy_block_seq" not in out.columns
    assert "a" in out.columns
