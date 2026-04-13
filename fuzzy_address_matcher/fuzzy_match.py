from datetime import datetime
from typing import Dict, List, Optional, Tuple, Type

import gradio as gr
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from tqdm import tqdm

from fuzzy_address_matcher.config import (
    fuzzy_match_limit as default_fuzzy_match_limit,
)
from fuzzy_address_matcher.config import (
    no_number_fuzzy_match_limit,
)

PandasDataFrame = Type[pd.DataFrame]
PandasSeries = Type[pd.Series]
MatchedResults = Dict[str, Tuple[str, int]]
array = List[str]

today = datetime.now().strftime("%d%m%Y")
today_rev = datetime.now().strftime("%Y%m%d")


def _first_series_for_col(df: pd.DataFrame, col_name: str) -> Optional[pd.Series]:
    """Single Series when ``col_name`` is duplicated (DataFrame) or missing."""
    if col_name not in df.columns:
        return None
    col = df[col_name]
    if isinstance(col, pd.DataFrame):
        if col.shape[1] == 0:
            return None
        col = col.iloc[:, 0]
    return col if isinstance(col, pd.Series) else None


def add_fuzzy_block_sequence_col(df: pd.DataFrame, block_col: str) -> pd.DataFrame:
    """
    0..n-1 within each blocker value (postcode or street), stable on original row order.

    Must match iteration order in ``string_match_by_post_code_multiple`` (sort by block
    key, then original position) so fuzzy output rows can join back without ambiguity
    when several records share the same standardised search address within a block.
    """
    if df.empty:
        out = df.copy()
        out["_fuzzy_block_seq"] = pd.Series(dtype="Int64")
        return out
    blk = _first_series_for_col(df, block_col)
    if blk is None:
        out = df.copy()
        out["_fuzzy_block_seq"] = np.int64(0)
        return out
    tmp = df.copy()
    tmp["__fz_orig"] = np.arange(len(tmp), dtype=np.int64)
    tmp["__fz_block_key"] = blk
    tmp = tmp.sort_values(["__fz_block_key", "__fz_orig"], kind="mergesort")
    tmp["_fuzzy_block_seq"] = (
        tmp.groupby("__fz_block_key", sort=False).cumcount().astype(np.int64)
    )
    out = tmp.sort_values("__fz_orig").drop(
        columns=["__fz_orig", "__fz_block_key"], errors="ignore"
    )
    return out


def _create_frame(
    matched_results: Dict[str, Tuple[str, int]],
    index_name: str,
    matched_name: str,
) -> pd.DataFrame:
    """
    Convert RapidFuzz extractOne-style outputs into a DataFrame.
    """
    rows = []
    for k, v in matched_results.items():
        if v is None:
            rows.append((k, None, None))
        else:
            # v is (match, score, optional_index)
            rows.append((k, v[0], v[1]))
    return pd.DataFrame(rows, columns=[index_name, matched_name, "score"])


def string_match_array(
    to_match: array, choices: array, index_name: str, matched_name: str
) -> PandasDataFrame:

    temp = {name: process.extractOne(name, choices) for name in to_match}

    return _create_frame(
        matched_results=temp, index_name=index_name, matched_name=matched_name
    )


# Fuzzy match algorithm
def create_fuzzy_matched_col(
    df: PandasDataFrame,
    orig_match_address_series: PandasSeries,
    pred_match_address_series: PandasSeries,
    fuzzy_method: str,
    match_score=95,
):

    results = []

    for orig_index, orig_string in df[orig_match_address_series].items():

        predict_string = df[pred_match_address_series][orig_index]

        if (orig_string == "") and (predict_string == ""):
            results.append(np.nan)

        else:
            fuzz_score = process.extract(
                orig_string, [predict_string], scorer=getattr(fuzz, fuzzy_method)
            )
            results.append(fuzz_score[0][1])

    new_result_col_score = orig_match_address_series + "_fuzz_score"
    new_result_col_match = orig_match_address_series + "_fuzz_match"

    df[new_result_col_score] = results
    df[new_result_col_match] = df[new_result_col_score] >= match_score
    # df[new_result_col_match][df[new_result_col_score].isna()] = np.nan
    df.loc[df[new_result_col_score].isna(), new_result_col_match] = np.nan

    return df


def string_match_by_post_code_multiple(
    match_address_series: PandasSeries,
    reference_address_series: PandasSeries,
    search_limit=100,
    scorer_name="token_set_ratio",
    fuzzy_match_limit: Optional[float] = None,
    show_progress: bool = True,
    progress=gr.Progress(track_tqdm=True),
) -> MatchedResults:
    """
    Matches by Series values; for example idx is post code and
    values address. Search field is reduced by comparing same post codes address reference_address_series.

    Default scorer is fuzz.Wratio. This tries to weight the different algorithms
    to give the best score.
    Choice of ratio type seems to make a big difference. Looking at this link:
    https://chairnerd.seatgeek.com/fuzzywuzzy-fuzzy-string-matching-in-python/
    and this one:
    https://stackoverflow.com/questions/31806695/when-to-use-which-fuzz-function-to-compare-2-strings

    """
    _score_cutoff = (
        default_fuzzy_match_limit
        if fuzzy_match_limit is None
        else float(fuzzy_match_limit)
    )

    def do_one_match(
        reference_addresses: pd.Series,
        scorer: callable,
        search_limit: int,
        postcode_match: str,
        search_addresses: pd.Series,
    ) -> MatchedResults:

        def _prepare_results(
            search_addresses, reference_addresses, matched, postcode_match
        ):

            # Create a list to store the results
            results = []

            # Iterate through the matched dataframe and store results in the list
            for i, search_address in enumerate(search_addresses):
                for j, reference_address in enumerate(reference_addresses):
                    score = matched[i][j]
                    results.append(
                        (
                            postcode_match,
                            i,
                            search_address,
                            reference_address,
                            score,
                        )
                    )

            # Create a dataframe from the results list
            matched_out = pd.DataFrame(
                results,
                columns=[
                    "postcode_search",
                    "fuzzy_search_block_seq",
                    "fuzzy_match_search_address",
                    "fuzzy_match_reference_address",
                    "fuzzy_score",
                ],
            )

            return matched_out

        try:
            if isinstance(
                reference_addresses, str
            ):  # reference_addresses can be a str-> 1 address per postcode
                matched = process.cdist(
                    search_addresses.values,
                    [reference_addresses],
                    scorer=scorer,
                    score_cutoff=_score_cutoff,
                    workers=-1,
                )

                # Transform results into a dataframe
                matched_out = _prepare_results(
                    search_addresses, reference_addresses, matched, postcode_match
                )

            else:  # 1+ addresses
                matched = process.cdist(
                    search_addresses.values,
                    reference_addresses.values,
                    scorer=scorer,
                    score_cutoff=_score_cutoff,
                    workers=-1,
                )

                # Transform results into a dataframe
                matched_out = _prepare_results(
                    search_addresses, reference_addresses, matched, postcode_match
                )

            # Sort the matched results by score in descending order
            matched_out = matched_out.sort_values(by="fuzzy_score", ascending=False)

            # Keep only the top search_limit number of results - doesn't work anymore when working with multiple results
            # matched_out = matched_out.head(search_limit)

        except KeyError:
            matched_out = pd.DataFrame()

        return matched_out

    def apply_fuzzy_matching(
        postcode_match: str,
        search_addresses: PandasSeries,
        reference_addresses: PandasSeries,
        scorer: callable,
        search_limit: int,
    ) -> tuple:

        try:
            matched = do_one_match(
                reference_addresses,
                scorer,
                search_limit,
                postcode_match,
                search_addresses,
            )
            return matched
        except KeyError:
            matched = (
                pd.DataFrame()
            )  # [("NA", 0)] # for _ in range(1, search_limit + 1)]
            return matched

    match_address_series = match_address_series.rename_axis("postcode_search")
    match_address_df = pd.DataFrame(match_address_series.reset_index())
    match_address_df["index"] = list(range(0, len(match_address_df)))

    reference_address_series = reference_address_series.rename_axis("postcode_search")
    reference_address_df = pd.DataFrame(reference_address_series.reset_index())
    reference_address_df["index"] = list(range(0, len(reference_address_df)))

    # Apply the match functions to each address
    scorer = getattr(fuzz, scorer_name)
    # counter = 0

    index_list = []
    match_list = []
    search_addresses_list = []
    reference_addresses_list = []

    unique_postcodes = pd.unique(match_address_df["postcode_search"])

    for postcode_match in tqdm(
        unique_postcodes,
        desc="Fuzzy matching",
        unit="blocks (postcodes or streets)",
        disable=not show_progress,
    ):

        postcode_match_list = [postcode_match]
        search_indexes = pd.Series()
        search_addresses = pd.Series()
        reference_addresses = pd.Series()

        try:
            search_indexes = match_address_df.loc[
                match_address_df["postcode_search"].isin(postcode_match_list), "index"
            ]
            search_addresses = match_address_df.loc[
                match_address_df["postcode_search"].isin(postcode_match_list),
                "search_address_stand",
            ]
            reference_addresses = reference_address_df.loc[
                reference_address_df["postcode_search"].isin(postcode_match_list),
                "ref_address_stand",
            ]

            if isinstance(
                reference_addresses, str
            ):  # reference_addresses can be a str-> 1 address per postcode
                reference_addresses = pd.Series(reference_addresses)
        except KeyError:
            reference_addresses = pd.Series("NA")

        matched = apply_fuzzy_matching(
            postcode_match, search_addresses, reference_addresses, scorer, search_limit
        )

        # Write to output lists
        match_list.extend([matched])
        index_list.extend(search_indexes.tolist())
        search_addresses_list.extend(search_addresses.tolist())
        reference_addresses_list.extend(reference_addresses.tolist())

    _dfs = [m for m in match_list if isinstance(m, pd.DataFrame)]
    _non_empty_matches = [m for m in _dfs if not m.empty]
    if not _non_empty_matches:
        if _dfs:
            return _dfs[0].iloc[0:0].copy()
        return pd.DataFrame()

    out_frame = pd.concat(_non_empty_matches, ignore_index=False)

    return out_frame


def _create_fuzzy_match_results_output(
    results: PandasDataFrame,
    search_df_after_stand: PandasDataFrame,
    ref_df_cleaned: PandasDataFrame,
    ref_df_after_stand: PandasDataFrame,
    fuzzy_match_limit: int,
    search_df_cleaned: PandasDataFrame,
    search_df_key_field: str,
    new_join_col: str,
    standardise: bool,
    blocker_col: str,
):
    """
    Take fuzzy match outputs, create shortlist dataframes, rearrange, return diagnostics and shortlist dataframes for export
    """

    ## Diagnostics

    diag_shortlist, diag_best_match = create_diagnostic_results(
        results_df=results,
        matched_df=search_df_after_stand,
        ref_list_df=ref_df_after_stand,
        fuzzy_match_limit=fuzzy_match_limit,
        blocker_col=blocker_col,
        search_df_key_field=search_df_key_field,
    )

    ## Fuzzy search results
    match_results_cols = [
        "search_orig_address",
        "search_merge_full_address",
        "reference_orig_address",
        "ref_index",
        "full_match",
        "full_number_match",
        "flat_number_match",
        "room_number_match",
        "block_number_match",
        "unit_number_match",
        "property_number_match",
        "close_postcode_match",
        "house_court_name_match",
        "fuzzy_score_match",
        "fuzzy_score",
        "wratio_score",
        "property_number_search",
        "property_number_reference",
        "flat_number_search",
        "flat_number_reference",
        "room_number_search",
        "room_number_reference",
        "unit_number_search",
        "unit_number_reference",
        "block_number_search",
        "block_number_reference",
        "house_court_name_search",
        "house_court_name_reference",
        "search_mod_address",
        "reference_mod_address",
        "Postcode",
    ]

    # Join results data onto the original housing list to create the full output.
    # Include the stable original-format display address when present so the
    # exported `search_orig_address` never reflects standardised/lowercased text.
    search_df_cleaned_join_cols = [search_df_key_field, "full_address", "postcode"]
    if "search_input_display" in search_df_cleaned.columns:
        search_df_cleaned_join_cols.append("search_input_display")

    _diag_cols = [c for c in match_results_cols if c in diag_best_match.columns]
    if (
        search_df_key_field
        and search_df_key_field in diag_best_match.columns
        and search_df_key_field not in _diag_cols
    ):
        _diag_cols.insert(0, search_df_key_field)

    _left_out = search_df_cleaned[search_df_cleaned_join_cols].copy()
    _diag_subset = diag_best_match[_diag_cols].copy()
    if search_df_key_field in _diag_subset.columns:
        _left_out[search_df_key_field] = _left_out[search_df_key_field].astype(str)
        _diag_subset[search_df_key_field] = _diag_subset[search_df_key_field].astype(
            str
        )
        match_results_output = _left_out.merge(
            _diag_subset, how="left", on=search_df_key_field
        )
    else:
        match_results_output = _left_out.merge(
            _diag_subset,
            how="left",
            left_on="full_address",
            right_on="search_merge_full_address",
        )

    match_results_output = match_results_output.drop(
        ["postcode", "search_merge_full_address"], axis=1, errors="ignore"
    )
    # `search_orig_address` from diagnostics is already display-oriented; optionally
    # overwrite with `search_input_display` when present. Always drop prepared
    # `full_address` so we do not carry duplicate address columns forward.
    if "search_input_display" in match_results_output.columns:
        _sid = match_results_output["search_input_display"]
        _has = (
            _sid.notna()
            & _sid.astype(str).str.strip().ne("")
            & ~_sid.astype(str).str.strip().str.lower().isin(("nan", "none", "<na>"))
        )
        _base = (
            match_results_output["search_orig_address"]
            if "search_orig_address" in match_results_output.columns
            else match_results_output["full_address"]
        )
        match_results_output["search_orig_address"] = _sid.where(_has, _base)
        match_results_output = match_results_output.drop(
            columns=["search_input_display"], errors="ignore"
        )
    elif "search_orig_address" not in match_results_output.columns:
        match_results_output["search_orig_address"] = match_results_output[
            "full_address"
        ]
    else:
        _soa = match_results_output["search_orig_address"]
        _need_fill = _soa.isna() | (_soa.astype(str).str.strip().eq(""))
        if _need_fill.any():
            match_results_output.loc[_need_fill, "search_orig_address"] = (
                match_results_output.loc[_need_fill, "full_address"]
            )
    match_results_output = match_results_output.drop(
        columns=["full_address"], errors="ignore"
    )

    # Join UPRN back onto the data from reference data
    joined_ref_cols = ["fulladdress", "Reference file"]
    joined_ref_cols.extend(new_join_col)

    # Keep only columns that exist in reference dataset
    joined_ref_cols = [col for col in joined_ref_cols if col in ref_df_cleaned.columns]

    match_results_output = pd.merge(
        match_results_output,
        ref_df_cleaned[joined_ref_cols].drop_duplicates("fulladdress"),
        how="left",
        left_on="reference_orig_address",
        right_on="fulladdress",
    ).drop("fulladdress", axis=1)

    # Convert long keys to string to avoid data loss
    match_results_output[search_df_key_field] = match_results_output[
        search_df_key_field
    ].astype("str")
    join_cols_out = (
        list(new_join_col)
        if isinstance(new_join_col, (list, tuple))
        else [new_join_col]
    )
    for jc in join_cols_out:
        if jc in match_results_output.columns:
            match_results_output[jc] = match_results_output[jc].astype("string")
    match_results_output["standardised_address"] = standardise

    match_results_output = match_results_output.sort_values(
        search_df_key_field, ascending=True
    )

    return match_results_output, diag_shortlist, diag_best_match


def create_diag_shortlist(
    results_df: PandasDataFrame,
    matched_col: str,
    fuzzy_match_limit: int,
    blocker_col: str,
    fuzzy_col: str = "fuzzy_score",
    search_mod_address: str = "search_mod_address",
    resolve_tie_breaks: bool = True,
    no_number_fuzzy_match_limit: int = no_number_fuzzy_match_limit,
) -> PandasDataFrame:
    """
    Create a shortlist of the best matches from a list of suggested matches
    """

    ## Calculate highest fuzzy score from all candidates, keep all candidates with matching highest fuzzy score
    results_max_fuzzy_score = (
        results_df.groupby(matched_col)[fuzzy_col]
        .max()
        .reset_index()
        .rename(columns={fuzzy_col: "max_fuzzy_score"})
        .drop_duplicates(subset=matched_col)
    )

    results_df = pd.merge(
        results_df, results_max_fuzzy_score, how="left", on=matched_col
    )

    _shortlist_mask = results_df[fuzzy_col] == results_df["max_fuzzy_score"]
    diag_shortlist = results_df.loc[_shortlist_mask, :].copy()

    # Fuzzy match limit for records with no numbers in it is 0.95 or the provided fuzzy_match_limit, whichever is higher
    # diag_shortlist["fuzzy_score_match"] = diag_shortlist[fuzzy_col] >= fuzzy_match_limit
    diag_shortlist.loc[
        diag_shortlist[fuzzy_col] >= fuzzy_match_limit, "fuzzy_score_match"
    ] = True

    ### Count number of numbers in search string
    # Using .loc
    diag_shortlist.loc[:, "number_count_search_string"] = diag_shortlist.loc[
        :, search_mod_address
    ].str.count(r"\d")
    diag_shortlist.loc[:, "no_numbers_in_search_string"] = (
        diag_shortlist.loc[:, "number_count_search_string"] == 0
    )

    # Replace fuzzy_score_match values for addresses with no numbers in them
    diag_shortlist.loc[
        (diag_shortlist["no_numbers_in_search_string"])
        & (diag_shortlist[fuzzy_col] >= no_number_fuzzy_match_limit),
        "fuzzy_score_match",
    ] = True
    diag_shortlist.loc[
        (diag_shortlist["no_numbers_in_search_string"])
        & (diag_shortlist[fuzzy_col] < no_number_fuzzy_match_limit),
        "fuzzy_score_match",
    ] = False

    # If blocking on street, don't match addresses with 0 numbers in. There are too many options and the matches are rarely good
    if blocker_col == "Street":
        diag_shortlist.loc[
            (diag_shortlist["no_numbers_in_search_string"]), "fuzzy_score_match"
        ] = False

    # Avoid filling the entire frame with "" (can silently coerce dtypes and triggers
    # pandas future downcasting warnings). Instead, normalise booleans explicitly and
    # fill only object/string-like columns for export friendliness.
    if "fuzzy_score_match" in diag_shortlist.columns:
        _fsm = diag_shortlist["fuzzy_score_match"]
        if _fsm.dtype == object:
            _fsm = _fsm.where(_fsm != "", pd.NA)
        diag_shortlist["fuzzy_score_match"] = (
            _fsm.astype("boolean").fillna(False).astype(bool)
        )

    diag_shortlist = diag_shortlist.infer_objects(copy=False)

    _obj_cols = diag_shortlist.select_dtypes(include=["object", "string"]).columns
    if len(_obj_cols) > 0:
        diag_shortlist.loc[:, _obj_cols] = diag_shortlist.loc[:, _obj_cols].fillna("")

    diag_shortlist = diag_shortlist.drop(
        ["number_count_search_string", "no_numbers_in_search_string"], axis=1
    )

    # Following considers full matches to be those that match on property number and flat number, and the postcode is relatively close.
    # Treat blank-like tokens and missing values as equivalent "no value" for
    # component-level comparisons.
    def _equal_or_both_missing(left: pd.Series, right: pd.Series) -> pd.Series:
        left_n = left.astype("string").str.strip().replace("", pd.NA)
        right_n = right.astype("string").str.strip().replace("", pd.NA)
        return (left_n == right_n) | (left_n.isna() & right_n.isna())

    diag_shortlist["property_number_match"] = _equal_or_both_missing(
        diag_shortlist["property_number_search"],
        diag_shortlist["property_number_reference"],
    )
    diag_shortlist["flat_number_match"] = _equal_or_both_missing(
        diag_shortlist["flat_number_search"],
        diag_shortlist["flat_number_reference"],
    )
    diag_shortlist["room_number_match"] = _equal_or_both_missing(
        diag_shortlist["room_number_search"],
        diag_shortlist["room_number_reference"],
    )
    diag_shortlist["block_number_match"] = _equal_or_both_missing(
        diag_shortlist["block_number_search"],
        diag_shortlist["block_number_reference"],
    )
    diag_shortlist["unit_number_match"] = _equal_or_both_missing(
        diag_shortlist["unit_number_search"],
        diag_shortlist["unit_number_reference"],
    )
    diag_shortlist["house_court_name_match"] = _equal_or_both_missing(
        diag_shortlist["house_court_name_search"],
        diag_shortlist["house_court_name_reference"],
    )

    # Full number match is currently considered only a match between property number and flat number

    diag_shortlist["full_number_match"] = (
        (diag_shortlist["property_number_match"])
        & (diag_shortlist["flat_number_match"])
        & (diag_shortlist["room_number_match"])
        & (diag_shortlist["block_number_match"])
        & (diag_shortlist["unit_number_match"])
        & (diag_shortlist["house_court_name_match"])
    )

    # Postcode closeness check:
    # - In street-only blocking, we should not require postcodes.
    # - Some inputs can produce duplicate column names (e.g. "postcode"), making df["postcode"] a DataFrame.
    if blocker_col == "Street":
        diag_shortlist["close_postcode_match"] = True
    else:
        postcode_left = diag_shortlist["postcode"]
        if isinstance(postcode_left, pd.DataFrame):
            postcode_left = postcode_left.iloc[:, 0]

        postcode_right = diag_shortlist["Postcode"]
        if isinstance(postcode_right, pd.DataFrame):
            postcode_right = postcode_right.iloc[:, 0]

        diag_shortlist["close_postcode_match"] = (
            postcode_left.astype(str).str.lower().str.replace(" ", "").str[:-2]
            == postcode_right.astype(str).str.lower().str.replace(" ", "").str[:-2]
        )

    diag_shortlist["full_match"] = (
        (diag_shortlist["fuzzy_score_match"])
        & (diag_shortlist["full_number_match"])
        & (diag_shortlist["close_postcode_match"])
    )

    diag_shortlist = diag_shortlist.rename(
        columns={"reference_list_address": "reference_mod_address"}
    )

    ### Dealing with tie breaks ##
    # Do a backup simple Wratio search on the open text to act as a tie breaker when the fuzzy scores are identical
    # fuzz.WRatio
    if resolve_tie_breaks:

        def compare_strings_wratio(row, scorer=fuzz.ratio, fuzzy_col=fuzzy_col):
            search_score = process.cdist(
                [row[search_mod_address]], [row["reference_mod_address"]], scorer=scorer
            )
            return search_score[0][0]

        diag_shortlist_dups = diag_shortlist[diag_shortlist["full_number_match"]]
        diag_shortlist_dups = diag_shortlist_dups.loc[
            diag_shortlist_dups.duplicated(
                subset=[
                    search_mod_address,
                    "full_number_match",
                    "room_number_search",
                    fuzzy_col,
                ],
                keep=False,
            )
        ]

        if not diag_shortlist_dups.empty:
            diag_shortlist_dups["wratio_score"] = diag_shortlist_dups.apply(
                compare_strings_wratio, axis=1
            )

            diag_shortlist = diag_shortlist.merge(
                diag_shortlist_dups[["wratio_score"]],
                left_index=True,
                right_index=True,
                how="left",
            )

    if "wratio_score" not in diag_shortlist.columns:
        diag_shortlist["wratio_score"] = None

    # Order by best score
    diag_shortlist = diag_shortlist.sort_values(
        [
            search_mod_address,
            "full_match",
            "full_number_match",
            fuzzy_col,
            "wratio_score",
        ],
        ascending=[True, False, False, False, False],
    )

    return diag_shortlist


def create_diagnostic_results(
    results_df: PandasDataFrame,
    matched_df: PandasDataFrame,
    ref_list_df: PandasDataFrame,
    matched_col="fuzzy_match_search_address",
    ref_list_col="fuzzy_match_reference_address",
    final_matched_address_col="search_address_stand",
    final_ref_address_col="ref_address_stand",
    orig_matched_address_col="full_address",
    orig_ref_address_col="fulladdress",
    fuzzy_match_limit=default_fuzzy_match_limit,
    blocker_col="Postcode",
    search_df_key_field: Optional[str] = None,
) -> PandasDataFrame:
    """
    This function takes a result file from the fuzzy search, then refines the 'matched results' according
    the score limit specified by the user and exports results list, matched and unmatched files.
    """

    # Rename score column
    results_df = results_df.rename(columns={"score": "fuzzy_score"})

    # Remove empty addresses
    results_df = results_df[results_df[matched_col] != 0]

    ### Join property number and flat/room number etc. onto results_df
    if "ref_index" not in ref_list_df.columns:
        print("Existing ref_index column not found")
        ref_list_df["ref_index"] = ref_list_df.index

    # Normalise postcode naming on reference side for custom schemas.
    if "Postcode" not in ref_list_df.columns and "postcode" in ref_list_df.columns:
        ref_list_df["Postcode"] = ref_list_df["postcode"]

    ref_join_cols = [
        "ref_index",
        final_ref_address_col,
        "property_number",
        "flat_number",
        "room_number",
        "block_number",
        "unit_number",
        "house_court_name",
        orig_ref_address_col,
        "Postcode",
    ]
    for _c in ref_join_cols:
        if _c not in ref_list_df.columns:
            ref_list_df[_c] = ""
    ref_list_df = ref_list_df[ref_join_cols].rename(
        columns={
            orig_ref_address_col: "reference_orig_address",
            final_ref_address_col: "reference_list_address",
        }
    )

    results_df = results_df.merge(
        ref_list_df, how="left", left_on=ref_list_col, right_on="reference_list_address"
    )

    ### Join on relevant details from the standardised match dataframe
    # `search_orig_address` is for human-facing output (preserve original casing/format).
    # `search_merge_full_address` must stay aligned with `search_df_cleaned["full_address"]`
    # so `_create_fuzzy_match_results_output` can merge match rows back without breaking
    # when display text differs from prepared `full_address`.
    matched_df = matched_df.copy()
    if "full_address" in matched_df.columns:
        matched_df["search_merge_full_address"] = matched_df["full_address"]
    else:
        matched_df["search_merge_full_address"] = ""
    if orig_matched_address_col == "full_address" and (
        "search_input_display" in matched_df.columns
    ):
        orig_matched_address_col = "search_input_display"

    _block_join_col = "postcode_search" if blocker_col == "Postcode" else "street"
    if _block_join_col not in matched_df.columns:
        matched_df[_block_join_col] = ""
    if "_fuzzy_block_seq" not in matched_df.columns:
        matched_df["_fuzzy_block_seq"] = np.nan

    matched_df_cols = [
        final_matched_address_col,
        "property_number",
        "flat_number",
        "room_number",
        "block_number",
        "unit_number",
        "house_court_name",
        orig_matched_address_col,
        "search_merge_full_address",
        "postcode",
    ]
    if search_df_key_field and search_df_key_field in matched_df.columns:
        if search_df_key_field not in matched_df_cols:
            matched_df_cols.insert(0, search_df_key_field)
    if _block_join_col not in matched_df_cols:
        matched_df_cols.append(_block_join_col)
    if "_fuzzy_block_seq" not in matched_df_cols:
        matched_df_cols.append("_fuzzy_block_seq")
    for _c in matched_df_cols:
        if _c not in matched_df.columns:
            matched_df[_c] = ""
    matched_df = matched_df[matched_df_cols].rename(
        columns={
            orig_matched_address_col: "search_orig_address",
            final_matched_address_col: "search_mod_address",
        }
    )

    if "fuzzy_search_block_seq" not in results_df.columns:
        results_df = results_df.copy()
        results_df["fuzzy_search_block_seq"] = np.int64(0)
    else:
        results_df = results_df.copy()
        results_df["fuzzy_search_block_seq"] = (
            pd.to_numeric(results_df["fuzzy_search_block_seq"], errors="coerce")
            .fillna(0)
            .astype(np.int64)
        )

    matched_m = matched_df.copy()
    matched_m["_fuzzy_block_seq"] = (
        pd.to_numeric(matched_m["_fuzzy_block_seq"], errors="coerce")
        .fillna(-1)
        .astype(np.int64)
    )
    matched_m["_fuzzy_merge_block"] = matched_m[_block_join_col].map(
        _norm_join_lookup_key
    )
    results_df["_fuzzy_merge_block"] = results_df["postcode_search"].map(
        _norm_join_lookup_key
    )

    # Avoid duplicate labels with `results_df["postcode_search"]` (block key from
    # fuzzy output). Pandas would add _reference/_search suffixes and break later
    # postcode / component logic — especially for street blocking where results
    # `postcode_search` holds the street token but matched rows carry real postcodes.
    _drop_from_matched = []
    if _block_join_col in matched_m.columns:
        _drop_from_matched.append(_block_join_col)
    if "postcode_search" in matched_m.columns:
        _drop_from_matched.append("postcode_search")
    if _drop_from_matched:
        matched_m = matched_m.drop(columns=list(dict.fromkeys(_drop_from_matched)))

    results_df = results_df.merge(
        matched_m,
        how="left",
        left_on=[matched_col, "_fuzzy_merge_block", "fuzzy_search_block_seq"],
        right_on=["search_mod_address", "_fuzzy_merge_block", "_fuzzy_block_seq"],
        suffixes=("_reference", "_search"),
    )
    results_df = results_df.drop(columns=["_fuzzy_merge_block"], errors="ignore")

    # Choose your best matches from the list of options
    diag_shortlist = create_diag_shortlist(
        results_df, matched_col, fuzzy_match_limit, blocker_col
    )

    ### Create matched results output ###
    # Columns for the output match_results file in order
    match_results_cols = [
        "search_orig_address",
        "search_merge_full_address",
        "reference_orig_address",
        "ref_index",
        "full_match",
        "full_number_match",
        "flat_number_match",
        "room_number_match",
        "block_number_match",
        "unit_number_match",
        "house_court_name_match",
        "property_number_match",
        "close_postcode_match",
        "fuzzy_score_match",
        "fuzzy_score",
        "wratio_score",
        "property_number_search",
        "property_number_reference",
        "flat_number_search",
        "flat_number_reference",
        "room_number_search",
        "room_number_reference",
        "block_number_search",
        "block_number_reference",
        "unit_number_search",
        "unit_number_reference",
        "house_court_name_search",
        "house_court_name_reference",
        "search_mod_address",
        "reference_mod_address",
        "postcode",
        "Postcode",
    ]

    if search_df_key_field and search_df_key_field in diag_shortlist.columns:
        if search_df_key_field not in match_results_cols:
            match_results_cols.insert(0, search_df_key_field)
    match_results_cols = [c for c in match_results_cols if c in diag_shortlist.columns]
    diag_shortlist = diag_shortlist[match_results_cols]

    diag_shortlist["ref_index"] = diag_shortlist["ref_index"].astype(
        int, errors="ignore"
    )
    diag_shortlist["wratio_score"] = diag_shortlist["wratio_score"].astype(
        float, errors="ignore"
    )

    # Choose best match from the shortlist that has been ordered according to score descending
    _dedupe_subset = (
        [search_df_key_field]
        if (search_df_key_field and search_df_key_field in diag_shortlist.columns)
        else ["search_mod_address"]
    )
    diag_best_match = diag_shortlist.drop_duplicates(
        subset=_dedupe_subset, keep="first"
    )

    return diag_shortlist, diag_best_match


def _norm_join_lookup_key(val) -> str:
    """Normalize stored IDs / keys for joining search 'existing match' values to reference rows."""
    if val is None:
        return ""
    if isinstance(val, float) and np.isnan(val):
        return ""
    if isinstance(val, (np.integer, np.floating)):
        if isinstance(val, np.floating) and np.isnan(val):
            return ""
        val = val.item()
    s = str(val).strip()
    if s.lower() in ("nan", "none", "<na>", ""):
        return ""
    if len(s) > 2 and s.endswith(".0") and s[:-2].lstrip("-").isdigit():
        s = s[:-2]
    return s


def fill_reference_join_columns_for_previously_matched(
    results_df: PandasDataFrame,
    ref_df_cleaned: PandasDataFrame,
    new_join_col: List[str],
    existing_match_col: Optional[str] = None,
) -> PandasDataFrame:
    """
    For rows with Excluded from search == 'Previously matched', populate `new_join_col`
    from `ref_df_cleaned`.

    Prefer joining on the Gradio `in_existing` column when it holds the same identifier as
    the reference join field (e.g. UPRN). Fall back to matching normalised Search data address
    to reference `fulladdress`.
    """
    if results_df is None or results_df.empty:
        return results_df
    if ref_df_cleaned is None or ref_df_cleaned.empty or not new_join_col:
        return results_df

    needed = [c for c in new_join_col if c in ref_df_cleaned.columns]
    if not needed:
        return results_df

    out = results_df.copy()
    if out.columns.duplicated().any():
        out = out.loc[:, ~out.columns.duplicated()].copy()

    ref = ref_df_cleaned.copy()
    if ref.columns.duplicated().any():
        ref = ref.loc[:, ~ref.columns.duplicated()].copy()

    mask = (
        out.get("Excluded from search", pd.Series("", index=out.index))
        .fillna("")
        .astype(str)
        .eq("Previously matched")
    )
    if not mask.any():
        return out

    jc0 = needed[0]

    # --- 1) Key-based lookup: value in `in_existing` matches reference join key column ---
    existing_series = None
    if existing_match_col:
        if existing_match_col in out.columns:
            existing_series = out[existing_match_col]
        else:
            _alt = f"__search_side_{existing_match_col}"
            if _alt in out.columns:
                existing_series = out[_alt]

    if existing_series is not None:
        ref_key = ref[needed].drop_duplicates(subset=[jc0], keep="first")
        by_primary: Dict[str, pd.Series] = {}
        for _, row in ref_key.iterrows():
            k = _norm_join_lookup_key(row[jc0])
            if k:
                by_primary[k] = row[needed]

        if existing_match_col in ref.columns:
            ex_cols = list(dict.fromkeys([existing_match_col] + needed))
            ref_ex = ref[ex_cols].drop_duplicates(
                subset=[existing_match_col], keep="first"
            )
            by_existing_name: Dict[str, pd.Series] = {}
            for _, row in ref_ex.iterrows():
                k = _norm_join_lookup_key(row[existing_match_col])
                if k:
                    by_existing_name[k] = row[needed]
        else:
            by_existing_name = {}

        for idx in out.index[mask]:
            k = _norm_join_lookup_key(existing_series.loc[idx])
            if not k:
                continue
            row_data = None
            if k in by_primary:
                row_data = by_primary[k]
            elif k in by_existing_name:
                row_data = by_existing_name[k]
            if row_data is None:
                continue
            for col in needed:
                if col in row_data.index:
                    v = row_data[col]
                    if pd.notna(v) and str(v).strip() != "":
                        out.at[idx, col] = v

    # --- 2) Address fallback where join columns are still empty ---
    if "Search data address" not in out.columns or "fulladdress" not in ref.columns:
        return out

    jc0 = needed[0]
    if jc0 not in out.columns:
        return out

    still_blank = mask & (
        out[jc0].isna()
        | (out[jc0].astype(str).str.strip().eq(""))
        | (out[jc0].astype(str).str.strip().str.lower().isin(["nan", "none"]))
    )
    if not still_blank.any():
        return out

    sub = out.loc[still_blank, ["Search data address"]].copy()
    sub["_norm_k"] = sub["Search data address"].astype(str).str.lower().str.strip()

    ref_l = ref[["fulladdress"] + needed].drop_duplicates(
        subset=["fulladdress"], keep="first"
    )
    ref_l["_norm_k"] = ref_l["fulladdress"].astype(str).str.lower().str.strip()
    ref_l = ref_l.drop_duplicates(subset=["_norm_k"], keep="first")

    merged = sub.merge(ref_l[["_norm_k"] + needed], on="_norm_k", how="left")
    for col in needed:
        if col not in merged.columns:
            continue
        fill_series = merged[col]
        has_val = fill_series.notna() & (fill_series.astype(str).str.strip().ne(""))
        if has_val.any():
            idx = fill_series.index[has_val]
            out.loc[idx, col] = fill_series.loc[idx]

    return out


def _overlay_pre_filter_address_for_display(
    search_df: PandasDataFrame,
    search_df_key_field: str,
    pre_filter_search_df: Optional[PandasDataFrame],
) -> PandasDataFrame:
    """
    Replace cleaned `full_address` with the raw input join (`address_cols_joined`) when available
    so 'Search data address' matches the original search file format for every row.
    """
    # Prefer using the pre-filter search snapshot (this preserves original case/format),
    # but fall back to `search_df` itself if it already carries `address_cols_joined`.
    # This avoids mixed casing in the output when some call paths don't pass
    # `pre_filter_search_df`.
    disp_src = None
    if (
        isinstance(pre_filter_search_df, pd.DataFrame)
        and (not pre_filter_search_df.empty)
        and ("address_cols_joined" in pre_filter_search_df.columns)
    ):
        disp_src = pre_filter_search_df
    elif isinstance(search_df, pd.DataFrame) and (
        "address_cols_joined" in search_df.columns
    ):
        disp_src = search_df

    if disp_src is None:
        return search_df
    if (
        search_df_key_field not in disp_src.columns
        or "full_address" not in search_df.columns
    ):
        return search_df

    disp = disp_src[[search_df_key_field, "address_cols_joined"]].drop_duplicates(
        subset=[search_df_key_field], keep="first"
    )
    k = search_df_key_field
    out = search_df.copy()
    out[k] = out[k].astype(str)
    disp = disp.copy()
    disp[k] = disp[k].astype(str)
    out = out.merge(disp, on=k, how="left")
    orig = out["address_cols_joined"]
    has_orig = (
        orig.notna()
        & orig.astype(str).str.strip().ne("")
        & ~orig.astype(str).str.strip().str.lower().isin(("nan", "none", "<na>"))
    )
    out["full_address"] = orig.where(has_orig, out["full_address"])
    return out.drop(columns=["address_cols_joined"], errors="ignore")


def create_results_df(
    match_results_output: PandasDataFrame,
    search_df: PandasDataFrame,
    search_df_key_field: str,
    new_join_col: List[str],
    ref_df_cleaned: Optional[PandasDataFrame] = None,
    existing_match_col: Optional[str] = None,
    pre_filter_search_df: Optional[PandasDataFrame] = None,
) -> PandasDataFrame:
    """
    Following the fuzzy match, join the match results back to the original search dataframe to create a results dataframe.
    """
    search_df = search_df.copy()
    search_df = _overlay_pre_filter_address_for_display(
        search_df, search_df_key_field, pre_filter_search_df
    )

    # If a stable original-format display address is present, prefer it for the
    # output 'Search data address' rather than the prepared/standardised `full_address`.
    if (
        "search_input_display" in search_df.columns
        and "full_address" in search_df.columns
    ):
        _sid = search_df["search_input_display"]
        _has = (
            _sid.notna()
            & _sid.astype(str).str.strip().ne("")
            & ~_sid.astype(str).str.strip().str.lower().isin(("nan", "none", "<na>"))
        )
        search_df["full_address"] = _sid.where(_has, search_df["full_address"])

    # Preserve the search-side "in_existing" values even when the column name collides
    # with `new_join_col` (and thus gets renamed and later dropped).
    if existing_match_col:
        if existing_match_col in search_df.columns:
            search_df[f"__search_existing_{existing_match_col}"] = search_df[
                existing_match_col
            ]
        else:
            _alt = f"__search_side_{existing_match_col}"
            if _alt in search_df.columns:
                search_df[f"__search_existing_{existing_match_col}"] = search_df[_alt]

    for col in new_join_col:
        if col in search_df.columns:
            search_df = search_df.rename(columns={col: f"__search_side_{col}"})

    _full_match_raw = match_results_output.get(
        "full_match", pd.Series(False, index=match_results_output.index)
    )
    if _full_match_raw.dtype == object:
        _full_match_raw = _full_match_raw.where(_full_match_raw != "", pd.NA)
    full_match_series = _full_match_raw.astype("boolean").fillna(False).astype(bool)
    match_results_output_success = match_results_output[full_match_series]

    # If you're joining to the original df on index you will need to recreate the index again

    match_results_output_success = match_results_output_success.rename(
        columns={
            "reference_orig_address": "Reference matched address",
            "full_match": "Matched with reference address",
            "uprn": "UPRN",
        },
        errors="ignore",
    )

    ref_df_after_stand_cols = [
        "ref_index",
        "Reference matched address",
        "Matched with reference address",
        "Reference file",
        search_df_key_field,
    ]
    ref_df_after_stand_cols.extend(new_join_col)

    ref_df_after_stand_cols = [
        c for c in ref_df_after_stand_cols if c in match_results_output_success.columns
    ]
    if search_df_key_field not in ref_df_after_stand_cols:
        raise ValueError(
            f"Match results missing join key column {search_df_key_field!r}; "
            "cannot build results dataframe."
        )

    if search_df_key_field == "index":
        # Check index is int
        # match_results_output_success[search_df_key_field] = match_results_output_success[search_df_key_field].astype(float).astype(int)
        results_for_orig_df_join = search_df.merge(
            match_results_output_success[ref_df_after_stand_cols],
            on=search_df_key_field,
            how="left",
            suffixes=("", "_y"),
        )
    else:
        results_for_orig_df_join = search_df.merge(
            match_results_output_success[ref_df_after_stand_cols],
            how="left",
            on=search_df_key_field,
            suffixes=("", "_y"),
        )

    # If the join columns already exist in the search_df, then use the new column to fill in the NAs in the original column, then delete the new column

    if "Reference matched address_y" in results_for_orig_df_join.columns:
        results_for_orig_df_join["Reference matched address"] = (
            results_for_orig_df_join["Reference matched address"]
            .fillna(results_for_orig_df_join["Reference matched address_y"])
            .infer_objects(copy=False)
        )

    if "Matched with reference address_y" in results_for_orig_df_join.columns:
        results_for_orig_df_join["Matched with reference address"] = pd.Series(
            np.where(
                results_for_orig_df_join["Matched with reference address_y"].notna(),
                results_for_orig_df_join["Matched with reference address_y"],
                results_for_orig_df_join["Matched with reference address"],
            )
        )

    if "Reference file_y" in results_for_orig_df_join.columns:
        results_for_orig_df_join["Reference file"] = (
            results_for_orig_df_join["Reference file"]
            .fillna(results_for_orig_df_join["Reference file_y"])
            .infer_objects(copy=False)
        )

    if "UPRN_y" in results_for_orig_df_join.columns:
        results_for_orig_df_join["UPRN"] = (
            results_for_orig_df_join["UPRN"]
            .fillna(results_for_orig_df_join["UPRN_y"])
            .infer_objects(copy=False)
        )

    # For any join columns that exist in BOTH search_df and reference output,
    # always prefer the reference-derived values (the `_y` columns).
    for col in new_join_col:
        col_y = f"{col}_y"
        if col_y in results_for_orig_df_join.columns:
            results_for_orig_df_join[col] = results_for_orig_df_join[col_y]

    # Drop columns that aren't useful
    results_for_orig_df_join = results_for_orig_df_join.drop(
        [
            "Reference matched address_y",
            "Matched with reference address_y",
            "Reference file_y",
            "search_df_key_field_y",
            "UPRN_y",
            *[f"{col}_y" for col in new_join_col],
            *[f"__search_side_{col}" for col in new_join_col],
            *(  # internal preservation columns
                [f"__search_existing_{existing_match_col}"]
                if existing_match_col
                else []
            ),
            "index_y",
            "full_address_search",
            "postcode_search",
            "full_address_1",
            "full_address_2",
            "address_stand",
            "property_number",
            "prop_number" "flat_number" "apart_number" "first_sec_number" "room_number",
        ],
        axis=1,
        errors="ignore",
    )

    results_for_orig_df_join.rename(
        columns={"full_address": "Search data address"}, inplace=True
    )

    # Surface a dedicated existing-match column from the search data in the results output.
    if existing_match_col:
        _src = f"__search_existing_{existing_match_col}"
        if _src in results_for_orig_df_join.columns:
            results_for_orig_df_join[f"{existing_match_col} (from search data)"] = (
                results_for_orig_df_join[_src]
            )

    if ref_df_cleaned is not None and not ref_df_cleaned.empty:
        results_for_orig_df_join = fill_reference_join_columns_for_previously_matched(
            results_for_orig_df_join,
            ref_df_cleaned,
            new_join_col,
            existing_match_col=existing_match_col,
        )

    results_for_orig_df_join["index"] = results_for_orig_df_join["index"].astype(
        int, errors="ignore"
    )
    results_for_orig_df_join["ref_index"] = results_for_orig_df_join[
        "ref_index"
    ].astype(int, errors="ignore")

    # Replace blank strings with NA only on object/string columns to avoid pandas'
    # future downcasting warnings on DataFrame-wide `replace`.
    _obj_cols = results_for_orig_df_join.select_dtypes(
        include=["object", "string"]
    ).columns
    if len(_obj_cols) > 0:
        for _c in _obj_cols:
            _s = results_for_orig_df_join[_c]
            _mask_blank = _s.notna() & _s.astype(str).str.fullmatch(r"\s*", na=False)
            if _mask_blank.any():
                results_for_orig_df_join.loc[_mask_blank, _c] = np.nan

    present_join = [c for c in new_join_col if c in results_for_orig_df_join.columns]
    if present_join:
        _pj = results_for_orig_df_join[present_join].astype(str)
        _pj = _pj.apply(
            lambda c: c.str.replace(".0", "", regex=False).str.replace(
                "nan", "", regex=False
            )
        )
        results_for_orig_df_join[present_join] = _pj

    # Replace cells with only 'nan' with blank (object/string columns only)
    _obj_cols2 = results_for_orig_df_join.select_dtypes(
        include=["object", "string"]
    ).columns
    if len(_obj_cols2) > 0:
        for _c in _obj_cols2:
            _s = results_for_orig_df_join[_c]
            # Keep missing values missing; only normalise literal 'nan' strings.
            _mask_nan_str = _s.notna() & _s.astype(str).str.fullmatch(
                r"nan", case=False, na=False
            )
            if _mask_nan_str.any():
                results_for_orig_df_join.loc[_mask_nan_str, _c] = ""

    return results_for_orig_df_join
