import copy
import math
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import gradio as gr
import numpy as np
import pandas as pd
from tqdm import tqdm


# ---- Pandas compatibility helpers ----
def _bool_mask(series: pd.Series, *, default: bool = False) -> pd.Series:
    """
    Return a strict-boolean mask from a Series that may be bool/object/nullable.

    This avoids pandas FutureWarnings about silent downcasting when using `.fillna(...)`
    on object dtype columns.
    """
    if series is None:
        return pd.Series([], dtype=bool)

    s = series
    if not isinstance(s, pd.Series):
        s = pd.Series(s)

    if s.dtype == object:
        # Common pattern in this codebase: blank string used as "missing"
        s = s.where(s != "", pd.NA)

    try:
        return s.astype("boolean").fillna(default).astype(bool)
    except (TypeError, ValueError):
        # Last resort: interpret common truthy strings
        return (
            s.astype(str)
            .str.strip()
            .str.lower()
            .isin(("1", "true", "t", "yes", "y"))
            .fillna(default)
            .astype(bool)
        )


# ---- Output naming helpers ----
def _safe_file_id(value: Optional[str], max_len: int = 40) -> str:
    """
    Build a filesystem-friendly identifier from a filename-like input.
    Keeps letters/numbers/underscore/dash, collapses other characters to '_'.
    Truncates to ``max_len`` characters (default 40) to reduce collisions between runs.
    """
    if value is None:
        return "unknown"
    s = str(value).strip()
    if not s:
        return "unknown"
    s = os.path.basename(s)
    for ext in (".csv.gz", ".csv", ".tsv", ".txt", ".parquet", ".xlsx", ".xls"):
        if s.lower().endswith(ext):
            s = s[: -len(ext)]
            break
    s = re.sub(r"[^A-Za-z0-9_-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return (s or "unknown")[:max_len]


def _ensure_str_list(value: Optional[object]) -> List[str]:
    """
    Normalise column-name parameters from callers that may pass a single string.

    Downstream code expects a list of column names; a bare string must become a
    one-element list (otherwise iteration would walk characters of the string).
    """
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        out: List[str] = []
        for x in value:
            if x is None:
                continue
            s = str(x).strip()
            if s:
                out.append(s)
        return out
    if isinstance(value, str):
        s = value.strip()
        return [s] if s else []
    s = str(value).strip()
    return [s] if s else []


# API functions
from fuzzy_address_matcher.addressbase_api_funcs import places_api_query
from fuzzy_address_matcher.config import (
    MAX_PARALLEL_WORKERS,
    OUTPUT_FOLDER,
    PRINT_MATCH_STAGE_SUMMARY_TO_CONSOLE,
    RUN_BATCHES_IN_PARALLEL,
    SAVE_INTERIM_FILES,
    SAVE_OUTPUT_FILES,
    USE_EXISTING_STANDARDISED_FILES,
    USE_POSTCODE_BLOCKER,
    batch_size,
    max_predict_len,
    output_folder,
    ref_batch_size,
)
from fuzzy_address_matcher.config import (
    fuzzy_match_limit as config_fuzzy_match_limit,
)
from fuzzy_address_matcher.constants import (
    InitMatch,
    MatcherClass,
    run_nnet_match,
)
from fuzzy_address_matcher.fuzzy_match import (
    _create_fuzzy_match_results_output,
    add_fuzzy_block_sequence_col,
    create_results_df,
    string_match_by_post_code_multiple,
)
from fuzzy_address_matcher.helper_functions import (
    initial_data_load,
    sum_numbers_before_seconds,
)

# Neural network functions
### Predict function for imported model
from fuzzy_address_matcher.model_predict import (
    full_predict_func,
    full_predict_torch,
    post_predict_clean,
)

# Imports (must be module-level)
from fuzzy_address_matcher.preparation import (
    check_no_number_addresses,
    extract_postcode,
    extract_street_name,
    prepare_ref_address,
    prepare_search_address,
    prepare_search_address_string,
    remove_non_postal,
)
from fuzzy_address_matcher.recordlinkage_funcs import score_based_match
from fuzzy_address_matcher.secure_path_utils import secure_path_join
from fuzzy_address_matcher.standardise import (
    remove_postcode,
    standardise_address,
)

# (max_predict_len, MatcherClass imported above)

# Type aliases / module globals (must come after imports)
PandasDataFrame = pd.DataFrame
PandasSeries = pd.Series
MatchedResults = Dict[str, Tuple[str, int]]
array = List[str]

today = datetime.now().strftime("%d%m%Y")
today_rev = datetime.now().strftime("%Y%m%d")
today_month_rev = datetime.now().strftime("%Y%m")

# Constants
run_fuzzy_match = True
run_standardise = True

# Load in data functions


def detect_file_type(filename: str) -> str:
    """Detect the file type based on its extension."""
    if (
        (filename.endswith(".csv"))
        | (filename.endswith(".csv.gz"))
        | (filename.endswith(".zip"))
    ):
        return "csv"
    elif filename.endswith(".xlsx"):
        return "xlsx"
    elif filename.endswith(".parquet"):
        return "parquet"
    else:
        raise ValueError("Unsupported file type.")


def read_file(filename: str) -> PandasDataFrame:
    """Read the file based on its detected type and convert to Pandas Dataframe. Supports csv, xlsx, and parquet."""
    file_type = detect_file_type(filename)

    if file_type == "csv":
        return pd.read_csv(filename, low_memory=False)
    elif file_type == "xlsx":
        return pd.read_excel(filename)
    elif file_type == "parquet":
        return pd.read_parquet(filename)


def get_file_name(in_name: str) -> str:
    """Get basename from a path-like string, handling Windows/Unix separators."""
    if not in_name:
        return None
    in_name = str(in_name).strip()
    if not in_name:
        return None
    # Normalise separators first so basenames are extracted consistently even
    # when incoming paths use a separator that differs from the host OS.
    normalized = in_name.replace("\\", "/").rstrip("/")
    base = os.path.basename(normalized)
    return base or None


def _safe_file_stem(name: str, default: str) -> str:
    raw = (name or "").strip()
    if not raw:
        raw = default
    stem = os.path.splitext(raw)[0]
    for ch in (":", "*", "?", '"', "<", ">", "|", "\\", "/"):
        stem = stem.replace(ch, "_")
    return stem


def _stand_cache_path(
    output_folder: str, input_name: str, stand_kind: str, blocker_mode: str
) -> str:
    """
    input_name: original input file name (or a descriptive label like 'API').
    stand_kind: 'stand_min' or 'stand_full'.
    blocker_mode: 'postcode' or 'street' (cache isolation by match mode).
    Produces <output_folder>/<input_stem>_<stand_kind>_<blocker_mode>.parquet
    """
    stem = _safe_file_stem(input_name, default="data")
    out_dir = output_folder or ""
    if out_dir and not out_dir.endswith((os.sep, "/")):
        out_dir = out_dir + os.sep
    mode = _safe_file_stem(blocker_mode, default="postcode")
    return os.path.join(out_dir, f"{stem}_{stand_kind}_{mode}.parquet")


def _standardise_search_df(
    search_df_cleaned: PandasDataFrame, *, standardise: bool
) -> PandasDataFrame:
    # Mirror `standardise_wrapper_func` but only for the search side.
    full_address_col = search_df_cleaned["full_address"]
    if isinstance(full_address_col, pd.DataFrame):
        full_address_col = full_address_col.iloc[:, 0]

    postcode_col = search_df_cleaned["postcode"]
    if isinstance(postcode_col, pd.DataFrame):
        postcode_col = postcode_col.iloc[:, 0]

    df = search_df_cleaned.copy()
    df["full_address_search"] = full_address_col.astype(str).str.lower().str.strip()
    df["postcode_search"] = (
        postcode_col.astype(str)
        .str.lower()
        .str.strip()
        .str.replace(r"\s+", "", regex=True)
    )

    df.loc[
        df["Excluded from search"] == "Excluded - non-postal address",
        "postcode_search",
    ] = ""

    return standardise_address(
        df,
        "full_address_search",
        "search_address_stand",
        standardise=standardise,
        out_london=True,
    )


def _standardise_ref_df(
    ref_df_cleaned: PandasDataFrame, *, standardise: bool
) -> PandasDataFrame:
    # Mirror `standardise_wrapper_func` but only for the reference side (fuzzy task).
    df = ref_df_cleaned.copy()
    df["full_address_search"] = df["fulladdress"].str.lower().str.strip()

    if "Postcode" in df.columns:
        has_any_postcode = df["Postcode"].replace("", pd.NA).notna().any()
        if has_any_postcode:
            df = df[df["Postcode"].notna()]
        df["postcode_search"] = (
            df["Postcode"].str.lower().str.strip().str.replace(r"\s+", "", regex=True)
        )
    else:
        df["postcode_search"] = ""

    return standardise_address(
        df,
        "full_address_search",
        "ref_address_stand",
        standardise=standardise,
        out_london=True,
    )


def filter_not_matched(
    matched_results: pd.DataFrame, search_df: pd.DataFrame, key_col: str
) -> pd.DataFrame:
    """Filters search_df to only rows with key_col not in matched_results"""

    # Validate inputs
    if not isinstance(matched_results, pd.DataFrame):
        raise TypeError("not_matched_results must be a Pandas DataFrame")

    if not isinstance(search_df, pd.DataFrame):
        raise TypeError("search_df must be a Pandas DataFrame")

    if not isinstance(key_col, str):
        raise TypeError("key_col must be a string")

    # Empty batches or aborted stages can leave diagnostics without a key column.
    # Treat as "no successful matches recorded" and return the full search frame.
    if matched_results.empty or key_col not in matched_results.columns:
        return search_df.copy() if search_df is not None else search_df

    if "full_match" not in matched_results.columns:
        raise ValueError("full_match not a column in matched_results")

    full_match_series = _bool_mask(matched_results["full_match"], default=False)
    matched_results_success = matched_results[full_match_series]

    # Filter search_df

    matched = (
        search_df[key_col]
        .astype(str)
        .isin(matched_results_success[key_col].astype(str))
    )

    return search_df.iloc[np.where(~matched)[0]]


def add_search_data_existing_col_to_results(
    results_df: pd.DataFrame,
    existing_match_col: Optional[str],
) -> pd.DataFrame:
    """
    Expose the `in_existing` value from the search data as a clearly labelled column in the
    results dataframe. The value lives under either the original column name or under
    `__search_side_<name>` (when the name collides with a `new_join_col` and was renamed).

    The output column is named `"<existing_match_col> (from search data)"` so it appears
    alongside the reference-derived join column without confusion.
    """
    if not existing_match_col or results_df is None or results_df.empty:
        return results_df

    out = results_df.copy()
    output_col = f"{existing_match_col} (from search data)"

    # Find the source: prefer original name, then the __search_side_ alias
    src_col = None
    if existing_match_col in out.columns:
        src_col = existing_match_col
    else:
        alt = f"__search_side_{existing_match_col}"
        if alt in out.columns:
            src_col = alt

    if src_col is None:
        return out

    out[output_col] = out[src_col]
    return out


def copy_existing_match_col_to_join_cols(
    results_df: pd.DataFrame,
    existing_match_col: Optional[str],
    new_join_col: List[str],
) -> pd.DataFrame:
    """
    For rows flagged as 'Previously matched', the search data already holds the reference
    identifier in the `existing_match_col` column (i.e. the value the user chose as
    `in_existing` in the UI). Copy that value directly into the first `new_join_col` column
    so it appears in the results CSV.

    If `new_join_col[0]` and `existing_match_col` are the same name, the value may already
    be present under `__search_side_<name>` due to earlier renaming; we handle that too.
    """
    if (
        results_df is None
        or results_df.empty
        or not existing_match_col
        or not new_join_col
    ):
        return results_df

    out = results_df.copy()

    previously_matched_mask = (
        out.get("Excluded from search", pd.Series("", index=out.index))
        .fillna("")
        .astype(str)
        .eq("Previously matched")
    )
    if not previously_matched_mask.any():
        return out

    # Locate the source series: prefer direct name, then the __search_side_ alias.
    src_col = None
    if existing_match_col in out.columns:
        src_col = existing_match_col
    else:
        alt = f"__search_side_{existing_match_col}"
        if alt in out.columns:
            src_col = alt

    if src_col is None:
        return out

    source_series = out.loc[previously_matched_mask, src_col]

    # Copy into each new_join_col that is missing / blank for these rows.
    for jc in new_join_col:
        if jc not in out.columns:
            out[jc] = pd.NA
        current = out.loc[previously_matched_mask, jc].astype(str).str.strip()
        is_blank = current.isin(["", "nan", "None", "<NA>"])
        fill_idx = source_series.index[
            is_blank.values
            & source_series.notna().values
            & source_series.astype(str).str.strip().ne("").values
        ]
        if len(fill_idx):
            out.loc[fill_idx, jc] = source_series.loc[fill_idx].values

    return out


def _rename_search_side_join_columns_overlap(
    df: pd.DataFrame,
    new_join_col: List[str],
    *,
    key_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    If search data contains columns with the same names as `new_join_col`, rename them so
    merges and concat with reference-side values never keep the search copy by accident.
    """
    if df is None or df.empty or not new_join_col:
        return df
    # Never rename the key column used to merge back for display/results.
    rename_map = {
        c: f"__search_side_{c}"
        for c in new_join_col
        if (c in df.columns) and (c != key_col)
    }
    if rename_map:
        return df.rename(columns=rename_map)
    return df


def _normalize_join_key_strings(series: pd.Series) -> pd.Series:
    """
    Stable string labels for join keys so values like 6199, 6199.0, Int64(6199),
    and the string '6199.0' compare equal after standardisation / parquet round-trips.
    """
    if series is None or len(series) == 0:
        return pd.Series(dtype=object)
    n = pd.to_numeric(series, errors="coerce")
    whole = n.notna() & (n == n.round())
    out = series.astype(str).copy()
    out.loc[whole] = n.loc[whole].round().astype("Int64").astype(str)
    non_num = n.isna()
    out.loc[non_num] = series.loc[non_num].astype(str)
    return out


def _resolve_column_series(df: pd.DataFrame, col_name: str) -> Optional[pd.Series]:
    """
    Return a single Series for ``col_name``, even when duplicate labels make
    ``df[col_name]`` a DataFrame (take the first occurrence).
    """
    if df is None or (not isinstance(df, pd.DataFrame)) or (col_name not in df.columns):
        return None
    col = df[col_name]
    if isinstance(col, pd.DataFrame):
        if col.shape[1] == 0:
            return None
        col = col.iloc[:, 0]
    return col if isinstance(col, pd.Series) else None


def _slice_frame_by_normalized_keys(
    df: pd.DataFrame,
    key_field: Optional[str],
    key_values: list,
) -> pd.DataFrame:
    """
    Subset ``df`` to rows whose join key appears in ``key_values``.

    Cached standardisation parquets are written with ``index=False``, so frames
    reload with a positional RangeIndex while ``create_batch_ranges`` builds
    ``search_range`` / ``ref_range`` from the **original** index labels. Slicing
    with ``df.index.isin(search_range)`` then drops every row when labels do not
    match positions; matching on ``key_field`` (e.g. ``index`` / ``ref_index``)
    avoids that.
    """
    if df is None or (not isinstance(df, pd.DataFrame)) or df.empty:
        return df
    keys_norm = set(
        _normalize_join_key_strings(pd.Series(list(key_values), dtype=object)).tolist()
    )
    if key_field and key_field in df.columns:
        col = _resolve_column_series(df, key_field)
        if col is not None:
            mask = _normalize_join_key_strings(col).isin(keys_norm)
            return df.loc[mask].reset_index(drop=True)
    return df.loc[df.index.isin(key_values)].reset_index(drop=True)


def _fuzzy_match_debug_enabled() -> bool:
    return os.environ.get("FUZZY_MATCH_DEBUG", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "y",
    )


def _log_postcode_blocker_debug(
    *,
    standardise: bool,
    use_postcode_blocker: bool,
    search_df_after_stand: pd.DataFrame,
    ref_df_after_stand: pd.DataFrame,
    can_run_postcode_blocker: bool,
) -> None:
    if not _fuzzy_match_debug_enabled():
        return

    def _dup_count(d: pd.DataFrame, c: str) -> int:
        if c not in d.columns:
            return 0
        return int(d.columns.tolist().count(c))

    def _nonempty_series(s: Optional[pd.Series]) -> int:
        if s is None:
            return 0
        return int(s.fillna("").astype(str).str.strip().ne("").sum())

    s_ser = _resolve_column_series(search_df_after_stand, "postcode_search")
    r_ser = _resolve_column_series(ref_df_after_stand, "postcode_search")
    s_ok = _column_has_usable_values(search_df_after_stand, "postcode_search")
    r_ok = _column_has_usable_values(ref_df_after_stand, "postcode_search")
    print(
        "[FUZZY_MATCH_DEBUG] postcode blocker: "
        f"standardise={standardise} use_postcode_blocker={use_postcode_blocker} "
        f"can_run={can_run_postcode_blocker}"
    )
    print(
        f"  search: rows={len(search_df_after_stand)} "
        f"nonempty_postcode_search={_nonempty_series(s_ser)} "
        f"usable_col={s_ok} dup_postcode_search_labels={_dup_count(search_df_after_stand, 'postcode_search')}"
    )
    print(
        f"  ref: rows={len(ref_df_after_stand)} "
        f"nonempty_postcode_search={_nonempty_series(r_ser)} "
        f"usable_col={r_ok} dup_postcode_search_labels={_dup_count(ref_df_after_stand, 'postcode_search')}"
    )
    if not can_run_postcode_blocker:
        reasons = []
        if not use_postcode_blocker:
            reasons.append("use_postcode_blocker=False")
        if not s_ok:
            reasons.append("search postcode_search not usable")
        if not r_ok:
            reasons.append("ref postcode_search not usable")
        print(f"  reasons: {', '.join(reasons)}")


def _street_overflow_unbatched_search_enabled() -> bool:
    """
    When postcode batching is on, also run street-only batches for search rows that
    never appear in any postcode-overlap batch. Default: enabled. Set
    STREET_OVERFLOW_UNBATCHED_SEARCH=0 to disable (faster, fewer comparisons).
    """
    raw = os.environ.get("STREET_OVERFLOW_UNBATCHED_SEARCH")
    if raw is None or not str(raw).strip():
        return True
    return str(raw).strip().lower() not in ("0", "false", "no", "n", "off")


def _postcode_batch_covered_search_keys_normalized(range_df: pd.DataFrame) -> set[str]:
    """Normalized join-key strings appearing in any postcode batch ``search_range``."""
    covered: set[str] = set()
    if range_df is None or range_df.empty or "search_range" not in range_df.columns:
        return covered
    for _, r in range_df.iterrows():
        sr = r.get("search_range")
        if not sr:
            continue
        for v in list(sr):
            covered.update(
                _normalize_join_key_strings(pd.Series([v], dtype=object)).tolist()
            )
    return covered


def _uncovered_search_key_values_for_street_overflow(
    matcher: MatcherClass,
    covered_norm: set[str],
) -> List:
    """Raw key values for search rows whose normalized key is not in ``covered_norm``."""
    k = matcher.search_df_key_field or "index"
    ser = _resolve_column_series(matcher.search_df_cleaned, k)
    if ser is None:
        ser = pd.Series(
            matcher.search_df_cleaned.index, index=matcher.search_df_cleaned.index
        )
    norm = _normalize_join_key_strings(ser)
    return ser.loc[~norm.isin(covered_norm)].tolist()


def _matcher_search_side_sliced_to_keys(
    matcher: MatcherClass,
    key_values: list,
) -> MatcherClass:
    """Shallow copy of ``matcher`` with search-side frames filtered to ``key_values``."""
    out = copy.copy(matcher)
    _k = out.search_df_key_field or "index"
    keys = list(key_values)
    out.search_df = _slice_frame_by_normalized_keys(out.search_df, _k, keys)
    out.search_df_not_matched = out.search_df.copy()
    out.search_df_cleaned = _slice_frame_by_normalized_keys(
        out.search_df_cleaned, _k, keys
    )
    out.search_df_after_stand = _slice_frame_by_normalized_keys(
        out.search_df_after_stand, _k, keys
    )
    out.search_df_after_full_stand = _slice_frame_by_normalized_keys(
        out.search_df_after_full_stand, _k, keys
    )
    return out


def _column_has_usable_values(df: pd.DataFrame, col_name: str) -> bool:
    """Return True when the column exists and has at least one non-empty value."""
    col = _resolve_column_series(df, col_name)
    if col is None:
        return False
    cleaned_series = col.fillna("").astype(str).str.strip()
    return bool(cleaned_series.ne("").any())


def _strip_runtime_fuzzy_cols_from_stand_cache(df: pd.DataFrame) -> pd.DataFrame:
    """Drop per-run columns that must not be reused from parquet standardisation caches."""
    return df.drop(columns=["_fuzzy_block_seq"], errors="ignore")


def _resolve_parallel_worker_count(
    number_of_batches: int, max_parallel_workers: Optional[int], includes_nnet: bool
) -> int:
    """
    Resolve a safe worker count for process-based batch parallelism.
    Caps workers for NN runs to reduce memory pressure.
    """
    cpu_count = os.cpu_count() or 1
    default_workers = max(1, min(number_of_batches, max(1, cpu_count - 1)))
    workers = (
        default_workers
        if max_parallel_workers is None
        else max(1, min(int(max_parallel_workers), number_of_batches))
    )

    if includes_nnet:
        workers = min(workers, 2)

    return max(1, workers)


def run_single_match_batch_worker(
    batch_n: int,
    batch_match: MatcherClass,
    total_batches: int,
    use_postcode_blocker: bool,
    print_match_stage_summary_to_console: bool,
) -> Tuple[int, str, MatcherClass]:
    """
    Worker wrapper for parallel execution.
    Must avoid writing files to shared output paths.
    """
    summary, batch_out = run_single_match_batch(
        batch_match,
        batch_n,
        total_batches,
        use_postcode_blocker=use_postcode_blocker,
        write_outputs=False,
        show_progress=False,
        print_match_stage_summary_to_console=print_match_stage_summary_to_console,
    )
    return batch_n, summary, batch_out


def query_addressbase_api(
    in_api_key: str,
    Matcher: MatcherClass,
    query_type: str,
    output_folder: Optional[str] = None,
    progress=gr.Progress(track_tqdm=True),
):

    final_api_output_file_name = ""

    if in_api_key == "":
        print("No API key provided, please provide one to continue")
        return Matcher, final_api_output_file_name
    else:
        # Call the API
        # Matcher.ref_df = pd.DataFrame()

        # Check if the ref_df file already exists
        def check_and_create_api_folder():
            # Check if the environmental variable is available
            file_path = os.environ.get("ADDRESSBASE_API_OUT")

            if file_path is None:
                # Environmental variable is not set
                print("API output environmental variable not set.")
                # Create the 'api/' folder if it doesn't already exist
                api_folder_path = "api/"
                if not os.path.exists(api_folder_path):
                    os.makedirs(api_folder_path)
                print(f"'{api_folder_path}' folder created.")
            else:
                # Environmental variable is set
                api_folder_path = file_path
                print(f"Environmental variable found: {api_folder_path}")

            return api_folder_path

        base_output_folder = output_folder or output_folder
        if not base_output_folder.endswith(("\\", "/")):
            base_output_folder = base_output_folder + os.sep

        api_output_folder = check_and_create_api_folder()

        # Check if the file exists
        # print("Matcher file name: ", Matcher.file_name)
        search_file_name_without_extension = re.sub(r"\.[^.]+$", "", Matcher.file_name)

        api_ref_save_loc = (
            api_output_folder
            + search_file_name_without_extension
            + "_api_"
            + today_month_rev
            + "_"
            + query_type
            + "_ckpt"
        )
        print("API reference save location: ", api_ref_save_loc)

        final_api_output_file_name = api_ref_save_loc + ".parquet"

        # Allow for csv, parquet and gzipped csv files
        if os.path.isfile(api_ref_save_loc + ".csv"):
            print("API reference CSV file found")
            Matcher.ref_df = pd.read_csv(api_ref_save_loc + ".csv")
        elif os.path.isfile(final_api_output_file_name):
            print("API reference Parquet file found")
            Matcher.ref_df = pd.read_parquet(api_ref_save_loc + ".parquet")
        elif os.path.isfile(api_ref_save_loc + ".csv.gz"):
            print("API reference gzipped CSV file found")
            Matcher.ref_df = pd.read_csv(
                api_ref_save_loc + ".csv.gz", compression="gzip"
            )
        else:
            print("API reference file not found, querying API for reference data.")

        def conduct_api_loop(
            in_query,
            in_api_key,
            query_type,
            i,
            api_ref_save_loc,
            loop_list,
            api_search_index,
        ):
            ref_addresses = places_api_query(in_query, in_api_key, query_type)

            ref_addresses["Address_row_number"] = api_search_index[i]

            loop_list.append(ref_addresses)

            if (i + 1) % 500 == 0:
                print("Saving api call checkpoint for query:", str(i + 1))
                if SAVE_INTERIM_FILES:
                    pd.concat(loop_list).to_parquet(
                        os.path.join(base_output_folder, api_ref_save_loc + ".parquet"),
                        index=False,
                    )

            return loop_list

        def check_postcode(postcode):
            # Remove spaces on the ends or in the middle of the postcode, and any symbols
            cleaned_postcode = re.sub(r"[^\w\s]|[\s]", "", postcode)
            # Ensure that the postcode meets the specified format
            postcode_pattern = r"\b(?:[A-Z][A-HJ-Y]?[0-9][0-9A-Z]?[0-9][A-Z]{2}|GIR0AA|GIR0A{2}|[A-Z][A-HJ-Y]?[0-9][0-9A-Z]?[0-9]{1}?)\b"
            match = re.match(postcode_pattern, cleaned_postcode)
            if match and len(cleaned_postcode) in (6, 7):
                return cleaned_postcode  # Return the matched postcode string
            else:
                return None  # Return None if no match is found

        if query_type == "Address":
            save_file = True
            # Do an API call for each unique address

            if not Matcher.ref_df.empty:
                api_search_df = Matcher.search_df.copy().drop(
                    list(set(Matcher.ref_df["Address_row_number"]))
                )

            else:
                print("Matcher ref_df data empty")
                api_search_df = Matcher.search_df.copy()

            i = 0
            loop_df = Matcher.ref_df
            loop_list = [Matcher.ref_df]

            for address in tqdm(
                api_search_df["full_address_postcode"],
                desc="Making API calls",
                unit="addresses",
                total=len(api_search_df["full_address_postcode"]),
            ):
                print("Query number: " + str(i + 1), "with address: ", address)

                api_search_index = api_search_df.index

                loop_list = conduct_api_loop(
                    address,
                    in_api_key,
                    query_type,
                    i,
                    api_ref_save_loc,
                    loop_list,
                    api_search_index,
                )

                i += 1

            loop_df = pd.concat(loop_list)
            Matcher.ref_df = loop_df.drop_duplicates(keep="first", ignore_index=True)

        elif query_type == "Postcode":
            save_file = True
            # Do an API call for each unique postcode. Each API call can only return 100 results maximum :/

            if not Matcher.ref_df.empty:
                print("Excluding postcodes that already exist in API call data.")

                # Retain original index values after filtering
                Matcher.search_df["index_keep"] = Matcher.search_df.index

                if (
                    "invalid_request" in Matcher.ref_df.columns
                    and "Address_row_number" in Matcher.ref_df.columns
                ):
                    print("Joining on invalid_request column")
                    Matcher.search_df = Matcher.search_df.merge(
                        Matcher.ref_df[
                            ["Address_row_number", "invalid_request"]
                        ].drop_duplicates(subset="Address_row_number"),
                        left_on=Matcher.search_df_key_field,
                        right_on="Address_row_number",
                        how="left",
                    )

                elif "invalid_request" not in Matcher.search_df.columns:
                    Matcher.search_df["invalid_request"] = False

                postcode_col = Matcher.search_postcode_col[0]

                # Check ref_df df against cleaned and non-cleaned postcodes
                Matcher.search_df[postcode_col] = Matcher.search_df[
                    postcode_col
                ].astype(str)
                search_df_cleaned_pcodes = Matcher.search_df[postcode_col].apply(
                    check_postcode
                )
                ref_df_cleaned_pcodes = (
                    Matcher.ref_df["POSTCODE_LOCATOR"].dropna().apply(check_postcode)
                )

                api_search_df = Matcher.search_df.copy().loc[
                    ~Matcher.search_df[postcode_col].isin(
                        Matcher.ref_df["POSTCODE_LOCATOR"]
                    )
                    & ~(Matcher.search_df["invalid_request"])
                    & ~(search_df_cleaned_pcodes.isin(ref_df_cleaned_pcodes)),
                    :,
                ]

                # api_search_index = api_search_df["index_keep"]
                # api_search_df.index = api_search_index

                print(
                    "Remaining invalid request count: ",
                    Matcher.search_df["invalid_request"].value_counts(),
                )

            else:
                print("Matcher ref_df data empty")
                api_search_df = Matcher.search_df.copy()
                api_search_index = api_search_df.index
                api_search_df["index_keep"] = api_search_index

                postcode_col = Matcher.search_postcode_col[0]

            unique_pcodes = api_search_df.loc[
                :, ["index_keep", postcode_col]
            ].drop_duplicates(subset=[postcode_col], keep="first")
            print("Unique postcodes: ", unique_pcodes[postcode_col])

            # Apply the function to each postcode in the Series
            unique_pcodes["cleaned_unique_postcodes"] = unique_pcodes[
                postcode_col
            ].apply(check_postcode)

            # Filter out the postcodes that comply with the specified format
            valid_unique_postcodes = unique_pcodes.dropna(
                subset=["cleaned_unique_postcodes"]
            )

            valid_postcode_search_index = valid_unique_postcodes["index_keep"]
            valid_postcode_search_index_list = valid_postcode_search_index.tolist()

            if not valid_unique_postcodes.empty:

                print("Unique valid postcodes: ", valid_unique_postcodes)
                print("Number of unique valid postcodes: ", len(valid_unique_postcodes))

                tic = time.perf_counter()

                i = 0
                loop_df = Matcher.ref_df
                loop_list = [Matcher.ref_df]

                for pcode in progress.tqdm(
                    valid_unique_postcodes["cleaned_unique_postcodes"],
                    desc="Making API calls",
                    unit="unique postcodes",
                    total=len(valid_unique_postcodes["cleaned_unique_postcodes"]),
                ):
                    # api_search_index = api_search_df.index

                    print(
                        "Query number: " + str(i + 1),
                        " with postcode: ",
                        pcode,
                        " and index: ",
                        valid_postcode_search_index_list[i],
                    )

                    loop_list = conduct_api_loop(
                        pcode,
                        in_api_key,
                        query_type,
                        i,
                        api_ref_save_loc,
                        loop_list,
                        valid_postcode_search_index_list,
                    )

                    i += 1

                loop_df = pd.concat(loop_list)
                Matcher.ref_df = loop_df.drop_duplicates(
                    keep="first", ignore_index=True
                )

                toc = time.perf_counter()
                print("API call time (seconds): ", round(toc - tic, 1))
            else:
                print("No valid postcodes found.")

        elif query_type == "UPRN":
            save_file = True
            # Do an API call for each unique address

            if not Matcher.ref_df.empty:
                api_search_df = Matcher.search_df.copy().drop(
                    list(set(Matcher.ref_df["Address_row_number"]))
                )

            else:
                print("Matcher ref_df data empty")
                api_search_df = Matcher.search_df.copy()

            i = 0
            loop_df = Matcher.ref_df
            loop_list = [Matcher.ref_df]
            uprn_check_col = "ADR_UPRN"

            for uprn in progress.tqdm(
                api_search_df[uprn_check_col],
                desc="Making API calls",
                unit="UPRNs",
                total=len(api_search_df[uprn_check_col]),
            ):
                print("Query number: " + str(i + 1), "with address: ", uprn)

                api_search_index = api_search_df.index

                loop_list = conduct_api_loop(
                    uprn,
                    in_api_key,
                    query_type,
                    i,
                    api_ref_save_loc,
                    loop_list,
                    api_search_index,
                )

                i += 1

            loop_df = pd.concat(loop_list)
            Matcher.ref_df = loop_df.drop_duplicates(keep="first", ignore_index=True)

        else:
            print("Reference file loaded from file, no API calls made.")
            save_file = False

        # Post API call processing

        Matcher.ref_name = "API"
        # Matcher.ref_df = Matcher.ref_df.reset_index(drop=True)
        Matcher.ref_df["Reference file"] = Matcher.ref_name

        if query_type == "Postcode":
            # print(Matcher.ref_df.columns)

            cols_of_interest = [
                "ADDRESS",
                "ORGANISATION",
                "SAO_TEXT",
                "SAO_START_NUMBER",
                "SAO_START_SUFFIX",
                "SAO_END_NUMBER",
                "SAO_END_SUFFIX",
                "PAO_TEXT",
                "PAO_START_NUMBER",
                "PAO_START_SUFFIX",
                "PAO_END_NUMBER",
                "PAO_END_SUFFIX",
                "STREET_DESCRIPTION",
                "TOWN_NAME",
                "ADMINISTRATIVE_AREA",
                "LOCALITY_NAME",
                "POSTCODE_LOCATOR",
                "UPRN",
                "PARENT_UPRN",
                "USRN",
                "LPI_KEY",
                "RPC",
                "LOGICAL_STATUS_CODE",
                "CLASSIFICATION_CODE",
                "LOCAL_CUSTODIAN_CODE",
                "COUNTRY_CODE",
                "POSTAL_ADDRESS_CODE",
                "BLPU_STATE_CODE",
                "LAST_UPDATE_DATE",
                "ENTRY_DATE",
                "STREET_STATE_CODE",
                "STREET_CLASSIFICATION_CODE",
                "LPI_LOGICAL_STATUS_CODE",
                "invalid_request",
                "Address_row_number",
                "Reference file",
            ]

            try:
                # Attempt to select only the columns of interest
                Matcher.ref_df = Matcher.ref_df[cols_of_interest]
            except KeyError as e:
                missing_columns = [
                    col
                    for col in e.args[0][1:-1].split(", ")
                    if col not in cols_of_interest
                ]
                # Handle the missing columns gracefully
                print(f"Some columns are missing: {missing_columns}")

            # if "LOCAL_CUSTODIAN_CODE" in Matcher.ref_df.columns:
            # These are items that are 'owned' by Ordnance Survey like telephone boxes, bus shelters
            # Matcher.ref_df = Matcher.ref_df.loc[Matcher.ref_df["LOCAL_CUSTODIAN_CODE"] != 7655,:]

        if save_file:
            os.makedirs(base_output_folder, exist_ok=True)
            final_api_output_file_name_pq = os.path.join(
                base_output_folder, api_ref_save_loc[:-5] + ".parquet"
            )
            final_api_output_file_name = os.path.join(
                base_output_folder, api_ref_save_loc[:-5] + ".csv"
            )
            print("Saving reference file to: " + api_ref_save_loc[:-5] + ".parquet")
            if SAVE_INTERIM_FILES:
                # Optional checkpoint/alternate format outputs for debugging/restarts.
                Matcher.ref_df.to_parquet(
                    os.path.join(base_output_folder, api_ref_save_loc + ".parquet"),
                    index=False,
                )
                Matcher.ref_df.to_parquet(final_api_output_file_name_pq, index=False)
            Matcher.ref_df.to_csv(final_api_output_file_name)

        if Matcher.ref_df.empty:
            print("No reference data found with API")
            return Matcher

    return Matcher, final_api_output_file_name


def load_ref_data(
    Matcher: MatcherClass,
    ref_data_state: PandasDataFrame,
    in_ref: List[str],
    in_refcol: List[str],
    in_api: List[str],
    in_api_key: str,
    query_type: str,
    use_postcode_blocker: bool = True,
    output_folder: Optional[str] = None,
    progress=gr.Progress(track_tqdm=True),
):
    """
    Check for reference address data, do some preprocessing, and load in from the Addressbase API if required.
    """
    final_api_output_file_name = ""

    # Prefer explicit uploaded/reference file name when available, even when
    # dataframe state is already populated from a prior step.
    if in_ref:
        inferred_ref_name = get_file_name(in_ref[0].name)
        if inferred_ref_name:
            Matcher.ref_name = inferred_ref_name

    # Check if reference data loaded, bring in if already there
    if not ref_data_state.empty:
        Matcher.ref_df = ref_data_state
        if in_ref:
            Matcher.ref_name = get_file_name(in_ref[0].name)

        if not Matcher.ref_name:
            Matcher.ref_name = ""

        Matcher.ref_df["Reference file"] = Matcher.ref_name

    # Otherwise check for file name and load in. If nothing found, fail
    else:
        Matcher.ref_df = pd.DataFrame()

        if not in_ref:
            if not in_api:
                print("No reference file provided, please provide one to continue")
                return Matcher, final_api_output_file_name
            # Check if api call required and api key is provided
            else:
                Matcher, final_api_output_file_name = query_addressbase_api(
                    in_api_key,
                    Matcher,
                    query_type,
                    output_folder=output_folder,
                )

        else:
            Matcher.ref_name = get_file_name(in_ref[0].name)
            if not Matcher.ref_name:
                Matcher.ref_name = ""

            # Concatenate all in reference files together
            for ref_file in in_ref:
                # print(ref_file.name)
                temp_ref_file = read_file(ref_file.name)

                file_name_out = get_file_name(ref_file.name)
                temp_ref_file["Reference file"] = file_name_out

                Matcher.ref_df = pd.concat([Matcher.ref_df, temp_ref_file])

    # For the neural net model to work, the llpg columns have to be in the LPI format (e.g. with columns SaoText, SaoStartNumber etc. Here we check if we have that format.

    if "Address_LPI" in Matcher.ref_df.columns:
        Matcher.ref_df = Matcher.ref_df.rename(
            columns={
                "Name_LPI": "PaoText",
                "Num_LPI": "PaoStartNumber",
                "Num_Suffix_LPI": "PaoStartSuffix",
                "Number End_LPI": "PaoEndNumber",
                "Number_End_Suffix_LPI": "PaoEndSuffix",
                "Secondary_Name_LPI": "SaoText",
                "Secondary_Num_LPI": "SaoStartNumber",
                "Secondary_Num_Suffix_LPI": "SaoStartSuffix",
                "Secondary_Num_End_LPI": "SaoEndNumber",
                "Secondary_Num_End_Suffix_LPI": "SaoEndSuffix",
                "Postcode_LPI": "Postcode",
                "Postal_Town_LPI": "PostTown",
                "UPRN_BLPU": "UPRN",
            }
        )

    # print("Matcher reference file: ", Matcher.ref_df['Reference file'])

    # Check if the source is the Addressbase places API
    if Matcher.ref_df is not None and len(Matcher.ref_df) > 0:
        first_row = Matcher.ref_df.iloc[0]
        # print(first_row)
        if first_row is not None and "Reference file" in first_row:
            if (
                first_row["Reference file"] == "API"
                or "_api_" in first_row["Reference file"]
            ):
                Matcher.ref_df = Matcher.ref_df.rename(
                    columns={
                        "ORGANISATION_NAME": "Organisation",
                        "ORGANISATION": "Organisation",
                        "PAO_TEXT": "PaoText",
                        "PAO_START_NUMBER": "PaoStartNumber",
                        "PAO_START_SUFFIX": "PaoStartSuffix",
                        "PAO_END_NUMBER": "PaoEndNumber",
                        "PAO_END_SUFFIX": "PaoEndSuffix",
                        "STREET_DESCRIPTION": "Street",
                        "SAO_TEXT": "SaoText",
                        "SAO_START_NUMBER": "SaoStartNumber",
                        "SAO_START_SUFFIX": "SaoStartSuffix",
                        "SAO_END_NUMBER": "SaoEndNumber",
                        "SAO_END_SUFFIX": "SaoEndSuffix",
                        "POSTCODE_LOCATOR": "Postcode",
                        "TOWN_NAME": "PostTown",
                        "LOCALITY_NAME": "LocalityName",
                        "ADMINISTRATIVE_AREA": "AdministrativeArea",
                    },
                    errors="ignore",
                )

    # Check ref_df file format
    # If standard format, or it's an API call
    if "SaoText" in Matcher.ref_df.columns or in_api:
        Matcher.standard_llpg_format = True
        Matcher.ref_address_cols = [
            "Organisation",
            "SaoStartNumber",
            "SaoStartSuffix",
            "SaoEndNumber",
            "SaoEndSuffix",
            "SaoText",
            "PaoStartNumber",
            "PaoStartSuffix",
            "PaoEndNumber",
            "PaoEndSuffix",
            "PaoText",
            "Street",
            "PostTown",
            "Postcode",
        ]
        # Add columns from the list if they don't exist
        for col in Matcher.ref_address_cols:
            if col not in Matcher.ref_df:
                Matcher.ref_df[col] = np.nan
    else:
        Matcher.standard_llpg_format = False
        Matcher.ref_address_cols = in_refcol
        # When postcode blocking is requested, treat the last selected column as the
        # postcode column (existing behaviour). When it is not requested, we only need
        # address columns — do not force a rename to "Postcode" unless the reference data
        # already contains one or the user has explicitly included a postcode column.
        if use_postcode_blocker and len(Matcher.ref_address_cols) == 1:
            # Single-column reference input in postcode mode:
            # extract a postcode from the single address column and store it in "Postcode".
            addr_col = Matcher.ref_address_cols[0]
            if "Postcode" not in Matcher.ref_df.columns:
                pc_df = extract_postcode(Matcher.ref_df, addr_col)
                try:
                    pc_series = pc_df.dropna(axis=1).iloc[:, 0]
                except Exception:
                    pc_series = pd.Series("", index=Matcher.ref_df.index)
                Matcher.ref_df["Postcode"] = pc_series.fillna("").astype(str)
            # Remove any embedded postcode from the address string so it doesn't pollute matching.
            Matcher.ref_df[addr_col] = remove_postcode(Matcher.ref_df, addr_col)
            # Downstream expects "Postcode" to exist; keep address col(s) plus "Postcode".
            Matcher.ref_address_cols = [addr_col, "Postcode"]
        else:
            last_col = Matcher.ref_address_cols[-1] if Matcher.ref_address_cols else ""
            last_col_looks_like_postcode = last_col.lower() in (
                "postcode",
                "post_code",
                "pcode",
                "postalcode",
                "postal_code",
            )
            if use_postcode_blocker or last_col_looks_like_postcode:
                Matcher.ref_df = Matcher.ref_df.rename(columns={last_col: "Postcode"})
                Matcher.ref_address_cols[-1] = "Postcode"
            else:
                # Street-only mode: add a blank Postcode column so downstream code that
                # expects the column doesn't crash, but it will be ignored during matching.
                if "Postcode" not in Matcher.ref_df.columns:
                    Matcher.ref_df["Postcode"] = ""

    # Reset index for ref_df as multiple files may have been combined with identical indices
    Matcher.ref_df = (
        Matcher.ref_df.reset_index()
    )  # .drop(["index","level_0"], axis = 1, errors="ignore").reset_index().drop(["index","level_0"], axis = 1, errors="ignore")
    Matcher.ref_df.index.name = "index"

    return Matcher, final_api_output_file_name


def load_match_data_and_filter(
    Matcher: MatcherClass,
    data_state: PandasDataFrame,
    results_data_state: PandasDataFrame,
    in_file: List[str],
    in_text: str,
    in_colnames: List[str],
    in_joincol: List[str],
    in_existing: List[str],
    in_api: List[str],
    use_postcode_blocker: bool = True,
):
    """
    Check if data to be matched exists. Filter it according to which records are relevant in the reference dataset
    """

    # Assign join field if not known
    if not Matcher.search_df_key_field:
        Matcher.search_df_key_field = "index"

    # Set search address cols as entered column names
    # print("In colnames in check match data: ", in_colnames)
    Matcher.search_address_cols = in_colnames

    # Prefer explicit uploaded search filename when available, even when
    # dataframe state is already populated from a prior step.
    if in_file:
        file_list = [string.name for string in in_file]
        data_file_names = [
            string for string in file_list if "results_" not in string.lower()
        ]
        if data_file_names:
            inferred_file_name = get_file_name(data_file_names[0])
            if inferred_file_name:
                Matcher.file_name = inferred_file_name

    # Check if data loaded already and bring it in
    if not data_state.empty:

        Matcher.search_df = data_state
        Matcher.search_df["index"] = Matcher.search_df.reset_index().index

    else:
        Matcher.search_df = pd.DataFrame()

    # If a single address entered into the text box, just load this instead
    if in_text:
        (
            Matcher.search_df,
            Matcher.search_df_key_field,
            Matcher.search_address_cols,
            Matcher.search_postcode_col,
        ) = prepare_search_address_string(in_text)

    # If no file loaded yet and a file has been selected
    if Matcher.search_df.empty and in_file:
        (
            output_message,
            drop1,
            drop2,
            Matcher.search_df,
            results_data_state,
            _data_file_names_end,
        ) = initial_data_load(in_file)

        file_list = [string.name for string in in_file]
        data_file_names = [
            string for string in file_list if "results_" not in string.lower()
        ]
        if not data_file_names:
            raise ValueError(
                "No search data file found after excluding results files. "
                "Please provide at least one non-results input file."
            )
        Matcher.file_name = get_file_name(data_file_names[0])

        # search_df makes column to use as index
        Matcher.search_df["index"] = Matcher.search_df.index

    # Join previously created results file onto search_df if previous results file exists
    if not results_data_state.empty:

        print("Joining on previous results file")
        _results_state = results_data_state.copy()
        if "index" in _results_state.columns:
            # Normalise key dtype and enforce one row per key to avoid many-to-many
            # row multiplication when merging prior results onto current search data.
            Matcher.search_df["index"] = Matcher.search_df["index"].astype(str)
            _results_state["index"] = _results_state["index"].astype(str)
            _dup_count = int(_results_state.duplicated(subset=["index"]).sum())
            if _dup_count > 0:
                print(
                    f"Prior results file has {_dup_count} duplicate index values; "
                    "collapsing to last occurrence per index before merge."
                )
                _results_state = _results_state.drop_duplicates(
                    subset=["index"], keep="last"
                )

        Matcher.results_on_orig_df = _results_state.copy()
        Matcher.search_df = Matcher.search_df.merge(
            _results_state, on="index", how="left"
        )

    # If no join on column suggested, assume the user wants the UPRN
    if not in_joincol:
        Matcher.new_join_col = ["UPRN"]

    else:
        Matcher.new_join_col = in_joincol

    if len(in_colnames) > 1:
        Matcher.search_postcode_col = [in_colnames[-1]]

        # print("Postcode col: ", Matcher.search_postcode_col)

    elif len(in_colnames) == 1:
        Matcher.search_df["full_address_postcode"] = Matcher.search_df[in_colnames[0]]
        Matcher.search_postcode_col = ["full_address_postcode"]
        # Do NOT append to search_address_cols: with a single input column this would
        # cause the address join step to concatenate the address to itself.

    search_postcode_col_name = (
        Matcher.search_postcode_col[0] if Matcher.search_postcode_col else ""
    )
    has_search_postcode_data = bool(
        search_postcode_col_name
    ) and _column_has_usable_values(Matcher.search_df, search_postcode_col_name)
    has_ref_postcode_data = _column_has_usable_values(Matcher.ref_df, "Postcode")
    requested_postcode_blocker = bool(use_postcode_blocker)
    can_use_postcode_blocker = (
        requested_postcode_blocker
        and has_search_postcode_data
        and has_ref_postcode_data
    )

    Matcher.use_postcode_blocker_requested = requested_postcode_blocker
    Matcher.use_postcode_blocker_effective = can_use_postcode_blocker

    # Check for column that indicates there are existing matches. The code will then search this column for entries, and will remove them from the data to be searched
    Matcher.existing_match_cols = in_existing

    previously_matched = pd.Series(False, index=Matcher.search_df.index)

    if in_existing:
        existing_col = in_existing[0] if isinstance(in_existing, list) else in_existing
        existing_series = Matcher.search_df[existing_col]

        # Treat blanks / None-like strings as NOT pre-existing matches
        existing_series_str = existing_series.astype(str).str.strip()
        is_blank_or_none_like = existing_series_str.isin(
            ["", "none", "nan", "null", "<na>"]
        )
        previously_matched = (~existing_series.isna()) & (~is_blank_or_none_like)
        if "Matched with reference address" in Matcher.search_df.columns:
            Matcher.search_df.loc[
                previously_matched, "Matched with reference address"
            ] = True
        else:
            Matcher.search_df["Matched with reference address"] = previously_matched

    print("Shape of ref_df before filtering is: ", Matcher.ref_df.shape)
    print("Shape of search_df before filtering: ", Matcher.search_df.shape)

    ### Filter addresses to those with length > 0
    zero_length_search_df = Matcher.search_df.copy()[Matcher.search_address_cols]
    zero_length_search_df = zero_length_search_df.fillna("").infer_objects(copy=False)
    # Join with spaces (not raw string concatenation) so excluded rows keep a readable
    # "Search data address" when this helper column is used downstream.
    Matcher.search_df["address_cols_joined"] = (
        zero_length_search_df.astype(str)
        .apply(lambda row: " ".join(row.values), axis=1)
        .str.replace(r"\s{2,}", " ", regex=True)
        .str.strip()
    )

    length_more_than_0 = Matcher.search_df["address_cols_joined"].str.len() > 0

    # Ensure Excluded from search exists so it can be propagated to outputs
    if "Excluded from search" not in Matcher.search_df.columns:
        Matcher.search_df["Excluded from search"] = "Included in search"
    Matcher.search_df.loc[previously_matched, "Excluded from search"] = (
        "Previously matched"
    )

    ### Filter addresses to match to postcode areas present in both search_df and ref_df_cleaned only (postcode without the last three characters). Only run if API call is false. When the API is called, relevant addresses and postcodes should be brought in by the API.
    if not in_api:
        if (Matcher.filter_to_lambeth_pcodes) and can_use_postcode_blocker:
            Matcher.search_df["postcode_search_area"] = (
                Matcher.search_df[Matcher.search_postcode_col[0]]
                .str.strip()
                .str.upper()
                .str.replace(" ", "")
                .str[:-2]
            )
            Matcher.ref_df["postcode_search_area"] = (
                Matcher.ref_df["Postcode"]
                .str.strip()
                .str.upper()
                .str.replace(" ", "")
                .str[:-2]
            )

            # Keep valid outward-code style areas, including one-letter areas
            # (e.g. E2 -> "E20" after current normalisation/slicing).
            unique_ref_pcode_area = (
                Matcher.ref_df["postcode_search_area"][
                    Matcher.ref_df["postcode_search_area"].str.len() > 2
                ]
            ).unique()
            postcode_found_in_search = Matcher.search_df["postcode_search_area"].isin(
                unique_ref_pcode_area
            )

            # Do not overwrite the previously matched flag set above
            Matcher.search_df.loc[
                ~(postcode_found_in_search), "Excluded from search"
            ] = "Postcode area not found"
            Matcher.search_df.loc[~(length_more_than_0), "Excluded from search"] = (
                "Address length 0"
            )
            Matcher.pre_filter_search_df = (
                Matcher.search_df.copy()
            )  # .drop(["index", "level_0"], axis = 1, errors = "ignore").reset_index()
            # Matcher.pre_filter_search_df = Matcher.pre_filter_search_df.drop("address_cols_joined", axis = 1)

            Matcher.excluded_df = Matcher.search_df.copy()[
                ~(postcode_found_in_search)
                | ~(length_more_than_0)
                | (previously_matched)
            ]
            Matcher.search_df = Matcher.search_df[
                (postcode_found_in_search)
                & (length_more_than_0)
                & ~(previously_matched)
            ]

            # Exclude records that have already been matched separately, i.e. if 'Matched with reference address' column exists, and has trues in it
            if "Matched with reference address" in Matcher.search_df.columns:
                previously_matched = Matcher.pre_filter_search_df[
                    "Matched with reference address"
                ]
                Matcher.pre_filter_search_df.loc[
                    previously_matched, "Excluded from search"
                ] = "Previously matched"

                Matcher.excluded_df = Matcher.search_df.copy()[
                    ~(postcode_found_in_search)
                    | ~(length_more_than_0)
                    | (previously_matched)
                ]
                Matcher.search_df = Matcher.search_df[
                    (postcode_found_in_search)
                    & (length_more_than_0)
                    & ~(previously_matched)
                ]

            else:
                Matcher.excluded_df = Matcher.search_df.copy()[
                    ~(postcode_found_in_search) | ~(length_more_than_0)
                ]
                Matcher.search_df = Matcher.search_df[
                    (postcode_found_in_search) & (length_more_than_0)
                ]

            unique_search_pcode_area = (
                Matcher.search_df["postcode_search_area"]
            ).unique()
            postcode_found_in_ref = Matcher.ref_df["postcode_search_area"].isin(
                unique_search_pcode_area
            )
            Matcher.ref_df = Matcher.ref_df[postcode_found_in_ref]

            Matcher.pre_filter_search_df = Matcher.pre_filter_search_df.drop(
                "postcode_search_area", axis=1
            )
            Matcher.search_df = Matcher.search_df.drop("postcode_search_area", axis=1)
            Matcher.ref_df = Matcher.ref_df.drop("postcode_search_area", axis=1)
            Matcher.excluded_df = Matcher.excluded_df.drop(
                "postcode_search_area", axis=1
            )
        else:
            if (
                Matcher.filter_to_lambeth_pcodes
                and requested_postcode_blocker
                and not can_use_postcode_blocker
            ):
                print(
                    "Postcode blocker requested but postcode data is unavailable. Falling back to street-only matching."
                )

            Matcher.pre_filter_search_df = Matcher.search_df.copy()
            Matcher.search_df.loc[~(length_more_than_0), "Excluded from search"] = (
                "Address length 0"
            )

            Matcher.excluded_df = Matcher.search_df[
                ~(length_more_than_0) | (previously_matched)
            ]
            Matcher.search_df = Matcher.search_df[
                (length_more_than_0) & ~(previously_matched)
            ]

    Matcher.search_df = Matcher.search_df.drop(
        "address_cols_joined", axis=1, errors="ignore"
    )
    Matcher.excluded_df = Matcher.excluded_df.drop(
        "address_cols_joined", axis=1, errors="ignore"
    )

    Matcher.search_df_not_matched = Matcher.search_df

    # If this is for an API call, we need to convert the search_df address columns to one column now. This is so the API call can be made and the reference dataframe created.
    if in_api:

        if in_file:
            _loaded = initial_data_load(in_file)
            # `initial_data_load` may return 5 or 6 values depending on UI needs.
            # We only need the first 5 here.
            output_message, drop1, drop2, df, results_data_state = _loaded[:5]

            file_list = [string.name for string in in_file]
            data_file_names = [
                string for string in file_list if "results_" not in string.lower()
            ]

            Matcher.file_name = get_file_name(data_file_names[0])

            print("File list in in_api bit: ", file_list)
            print("data_file_names in in_api bit: ", data_file_names)
            print("Matcher.file_name in in_api bit: ", Matcher.file_name)

        else:
            if in_text:
                Matcher.file_name = in_text
            else:
                Matcher.file_name = "API call"

        # Exclude records that have already been matched separately, i.e. if 'Matched with reference address' column exists, and has trues in it
        if in_existing:
            print("Checking for previously matched records")
            Matcher.pre_filter_search_df = Matcher.search_df.copy()
            previously_matched = ~Matcher.pre_filter_search_df[in_existing].isnull()
            Matcher.pre_filter_search_df.loc[
                previously_matched, "Excluded from search"
            ] = "Previously matched"

            Matcher.excluded_df = Matcher.search_df.copy()[
                ~(length_more_than_0) | (previously_matched)
            ]
            Matcher.search_df = Matcher.search_df[
                (length_more_than_0) & ~(previously_matched)
            ]

        if isinstance(Matcher.search_df, str):
            search_df_cleaned, search_df_key_field, search_address_cols = (
                prepare_search_address_string(Matcher.search_df)
            )
        else:
            search_df_cleaned = prepare_search_address(
                Matcher.search_df,
                Matcher.search_address_cols,
                Matcher.search_postcode_col,
                Matcher.search_df_key_field,
            )

        Matcher.search_df["full_address_postcode"] = search_df_cleaned["full_address"]

    Matcher.pre_filter_search_df = _rename_search_side_join_columns_overlap(
        Matcher.pre_filter_search_df,
        Matcher.new_join_col or [],
        key_col=Matcher.search_df_key_field,
    )
    Matcher.excluded_df = _rename_search_side_join_columns_overlap(
        Matcher.excluded_df,
        Matcher.new_join_col or [],
        key_col=Matcher.search_df_key_field,
    )

    return Matcher


def load_matcher_data(
    in_text: str,
    in_file: str,
    in_ref: str,
    data_state: PandasDataFrame,
    results_data_state: PandasDataFrame,
    ref_data_state: PandasDataFrame,
    in_colnames: list,
    in_refcol: list,
    in_joincol: list,
    in_existing: list,
    Matcher: MatcherClass,
    in_api: str,
    in_api_key: str,
    use_postcode_blocker: bool = True,
    output_folder: Optional[str] = None,
) -> tuple:
    """
    Load and preprocess user inputs from the Gradio interface for address matching.

    This function standardises all input types (single address string, file uploads, etc.) into a consistent data format
    suitable for downstream fuzzy matching. It handles both search and reference data, including API-based reference data retrieval
    if requested.

    Args:
        in_text (str): Single address input as text, if provided.
        in_file: Uploaded file(s) containing addresses to match.
        in_ref: Uploaded reference file(s) or None if using API.
        data_state (PandasDataFrame): Current state of the search data.
        results_data_state (PandasDataFrame): Current state of the results data.
        ref_data_state (PandasDataFrame): Current state of the reference data.
        in_colnames (list): List of column names that make up the address in the search data.
        in_refcol (list): List of column names that make up the address in the reference data.
        in_joincol (list): List of columns to join on between search and reference data.
        in_existing (list): List of columns indicating existing matches.
        Matcher (MatcherClass): Matcher object to store and process data.
        in_api: Flag or value indicating whether to use the API for reference data.
        in_api_key (str): API key for reference data retrieval, if applicable.

    Returns:
        tuple: (Matcher, final_api_output_file_name)
            Matcher: The updated Matcher object with loaded and preprocessed data.
            final_api_output_file_name (str): The filename of the reference data if loaded from API, else empty string.
    """

    final_api_output_file_name = ""

    today_rev = datetime.now().strftime("%Y%m%d")

    # Abort flag for if it's not even possible to attempt the first stage of the match for some reason
    Matcher.abort_flag = False

    effective_output_folder = output_folder or output_folder
    if effective_output_folder and (not effective_output_folder.endswith(("\\", "/"))):
        effective_output_folder = effective_output_folder + os.sep
    Matcher.output_folder = effective_output_folder

    ### ref_df FILES ###
    # If not an API call, run this first
    if not in_api:
        Matcher, final_api_output_file_name = load_ref_data(
            Matcher,
            ref_data_state,
            in_ref,
            in_refcol,
            in_api,
            in_api_key,
            query_type=in_api,
            use_postcode_blocker=use_postcode_blocker,
            output_folder=effective_output_folder,
        )

    ### MATCH/SEARCH FILES ###
    # If doing API calls, we need to know the search data before querying for specific addresses/postcodes
    Matcher = load_match_data_and_filter(
        Matcher,
        data_state,
        results_data_state,
        in_file,
        in_text,
        in_colnames,
        in_joincol,
        in_existing,
        in_api,
        use_postcode_blocker=use_postcode_blocker,
    )

    # If an API call, ref_df data is loaded after
    if in_api:
        Matcher, final_api_output_file_name = load_ref_data(
            Matcher,
            ref_data_state,
            in_ref,
            in_refcol,
            in_api,
            in_api_key,
            query_type=in_api,
            use_postcode_blocker=use_postcode_blocker,
            output_folder=effective_output_folder,
        )

    print("Shape of ref_df after filtering is: ", Matcher.ref_df.shape)
    print("Shape of search_df after filtering is: ", Matcher.search_df.shape)

    Matcher.match_outputs_name = (
        effective_output_folder + "diagnostics_initial_" + today_rev + ".csv"
    )
    Matcher.results_orig_df_name = (
        effective_output_folder + "results_initial_" + today_rev + ".csv"
    )

    if "fuzzy_score" in Matcher.match_results_output.columns:
        Matcher.match_results_output["fuzzy_score"] = pd.to_numeric(
            Matcher.match_results_output["fuzzy_score"], errors="coerce"
        ).round(2)
    if "wratio_score" in Matcher.match_results_output.columns:
        Matcher.match_results_output["wratio_score"] = pd.to_numeric(
            Matcher.match_results_output["wratio_score"], errors="coerce"
        ).round(2)

    if SAVE_INTERIM_FILES:
        Matcher.match_results_output.to_csv(Matcher.match_outputs_name, index=None)
        Matcher.results_on_orig_df.to_csv(Matcher.results_orig_df_name, index=None)

    return Matcher, final_api_output_file_name


# Run whole matcher process
def fuzzy_address_match(
    in_text: Optional[str] = None,
    in_file=None,
    in_ref=None,
    data_state: Optional[PandasDataFrame] = None,
    results_data_state: Optional[PandasDataFrame] = None,
    ref_data_state: Optional[PandasDataFrame] = None,
    in_colnames: Optional[Union[str, List[str]]] = None,
    in_refcol: Optional[Union[str, List[str]]] = None,
    in_joincol: Optional[Union[str, List[str]]] = None,
    in_existing: Optional[Union[str, List[str]]] = None,
    in_api: Optional[str] = None,
    in_api_key: Optional[str] = None,
    use_postcode_blocker: bool = USE_POSTCODE_BLOCKER,
    fuzzy_match_limit: Optional[Union[int, float]] = None,
    run_street_matching: bool = True,
    output_folder: Optional[str] = None,
    save_output_files: bool = SAVE_OUTPUT_FILES,
    print_match_stage_summary_to_console: bool = PRINT_MATCH_STAGE_SUMMARY_TO_CONSOLE,
    run_batches_in_parallel: bool = RUN_BATCHES_IN_PARALLEL,
    max_parallel_workers: Optional[int] = MAX_PARALLEL_WORKERS,
    InitMatch: MatcherClass = InitMatch,
    progress: gr.Progress = gr.Progress(track_tqdm=True),
    search_df: Optional[PandasDataFrame] = None,
    ref_df: Optional[PandasDataFrame] = None,
    results_df: Optional[PandasDataFrame] = None,
):
    """
    Run the end-to-end address matching pipeline.

    This orchestrates the full workflow: load and filter the search/reference data (optionally
    fetching reference data via an API), prepare and standardise addresses, split the work into
    batches, run matching for each batch, and write out diagnostic/results files.

    Usage:
        Pass dataframes directly (recommended for Python use):
            msg, output_files, est = fuzzy_address_match(
                search_df=search_df,
                ref_df=ref_df,
                in_colnames=["search_addr_col1", "search_addr_col2"],
                in_refcol=["ref_addr_col1", "ref_addr_col2"],
            )

        Or pass file paths (the function will load them):
            msg, output_files, est = fuzzy_address_match(
                in_file=r"C:\\path\\to\\search.csv",
                in_ref=r"C:\\path\\to\\reference.csv",
                in_colnames=["address"],
                in_refcol=["full_address"],
            )

    Args:
        in_text: Free-text address input (used when the user provides a single address string).
        in_file: Search dataset input. May be a Gradio-uploaded file list, a single file path, or a list of file paths.
        in_ref: Reference dataset input. May be a Gradio-uploaded file list, a single file path, or a list of file paths.
        data_state: Search dataframe state (used by the UI pathway). Optional if `search_df` is provided.
        results_data_state: Results dataframe state (used by the UI pathway). Optional if `results_df` is provided.
        ref_data_state: Reference dataframe state (used by the UI pathway). Optional if `ref_df` is provided.
        in_colnames: Column names from the search data used to construct the searchable address.
            May be a single string or a list of strings.
        in_refcol: Column names from the reference data used to construct the reference address.
            May be a single string or a list of strings.
        in_joincol: Column name(s) used as join/key fields between intermediate and original data.
            May be a single string (e.g. 'UPRN') or a list of strings; a string is wrapped as a one-element list.
        in_existing: Column name(s) for any existing identifiers/fields to preserve through matching.
            May be a single string or a list of strings.
        in_api: API mode / query type. If falsy, the reference data is loaded from `in_ref`.
        in_api_key: API key used when `in_api` is provided.
        use_postcode_blocker: If True, apply postcode-based blocking to reduce candidate comparisons.
        fuzzy_match_limit: Minimum RapidFuzz score (0–100) used as ``score_cutoff`` during
            blocked fuzzy comparison and for ``fuzzy_score_match`` in diagnostics. When omitted,
            uses ``fuzzy_match_limit`` from config / environment.
        run_street_matching: When postcode blocking is in use, also run the street-blocked fuzzy
            pass for rows that are not fully matched after postcode. Ignored when postcode blocking
            is off (street-only matching still runs as the primary pass).
        output_folder: Optional override for the output folder.
        save_output_files: If True, save final output CSV files (results/diagnostics/summary).
        print_match_stage_summary_to_console: If True, print per-stage match statistics
            (e.g. "For the Fuzzy standardised dataset...") to stdout during the run.
        run_batches_in_parallel: If True, process batches concurrently using multiple workers.
        max_parallel_workers: Maximum number of parallel workers (only used when parallel batching is enabled).
        InitMatch: Matcher object (or class instance) that carries configuration and intermediate state.
        progress: Gradio progress reporter used to update UI progress.
        search_df: Direct search dataframe input (Python-function pathway).
        ref_df: Direct reference dataframe input (Python-function pathway).
        results_df: Optional existing results dataframe (Python-function pathway).

    Returns:
        A tuple of:
        - out_message: Status message (string) when there is nothing to match; otherwise the matcher's output message.
        - output_files: List of output file paths produced during the run.
        - estimate_total_processing_time: Estimated total processing time in seconds.
    """

    class _FilePathLike:
        def __init__(self, name: str):
            self.name = name

    def _as_file_list(value):
        """
        Normalise a file input into the list-of-objects-with-.name convention used by the Gradio codepath.

        Accepts:
        - None / "" -> []
        - "C:/path/file.csv" -> [_FilePathLike(...)]
        - ["a.csv", "b.csv"] -> [_FilePathLike(...), _FilePathLike(...)]
        - already-file-like objects (with .name) -> returned unchanged
        """
        if value is None or value == "":
            return []
        if isinstance(value, str):
            return [_FilePathLike(value)]
        if isinstance(value, list):
            if not value:
                return []
            if all(isinstance(v, str) for v in value):
                return [_FilePathLike(v) for v in value]
            return value
        return [_FilePathLike(str(value))]

    # Resolve output folder robustly for UI, scripts, and CI.
    # SECURITY: user-specified `output_folder` must be within configured OUTPUT_FOLDER.
    def _resolve_output_folder_within_base(
        base_folder: str, user_folder: Optional[str]
    ) -> str:
        base = str(base_folder or "").strip() or "output"

        # Ensure base exists and is absolute/resolved
        base_path = os.path.abspath(base)
        os.makedirs(base_path, exist_ok=True)

        if user_folder is None or not str(user_folder).strip():
            out_path = base_path
        else:
            uf = str(user_folder).strip()
            if os.path.isabs(uf):
                cand = os.path.abspath(uf)
                try:
                    # Containment check
                    os.path.commonpath([base_path, cand])
                except ValueError as e:
                    raise PermissionError(f"Invalid output_folder path {uf!r}") from e
                if os.path.commonpath([base_path, cand]) != base_path:
                    raise PermissionError(
                        f"output_folder must be within OUTPUT_FOLDER ({base_path}). Got: {uf}"
                    )
                out_path = cand
            else:
                out_path = str(secure_path_join(base_path, uf))

        os.makedirs(out_path, exist_ok=True)
        if not out_path.endswith(("\\", "/")):
            out_path = out_path + os.sep
        return out_path

    if not use_postcode_blocker and not run_street_matching:
        raise ValueError(
            "Neither postcode blocking nor street-based matching is enabled. Please enable at least one matching method to proceed."
        )

    effective_output_folder = _resolve_output_folder_within_base(
        OUTPUT_FOLDER, output_folder
    )
    InitMatch.output_folder = effective_output_folder

    if fuzzy_match_limit is None:
        _fuzzy_lim = int(config_fuzzy_match_limit)
    else:
        try:
            _fuzzy_lim = int(round(float(fuzzy_match_limit)))
        except (TypeError, ValueError):
            _fuzzy_lim = int(config_fuzzy_match_limit)
    _fuzzy_lim = max(0, min(100, _fuzzy_lim))
    InitMatch.fuzzy_match_limit = _fuzzy_lim
    InitMatch.run_street_matching = bool(run_street_matching)

    def _notify_ui(level: str, message: str) -> None:
        """
        Best-effort UI notification for Gradio, while still being usable from pure Python.
        """
        fn = getattr(gr, level, None)
        if callable(fn):
            try:
                fn(message)
            except Exception:
                # Never let UI notifications break the pipeline.
                pass
        print(message)

    output_files = []

    estimate_total_processing_time = 0.0

    overall_tic = time.perf_counter()

    # ----- Input normalisation (Python-function + UI) -----
    # If the caller provided direct dataframes, prefer those.
    if search_df is not None:
        data_state = search_df
    if ref_df is not None:
        ref_data_state = ref_df
    if results_df is not None:
        results_data_state = results_df

    # Ensure all dataframe-state inputs are real DataFrames for downstream `.empty` checks.
    if data_state is None:
        data_state = pd.DataFrame()
    if results_data_state is None:
        results_data_state = pd.DataFrame()
    if ref_data_state is None:
        ref_data_state = pd.DataFrame()

    # Ensure lists are always lists (avoids mutable default args and None handling).
    # Single strings must become one-element lists so we never iterate characters.
    in_colnames = _ensure_str_list(in_colnames)
    in_refcol = _ensure_str_list(in_refcol)
    in_joincol = _ensure_str_list(in_joincol)
    in_existing = _ensure_str_list(in_existing)

    # Normalise file inputs (paths or Gradio uploads) into list-of-file-like objects.
    in_file = _as_file_list(in_file)
    in_ref = _as_file_list(in_ref)

    # ----- Guards: ensure we have enough inputs to run -----
    has_search_input = (
        bool(in_text)
        or (data_state is not None and not data_state.empty)
        or bool(in_file)
    )
    if not has_search_input:
        out_message = "No search data provided. Please upload an input file or enter a single address as text."
        _notify_ui("Warning", out_message)
        return out_message, None, estimate_total_processing_time, ""

    has_ref_input = (ref_data_state is not None and not ref_data_state.empty) or bool(
        in_ref
    )
    if not has_ref_input and not in_api:
        out_message = "No reference data provided. Please upload a reference file, or choose an API option."
        _notify_ui("Warning", out_message)
        return out_message, None, estimate_total_processing_time, ""

    # If matching against a reference file (not API), the user must specify at least one
    # column that makes up the reference address; otherwise later outputs/diagnostics
    # can fail due to missing match fields.
    if (
        (not in_api)
        and has_ref_input
        and (len(in_refcol) == 0)
        and not any("sao" in col.lower() for col in ref_data_state.columns)
    ):
        out_message = (
            "No reference address columns selected.\n\n"
            "Please select at least one column that makes up the reference address "
            "(and put postcode last if you include it), then try again."
        )
        _notify_ui("Warning", out_message)
        return out_message, None, estimate_total_processing_time, ""

    if any("sao" in col.lower() for col in ref_data_state.columns):
        print("Addressbase format found in reference data")

    # If using file/dataframe search input (not single-text mode), require at least one
    # search address column to build the searchable address.
    if (not in_text or str(in_text).strip() == "") and (len(in_colnames) == 0):
        out_message = (
            "No search address columns selected.\n\n"
            "Please select at least one search address column (or enter a single address as text), "
            "then try again."
        )
        _notify_ui("Warning", out_message)
        return out_message, None, estimate_total_processing_time, ""

    # If the caller is using the "pure python" dataframe pathway and didn't supply any file inputs,
    # give `load_ref_data` something to name the reference dataset with when `ref_data_state` is non-empty.
    if ref_data_state is not None and not ref_data_state.empty and not in_ref:
        in_ref = [_FilePathLike("in_memory_reference")]

    # Load in initial data. This will filter to relevant addresses in the search and reference datasets that can potentially be matched, and will pull in API data if asked for.
    InitMatch, final_api_output_file_name = load_matcher_data(
        in_text or "",
        in_file,
        in_ref,
        data_state,
        results_data_state,
        ref_data_state,
        in_colnames,
        in_refcol,
        in_joincol,
        in_existing,
        InitMatch,
        in_api,
        in_api_key,
        use_postcode_blocker=use_postcode_blocker,
        output_folder=effective_output_folder,
    )

    if final_api_output_file_name:
        output_files.append(final_api_output_file_name)

    if InitMatch.search_df.empty or InitMatch.ref_df.empty:
        out_message = (
            "Nothing to match! Search data frame or reference data frame are empty"
        )
        print(out_message)

        output_files.extend(
            [InitMatch.results_orig_df_name, InitMatch.match_outputs_name]
        )
        return out_message, output_files, estimate_total_processing_time, ""

    # Run initial address preparation and standardisation processes
    # Prepare address format

    # Prepare all search addresses
    if isinstance(InitMatch.search_df, str):
        (
            InitMatch.search_df_cleaned,
            InitMatch.search_df_key_field,
            InitMatch.search_address_cols,
        ) = prepare_search_address_string(InitMatch.search_df)
    else:
        InitMatch.search_df_cleaned = prepare_search_address(
            InitMatch.search_df,
            InitMatch.search_address_cols,
            InitMatch.search_postcode_col,
            InitMatch.search_df_key_field,
        )

        # Remove addresses that are not postal addresses
    InitMatch.search_df_cleaned = remove_non_postal(
        InitMatch.search_df_cleaned, "full_address"
    )

    # Remove addresses that have no numbers in from consideration
    InitMatch.search_df_cleaned = check_no_number_addresses(
        InitMatch.search_df_cleaned, "full_address"
    )

    # Preserve a stable, original-format search address for display in outputs.
    # Prefer the raw joined input from the pre-filter search snapshot when available.
    _pfs = getattr(InitMatch, "pre_filter_search_df", None)
    if (
        isinstance(_pfs, pd.DataFrame)
        and (not _pfs.empty)
        and (InitMatch.search_df_key_field in _pfs.columns)
        and ("address_cols_joined" in _pfs.columns)
        and (InitMatch.search_df_key_field in InitMatch.search_df_cleaned.columns)
    ):
        _disp = (
            _pfs[[InitMatch.search_df_key_field, "address_cols_joined"]]
            .copy()
            .drop_duplicates(subset=[InitMatch.search_df_key_field], keep="first")
        )
        _k = InitMatch.search_df_key_field
        _disp[_k] = _disp[_k].astype(str)
        InitMatch.search_df_cleaned[_k] = InitMatch.search_df_cleaned[_k].astype(str)
        InitMatch.search_df_cleaned = InitMatch.search_df_cleaned.merge(
            _disp, on=_k, how="left"
        )
        _orig = InitMatch.search_df_cleaned["address_cols_joined"]
        _has_orig = (
            _orig.notna()
            & _orig.astype(str).str.strip().ne("")
            & ~_orig.astype(str).str.strip().str.lower().isin(("nan", "none", "<na>"))
        )
        InitMatch.search_df_cleaned["search_input_display"] = _orig.where(
            _has_orig, InitMatch.search_df_cleaned["full_address"]
        )
        InitMatch.search_df_cleaned = InitMatch.search_df_cleaned.drop(
            columns=["address_cols_joined"], errors="ignore"
        )
    else:
        # Fall back to the prepared address when we cannot access the pre-filter join.
        if "search_input_display" not in InitMatch.search_df_cleaned.columns:
            InitMatch.search_df_cleaned["search_input_display"] = (
                InitMatch.search_df_cleaned["full_address"]
            )

    # Initial preparation of reference addresses
    InitMatch.ref_df_cleaned = prepare_ref_address(
        InitMatch.ref_df, InitMatch.ref_address_cols, InitMatch.new_join_col
    )

    # Standardise addresses
    # Standardise - minimal

    tic = time.perf_counter()

    progress(0.1, desc="Performing minimal standardisation")

    _stand_out = getattr(InitMatch, "output_folder", None) or output_folder
    _cache_mode = "postcode" if use_postcode_blocker else "street"
    _path_min_s = _stand_cache_path(
        _stand_out, InitMatch.file_name, "stand_min", _cache_mode
    )
    _path_min_r = _stand_cache_path(
        _stand_out, InitMatch.ref_name, "stand_min", _cache_mode
    )

    def _cache_is_compatible(
        cached_df: pd.DataFrame,
        expected_df: pd.DataFrame,
        *,
        key_col: str,
        required_col: str,
        compare_cols: Optional[List[str]] = None,
    ) -> bool:
        """
        Validate that a cached standardisation frame matches the current run.

        We require:
        - required_col exists (e.g. search_address_stand/ref_address_stand)
        - key_col exists (e.g. index/ref_index)
        - key set matches (prevents reusing cache from a different input)
        - if compare_cols are available on both frames, a key-sorted value signature
          also matches (guards against stale cache with same key set)
        """
        if (
            cached_df is None
            or cached_df.empty
            or expected_df is None
            or expected_df.empty
        ):
            return False
        if required_col not in cached_df.columns:
            return False
        if key_col not in cached_df.columns:
            return False
        if key_col not in expected_df.columns:
            return False

        # Reject non-unique keys to avoid many-to-many ambiguities.
        if cached_df[key_col].astype(str).duplicated().any():
            return False
        if expected_df[key_col].astype(str).duplicated().any():
            return False

        cached_keys = set(cached_df[key_col].astype(str).tolist())
        expected_keys = set(expected_df[key_col].astype(str).tolist())
        if cached_keys != expected_keys:
            return False

        if compare_cols:
            # Compare a deterministic signature over key-sorted value tuples for
            # whichever columns are present on both sides.
            cols_present = [
                c
                for c in compare_cols
                if (c in cached_df.columns) and (c in expected_df.columns)
            ]
            if cols_present:
                lhs = (
                    expected_df[[key_col] + cols_present]
                    .copy()
                    .assign(**{key_col: expected_df[key_col].astype(str)})
                    .sort_values(by=[key_col], kind="stable")
                    .fillna("")
                    .astype(str)
                )
                rhs = (
                    cached_df[[key_col] + cols_present]
                    .copy()
                    .assign(**{key_col: cached_df[key_col].astype(str)})
                    .sort_values(by=[key_col], kind="stable")
                    .fillna("")
                    .astype(str)
                )
                if not lhs.equals(rhs):
                    return False

        return True

    _loaded_min_s = False
    if USE_EXISTING_STANDARDISED_FILES and os.path.isfile(_path_min_s):
        try:
            cached = pd.read_parquet(_path_min_s)
        except Exception as e:
            print(f"Failed to read minimal search cache; rebuilding. ({e})")
            cached = None

        if (cached is not None) and _cache_is_compatible(
            cached,
            InitMatch.search_df_cleaned,
            key_col=InitMatch.search_df_key_field,
            required_col="search_address_stand",
            compare_cols=["full_address", "postcode"],
        ):
            InitMatch.search_df_after_stand = (
                _strip_runtime_fuzzy_cols_from_stand_cache(cached)
            )
            _loaded_min_s = True
            print(
                f"Loaded minimal search standardisation from cache: {os.path.basename(_path_min_s)}"
            )
        else:
            print(
                "Cached minimal search standardisation is missing expected columns or "
                "does not match current input keys; rebuilding cache."
            )
            _loaded_min_s = False
    if not _loaded_min_s:
        InitMatch.search_df_after_stand = _standardise_search_df(
            InitMatch.search_df_cleaned, standardise=False
        )
        InitMatch.search_df_after_stand.to_parquet(_path_min_s, index=False)

    _loaded_min_r = False
    if USE_EXISTING_STANDARDISED_FILES and os.path.isfile(_path_min_r):
        try:
            cached = pd.read_parquet(_path_min_r)
        except Exception as e:
            print(f"Failed to read minimal reference cache; rebuilding. ({e})")
            cached = None

        if (cached is not None) and _cache_is_compatible(
            cached,
            InitMatch.ref_df_cleaned,
            key_col="ref_index",
            required_col="ref_address_stand",
            compare_cols=["fulladdress", "Postcode"],
        ):
            InitMatch.ref_df_after_stand = _strip_runtime_fuzzy_cols_from_stand_cache(
                cached
            )
            _loaded_min_r = True
            print(
                f"Loaded minimal reference standardisation from cache: {os.path.basename(_path_min_r)}"
            )
        else:
            print(
                "Cached minimal reference standardisation is missing expected columns or "
                "does not match current input keys; rebuilding cache."
            )
            _loaded_min_r = False
    if not _loaded_min_r:
        InitMatch.ref_df_after_stand = _standardise_ref_df(
            InitMatch.ref_df_cleaned, standardise=False
        )
        InitMatch.ref_df_after_stand.to_parquet(_path_min_r, index=False)

    toc = time.perf_counter()
    if _loaded_min_s and _loaded_min_r:
        print(f"Minimal standardisation (cache) in {toc - tic:0.1f} seconds")
    elif _loaded_min_s or _loaded_min_r:
        print(f"Minimal standardisation (partial cache) in {toc - tic:0.1f} seconds")
    else:
        print(f"Performed the minimal standardisation step in {toc - tic:0.1f} seconds")

    progress(0.1, desc="Performing full standardisation")

    # Standardise - full
    tic = time.perf_counter()
    _path_full_s = _stand_cache_path(
        _stand_out, InitMatch.file_name, "stand_full", _cache_mode
    )
    _path_full_r = _stand_cache_path(
        _stand_out, InitMatch.ref_name, "stand_full", _cache_mode
    )

    _loaded_full_s = False
    if USE_EXISTING_STANDARDISED_FILES and os.path.isfile(_path_full_s):
        try:
            cached = pd.read_parquet(_path_full_s)
        except Exception as e:
            print(f"Failed to read full search cache; rebuilding. ({e})")
            cached = None

        if (cached is not None) and _cache_is_compatible(
            cached,
            InitMatch.search_df_cleaned,
            key_col=InitMatch.search_df_key_field,
            required_col="search_address_stand",
            compare_cols=["full_address", "postcode"],
        ):
            InitMatch.search_df_after_full_stand = (
                _strip_runtime_fuzzy_cols_from_stand_cache(cached)
            )
            _loaded_full_s = True
            print(
                f"Loaded full search standardisation from cache: {os.path.basename(_path_full_s)}"
            )
        else:
            print(
                "Cached full search standardisation is missing expected columns or "
                "does not match current input keys; rebuilding cache."
            )
            _loaded_full_s = False
    if not _loaded_full_s:
        InitMatch.search_df_after_full_stand = _standardise_search_df(
            InitMatch.search_df_cleaned, standardise=True
        )
        InitMatch.search_df_after_full_stand.to_parquet(_path_full_s, index=False)

    _loaded_full_r = False
    if USE_EXISTING_STANDARDISED_FILES and os.path.isfile(_path_full_r):
        try:
            cached = pd.read_parquet(_path_full_r)
        except Exception as e:
            print(f"Failed to read full reference cache; rebuilding. ({e})")
            cached = None

        if (cached is not None) and _cache_is_compatible(
            cached,
            InitMatch.ref_df_cleaned,
            key_col="ref_index",
            required_col="ref_address_stand",
            compare_cols=["fulladdress", "Postcode"],
        ):
            InitMatch.ref_df_after_full_stand = (
                _strip_runtime_fuzzy_cols_from_stand_cache(cached)
            )
            _loaded_full_r = True
            print(
                f"Loaded full reference standardisation from cache: {os.path.basename(_path_full_r)}"
            )
        else:
            print(
                "Cached full reference standardisation is missing expected columns or "
                "does not match current input keys; rebuilding cache."
            )
            _loaded_full_r = False
    if not _loaded_full_r:
        InitMatch.ref_df_after_full_stand = _standardise_ref_df(
            InitMatch.ref_df_cleaned, standardise=True
        )
        InitMatch.ref_df_after_full_stand.to_parquet(_path_full_r, index=False)

    toc = time.perf_counter()
    if _loaded_full_s and _loaded_full_r:
        print(f"Full standardisation (cache) in {toc - tic:0.1f} seconds")
    elif _loaded_full_s or _loaded_full_r:
        print(f"Full standardisation (partial cache) in {toc - tic:0.1f} seconds")
    else:
        print(f"Performed the full standardisation step in {toc - tic:0.1f} seconds")

    use_postcode_blocker_effective = bool(use_postcode_blocker)
    if use_postcode_blocker_effective:
        has_search_postcode_search = _column_has_usable_values(
            InitMatch.search_df_after_stand, "postcode_search"
        )
        has_ref_postcode_search = _column_has_usable_values(
            InitMatch.ref_df_after_stand, "postcode_search"
        )
        if not (has_search_postcode_search and has_ref_postcode_search):
            print(
                "Postcode blocker requested but postcode values are unavailable after preparation. Falling back to street-only matching."
            )
            use_postcode_blocker_effective = False

    # Determine batch ranges. Postcode mode uses postcode-grouped batching;
    # street-only mode batches search rows and compares each batch against all refs.
    if use_postcode_blocker_effective:
        range_df = create_batch_ranges(
            InitMatch.search_df_cleaned.copy(),
            InitMatch.ref_df_cleaned.copy(),
            batch_size,
            ref_batch_size,
            "postcode",
            "Postcode",
            search_df_key_field=InitMatch.search_df_key_field or "index",
            ref_key_field="ref_index",
        )
    else:
        range_df = create_street_batch_ranges(
            InitMatch.search_df_cleaned.copy(),
            InitMatch.ref_df_cleaned.copy(),
            batch_size,
        )

    OutputMatch = copy.copy(InitMatch)

    batch_inputs: List[Tuple[int, MatcherClass, bool]] = []
    for row in range(0, range_df.shape[0]):
        search_range = range_df.iloc[row]["search_range"]
        ref_range = range_df.iloc[row]["ref_range"]

        BatchMatch = copy.copy(InitMatch)
        if use_postcode_blocker_effective:
            _search_key = BatchMatch.search_df_key_field or "index"
            BatchMatch.search_df = _slice_frame_by_normalized_keys(
                BatchMatch.search_df, _search_key, list(search_range)
            )
            BatchMatch.search_df_not_matched = BatchMatch.search_df.copy()
            BatchMatch.search_df_cleaned = _slice_frame_by_normalized_keys(
                BatchMatch.search_df_cleaned, _search_key, list(search_range)
            )
            BatchMatch.search_df_after_stand = _slice_frame_by_normalized_keys(
                BatchMatch.search_df_after_stand, _search_key, list(search_range)
            )
            BatchMatch.search_df_after_full_stand = _slice_frame_by_normalized_keys(
                BatchMatch.search_df_after_full_stand,
                _search_key,
                list(search_range),
            )
        else:
            # Street-only: `search_range` is a list of integer **positions** (see
            # `create_street_batch_ranges`). All search-side frames share row order
            # and length with `search_df_cleaned`.
            _pos = list(search_range)
            BatchMatch.search_df = BatchMatch.search_df.iloc[_pos].reset_index(
                drop=True
            )
            BatchMatch.search_df_not_matched = BatchMatch.search_df.copy()
            BatchMatch.search_df_cleaned = BatchMatch.search_df_cleaned.iloc[
                _pos
            ].reset_index(drop=True)
            BatchMatch.search_df_after_stand = BatchMatch.search_df_after_stand.iloc[
                _pos
            ].reset_index(drop=True)
            BatchMatch.search_df_after_full_stand = (
                BatchMatch.search_df_after_full_stand.iloc[_pos].reset_index(drop=True)
            )

        if use_postcode_blocker_effective:
            _ref_key = "ref_index" if "ref_index" in BatchMatch.ref_df.columns else None
            BatchMatch.ref_df = _slice_frame_by_normalized_keys(
                BatchMatch.ref_df, _ref_key, list(ref_range)
            )
            BatchMatch.ref_df_cleaned = _slice_frame_by_normalized_keys(
                BatchMatch.ref_df_cleaned, _ref_key, list(ref_range)
            )
            BatchMatch.ref_df_after_stand = _slice_frame_by_normalized_keys(
                BatchMatch.ref_df_after_stand, _ref_key, list(ref_range)
            )
            BatchMatch.ref_df_after_full_stand = _slice_frame_by_normalized_keys(
                BatchMatch.ref_df_after_full_stand, _ref_key, list(ref_range)
            )
        else:
            BatchMatch.ref_df = BatchMatch.ref_df[
                BatchMatch.ref_df.index.isin(ref_range)
            ].reset_index(drop=True)
            BatchMatch.ref_df_cleaned = BatchMatch.ref_df_cleaned[
                BatchMatch.ref_df_cleaned.index.isin(ref_range)
            ].reset_index(drop=True)
            BatchMatch.ref_df_after_stand = BatchMatch.ref_df_after_stand[
                BatchMatch.ref_df_after_stand.index.isin(ref_range)
            ].reset_index(drop=True)
            BatchMatch.ref_df_after_full_stand = BatchMatch.ref_df_after_full_stand[
                BatchMatch.ref_df_after_full_stand.index.isin(ref_range)
            ].reset_index(drop=True)

        batch_inputs.append((row, BatchMatch, use_postcode_blocker_effective))

    pc_batch_count = len(batch_inputs)
    overflow_range_df: Optional[pd.DataFrame] = None
    overflow_uncovered_n: Optional[int] = None
    if (
        use_postcode_blocker_effective
        and _street_overflow_unbatched_search_enabled()
        and bool(getattr(InitMatch, "run_street_matching", True))
    ):
        _cov = _postcode_batch_covered_search_keys_normalized(range_df)
        _uncovered_vals = _uncovered_search_key_values_for_street_overflow(
            InitMatch, _cov
        )
        if _uncovered_vals:
            OverflowInit = _matcher_search_side_sliced_to_keys(
                InitMatch, _uncovered_vals
            )
            if not OverflowInit.search_df.empty:
                overflow_range_df = create_street_batch_ranges(
                    OverflowInit.search_df_cleaned.copy(),
                    InitMatch.ref_df_cleaned.copy(),
                    batch_size,
                )
                overflow_uncovered_n = len(_uncovered_vals)
                _n_ov = int(overflow_range_df.shape[0])
                if _fuzzy_match_debug_enabled():
                    print(
                        "[FUZZY_MATCH_DEBUG] street overflow: "
                        f"uncovered_rows={overflow_uncovered_n} overflow_batches={_n_ov}"
                    )
                for ov_row in range(_n_ov):
                    search_range = overflow_range_df.iloc[ov_row]["search_range"]
                    ref_range = overflow_range_df.iloc[ov_row]["ref_range"]
                    BatchMatch = copy.copy(OverflowInit)
                    _pos = list(search_range)
                    BatchMatch.search_df = BatchMatch.search_df.iloc[_pos].reset_index(
                        drop=True
                    )
                    BatchMatch.search_df_not_matched = BatchMatch.search_df.copy()
                    BatchMatch.search_df_cleaned = BatchMatch.search_df_cleaned.iloc[
                        _pos
                    ].reset_index(drop=True)
                    BatchMatch.search_df_after_stand = (
                        BatchMatch.search_df_after_stand.iloc[_pos].reset_index(
                            drop=True
                        )
                    )
                    BatchMatch.search_df_after_full_stand = (
                        BatchMatch.search_df_after_full_stand.iloc[_pos].reset_index(
                            drop=True
                        )
                    )
                    BatchMatch.ref_df = BatchMatch.ref_df[
                        BatchMatch.ref_df.index.isin(ref_range)
                    ].reset_index(drop=True)
                    BatchMatch.ref_df_cleaned = BatchMatch.ref_df_cleaned[
                        BatchMatch.ref_df_cleaned.index.isin(ref_range)
                    ].reset_index(drop=True)
                    BatchMatch.ref_df_after_stand = BatchMatch.ref_df_after_stand[
                        BatchMatch.ref_df_after_stand.index.isin(ref_range)
                    ].reset_index(drop=True)
                    BatchMatch.ref_df_after_full_stand = (
                        BatchMatch.ref_df_after_full_stand[
                            BatchMatch.ref_df_after_full_stand.index.isin(ref_range)
                        ].reset_index(drop=True)
                    )
                    batch_inputs.append((pc_batch_count + ov_row, BatchMatch, False))

    _print_batches_to_run_overview(
        range_df,
        use_postcode_mode=use_postcode_blocker_effective,
        overflow_range_df=overflow_range_df,
    )
    if overflow_uncovered_n is not None and overflow_range_df is not None:
        _n_ov = int(overflow_range_df.shape[0])
        print(
            "Street-only overflow: "
            f"{overflow_uncovered_n} search row(s) were not in any postcode-overlap batch; "
            f"running {_n_ov} extra batch(es) against the full reference "
            "(postcode blocking off for these batches). "
            "Set STREET_OVERFLOW_UNBATCHED_SEARCH=0 to skip."
        )

    number_of_batches = len(batch_inputs)
    run_parallel = bool(run_batches_in_parallel) and number_of_batches > 1
    worker_count = 1
    if run_parallel:
        worker_count = _resolve_parallel_worker_count(
            number_of_batches, max_parallel_workers, includes_nnet=run_nnet_match
        )
        # ProcessPoolExecutor still spawns subprocesses when max_workers==1; run in-process
        # instead so MAX_PARALLEL_WORKERS=1 avoids Windows spawn/__main__ footguns.
        if worker_count < 2:
            run_parallel = False

    if run_parallel:
        try:
            print(f"Running batches in parallel with {worker_count} worker processes")
            future_results = []

            with ProcessPoolExecutor(max_workers=worker_count) as executor:
                submitted = [
                    executor.submit(
                        run_single_match_batch_worker,
                        batch_n,
                        batch_match,
                        number_of_batches,
                        use_pc,
                        print_match_stage_summary_to_console,
                    )
                    for batch_n, batch_match, use_pc in batch_inputs
                ]

                for future in progress.tqdm(
                    as_completed(submitted),
                    desc="Matching addresses in parallel batches",
                    unit="batches",
                    total=number_of_batches,
                ):
                    future_results.append(future.result())

            for batch_n, summary_of_summaries, BatchMatch_out in sorted(
                future_results, key=lambda x: x[0]
            ):
                OutputMatch = combine_two_matches(
                    OutputMatch,
                    BatchMatch_out,
                    "All up to and including batch " + str(batch_n + 1),
                    write_outputs=False,
                    print_match_stage_summary_to_console=print_match_stage_summary_to_console,
                )

        except Exception as parallel_error:
            print(
                f"Parallel batch execution failed ({parallel_error}). Falling back to sequential batching."
            )
            run_parallel = False

    if not run_parallel:
        for batch_n, BatchMatch, use_pc in progress.tqdm(
            batch_inputs,
            desc="Matching addresses in batches",
            unit="batches",
            total=number_of_batches,
        ):
            print("Running batch", str(batch_n + 1))

            if BatchMatch.search_df.empty or BatchMatch.ref_df.empty:
                out_message = "Nothing to match for batch: " + str(batch_n)
                print(out_message)
                BatchMatch_out = BatchMatch
                _batch_key = getattr(BatchMatch, "search_df_key_field", None) or "index"
                if (
                    not BatchMatch.search_df.empty
                    and _batch_key in BatchMatch.search_df.columns
                ):
                    _sdf = BatchMatch.search_df
                    _excl = (
                        _sdf["Excluded from search"]
                        if "Excluded from search" in _sdf.columns
                        else "Included in search"
                    )
                    BatchMatch_out.results_on_orig_df = pd.DataFrame(
                        {
                            _batch_key: _sdf[_batch_key],
                            "Excluded from search": _excl,
                            "Matched with reference address": False,
                        }
                    )
                    BatchMatch_out.match_results_output = pd.DataFrame(
                        {
                            _batch_key: _sdf[_batch_key].astype(str),
                            "full_match": False,
                        }
                    )
                else:
                    BatchMatch_out.results_on_orig_df = pd.DataFrame()
                    BatchMatch_out.match_results_output = pd.DataFrame(
                        columns=[_batch_key, "full_match"]
                    )
            else:
                summary_of_summaries, BatchMatch_out = run_single_match_batch(
                    BatchMatch,
                    batch_n,
                    number_of_batches,
                    use_postcode_blocker=use_pc,
                    write_outputs=SAVE_INTERIM_FILES,
                    print_match_stage_summary_to_console=print_match_stage_summary_to_console,
                )

            OutputMatch = combine_two_matches(
                OutputMatch,
                BatchMatch_out,
                "All up to and including batch " + str(batch_n + 1),
                write_outputs=SAVE_INTERIM_FILES,
                print_match_stage_summary_to_console=print_match_stage_summary_to_console,
            )

    if in_api:
        OutputMatch.results_on_orig_df["Matched with reference address"] = (
            OutputMatch.results_on_orig_df["Matched with reference address"].replace(
                {1: True, 0: False}
            )
        )
        OutputMatch.results_on_orig_df["Excluded from search"] = (
            OutputMatch.results_on_orig_df["Excluded from search"]
            .replace("nan", False)
            .fillna(False)
        )

    # Remove any duplicates, prioritise successful matches, and keep a consistent
    # ordering across outputs (prefer numeric ordering of the key where possible).
    _key_col = (
        OutputMatch.search_df_key_field
        if (
            hasattr(OutputMatch, "search_df_key_field")
            and isinstance(OutputMatch.search_df_key_field, str)
            and (
                OutputMatch.search_df_key_field
                in OutputMatch.results_on_orig_df.columns
            )
        )
        else ("index" if "index" in OutputMatch.results_on_orig_df.columns else None)
    )
    if _key_col is not None:
        OutputMatch.results_on_orig_df["__key_num"] = pd.to_numeric(
            OutputMatch.results_on_orig_df[_key_col], errors="coerce"
        )
        _sort_by = ["__key_num", _key_col, "Matched with reference address"]
        _sort_asc = [True, True, False]
        OutputMatch.results_on_orig_df = (
            OutputMatch.results_on_orig_df.sort_values(
                by=_sort_by, ascending=_sort_asc, kind="stable"
            )
            .drop_duplicates(subset=_key_col, keep="first")
            .drop(columns=["__key_num"], errors="ignore")
        )

    # Ensure diagnostics contains exclusion info and a row for every input record
    # represented in results_on_orig_df.
    if ("Excluded from search" in OutputMatch.results_on_orig_df.columns) and (
        OutputMatch.search_df_key_field in OutputMatch.results_on_orig_df.columns
    ):
        diag_df = OutputMatch.match_results_output.copy()
        key_col = OutputMatch.search_df_key_field

        # Build canonical key/address lookup from results (one row per key).
        results_cols = [key_col, "Excluded from search"]
        if "Search data address" in OutputMatch.results_on_orig_df.columns:
            results_cols.append("Search data address")
        if "Matched with reference address" in OutputMatch.results_on_orig_df.columns:
            results_cols.append("Matched with reference address")
        results_keyed = OutputMatch.results_on_orig_df[results_cols].drop_duplicates(
            subset=key_col, keep="first"
        )
        results_keyed["__key_str"] = results_keyed[key_col].astype(str)

        # Ensure diagnostics has a key for mapping/filling.
        if key_col not in diag_df.columns:
            diag_df[key_col] = pd.Series(dtype="string")
        diag_df["__key_str"] = diag_df[key_col].astype(str)
        # Ensure diagnostics has the exclusion column so it can be populated even when
        # the underlying diagnostic output doesn't include it by default.
        if "Excluded from search" not in diag_df.columns:
            diag_df["Excluded from search"] = ""

        # If address->key mapping is unambiguous, correct any mis-keyed diagnostics rows.
        if ("search_orig_address" in diag_df.columns) and (
            "Search data address" in results_keyed.columns
        ):
            _addr = results_keyed["Search data address"].fillna("").astype(str)
            _addr_counts = _addr.value_counts()
            _unique_addrs = _addr_counts[_addr_counts == 1].index
            _addr_to_key = (
                results_keyed.loc[_addr.isin(_unique_addrs)]
                .drop_duplicates(subset=["Search data address"], keep="first")
                .set_index("Search data address")["__key_str"]
            )
            _mapped_key = (
                diag_df["search_orig_address"].fillna("").astype(str).map(_addr_to_key)
            )
            diag_df.loc[_mapped_key.notna(), "__key_str"] = _mapped_key[
                _mapped_key.notna()
            ]

        # Add placeholder rows for any keys missing from diagnostics.
        missing_rows = results_keyed.loc[
            ~results_keyed["__key_str"].isin(diag_df["__key_str"].tolist())
        ].copy()
        if not missing_rows.empty:
            diag_add = pd.DataFrame(columns=list(diag_df.columns))
            diag_add["__key_str"] = missing_rows["__key_str"]
            diag_add[key_col] = missing_rows[key_col]

            if ("search_orig_address" in diag_add.columns) and (
                "Search data address" in missing_rows.columns
            ):
                diag_add["search_orig_address"] = missing_rows["Search data address"]
            if "Excluded from search" in diag_add.columns:
                diag_add["Excluded from search"] = missing_rows["Excluded from search"]
            if ("full_match" in diag_add.columns) and (
                "Matched with reference address" in missing_rows.columns
            ):
                diag_add["full_match"] = _bool_mask(
                    missing_rows["Matched with reference address"],
                    default=False,
                )
            if "fuzzy_score" in diag_add.columns:
                diag_add["fuzzy_score"] = 0.0
            if "wratio_score" in diag_add.columns:
                diag_add["wratio_score"] = 0.0
            if "standardised_address" in diag_add.columns:
                diag_add["standardised_address"] = False
            if "match_method" in diag_add.columns:
                excluded_reason = (
                    missing_rows["Excluded from search"].fillna("").astype(str)
                )
                diag_add["match_method"] = np.where(
                    excluded_reason.eq("Previously matched"),
                    "Pre-existing match",
                    np.where(
                        excluded_reason.eq("Included in search"),
                        "No successful match",
                        "Excluded from matching",
                    ),
                )

            # Drop all-NA columns in the placeholder frame so concat does not trigger the
            # pandas FutureWarning about empty/all-NA entries affecting result dtypes.
            diag_add = diag_add.dropna(axis=1, how="all")
            diag_df = pd.concat([diag_df, diag_add], axis=0, ignore_index=True)

        # Canonicalise key/address/exclusion from results so diagnostics aligns exactly.
        _exclusion_map = results_keyed.set_index("__key_str")["Excluded from search"]
        if "Excluded from search" in diag_df.columns:
            _excl_m = diag_df["__key_str"].map(_exclusion_map)
            diag_df["Excluded from search"] = _excl_m.where(
                _excl_m.notna(), diag_df["Excluded from search"]
            )
        if ("search_orig_address" in diag_df.columns) and (
            "Search data address" in results_keyed.columns
        ):
            _addr_map = results_keyed.set_index("__key_str")["Search data address"]
            _addr_m = diag_df["__key_str"].map(_addr_map)
            diag_df["search_orig_address"] = _addr_m.where(
                _addr_m.notna(), diag_df["search_orig_address"]
            )
        if ("full_match" in diag_df.columns) and (
            "Matched with reference address" in results_keyed.columns
        ):
            _matched_map = results_keyed.set_index("__key_str")[
                "Matched with reference address"
            ]
            _fm_m = diag_df["__key_str"].map(_matched_map)
            diag_df["full_match"] = _fm_m.where(_fm_m.notna(), diag_df["full_match"])
            diag_df["full_match"] = _bool_mask(diag_df["full_match"], default=False)
        _key_map = results_keyed.set_index("__key_str")[key_col]
        _key_m = diag_df["__key_str"].map(_key_map)
        diag_df[key_col] = _key_m.where(_key_m.notna(), diag_df[key_col])

        # One row per key in diagnostics; prefer successful/high-score rows.
        if "fuzzy_score" in diag_df.columns:
            diag_df["fuzzy_score"] = pd.to_numeric(
                diag_df["fuzzy_score"], errors="coerce"
            )
        if "wratio_score" in diag_df.columns:
            diag_df["wratio_score"] = pd.to_numeric(
                diag_df["wratio_score"], errors="coerce"
            )
        _sort_cols = []
        _ascending = []
        diag_df["__key_num"] = pd.to_numeric(diag_df[key_col], errors="coerce")
        if diag_df["__key_num"].notna().any():
            _sort_cols.append("__key_num")
            _ascending.append(True)
        _sort_cols.append("__key_str")
        _ascending.append(True)
        if "full_match" in diag_df.columns:
            _sort_cols.append("full_match")
            _ascending.append(False)
        if "fuzzy_score" in diag_df.columns:
            _sort_cols.append("fuzzy_score")
            _ascending.append(False)
        if "wratio_score" in diag_df.columns:
            _sort_cols.append("wratio_score")
            _ascending.append(False)
        diag_df = diag_df.sort_values(
            by=_sort_cols, ascending=_ascending, kind="stable"
        ).drop_duplicates(subset="__key_str", keep="first")
        diag_df = diag_df.drop(columns=["__key_str", "__key_num"], errors="ignore")

        OutputMatch.match_results_output = diag_df

    overall_toc = time.perf_counter()
    time_out = (
        f"The overall match (all batches) took {overall_toc - overall_tic:0.1f} seconds"
    )

    if print_match_stage_summary_to_console:
        print(OutputMatch.output_summary)

    if OutputMatch.output_summary == "":
        OutputMatch.output_summary = "No matches were found."

    final_summary = build_run_summary_text(
        OutputMatch.results_on_orig_df,
        diagnostics_df=OutputMatch.match_results_output,
        key_field=OutputMatch.search_df_key_field,
        use_postcode_blocker_requested=getattr(
            OutputMatch, "use_postcode_blocker_requested", None
        ),
        use_postcode_blocker_effective=getattr(
            OutputMatch, "use_postcode_blocker_effective", None
        ),
    )

    # Prepend run context for the UI (matched files + columns used).
    _search_display = (
        "single_address"
        if (in_text and str(in_text).strip())
        else getattr(OutputMatch, "file_name", None)
    )
    if (search_df is not None) and (len(in_file) == 0) and (not _search_display):
        _search_display = "search_df"
    _ref_display = "api" if in_api else getattr(OutputMatch, "ref_name", None)
    if not _ref_display:
        _ref_display = (
            "reference_df" if (ref_df is not None and len(in_ref) == 0) else "reference"
        )

    _search_cols = in_colnames or []
    _ref_cols = in_refcol or []
    _context_md = (
        "## Run context\n"
        f"- **Search input**: `{_search_display}`\n"
        f"- **Reference input**: `{_ref_display}`\n"
        f"- **Search address columns**: `{', '.join(map(str, _search_cols))}`\n"
        f"- **Reference address columns**: `{', '.join(map(str, _ref_cols))}`\n"
    )

    final_summary = _context_md + "\n" + final_summary + "\n\n" + time_out

    estimate_total_processing_time = sum_numbers_before_seconds(time_out)
    print("Estimated total processing time:", str(estimate_total_processing_time))

    # Build/export a results-based summary table (eligible-only percentages)
    summary_table_df = build_results_summary_table(OutputMatch.results_on_orig_df)
    summary_table_md = "## Summary table\n\n" + results_summary_table_to_markdown(
        summary_table_df
    )
    _out_folder = getattr(OutputMatch, "output_folder", None) or output_folder
    if _out_folder and (not _out_folder.endswith(("\\", "/"))):
        _out_folder = _out_folder + os.sep
    # Use user-meaningful names when we don't have real filenames (e.g. dataframe/text inputs).
    _search_name_for_id = getattr(OutputMatch, "file_name", None)
    if in_text and str(in_text).strip():
        _search_name_for_id = "single_address"
    elif (search_df is not None) and (len(in_file) == 0):
        _search_name_for_id = "search_df"

    _ref_name_for_id = getattr(OutputMatch, "ref_name", None)
    if in_api:
        _ref_name_for_id = "api"

    _search_id = _safe_file_id(_search_name_for_id)
    _ref_id = _safe_file_id(_ref_name_for_id)
    _run_id = f"{_search_id}_{_ref_id}_{today_rev}"

    # Final output files (these are the only CSVs we always want to write).
    OutputMatch.results_orig_df_name = _out_folder + f"results_{_run_id}.csv"
    OutputMatch.match_outputs_name = _out_folder + f"diagnostics_{_run_id}.csv"
    summary_table_name = _out_folder + f"summary_{_run_id}.csv"
    if save_output_files:
        summary_table_df.to_csv(summary_table_name, index=False)

    _em_col = None
    if OutputMatch.existing_match_cols:
        _em_col = (
            OutputMatch.existing_match_cols[0]
            if isinstance(OutputMatch.existing_match_cols, list)
            else OutputMatch.existing_match_cols
        )

    # Surface the in_existing value from the search data as a dedicated column so it
    # is always visible in the results CSV alongside the reference-derived join column.
    OutputMatch.results_on_orig_df = add_search_data_existing_col_to_results(
        OutputMatch.results_on_orig_df,
        _em_col,
    )

    # Defensive de-duplication: in some UI/state paths (especially street-only runs),
    # upstream merges can occasionally introduce duplicate keys. Ensure the final
    # results contain at most one row per key.
    _final_key = (
        OutputMatch.search_df_key_field
        if (
            hasattr(OutputMatch, "search_df_key_field")
            and isinstance(OutputMatch.search_df_key_field, str)
            and (
                OutputMatch.search_df_key_field
                in OutputMatch.results_on_orig_df.columns
            )
        )
        else ("index" if "index" in OutputMatch.results_on_orig_df.columns else None)
    )
    if _final_key is not None:
        _dup_mask = OutputMatch.results_on_orig_df.duplicated(subset=[_final_key])
        _dup_count = int(_dup_mask.sum())
        if _dup_count > 0:
            print(
                f"Final results contain {_dup_count} duplicate key rows on "
                f"{_final_key!r}; collapsing to one row per key."
            )
            _sort_cols = [_final_key]
            _ascending = [True]
            if (
                "Matched with reference address"
                in OutputMatch.results_on_orig_df.columns
            ):
                _sort_cols.append("Matched with reference address")
                _ascending.append(False)
            OutputMatch.results_on_orig_df = (
                OutputMatch.results_on_orig_df.sort_values(
                    by=_sort_cols, ascending=_ascending, kind="stable"
                )
                .drop_duplicates(subset=[_final_key], keep="first")
                .reset_index(drop=True)
            )

    # Also attach the search-side in_existing value onto the diagnostics output so it can be
    # audited alongside match scores/methods. Use the original/pre-filter search df because
    # `search_df_cleaned` typically only contains key/full_address/postcode.
    if _em_col:
        diag_output_col = f"{_em_col} (from search data)"
        needs_fill = (
            diag_output_col not in OutputMatch.match_results_output.columns
        ) or OutputMatch.match_results_output[diag_output_col].isna().all()
        if needs_fill and (
            OutputMatch.search_df_key_field in OutputMatch.match_results_output.columns
        ):
            source_df = getattr(OutputMatch, "pre_filter_search_df", pd.DataFrame())
            if source_df is None or source_df.empty:
                source_df = getattr(OutputMatch, "search_df", pd.DataFrame())

            _src_col = None
            if isinstance(source_df, pd.DataFrame) and (not source_df.empty):
                if _em_col in source_df.columns:
                    _src_col = _em_col
                else:
                    _alt = f"__search_side_{_em_col}"
                    if _alt in source_df.columns:
                        _src_col = _alt

            if _src_col and (OutputMatch.search_df_key_field in source_df.columns):
                # Use a key->value map to avoid merge column collisions (`_x`/`_y`).
                _map = (
                    source_df[[OutputMatch.search_df_key_field, _src_col]]
                    .copy()
                    .drop_duplicates(
                        subset=[OutputMatch.search_df_key_field], keep="first"
                    )
                )
                _map[OutputMatch.search_df_key_field] = _map[
                    OutputMatch.search_df_key_field
                ].astype(str)
                _series_map = _map.set_index(OutputMatch.search_df_key_field)[_src_col]

                _keys = OutputMatch.match_results_output[
                    OutputMatch.search_df_key_field
                ].astype(str)
                mapped_vals = _keys.map(_series_map)

                if diag_output_col in OutputMatch.match_results_output.columns:
                    _cur = OutputMatch.match_results_output[diag_output_col]
                    OutputMatch.match_results_output[diag_output_col] = (
                        mapped_vals.where(mapped_vals.notna(), _cur)
                    )
                else:
                    OutputMatch.match_results_output[diag_output_col] = mapped_vals

    # Ensure final output files are written once from the merged result.
    essential_results_cols = [
        OutputMatch.search_df_key_field,
        "Search data address",
        "Excluded from search",
        "Matched with reference address",
        "ref_index",
        "Reference matched address",
        "Reference file",
    ]
    essential_results_cols.extend(OutputMatch.new_join_col)
    if _em_col:
        essential_results_cols.append(f"{_em_col} (from search data)")
    essential_results_cols = [
        col
        for col in essential_results_cols
        if col in OutputMatch.results_on_orig_df.columns
    ]

    if "fuzzy_score" in OutputMatch.match_results_output.columns:
        OutputMatch.match_results_output["fuzzy_score"] = pd.to_numeric(
            OutputMatch.match_results_output["fuzzy_score"], errors="coerce"
        ).round(2)
    if "wratio_score" in OutputMatch.match_results_output.columns:
        OutputMatch.match_results_output["wratio_score"] = pd.to_numeric(
            OutputMatch.match_results_output["wratio_score"], errors="coerce"
        ).round(2)

    if save_output_files:
        OutputMatch.match_results_output.to_csv(
            OutputMatch.match_outputs_name, index=None
        )
        if essential_results_cols:
            OutputMatch.results_on_orig_df[essential_results_cols].to_csv(
                OutputMatch.results_orig_df_name, index=None
            )
        else:
            OutputMatch.results_on_orig_df.to_csv(
                OutputMatch.results_orig_df_name, index=None
            )

        output_files.extend(
            [
                OutputMatch.results_orig_df_name,
                OutputMatch.match_outputs_name,
                summary_table_name,
            ]
        )

    print("Final matching summary:", final_summary)
    print("Summary table:", summary_table_md)

    return final_summary, output_files, estimate_total_processing_time, summary_table_md


# Run a match run for a single batch
def create_simple_batch_ranges(
    df: PandasDataFrame, ref_df: PandasDataFrame, batch_size: int, ref_batch_size: int
):
    total_rows = df.shape[0]
    ref_total_rows = ref_df.shape[0]

    # Creating bottom and top limits for search data
    search_ranges = []
    for start in range(0, total_rows, batch_size):
        end = min(
            start + batch_size - 1, total_rows - 1
        )  # Adjusted to get the top limit
        search_ranges.append((start, end))

    # Creating bottom and top limits for reference data
    ref_ranges = []
    for start in range(0, ref_total_rows, ref_batch_size):
        end = min(
            start + ref_batch_size - 1, ref_total_rows - 1
        )  # Adjusted to get the top limit
        ref_ranges.append((start, end))

    # Create DataFrame with combinations of search_range and ref_range
    result_data = []
    for search_range in search_ranges:
        for ref_range in ref_ranges:
            result_data.append((search_range, ref_range))

    range_df = pd.DataFrame(result_data, columns=["search_range", "ref_range"])

    return range_df


def create_street_batch_ranges(
    df: PandasDataFrame, ref_df: PandasDataFrame, batch_size: int
) -> PandasDataFrame:
    """
    Create search batches that compare against the full reference index list.
    Used when postcode blocking is disabled/unavailable.
    """
    # Use **positional** row indices (0..n-1), not `df.index` labels. After
    # `prepare_search_address`, `search_df_cleaned` often has a default RangeIndex
    # while `InitMatch.search_df` keeps the original file/filter index; batching
    # on cleaned index labels and then `.index.isin(...)` on `search_df` yields
    # empty batches.
    n_search = len(df)
    ref_indexes = ref_df.index.tolist()

    if n_search == 0:
        return pd.DataFrame(
            data={
                "search_range": [[]],
                "ref_range": [ref_indexes],
                "batch_length": [0],
                "ref_length": [len(ref_indexes)],
            }
        )

    search_chunks = [
        list(range(i, min(i + batch_size, n_search)))
        for i in range(0, n_search, batch_size)
    ]
    out_df = pd.DataFrame(
        data={
            "search_range": search_chunks,
            "ref_range": [ref_indexes] * len(search_chunks),
            "batch_length": [len(chunk) for chunk in search_chunks],
            "ref_length": [len(ref_indexes)] * len(search_chunks),
        }
    )

    return out_df


def create_batch_ranges(
    df: PandasDataFrame,
    ref_df: PandasDataFrame,
    batch_size: int,
    ref_batch_size: int,
    search_postcode_col: str,
    ref_postcode_col: str,
    search_df_key_field: Optional[str] = None,
    ref_key_field: Optional[str] = None,
):
    """
    Create batches of address indexes for search and reference dataframes based on shortened postcodes.
    """

    _search_k = search_df_key_field or "index"
    _ref_k = ref_key_field or "ref_index"

    def _batch_id_series(side: pd.DataFrame, key_field: str) -> pd.Series:
        if key_field in side.columns:
            ser = _resolve_column_series(side, key_field)
            if ser is not None:
                return ser
        return pd.Series(side.index, index=side.index)

    def _clean_postcode_minus_last(series: pd.Series) -> pd.Series:
        return (
            series.fillna("")
            .astype(str)
            .str.lower()
            .str.strip()
            .str.replace(r"\s+", "", regex=True)
            .str[:-1]
        )

    # If df sizes are smaller than the batch size limits, no need to run through everything
    if len(df) < batch_size and len(ref_df) < ref_batch_size:
        print(
            "Dataframe sizes are smaller than maximum batch sizes, no need to split data."
        )
        lengths_df = pd.DataFrame(
            data={
                "search_range": [_batch_id_series(df, _search_k).tolist()],
                "ref_range": [_batch_id_series(ref_df, _ref_k).tolist()],
                "batch_length": len(df),
                "ref_length": len(ref_df),
            }
        )
        return lengths_df

    # df.index = df[search_postcode_col]

    df["index"] = _batch_id_series(df, _search_k)
    ref_df["index"] = _batch_id_series(ref_df, _ref_k)

    # Remove the last character of postcode
    df["postcode_minus_last_character"] = _clean_postcode_minus_last(
        df[search_postcode_col]
    )
    ref_df["postcode_minus_last_character"] = _clean_postcode_minus_last(
        ref_df[ref_postcode_col]
    )

    # ---- Explain why some eligible search rows are not batched ----
    # Postcode batching only considers postcodes with length >= 4 (after removing spaces and last char)
    # and only builds batches for postcode groups that exist in BOTH search and reference.
    try:
        _search_total = len(df)
        _ref_total = len(ref_df)
        _s_pc = df["postcode_minus_last_character"].fillna("").astype(str)
        _r_pc = ref_df["postcode_minus_last_character"].fillna("").astype(str)
        _s_ok = _s_pc.str.len().ge(4)
        _r_ok = _r_pc.str.len().ge(4)

        _search_rows_short_or_blank = int((~_s_ok).sum())
        _ref_rows_short_or_blank = int((~_r_ok).sum())
        _search_rows_eligible = int(_s_ok.sum())
        _ref_rows_eligible = int(_r_ok.sum())

        _s_counts = _s_pc[_s_ok].value_counts(dropna=False)
        _r_counts = _r_pc[_r_ok].value_counts(dropna=False)
        _s_postcodes = set(_s_counts.index.tolist())
        _r_postcodes = set(_r_counts.index.tolist())
        _common_postcodes = _s_postcodes.intersection(_r_postcodes)

        _search_rows_in_common = (
            int(_s_counts.loc[list(_common_postcodes)].sum())
            if _common_postcodes
            else 0
        )
        _search_rows_not_in_ref = _search_rows_eligible - _search_rows_in_common

        if _fuzzy_match_debug_enabled():
            print("[FUZZY_MATCH_DEBUG] batch planning (postcode mode)")
            print(
                f"  search rows total={_search_total} "
                f"eligible_postcode_rows={_search_rows_eligible} "
                f"short_or_blank_postcode_rows={_search_rows_short_or_blank}"
            )
            print(
                f"  ref rows total={_ref_total} "
                f"eligible_postcode_rows={_ref_rows_eligible} "
                f"short_or_blank_postcode_rows={_ref_rows_short_or_blank}"
            )
            print(
                f"  unique truncated postcodes: search={len(_s_postcodes)} "
                f"ref={len(_r_postcodes)} common={len(_common_postcodes)}"
            )
            print(
                f"  search rows with eligible truncated postcode NOT in reference={_search_rows_not_in_ref}"
            )
    except Exception:
        # Never fail the run due to debug stats.
        pass

    unique_postcodes = (
        df["postcode_minus_last_character"][
            df["postcode_minus_last_character"].str.len() >= 4
        ]
        .unique()
        .tolist()
    )

    df = df.set_index("postcode_minus_last_character")
    ref_df = ref_df.set_index("postcode_minus_last_character")

    df = df.sort_index()
    ref_df = ref_df.sort_index()

    # Overall batch variables
    batch_indexes = []
    ref_indexes = []
    batch_lengths = []
    ref_lengths = []

    # Current batch variables for loop
    current_batch = []
    current_ref_batch = []
    current_batch_length = []
    current_ref_length = []

    unique_postcodes_iterator = unique_postcodes.copy()

    while unique_postcodes_iterator:

        unique_postcodes_loop = unique_postcodes_iterator.copy()

        # print("Current loop postcodes: ", unique_postcodes_loop)

        for current_postcode in unique_postcodes_loop:

            if (
                len(current_batch) >= batch_size
                or len(current_ref_batch) >= ref_batch_size
            ):
                # print("Batch length reached - breaking")
                break

            # Only batch postcode groups that exist in BOTH search and reference.
            # If a truncated postcode is only present on the search side, it will
            # not be attempted in postcode-blocked matching (it may still be
            # matchable via street-only fallback elsewhere).
            if (current_postcode in df.index) and (current_postcode in ref_df.index):
                current_postcode_search_data_add = df.loc[[current_postcode]]
                current_postcode_ref_data_add = ref_df.loc[[current_postcode]]

                if not current_postcode_search_data_add.empty:
                    current_batch.extend(current_postcode_search_data_add["index"])
                if not current_postcode_ref_data_add.empty:
                    current_ref_batch.extend(current_postcode_ref_data_add["index"])

            unique_postcodes_iterator.remove(current_postcode)

        # Append the batch data to the master lists and reset lists
        batch_indexes.append(current_batch)
        ref_indexes.append(current_ref_batch)

        current_batch_length = len(current_batch)
        current_ref_length = len(current_ref_batch)

        batch_lengths.append(current_batch_length)
        ref_lengths.append(current_ref_length)

        current_batch = []
        current_ref_batch = []
        current_batch_length = []
        current_ref_length = []

    # Create df to store lengths
    lengths_df = pd.DataFrame(
        data={
            "search_range": batch_indexes,
            "ref_range": ref_indexes,
            "batch_length": batch_lengths,
            "ref_length": ref_lengths,
        }
    )

    # Report coverage: how many eligible search rows were actually assigned to batches.
    try:
        if _fuzzy_match_debug_enabled():
            _all_search_keys = set()
            for chunk in batch_indexes:
                _all_search_keys.update(chunk)
            print(
                "[FUZZY_MATCH_DEBUG] batch planning coverage (postcode mode)\n"
                f"  batches={len(batch_indexes)} "
                f"search rows covered by batches={len(_all_search_keys)} "
                f"of total search rows={len(df)}"
            )
    except Exception:
        pass

    return lengths_df


def _print_batches_to_run_overview(
    range_df: pd.DataFrame,
    *,
    use_postcode_mode: bool,
    overflow_range_df: Optional[pd.DataFrame],
) -> None:
    """
    Console table of all batches: postcode and/or street, plus optional street-overflow
    chunks (search rows not in any postcode-overlap batch).
    """
    try:
        rows_out: List[dict] = []
        for i in range(range_df.shape[0]):
            r = range_df.iloc[i]
            sr = r["search_range"]
            rr = r["ref_range"]
            rows_out.append(
                {
                    "batch_type": "postcode" if use_postcode_mode else "street",
                    "search_range_summary": _summarise_index_list(sr),
                    "batch_length": r.get("batch_length", len(list(sr))),
                    "ref_range_summary": _summarise_index_list(rr),
                    "ref_length": r.get("ref_length", len(list(rr))),
                }
            )
        if overflow_range_df is not None and not overflow_range_df.empty:
            for j in range(overflow_range_df.shape[0]):
                r = overflow_range_df.iloc[j]
                sr = r["search_range"]
                rr = r["ref_range"]
                rows_out.append(
                    {
                        "batch_type": "street_overflow",
                        "search_range_summary": _summarise_index_list(sr),
                        "batch_length": r.get("batch_length", len(list(sr))),
                        "ref_range_summary": _summarise_index_list(rr),
                        "ref_length": r.get("ref_length", len(list(rr))),
                    }
                )
        overview = pd.DataFrame(rows_out)
        cols = [
            "batch_type",
            "search_range_summary",
            "batch_length",
            "ref_range_summary",
            "ref_length",
        ]
        print("Batches to run in this session:")
        print(overview[cols].to_string(index=True))
    except Exception:
        print("Batches to run in this session: ", range_df)


def _summarise_index_list(values) -> str:
    """
    Summarise a list/iterable of numeric indices for console display.

    Examples:
    - [] -> "∅ (n=0)"
    - [10, 11, 12] -> "10–12 (n=3)"
    """
    if values is None:
        return "∅ (n=0)"

    try:
        seq = list(values)
    except TypeError:
        seq = [values]

    if not seq:
        return "∅ (n=0)"

    try:
        mn = min(seq)
        mx = max(seq)
    except TypeError:
        return f"{seq[0]}…{seq[-1]} (n={len(seq)})"

    if mn == mx:
        return f"{mn} (n={len(seq)})"

    return f"{mn}–{mx} (n={len(seq)})"


def run_single_match_batch(
    InitialMatch: MatcherClass,
    batch_n: int,
    total_batches: int,
    use_postcode_blocker: bool = True,
    write_outputs: bool = True,
    show_progress: bool = True,
    progress=gr.Progress(track_tqdm=True),
    print_match_stage_summary_to_console: bool = PRINT_MATCH_STAGE_SUMMARY_TO_CONSOLE,
):
    """
    Over-arching function for running a single batch of data through the full matching process. Calls fuzzy matching, then neural network match functions in order. It outputs a summary of the match, and a MatcherClass with the matched data included.
    """

    progress_update = progress if show_progress else (lambda *args, **kwargs: None)

    if run_fuzzy_match:

        overall_tic = time.perf_counter()

        progress_update(
            0,
            desc="Batch "
            + str(batch_n + 1)
            + " of "
            + str(total_batches)
            + ". Fuzzy match - non-standardised dataset",
        )
        df_name = "Fuzzy not standardised"

        """ FUZZY MATCHING """

        """ Run fuzzy match on non-standardised dataset """

        FuzzyNotStdMatch = orchestrate_single_match_batch(
            Matcher=copy.copy(InitialMatch),
            standardise=False,
            nnet=False,
            file_stub="not_std_",
            df_name=df_name,
            use_postcode_blocker=use_postcode_blocker,
            write_outputs=write_outputs,
            show_block_progress=show_progress,
        )

        if FuzzyNotStdMatch.abort_flag:
            message = "Nothing to match! Aborting address check."
            print(message)
            return message, InitialMatch

        FuzzyNotStdMatch = combine_two_matches(
            InitialMatch,
            FuzzyNotStdMatch,
            df_name,
            write_outputs=write_outputs,
            print_match_stage_summary_to_console=print_match_stage_summary_to_console,
        )

        full_match_series = _bool_mask(
            FuzzyNotStdMatch.match_results_output["full_match"], default=False
        )
        if (len(FuzzyNotStdMatch.search_df_not_matched) == 0) | (
            sum(
                FuzzyNotStdMatch.match_results_output[~full_match_series]["fuzzy_score"]
            )
            == 0
        ):
            overall_toc = time.perf_counter()
            time_out = (
                f"The fuzzy match script took {overall_toc - overall_tic:0.1f} seconds"
            )
            FuzzyNotStdMatch.output_summary = (
                FuzzyNotStdMatch.output_summary + " Neural net match not attempted. "
            )  # + time_out
            return FuzzyNotStdMatch.output_summary, FuzzyNotStdMatch

        """ Run fuzzy match on standardised dataset """

        progress_update(
            0.25,
            desc="Batch "
            + str(batch_n + 1)
            + " of "
            + str(total_batches)
            + ". Fuzzy match - standardised dataset",
        )
        df_name = "Fuzzy standardised"

        FuzzyStdMatch = orchestrate_single_match_batch(
            Matcher=copy.copy(FuzzyNotStdMatch),
            standardise=True,
            nnet=False,
            file_stub="std_",
            df_name=df_name,
            use_postcode_blocker=use_postcode_blocker,
            write_outputs=write_outputs,
            show_block_progress=show_progress,
        )
        FuzzyStdMatch = combine_two_matches(
            FuzzyNotStdMatch,
            FuzzyStdMatch,
            df_name,
            write_outputs=write_outputs,
            print_match_stage_summary_to_console=print_match_stage_summary_to_console,
        )

        """ Continue if reference file in correct format, and neural net model exists. Also if data not too long """
        if (
            (len(FuzzyStdMatch.search_df_not_matched) == 0)
            | (not FuzzyStdMatch.standard_llpg_format)
            | (not os.path.exists(FuzzyStdMatch.model_dir_name + "/saved_model.zip"))
            | (not run_nnet_match)
        ):
            overall_toc = time.perf_counter()
            time_out = (
                f"The fuzzy match script took {overall_toc - overall_tic:0.1f} seconds"
            )
            FuzzyStdMatch.output_summary = (
                FuzzyStdMatch.output_summary + " Neural net match not attempted. "
            )  # + time_out
            return FuzzyStdMatch.output_summary, FuzzyStdMatch

    if run_nnet_match:

        """NEURAL NET"""

        if not run_fuzzy_match:
            FuzzyStdMatch = copy.copy(InitialMatch)
            overall_tic = time.perf_counter()

        """ First on non-standardised addresses """
        progress_update(
            0.50,
            desc="Batch "
            + str(batch_n + 1)
            + " of "
            + str(total_batches)
            + ". Neural net - non-standardised dataset",
        )
        df_name = "Neural net not standardised"

        FuzzyNNetNotStdMatch = orchestrate_single_match_batch(
            Matcher=copy.copy(FuzzyStdMatch),
            standardise=False,
            nnet=True,
            file_stub="nnet_not_std_",
            df_name=df_name,
            write_outputs=write_outputs,
            show_block_progress=show_progress,
        )
        FuzzyNNetNotStdMatch = combine_two_matches(
            FuzzyStdMatch,
            FuzzyNNetNotStdMatch,
            df_name,
            write_outputs=write_outputs,
            print_match_stage_summary_to_console=print_match_stage_summary_to_console,
        )

        if len(FuzzyNNetNotStdMatch.search_df_not_matched) == 0:
            overall_toc = time.perf_counter()
            time_out = (
                f"The whole match script took {overall_toc - overall_tic:0.1f} seconds"
            )
            FuzzyNNetNotStdMatch.output_summary = (
                FuzzyNNetNotStdMatch.output_summary
            )  # + time_out
            return FuzzyNNetNotStdMatch.output_summary, FuzzyNNetNotStdMatch

        """ Next on standardised addresses """
        progress_update(
            0.75,
            desc="Batch "
            + str(batch_n + 1)
            + " of "
            + str(total_batches)
            + ". Neural net - standardised dataset",
        )
        df_name = "Neural net standardised"

        FuzzyNNetStdMatch = orchestrate_single_match_batch(
            Matcher=copy.copy(FuzzyNNetNotStdMatch),
            standardise=True,
            nnet=True,
            file_stub="nnet_std_",
            df_name=df_name,
            write_outputs=write_outputs,
            show_block_progress=show_progress,
        )
        FuzzyNNetStdMatch = combine_two_matches(
            FuzzyNNetNotStdMatch,
            FuzzyNNetStdMatch,
            df_name,
            write_outputs=write_outputs,
            print_match_stage_summary_to_console=print_match_stage_summary_to_console,
        )

        if not run_fuzzy_match:
            overall_toc = time.perf_counter()
            time_out = f"The neural net match script took {overall_toc - overall_tic:0.1f} seconds"
            FuzzyNNetStdMatch.output_summary = (
                FuzzyNNetStdMatch.output_summary + " Only Neural net match attempted. "
            )  # + time_out
            return FuzzyNNetStdMatch.output_summary, FuzzyNNetStdMatch

    overall_toc = time.perf_counter()
    time_out = f"The whole match script took {overall_toc - overall_tic:0.1f} seconds"

    summary_of_summaries = (
        FuzzyNotStdMatch.output_summary
        + "\n"
        + FuzzyStdMatch.output_summary
        + "\n"
        + FuzzyNNetStdMatch.output_summary
        + "\n"
        + time_out
    )

    return summary_of_summaries, FuzzyNNetStdMatch


# Overarching functions
def orchestrate_single_match_batch(
    Matcher: MatcherClass,
    standardise=False,
    nnet=False,
    file_stub="not_std_",
    df_name="Fuzzy not standardised",
    use_postcode_blocker: bool = True,
    write_outputs: bool = True,
    show_block_progress: bool = True,
):

    today_rev = datetime.now().strftime("%Y%m%d")

    # print(Matcher.standardise)
    Matcher.standardise = standardise
    retain_prior_fuzzy_outputs = False

    if Matcher.search_df_not_matched.empty:
        print("Nothing to match! At start of preparing run.")
        return Matcher

    if not nnet:
        _existing_col = None
        if Matcher.existing_match_cols:
            _existing_col = (
                Matcher.existing_match_cols[0]
                if isinstance(Matcher.existing_match_cols, list)
                else Matcher.existing_match_cols
            )
        (
            diag_shortlist,
            diag_best_match,
            match_results_output,
            results_on_orig_df,
            summary,
            search_address_cols,
        ) = full_fuzzy_match(
            Matcher.search_df_not_matched.copy(),
            Matcher.standardise,
            Matcher.search_df_key_field,
            Matcher.search_address_cols,
            Matcher.search_df_cleaned,
            Matcher.search_df_after_stand,
            Matcher.search_df_after_full_stand,
            Matcher.ref_df_cleaned,
            Matcher.ref_df_after_stand,
            Matcher.ref_df_after_full_stand,
            Matcher.fuzzy_match_limit,
            Matcher.fuzzy_scorer_used,
            Matcher.new_join_col,
            use_postcode_blocker=use_postcode_blocker,
            run_street_matching=getattr(Matcher, "run_street_matching", True),
            existing_match_col=_existing_col,
            pre_filter_search_df=getattr(Matcher, "pre_filter_search_df", None),
            show_block_progress=show_block_progress,
        )
        if match_results_output.empty:
            _prior = Matcher.match_results_output
            _prior_nonempty = isinstance(_prior, pd.DataFrame) and not _prior.empty
            if standardise and _prior_nonempty:
                print(
                    "Standardised fuzzy produced no new match rows; retaining prior fuzzy outputs."
                )
                Matcher.abort_flag = False
                retain_prior_fuzzy_outputs = True
            else:
                print("Match results empty")
                Matcher.abort_flag = True
                return Matcher
        else:
            Matcher.diag_shortlist = diag_shortlist
            Matcher.diag_best_match = diag_best_match
            Matcher.match_results_output = match_results_output

    else:
        _existing_col = None
        if Matcher.existing_match_cols:
            _existing_col = (
                Matcher.existing_match_cols[0]
                if isinstance(Matcher.existing_match_cols, list)
                else Matcher.existing_match_cols
            )
        match_results_output, results_on_orig_df, summary, predict_df_nnet = (
            full_nn_match(
                Matcher.ref_address_cols,
                Matcher.search_df_not_matched.copy(),
                Matcher.search_address_cols,
                Matcher.search_df_key_field,
                Matcher.standardise,
                Matcher.exported_model[0],
                Matcher.matching_variables,
                Matcher.text_columns,
                Matcher.weights,
                Matcher.fuzzy_method,
                Matcher.score_cut_off,
                Matcher.match_results_output.copy(),
                Matcher.filter_to_lambeth_pcodes,
                Matcher.model_type,
                Matcher.word_to_index,
                Matcher.cat_to_idx,
                Matcher.device,
                Matcher.vocab,
                Matcher.labels_list,
                Matcher.search_df_cleaned,
                Matcher.ref_df_cleaned,
                Matcher.ref_df_after_stand,
                Matcher.search_df_after_stand,
                Matcher.search_df_after_full_stand,
                Matcher.new_join_col,
                existing_match_col=_existing_col,
                pre_filter_search_df=getattr(Matcher, "pre_filter_search_df", None),
            )
        )

        if match_results_output.empty:
            print("Match results empty")
            Matcher.abort_flag = True
            return Matcher
        else:
            Matcher.match_results_output = match_results_output
            Matcher.predict_df_nnet = predict_df_nnet

    # Save to file
    if nnet or (not retain_prior_fuzzy_outputs):
        Matcher.results_on_orig_df = results_on_orig_df
        Matcher.summary = summary
    Matcher.output_summary = create_match_summary(
        Matcher.match_results_output, df_name=df_name
    )

    _out_folder = getattr(Matcher, "output_folder", None) or output_folder
    if _out_folder and (not _out_folder.endswith(("\\", "/"))):
        _out_folder = _out_folder + os.sep
    Matcher.match_outputs_name = (
        _out_folder + "diagnostics_" + file_stub + today_rev + ".csv"
    )
    Matcher.results_orig_df_name = (
        _out_folder + "results_" + file_stub + today_rev + ".csv"
    )

    if "fuzzy_score" in Matcher.match_results_output.columns:
        Matcher.match_results_output["fuzzy_score"] = pd.to_numeric(
            Matcher.match_results_output["fuzzy_score"], errors="coerce"
        ).round(2)
    if "wratio_score" in Matcher.match_results_output.columns:
        Matcher.match_results_output["wratio_score"] = pd.to_numeric(
            Matcher.match_results_output["wratio_score"], errors="coerce"
        ).round(2)
    if write_outputs:
        Matcher.match_results_output.to_csv(Matcher.match_outputs_name, index=None)
        Matcher.results_on_orig_df.to_csv(Matcher.results_orig_df_name, index=None)

    return Matcher


# Overarching fuzzy match function
def full_fuzzy_match(
    search_df: PandasDataFrame,
    standardise: bool,
    search_df_key_field: str,
    search_address_cols: List[str],
    search_df_cleaned: PandasDataFrame,
    search_df_after_stand: PandasDataFrame,
    search_df_after_full_stand: PandasDataFrame,
    ref_df_cleaned: PandasDataFrame,
    ref_df_after_stand: PandasDataFrame,
    ref_df_after_full_stand: PandasDataFrame,
    fuzzy_match_limit: float,
    fuzzy_scorer_used: str,
    new_join_col: List[str],
    fuzzy_search_addr_limit: float = 100,
    filter_to_lambeth_pcodes: bool = False,
    use_postcode_blocker: bool = True,
    run_street_matching: bool = True,
    existing_match_col: Optional[str] = None,
    pre_filter_search_df: Optional[PandasDataFrame] = None,
    show_block_progress: bool = True,
):
    """
    Compare addresses in a 'search address' dataframe with a 'reference address' dataframe by using fuzzy matching from the rapidfuzz package, blocked by postcode and then street.
    """

    # Break if search item has length 0
    if search_df.empty:
        out_error = "Nothing to match! Just started fuzzy match."
        print(out_error)
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            out_error,
            search_address_cols,
        )

    # If standardise is true, replace relevant variables with standardised versions
    if standardise:
        df_name = "standardised address"
        search_df_after_stand = search_df_after_full_stand
        ref_df_after_stand = ref_df_after_full_stand
    else:
        df_name = "non-standardised address"

    print("Fuzzy matching with " + df_name)

    # RUN WITH POSTCODE AS A BLOCKER #
    can_run_postcode_blocker = (
        use_postcode_blocker
        and _column_has_usable_values(search_df_after_stand, "postcode_search")
        and _column_has_usable_values(ref_df_after_stand, "postcode_search")
    )
    postcode_blocking_in_use = bool(can_run_postcode_blocker)
    _log_postcode_blocker_debug(
        standardise=standardise,
        use_postcode_blocker=use_postcode_blocker,
        search_df_after_stand=search_df_after_stand,
        ref_df_after_stand=ref_df_after_stand,
        can_run_postcode_blocker=can_run_postcode_blocker,
    )

    if can_run_postcode_blocker:
        # Exclude blank postcodes from postcode-blocked matching.
        # (They can still be considered later in street-blocked matching.)
        _s_pc = _resolve_column_series(search_df_after_stand, "postcode_search")
        _r_pc = _resolve_column_series(ref_df_after_stand, "postcode_search")
        search_pc = (
            _s_pc.fillna("").astype(str).str.strip()
            if _s_pc is not None
            else pd.Series("", index=search_df_after_stand.index)
        )
        ref_pc = (
            _r_pc.fillna("").astype(str).str.strip()
            if _r_pc is not None
            else pd.Series("", index=ref_df_after_stand.index)
        )
        search_df_after_stand_pc = search_df_after_stand.loc[search_pc.ne("")].copy()
        ref_df_after_stand_pc = ref_df_after_stand.loc[ref_pc.ne("")].copy()

        search_df_after_stand_pc = add_fuzzy_block_sequence_col(
            search_df_after_stand_pc, "postcode_search"
        )
        search_df_after_stand = search_df_after_stand.copy()
        search_df_after_stand["_fuzzy_block_seq"] = np.nan
        search_df_after_stand.loc[
            search_df_after_stand_pc.index, "_fuzzy_block_seq"
        ] = search_df_after_stand_pc["_fuzzy_block_seq"].values

        search_df_after_stand_series = search_df_after_stand_pc.set_index(
            "postcode_search"
        )["search_address_stand"].sort_index()
        ref_df_after_stand_series = ref_df_after_stand_pc.set_index("postcode_search")[
            "ref_address_stand"
        ].sort_index()
        ref_df_after_stand_series_checked = ref_df_after_stand_series.copy()[
            ref_df_after_stand_series.index.isin(
                search_df_after_stand_series.index.tolist()
            )
        ]

        if len(ref_df_after_stand_series_checked) == 0:
            print(
                "No relevant postcode groups found; switching to street-only blocker."
            )
            can_run_postcode_blocker = False
            diag_shortlist = pd.DataFrame()
            diag_best_match = pd.DataFrame()
            match_results_output = pd.DataFrame()
            search_df_not_matched = search_df_after_stand.copy()
        else:
            print("Fuzzy matching with postcode as blocker")

            time.perf_counter()
            results = string_match_by_post_code_multiple(
                match_address_series=search_df_after_stand_series.copy(),
                reference_address_series=ref_df_after_stand_series_checked,
                search_limit=fuzzy_search_addr_limit,
                scorer_name=fuzzy_scorer_used,
                fuzzy_match_limit=fuzzy_match_limit,
                show_progress=show_block_progress,
            )

            time.perf_counter()
            # print(f"Performed the fuzzy match in {toc - tic:0.1f} seconds")

            # Create result dfs
            match_results_output, diag_shortlist, diag_best_match = (
                _create_fuzzy_match_results_output(
                    results,
                    search_df_after_stand,
                    ref_df_cleaned,
                    ref_df_after_stand,
                    fuzzy_match_limit,
                    search_df_cleaned,
                    search_df_key_field,
                    new_join_col,
                    standardise,
                    blocker_col="Postcode",
                )
            )
            match_results_output["match_method"] = "Fuzzy match - postcode"
            search_df_not_matched = filter_not_matched(
                match_results_output, search_df_after_stand, search_df_key_field
            )
    else:
        print("Skipping postcode blocker and running street-only fuzzy matching.")
        diag_shortlist = pd.DataFrame()
        diag_best_match = pd.DataFrame()
        match_results_output = pd.DataFrame()
        search_df_not_matched = search_df_after_stand.copy()

    # If nothing left to match, break
    if not match_results_output.empty:
        full_match_series = _bool_mask(
            match_results_output["full_match"], default=False
        )

    if (not match_results_output.empty) and (
        (sum(~full_match_series) == 0)
        | (sum(match_results_output[~full_match_series]["fuzzy_score"]) == 0)
    ):
        print("Nothing left to match!")

        summary = create_match_summary(match_results_output, df_name)

        if not isinstance(search_df, str):
            results_on_orig_df = create_results_df(
                match_results_output,
                search_df_cleaned,
                search_df_key_field,
                new_join_col,
                ref_df_cleaned=ref_df_cleaned,
                existing_match_col=existing_match_col,
                pre_filter_search_df=pre_filter_search_df,
            )
        else:
            results_on_orig_df = match_results_output

        return (
            diag_shortlist,
            diag_best_match,
            match_results_output,
            results_on_orig_df,
            summary,
            search_address_cols,
        )

    if (not run_street_matching) and postcode_blocking_in_use:
        if match_results_output.empty:
            out_msg = (
                "Street-based matching is disabled and there are no postcode-blocked "
                "fuzzy results (e.g. no overlapping postcode groups). "
                "Enable street-based matching or check postcode overlap between search and reference."
            )
            print(out_msg)
            return (
                diag_shortlist,
                diag_best_match,
                match_results_output,
                pd.DataFrame(),
                out_msg,
                search_address_cols,
            )
        summary = create_match_summary(match_results_output, df_name)
        if not isinstance(search_df, str):
            results_on_orig_df = create_results_df(
                match_results_output,
                search_df_cleaned,
                search_df_key_field,
                new_join_col,
                ref_df_cleaned=ref_df_cleaned,
                existing_match_col=existing_match_col,
                pre_filter_search_df=pre_filter_search_df,
            )
        else:
            results_on_orig_df = match_results_output
        print("Skipping street-based fuzzy matching (disabled).")
        return (
            diag_shortlist,
            diag_best_match,
            match_results_output,
            results_on_orig_df,
            summary,
            search_address_cols,
        )

    # RUN WITH STREET AS A BLOCKER #

    ### Redo with street as blocker
    search_df_after_stand_street = search_df_not_matched.copy()
    if ("postcode_search" in search_df_after_stand_street.columns) and (
        "postcode_search" in ref_df_after_stand.columns
    ):
        search_df_after_stand_street["search_address_stand_w_pcode"] = (
            search_df_after_stand_street["search_address_stand"]
            + " "
            + search_df_after_stand_street["postcode_search"]
        )
        ref_df_after_stand["ref_address_stand_w_pcode"] = (
            ref_df_after_stand["ref_address_stand"]
            + " "
            + ref_df_after_stand["postcode_search"]
        )
    else:
        search_df_after_stand_street["search_address_stand_w_pcode"] = (
            search_df_after_stand_street["search_address_stand"]
        )
        ref_df_after_stand["ref_address_stand_w_pcode"] = ref_df_after_stand[
            "ref_address_stand"
        ]

    # Prefer an existing street column (if user supplied one), otherwise extract.
    # Note: user-selected input columns can include "street"/"Street".
    if "street" in search_df_after_stand_street.columns:
        street_source_series = search_df_after_stand_street["street"]
    elif "Street" in search_df_after_stand_street.columns:
        street_source_series = search_df_after_stand_street["Street"]
    else:
        street_source_series = search_df_after_stand_street[
            "full_address_search"
        ].apply(extract_street_name)

    search_df_after_stand_street["street"] = (
        street_source_series.fillna("").astype(str).str.strip()
    )

    # Reference: ensure we have a usable "Street" column for street-blocking.
    if "Street" not in ref_df_after_stand.columns:
        if "street" in ref_df_after_stand.columns:
            ref_df_after_stand["Street"] = (
                ref_df_after_stand["street"].fillna("").astype(str).str.strip()
            )
        else:
            ref_df_after_stand["Street"] = (
                ref_df_after_stand["full_address_search"]
                .apply(extract_street_name)
                .fillna("")
                .astype(str)
                .str.strip()
            )
    # Exclude non-postal addresses from street-blocked search
    search_df_after_stand_street.loc[
        search_df_after_stand_street["Excluded from search"]
        == "Excluded - non-postal address",
        "street",
    ] = ""

    # If no street name can be extracted/provided, do not attempt street-blocked matching
    # (e.g. addresses like "Flat 3" should not be considered here).
    no_street_extracted = search_df_after_stand_street["street"].eq("")
    search_df_after_stand_street.loc[
        no_street_extracted
        & (
            search_df_after_stand_street["Excluded from search"] == "Included in search"
        ),
        "Excluded from search",
    ] = "Excluded - no street name extracted"
    search_df_after_stand_street = search_df_after_stand_street.loc[
        ~no_street_extracted
    ].copy()

    search_df_after_stand_street = add_fuzzy_block_sequence_col(
        search_df_after_stand_street, "street"
    )

    ### Create lookup lists
    search_df_match_series_street = search_df_after_stand_street.copy().set_index(
        "street"
    )["search_address_stand"]
    ref_df_after_stand_series_street = ref_df_after_stand.copy().set_index("Street")[
        "ref_address_stand"
    ]

    # Remove rows where street is not in ref_df df
    # index_check = ref_df_after_stand_series_street.index.isin(search_df_match_series_street.index)
    # ref_df_after_stand_series_street_checked = ref_df_after_stand_series_street.copy()[index_check == True]

    ref_df_after_stand_series_street_checked = ref_df_after_stand_series_street.copy()[
        ref_df_after_stand_series_street.index.isin(
            search_df_match_series_street.index.tolist()
        )
    ]

    # If nothing left to match, break
    if (len(ref_df_after_stand_series_street_checked) == 0) | (
        (len(search_df_match_series_street) == 0)
    ):
        if match_results_output.empty:
            print("Nothing relevant in reference data to match with street blocker!")
            return (
                pd.DataFrame(),
                pd.DataFrame(),
                pd.DataFrame(),
                pd.DataFrame(),
                "Nothing relevant in reference data to match!",
                search_address_cols,
            )

        summary = create_match_summary(match_results_output, df_name)

        if not isinstance(search_df, str):
            results_on_orig_df = create_results_df(
                match_results_output,
                search_df_after_stand,
                search_df_key_field,
                new_join_col,
                ref_df_cleaned=ref_df_cleaned,
                existing_match_col=existing_match_col,
                pre_filter_search_df=pre_filter_search_df,
            )
        else:
            results_on_orig_df = match_results_output

        return (
            diag_shortlist,
            diag_best_match,
            match_results_output,
            results_on_orig_df,
            summary,
            search_address_cols,
        )

    print("Fuzzy matching with street as blocker")

    time.perf_counter()
    results_st = string_match_by_post_code_multiple(
        match_address_series=search_df_match_series_street.copy(),
        reference_address_series=ref_df_after_stand_series_street_checked.copy(),
        search_limit=fuzzy_search_addr_limit,
        scorer_name=fuzzy_scorer_used,
        fuzzy_match_limit=fuzzy_match_limit,
        show_progress=show_block_progress,
    )

    time.perf_counter()

    # print(f"Performed the fuzzy match in {toc - tic:0.1f} seconds")

    match_results_output_st, diag_shortlist_st, diag_best_match_st = (
        _create_fuzzy_match_results_output(
            results_st,
            search_df_after_stand_street,
            ref_df_cleaned,
            ref_df_after_stand,
            fuzzy_match_limit,
            search_df_cleaned,
            search_df_key_field,
            new_join_col,
            standardise,
            blocker_col="Street",
        )
    )
    match_results_output_st["match_method"] = "Fuzzy match - street"

    if match_results_output.empty:
        match_results_output_st_out = match_results_output_st
    else:
        match_results_output_st_out = combine_dfs_and_remove_dups(
            match_results_output, match_results_output_st, index_col=search_df_key_field
        )

    match_results_output = match_results_output_st_out

    summary = create_match_summary(match_results_output, df_name)

    ### Join URPN back onto orig df

    if not isinstance(search_df, str):
        results_on_orig_df = create_results_df(
            match_results_output,
            search_df_cleaned,
            search_df_key_field,
            new_join_col,
            ref_df_cleaned=ref_df_cleaned,
            existing_match_col=existing_match_col,
            pre_filter_search_df=pre_filter_search_df,
        )
    else:
        results_on_orig_df = match_results_output

    return (
        diag_shortlist,
        diag_best_match,
        match_results_output,
        results_on_orig_df,
        summary,
        search_address_cols,
    )


# Overarching NN function
def full_nn_match(
    ref_address_cols: List[str],
    search_df: PandasDataFrame,
    search_address_cols: List[str],
    search_df_key_field: str,
    standardise: bool,
    exported_model: list,
    matching_variables: List[str],
    text_columns: List[str],
    weights: dict,
    fuzzy_method: str,
    score_cut_off: float,
    match_results: PandasDataFrame,
    filter_to_lambeth_pcodes: bool,
    model_type: str,
    word_to_index: dict,
    cat_to_idx: dict,
    device: str,
    vocab: List[str],
    labels_list: List[str],
    search_df_cleaned: PandasDataFrame,
    ref_df_cleaned: PandasDataFrame,
    ref_df_after_stand: PandasDataFrame,
    search_df_after_stand: PandasDataFrame,
    search_df_after_full_stand: PandasDataFrame,
    new_join_col: List[str],
    existing_match_col: Optional[str] = None,
    pre_filter_search_df: Optional[PandasDataFrame] = None,
):
    """
    Use a neural network model to partition 'search addresses' into consituent parts in the format of UK Ordnance Survey Land Property Identifier (LPI) addresses. These address components are compared individually against reference addresses in the same format to give an overall match score using the recordlinkage package.
    """

    # Break if search item has length 0
    if search_df.empty:
        out_error = "Nothing to match! At neural net matching stage."
        print(out_error)
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            out_error,
            search_address_cols,
        )

    # If it is the standardisation step, or you have come from the fuzzy match area
    if standardise:  # | (run_fuzzy_match == True & standardise == False):
        df_name = "standardised address"

        search_df_after_stand = search_df_after_full_stand

    else:
        df_name = "non-standardised address"

    # Predict on search data to extract LPI address components

    # predict_len = len(search_df_cleaned["full_address"])
    all_columns = list(search_df_cleaned)  # Creates list of all column headers
    search_df_cleaned[all_columns] = search_df_cleaned[all_columns].astype(str)
    predict_data = list(search_df_after_stand["search_address_stand"])

    ### Run predict function
    print("Starting neural net prediction for " + str(len(predict_data)) + " addresses")

    tic = time.perf_counter()

    # Determine the number of chunks
    num_chunks = math.ceil(len(predict_data) / max_predict_len)
    list_out_all = []
    predict_df_all = []

    for i in range(num_chunks):
        print(
            "Starting to predict batch "
            + str(i + 1)
            + " of "
            + str(num_chunks)
            + " batches."
        )

        start_idx = i * max_predict_len
        end_idx = start_idx + max_predict_len

        # Extract the current chunk of data
        chunk_data = predict_data[start_idx:end_idx]

        # Replace blank strings with a single space
        chunk_data = [" " if s in ("") else s for s in chunk_data]

        if (model_type == "gru") | (model_type == "lstm"):
            list_out, predict_df = full_predict_torch(
                model=exported_model,
                model_type=model_type,
                input_text=chunk_data,
                word_to_index=word_to_index,
                cat_to_idx=cat_to_idx,
                device=device,
            )
        else:
            list_out, predict_df = full_predict_func(
                chunk_data, exported_model, vocab, labels_list
            )

        # Append the results
        list_out_all.extend(list_out)
        predict_df_all.append(predict_df)

    # Concatenate all the results dataframes
    predict_df_all = pd.concat(predict_df_all, ignore_index=True)

    toc = time.perf_counter()

    print(f"Performed the NN prediction in {toc - tic:0.1f} seconds")

    predict_df = post_predict_clean(
        predict_df=predict_df_all,
        orig_search_df=search_df_cleaned,
        ref_address_cols=ref_address_cols,
        search_df_key_field=search_df_key_field,
    )

    # Score-based matching between neural net predictions and fuzzy match results

    # Example of recordlinkage package in use: https://towardsdatascience.com/how-to-perform-fuzzy-dataframe-row-matching-with-recordlinkage-b53ca0cb944c

    ## Run with Postcode as blocker column

    blocker_column = ["Postcode"]

    scoresSBM_best_pc, matched_output_SBM_pc = score_based_match(
        predict_df_search=predict_df.copy(),
        ref_search=ref_df_after_stand.copy(),
        orig_search_df=search_df_after_stand,
        matching_variables=matching_variables,
        text_columns=text_columns,
        blocker_column=blocker_column,
        weights=weights,
        fuzzy_method=fuzzy_method,
        score_cut_off=score_cut_off,
        search_df_key_field=search_df_key_field,
        standardise=standardise,
        new_join_col=new_join_col,
    )

    if matched_output_SBM_pc.empty:
        error_message = "Match results empty. Exiting neural net match."
        print(error_message)

        return pd.DataFrame(), pd.DataFrame(), error_message, predict_df

    else:
        matched_output_SBM_pc["match_method"] = "Neural net - Postcode"

        match_results_output_final_pc = combine_dfs_and_remove_dups(
            match_results, matched_output_SBM_pc, index_col=search_df_key_field
        )

    create_match_summary(
        match_results_output_final_pc, df_name="NNet blocked by Postcode " + df_name
    )
    # print(summary_pc)

    ## Run with Street as blocker column

    blocker_column = ["Street"]

    scoresSBM_best_st, matched_output_SBM_st = score_based_match(
        predict_df_search=predict_df.copy(),
        ref_search=ref_df_after_stand.copy(),
        orig_search_df=search_df_after_stand,
        matching_variables=matching_variables,
        text_columns=text_columns,
        blocker_column=blocker_column,
        weights=weights,
        fuzzy_method=fuzzy_method,
        score_cut_off=score_cut_off,
        search_df_key_field=search_df_key_field,
        standardise=standardise,
        new_join_col=new_join_col,
    )

    # If no matching pairs are found in the function above then it returns 0 - below we replace these values with the postcode blocker values (which should almost always find at least one pair unless it's a very unusual situation)
    if isinstance(matched_output_SBM_st, int) or matched_output_SBM_st.empty:
        print("Nothing to match for street block")

        matched_output_SBM_st = matched_output_SBM_pc
        matched_output_SBM_st["match_method"] = (
            "Neural net - Postcode"  # + standard_label
        )
    else:
        matched_output_SBM_st["match_method"] = (
            "Neural net - Street"  # + standard_label
        )

    ### Join together old match df with new (model) match df

    match_results_output_final_st = combine_dfs_and_remove_dups(
        match_results_output_final_pc,
        matched_output_SBM_st,
        index_col=search_df_key_field,
    )

    create_match_summary(
        match_results_output_final_st, df_name="NNet blocked by Street " + df_name
    )
    # print(summary_street)

    # I decided in the end not to use PaoStartNumber as a blocker column. I get only a couple more matches in general for a big increase in processing time

    matched_output_SBM_po = matched_output_SBM_st
    matched_output_SBM_po["match_method"] = "Neural net - Street"  # + standard_label

    match_results_output_final_po = match_results_output_final_st
    match_results_output_final_three = match_results_output_final_po

    summary_three = create_match_summary(
        match_results_output_final_three,
        df_name="fuzzy and nn model street + postcode " + df_name,
    )

    ### Join URPN back onto orig df

    if not isinstance(search_df, str):
        results_on_orig_df = create_results_df(
            match_results_output_final_three,
            search_df_after_stand,
            search_df_key_field,
            new_join_col,
            ref_df_cleaned=ref_df_cleaned,
            existing_match_col=existing_match_col,
            pre_filter_search_df=pre_filter_search_df,
        )
    else:
        results_on_orig_df = match_results_output_final_three

    return (
        match_results_output_final_three,
        results_on_orig_df,
        summary_three,
        predict_df,
    )


# Combiner/summary functions
def combine_dfs_and_remove_dups(
    orig_df: PandasDataFrame,
    new_df: PandasDataFrame,
    index_col: str = "search_orig_address",
    match_address_series: str = "full_match",
    keep_only_duplicated: bool = False,
) -> PandasDataFrame:
    """
    Combine two Pandas dataframes and remove duplicates according to a specified 'index' column.
    `orig_df` is typically the pre-filter search snapshot; `new_df` is matcher output. For the
    same `index_col`, we prefer rows with a positive match flag, then the matcher row over the
    pre-filter row so reference join fields from `create_results_df` are not replaced by NaNs.
    """

    # If one of the dataframes is empty, break
    if (orig_df.empty) & (new_df.empty):
        return orig_df

    # Defensive: concatenation fails if either df has duplicate column labels (common after merges).
    # Keep the first occurrence of each duplicated column name.
    if orig_df.columns.duplicated().any():
        orig_df = orig_df.loc[:, ~orig_df.columns.duplicated()].copy()
    if new_df.columns.duplicated().any():
        new_df = new_df.loc[:, ~new_df.columns.duplicated()].copy()

    # Ensure that the original search result is returned
    if "Search data address" not in orig_df.columns:
        if "search_orig_address" in orig_df.columns:
            orig_df["Search data address"] = orig_df["search_orig_address"]
        elif "address_cols_joined" in orig_df.columns:
            orig_df["Search data address"] = orig_df["address_cols_joined"]

    if "Search data address" not in new_df.columns:
        if "search_orig_address" in new_df.columns:
            new_df["Search data address"] = new_df["search_orig_address"]
        elif "address_cols_joined" in new_df.columns:
            new_df["Search data address"] = new_df["address_cols_joined"]

    orig_df = orig_df.copy()
    new_df = new_df.copy()
    orig_df["_combine_source"] = 0
    new_df["_combine_source"] = 1

    combined_std_not_matches = pd.concat([orig_df, new_df], axis=0, ignore_index=True)

    # If no results were combined
    if combined_std_not_matches.empty:
        combined_std_not_matches[match_address_series] = False

        # if "full_address" in combined_std_not_matches.columns:
        #    combined_std_not_matches[index_col] = combined_std_not_matches["full_address"]
        combined_std_not_matches["fuzzy_score"] = 0
        return combined_std_not_matches

    # Convert index_col to string to ensure indexes from different sources are being compared correctly
    combined_std_not_matches[index_col] = combined_std_not_matches[index_col].astype(
        str
    )

    if match_address_series not in combined_std_not_matches.columns:
        combined_std_not_matches[match_address_series] = False

    # Sort so that for each index_col: unmatched rows first, matched rows last; then pre-filter
    # (0) before matcher (1). drop_duplicates(keep="last") then keeps the matcher row when both
    # exist, and the matched row when match flags differ.
    m_series = combined_std_not_matches[match_address_series]
    if pd.api.types.is_bool_dtype(m_series.dtype):
        m_bool = _bool_mask(m_series, default=False)
    elif pd.api.types.is_numeric_dtype(m_series.dtype):
        m_bool = m_series.fillna(0).astype(bool)
    else:
        m_bool = m_series.astype(str).str.lower().isin(("1", "true", "t", "yes"))
    combined_std_not_matches["_match_sort"] = m_bool

    combined_std_not_matches = combined_std_not_matches.sort_values(
        [index_col, "_match_sort", "_combine_source"],
        ascending=[True, True, True],
    )

    if keep_only_duplicated:
        combined_std_not_matches = combined_std_not_matches[
            combined_std_not_matches.duplicated(index_col)
        ]

    combined_std_not_matches_no_dups = combined_std_not_matches.drop_duplicates(
        index_col, keep="last"
    ).drop(columns=["_combine_source", "_match_sort"], errors="ignore")
    if index_col in combined_std_not_matches_no_dups.columns:
        _key_num = pd.to_numeric(
            combined_std_not_matches_no_dups[index_col], errors="coerce"
        )
        if _key_num.notna().any():
            combined_std_not_matches_no_dups = combined_std_not_matches_no_dups.assign(
                __key_num=_key_num
            ).sort_values(by=["__key_num", index_col], kind="stable")
            combined_std_not_matches_no_dups = combined_std_not_matches_no_dups.drop(
                columns=["__key_num"], errors="ignore"
            )
        else:
            combined_std_not_matches_no_dups = (
                combined_std_not_matches_no_dups.sort_values(
                    by=[index_col], kind="stable"
                )
            )
    combined_std_not_matches_no_dups = combined_std_not_matches_no_dups.reset_index(
        drop=True
    )

    return combined_std_not_matches_no_dups


def combine_two_matches(
    OrigMatchClass: MatcherClass,
    NewMatchClass: MatcherClass,
    df_name: str,
    write_outputs: bool = True,
    print_match_stage_summary_to_console: bool = PRINT_MATCH_STAGE_SUMMARY_TO_CONSOLE,
) -> MatcherClass:
    """
    Combine two MatcherClass objects to retain newest matches and drop duplicate addresses.
    """

    today_rev = datetime.now().strftime("%Y%m%d")

    NewMatchClass.match_results_output = combine_dfs_and_remove_dups(
        OrigMatchClass.match_results_output,
        NewMatchClass.match_results_output,
        index_col=NewMatchClass.search_df_key_field,
    )

    NewMatchClass.results_on_orig_df = combine_dfs_and_remove_dups(
        OrigMatchClass.pre_filter_search_df,
        NewMatchClass.results_on_orig_df,
        index_col=NewMatchClass.search_df_key_field,
        match_address_series="Matched with reference address",
    )

    # Filter out search results where a match was found
    NewMatchClass.pre_filter_search_df = NewMatchClass.results_on_orig_df

    matched_mask = _bool_mask(
        NewMatchClass.results_on_orig_df["Matched with reference address"],
        default=False,
    )

    # Normalise key values to integers where possible.
    # API inputs can yield string forms like "0.0" which break `.astype(int)`.
    found_index = pd.to_numeric(
        NewMatchClass.results_on_orig_df.loc[
            matched_mask, NewMatchClass.search_df_key_field
        ],
        errors="coerce",
    ).astype("Int64")

    key_field_values = pd.to_numeric(
        NewMatchClass.search_df_not_matched[NewMatchClass.search_df_key_field],
        errors="coerce",
    ).astype("Int64")

    rows_to_drop = (
        key_field_values[key_field_values.isin(found_index)].dropna().tolist()
    )
    NewMatchClass.search_df_not_matched = NewMatchClass.search_df_not_matched.loc[
        ~NewMatchClass.search_df_not_matched[NewMatchClass.search_df_key_field].isin(
            rows_to_drop
        ),
        :,
    ]  # .drop(rows_to_drop, axis = 0)

    # Filter out rows from NewMatchClass.search_* dataframes.
    #
    # IMPORTANT: do this by key value, not by positional boolean masks.
    # Cached standardisation frames can legitimately have different row counts
    # (e.g. cache from a previous run). Positional masks then raise:
    # "Boolean index has wrong length".
    _key = NewMatchClass.search_df_key_field
    _nm = NewMatchClass.search_df_not_matched.get(_key, pd.Series(dtype=object))
    _keep_keys = set(_normalize_join_key_strings(_nm).tolist())

    def _filter_by_keys(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        if _key not in df.columns:
            return df
        df_keys = _normalize_join_key_strings(df[_key])
        return df.loc[df_keys.isin(_keep_keys), :].copy()

    NewMatchClass.search_df_cleaned = _filter_by_keys(NewMatchClass.search_df_cleaned)
    NewMatchClass.search_df_after_stand = _filter_by_keys(
        NewMatchClass.search_df_after_stand
    )
    NewMatchClass.search_df_after_full_stand = _filter_by_keys(
        NewMatchClass.search_df_after_full_stand
    )

    ### Create lookup lists
    NewMatchClass.search_df_after_stand_series = (
        NewMatchClass.search_df_after_stand.copy()
        .set_index("postcode_search")["search_address_stand"]
        .str.lower()
        .str.strip()
    )
    NewMatchClass.search_df_after_stand_series_full_stand = (
        NewMatchClass.search_df_after_full_stand.copy()
        .set_index("postcode_search")["search_address_stand"]
        .str.lower()
        .str.strip()
    )

    # Some abort/empty-stage paths produce a match_results_output without fuzzy_score.
    # Guard this enrichment so combine does not fail on batches with no match rows.
    if (
        isinstance(NewMatchClass.match_results_output, pd.DataFrame)
        and ("fuzzy_score" in NewMatchClass.match_results_output.columns)
        and ("index" in NewMatchClass.match_results_output.columns)
        and ("index" in NewMatchClass.results_on_orig_df.columns)
    ):
        match_results_output_match_score_is_0 = NewMatchClass.match_results_output[
            NewMatchClass.match_results_output["fuzzy_score"] == 0.0
        ][["index", "fuzzy_score"]].drop_duplicates(subset="index")
        match_results_output_match_score_is_0["index"] = (
            match_results_output_match_score_is_0["index"].astype(str)
        )
        # NewMatchClass.results_on_orig_df["index"] = NewMatchClass.results_on_orig_df["index"].astype(str)
        NewMatchClass.results_on_orig_df["index"] = NewMatchClass.results_on_orig_df[
            "index"
        ].astype(str)
        NewMatchClass.results_on_orig_df = NewMatchClass.results_on_orig_df.merge(
            match_results_output_match_score_is_0, on="index", how="left"
        )

        NewMatchClass.results_on_orig_df.loc[
            NewMatchClass.results_on_orig_df["fuzzy_score"] == 0.0,
            "Excluded from search",
        ] = "Match score is 0"
        NewMatchClass.results_on_orig_df = NewMatchClass.results_on_orig_df.drop(
            "fuzzy_score", axis=1
        )

    # Drop any duplicates, prioritise any matches
    if "index" in NewMatchClass.results_on_orig_df.columns:
        NewMatchClass.results_on_orig_df["index"] = NewMatchClass.results_on_orig_df[
            "index"
        ].astype(int, errors="ignore")
    # `combine_dfs_and_remove_dups` can leave only pre-filter rows if the incoming
    # `results_on_orig_df` was empty or never carried match columns (no `ref_index`).
    if "ref_index" in NewMatchClass.results_on_orig_df.columns:
        NewMatchClass.results_on_orig_df["ref_index"] = (
            NewMatchClass.results_on_orig_df["ref_index"].astype(int, errors="ignore")
        )
    else:
        NewMatchClass.results_on_orig_df["ref_index"] = pd.NA

    NewMatchClass.results_on_orig_df = NewMatchClass.results_on_orig_df.sort_values(
        by=["index", "Matched with reference address"], ascending=[True, False]
    ).drop_duplicates(subset="index")

    _em_col = None
    if NewMatchClass.existing_match_cols:
        _em_col = (
            NewMatchClass.existing_match_cols[0]
            if isinstance(NewMatchClass.existing_match_cols, list)
            else NewMatchClass.existing_match_cols
        )
    NewMatchClass.results_on_orig_df = add_search_data_existing_col_to_results(
        NewMatchClass.results_on_orig_df,
        _em_col,
    )

    NewMatchClass.output_summary = create_match_summary(
        NewMatchClass.match_results_output, df_name=df_name
    )
    if print_match_stage_summary_to_console:
        print(NewMatchClass.output_summary)

    NewMatchClass.search_df_not_matched = filter_not_matched(
        NewMatchClass.match_results_output,
        NewMatchClass.search_df,
        NewMatchClass.search_df_key_field,
    )

    ### Rejoin the excluded matches onto the output file
    # NewMatchClass.results_on_orig_df = pd.concat([NewMatchClass.results_on_orig_df, NewMatchClass.excluded_df])

    _out_folder = getattr(NewMatchClass, "output_folder", None) or output_folder
    if _out_folder and (not _out_folder.endswith(("\\", "/"))):
        _out_folder = _out_folder + os.sep
    NewMatchClass.match_outputs_name = _out_folder + "diagnostics_" + today_rev + ".csv"
    NewMatchClass.results_orig_df_name = _out_folder + "results_" + today_rev + ".csv"

    _em_col_btch = None
    if NewMatchClass.existing_match_cols:
        _em_col_btch = (
            NewMatchClass.existing_match_cols[0]
            if isinstance(NewMatchClass.existing_match_cols, list)
            else NewMatchClass.existing_match_cols
        )
    NewMatchClass.results_on_orig_df = add_search_data_existing_col_to_results(
        NewMatchClass.results_on_orig_df,
        _em_col_btch,
    )

    # Attach search-side in_existing value onto diagnostics output as well. Use the
    # original/pre-filter search df because `search_df_cleaned` typically only contains
    # key/full_address/postcode.
    if _em_col_btch:
        diag_output_col = f"{_em_col_btch} (from search data)"
        needs_fill = (
            diag_output_col not in NewMatchClass.match_results_output.columns
        ) or NewMatchClass.match_results_output[diag_output_col].isna().all()
        if needs_fill and (
            NewMatchClass.search_df_key_field
            in NewMatchClass.match_results_output.columns
        ):
            source_df = getattr(NewMatchClass, "pre_filter_search_df", pd.DataFrame())
            if source_df is None or source_df.empty:
                source_df = getattr(NewMatchClass, "search_df", pd.DataFrame())

            _src_col = None
            if isinstance(source_df, pd.DataFrame) and (not source_df.empty):
                if _em_col_btch in source_df.columns:
                    _src_col = _em_col_btch
                else:
                    _alt = f"__search_side_{_em_col_btch}"
                    if _alt in source_df.columns:
                        _src_col = _alt

            if _src_col and (NewMatchClass.search_df_key_field in source_df.columns):
                # Use a key->value map to avoid merge column collisions (`_x`/`_y`).
                _map = (
                    source_df[[NewMatchClass.search_df_key_field, _src_col]]
                    .copy()
                    .drop_duplicates(
                        subset=[NewMatchClass.search_df_key_field], keep="first"
                    )
                )
                _map[NewMatchClass.search_df_key_field] = _map[
                    NewMatchClass.search_df_key_field
                ].astype(str)
                _series_map = _map.set_index(NewMatchClass.search_df_key_field)[
                    _src_col
                ]

                _keys = NewMatchClass.match_results_output[
                    NewMatchClass.search_df_key_field
                ].astype(str)
                mapped_vals = _keys.map(_series_map)

                if diag_output_col in NewMatchClass.match_results_output.columns:
                    NewMatchClass.match_results_output[diag_output_col] = (
                        NewMatchClass.match_results_output[diag_output_col].fillna(
                            mapped_vals
                        )
                    )
                else:
                    NewMatchClass.match_results_output[diag_output_col] = mapped_vals

    # Only keep essential columns
    essential_results_cols = [
        NewMatchClass.search_df_key_field,
        "Search data address",
        "Excluded from search",
        "Matched with reference address",
        "ref_index",
        "Reference matched address",
        "Reference file",
    ]
    essential_results_cols.extend(NewMatchClass.new_join_col)
    if _em_col_btch:
        essential_results_cols.append(f"{_em_col_btch} (from search data)")
    essential_results_cols = [
        col
        for col in essential_results_cols
        if col in NewMatchClass.results_on_orig_df.columns
    ]

    if "fuzzy_score" in NewMatchClass.match_results_output.columns:
        NewMatchClass.match_results_output["fuzzy_score"] = pd.to_numeric(
            NewMatchClass.match_results_output["fuzzy_score"], errors="coerce"
        ).round(2)
    if "wratio_score" in NewMatchClass.match_results_output.columns:
        NewMatchClass.match_results_output["wratio_score"] = pd.to_numeric(
            NewMatchClass.match_results_output["wratio_score"], errors="coerce"
        ).round(2)

    if write_outputs:
        NewMatchClass.match_results_output.to_csv(
            NewMatchClass.match_outputs_name, index=None
        )
        out_cols = [
            c
            for c in essential_results_cols
            if c in NewMatchClass.results_on_orig_df.columns
        ]
        NewMatchClass.results_on_orig_df[out_cols].to_csv(
            NewMatchClass.results_orig_df_name, index=None
        )

    return NewMatchClass


def create_match_summary(match_results_output: PandasDataFrame, df_name: str) -> str:
    """
    Create a text summary of the matching process results to export to a text box or log file.
    """

    if (
        (not isinstance(match_results_output, pd.DataFrame))
        or match_results_output.empty
        or ("full_match" not in match_results_output.columns)
    ):
        return f"For the {df_name} dataset (0 records), the fuzzy matching algorithm successfully matched 0 records (0%). The algorithm could not attempt to match 0 records (0%). There are 0 records left to potentially match."

    full_match_series = _bool_mask(match_results_output["full_match"], default=False)

    """ Create a summary paragraph """
    full_match_count = int(full_match_series.sum())
    match_fail_count = int((~full_match_series).sum())
    records_attempted = int(
        sum(
            (match_results_output["fuzzy_score"] != 0.0)
            & ~(match_results_output["fuzzy_score"].isna())
        )
    )
    dataset_length = int(len(full_match_series))
    records_not_attempted = int(dataset_length - records_attempted)
    match_rate = str(round((full_match_count / dataset_length) * 100, 1))
    match_fail_count_without_excluded = match_fail_count - records_not_attempted
    str(round(((match_fail_count_without_excluded) / dataset_length) * 100, 1))
    not_attempted_rate = str(round((records_not_attempted / dataset_length) * 100, 1))

    summary = (
        "For the "
        + df_name
        + " dataset ("
        + str(dataset_length)
        + " records), the fuzzy matching algorithm successfully matched "
        + str(full_match_count)
        + " records ("
        + match_rate
        + "%). The algorithm could not attempt to match "
        + str(records_not_attempted)
        + " records ("
        + not_attempted_rate
        + "%). There are "
        + str(match_fail_count_without_excluded)
        + " records left to potentially match."
    )

    return summary


def build_run_summary_text(
    results_df: pd.DataFrame,
    diagnostics_df: Optional[pd.DataFrame],
    key_field: str,
    use_postcode_blocker_requested: Optional[bool] = None,
    use_postcode_blocker_effective: Optional[bool] = None,
) -> str:
    """
    Create a clearer, user-facing summary for the GUI.

    Percentages are calculated over *eligible* records only (excluding 'Previously matched'
    and other excluded categories).
    """
    if results_df is None or results_df.empty:
        return "No results were produced."

    out = results_df.copy()
    if "Matched with reference address" not in out.columns:
        out["Matched with reference address"] = False

    eligible_mask = _eligible_mask_from_results(out)
    eligible_total = int(eligible_mask.sum())
    all_total = int(len(out))
    excluded_total = int(all_total - eligible_total)
    matched_total = int(
        _bool_mask(
            out.loc[eligible_mask, "Matched with reference address"],
            default=False,
        ).sum()
    )
    eligible_unmatched = int(eligible_total - matched_total)

    attempted_total = None
    not_attempted_total = None
    stage_lines = []
    neural_net_ran = False
    if (
        diagnostics_df is not None
        and isinstance(diagnostics_df, pd.DataFrame)
        and (not diagnostics_df.empty)
        and key_field in diagnostics_df.columns
        and "fuzzy_score" in diagnostics_df.columns
    ):
        # Attempted = eligible keys that have a non-zero fuzzy_score recorded in diagnostics.
        diag = diagnostics_df[[key_field, "fuzzy_score"]].copy()
        diag[key_field] = diag[key_field].astype(str)
        # Reduce to best available signal per record
        diag_best = diag.groupby(key_field, dropna=False)["fuzzy_score"].max()
        eligible_keys = set(out.loc[eligible_mask, key_field].astype(str).tolist())
        attempted_mask = (
            diag_best.index.isin(eligible_keys) & diag_best.notna() & (diag_best != 0.0)
        )
        attempted_total = int(attempted_mask.sum())
        not_attempted_total = max(eligible_total - attempted_total, 0)

        # Matched-by-stage counts (eligible keys only), if stage columns exist.
        if (
            "full_match" in diagnostics_df.columns
            and "match_method" in diagnostics_df.columns
            and "standardised_address" in diagnostics_df.columns
        ):
            eligible_keys = set(out.loc[eligible_mask, key_field].astype(str).tolist())
            diag_stage = diagnostics_df[
                [key_field, "match_method", "standardised_address", "full_match"]
            ].copy()
            diag_stage[key_field] = diag_stage[key_field].astype(str)
            diag_stage["full_match"] = _bool_mask(
                diag_stage["full_match"], default=False
            )
            diag_stage["standardised_address"] = _bool_mask(
                diag_stage["standardised_address"], default=False
            )
            diag_stage = diag_stage[diag_stage[key_field].isin(eligible_keys)]

            def _stage_matched(mask: pd.Series) -> int:
                if mask is None or not mask.any():
                    return 0
                keys = diag_stage.loc[
                    mask & diag_stage["full_match"], key_field
                ].unique()
                return int(len(keys))

            mm = diag_stage["match_method"].fillna("").astype(str)
            fuzzy_mask = mm.str.contains("Fuzzy match", na=False)
            nn_mask = mm.str.contains("Neural net", na=False)
            neural_net_ran = bool(nn_mask.any())

            fuzzy_not_std_count = _stage_matched(
                fuzzy_mask & (~diag_stage["standardised_address"])
            )
            fuzzy_std_count = _stage_matched(
                fuzzy_mask & (diag_stage["standardised_address"])
            )
            nn_std_count = _stage_matched(nn_mask)

            stage_lines.append("## Matched by stage (eligible records only)")
            stage_lines.append(
                f"- **Fuzzy (not standardised)**: {fuzzy_not_std_count}"
                + (
                    f" ({round(fuzzy_not_std_count/eligible_total*100,1)}%)"
                    if eligible_total
                    else ""
                )
            )
            stage_lines.append(
                f"- **Fuzzy (standardised)**: {fuzzy_std_count}"
                + (
                    f" ({round(fuzzy_std_count/eligible_total*100,1)}%)"
                    if eligible_total
                    else ""
                )
            )
            (
                stage_lines.append(
                    f"- **Neural net (standardised)**: {nn_std_count}"
                    + (
                        f" ({round(nn_std_count/eligible_total*100,1)}%)"
                        if eligible_total
                        else ""
                    )
                )
                if neural_net_ran
                else None
            )

            # Remove any Nones introduced by conditional stage lines
            stage_lines = [ln for ln in stage_lines if ln]

    lines = []
    lines.append("## Match summary (eligible records only)")
    lines.append(f"- **Total input records**: {all_total}")
    lines.append(
        f"- **Excluded from matching**: {excluded_total} (e.g. previously matched, blank/invalid address)"
    )
    lines.append(f"- **Eligible for matching**: {eligible_total}")
    if eligible_total > 0:
        lines.append(
            f"- **Matched**: {matched_total} ({round(matched_total/eligible_total*100,1)}%)"
        )
        lines.append(
            f"- **Eligible but unmatched**: {eligible_unmatched} ({round(eligible_unmatched/eligible_total*100,1)}%)"
        )
    else:
        lines.append("- **Matched**: 0 (0.0%)")
        lines.append("- **Eligible but unmatched**: 0 (0.0%)")

    if (
        attempted_total is not None
        and not_attempted_total is not None
        and eligible_total > 0
    ):
        lines.append(
            f"- **Attempted**: {attempted_total} ({round(attempted_total/eligible_total*100,1)}%)"
        )
        lines.append(
            f"- **Not attempted**: {not_attempted_total} ({round(not_attempted_total/eligible_total*100,1)}%)"
        )
        lines.append(
            "- **What 'not attempted' means**: the record was eligible, but no candidate match was scored (often because blocking produced no candidates)."
        )

    if use_postcode_blocker_requested is not None:
        mode_bits = []
        mode_bits.append(
            f"requested={'postcode' if use_postcode_blocker_requested else 'street-only'}"
        )
        if use_postcode_blocker_effective is not None:
            mode_bits.append(
                f"effective={'postcode' if use_postcode_blocker_effective else 'street-only'}"
            )
        lines.append(f"- **Blocking mode**: {', '.join(mode_bits)}")

    lines.append("")
    lines.append("## What the stages mean")
    lines.append(
        "- **Fuzzy not standardised**: fuzzy matching on minimally cleaned text."
    )
    lines.append(
        "- **Fuzzy standardised**: fuzzy matching after full standardisation (more normalization; often improves recall)."
    )
    if neural_net_ran:
        lines.append(
            "- **Neural net standardised**: ML model stage; uses standardised inputs and can confirm/extend fuzzy matches."
        )

    # Top exclusion reasons
    excl_counts = (
        out.loc[~eligible_mask, "Excluded from search"]
        .fillna("")
        .astype(str)
        .str.strip()
        .replace("", "Excluded (unspecified)")
        .value_counts()
        .head(5)
    )
    if not excl_counts.empty:
        lines.append("")
        lines.append("## Top exclusion reasons")
        for k, v in excl_counts.items():
            lines.append(f"- **{k}**: {int(v)}")

    if stage_lines:
        lines.append("")
        lines.extend(stage_lines)

    return "\n".join(lines)


def _eligible_mask_from_results(results_df: pd.DataFrame) -> pd.Series:
    """
    Determine which records are eligible for matching-rate denominators.

    We exclude rows that were not searched (e.g. 'Previously matched', blank/invalid address,
    missing blocker values, etc.) based on the 'Excluded from search' field.
    """
    if results_df is None or results_df.empty:
        return pd.Series([], dtype=bool)

    excluded = (
        results_df.get("Excluded from search", pd.Series("", index=results_df.index))
        .fillna("")
        .astype(str)
        .str.strip()
    )
    # Treat these as not eligible for measuring matcher effectiveness.
    not_eligible_values = {
        "Previously matched",
        "Excluded - blank address",
        "Excluded - non-postal address",
        "Excluded - blank street",
        "Excluded - blank postcode",
    }
    return ~excluded.isin(not_eligible_values)


def build_results_summary_table(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a compact, results-based summary table with counts and pct_of_eligible.
    """
    if results_df is None or results_df.empty:
        return pd.DataFrame(
            columns=[
                "Included in search",
                "Reason for exclusion",
                "Matched with reference address",
                "count",
                "pct_of_eligible",
            ]
        )

    out = results_df.copy()
    if "Matched with reference address" not in out.columns:
        out["Matched with reference address"] = False

    eligible_mask = _eligible_mask_from_results(out)
    eligible_total = int(eligible_mask.sum())

    # Derive user-friendly grouping columns
    raw_excl = (
        out.get("Excluded from search", pd.Series("", index=out.index))
        .fillna("")
        .astype(str)
        .str.strip()
    )
    # Normalise common “included” representations so reasons don’t get lost.
    is_included = raw_excl.isin(["", "False", "0", "Included in search"])
    excl_reason = raw_excl.where(~is_included, other="Included in search").replace(
        "", "Included in search"
    )

    out["_included_excluded"] = np.where(
        is_included, "Included in search", "Excluded from search"
    )
    out["_reason_for_exclusion"] = excl_reason.where(
        ~is_included, other="Included in search"
    )

    grp = (
        out.assign(_eligible=eligible_mask)
        .groupby(
            [
                "_included_excluded",
                "_reason_for_exclusion",
                "Matched with reference address",
                "_eligible",
            ],
            dropna=False,
        )
        .size()
        .reset_index(name="count")
    )

    # Only compute pct_of_eligible for eligible rows; excluded rows get blank.
    grp["pct_of_eligible"] = ""
    if eligible_total > 0:
        eligible_rows = grp["_eligible"].astype(bool)
        grp.loc[eligible_rows, "pct_of_eligible"] = (
            (grp.loc[eligible_rows, "count"] / eligible_total * 100)
            .round(1)
            .astype(str)
        )

    grp = grp.drop(columns=["_eligible"])
    grp = grp.rename(
        columns={
            "_included_excluded": "Included in search",
            "_reason_for_exclusion": "Reason for exclusion",
            "pct_of_eligible": "Percentage of eligible records (%)",
            "count": "Count",
        }
    )

    # Add grand total row at bottom
    total_row = pd.DataFrame(
        [
            {
                "Included in search": "Grand total",
                "Reason for exclusion": "",
                "Matched with reference address": "",
                "Count": int(len(out)),
                "Percentage of eligible records (%)": "",
            }
        ]
    )

    grp = grp.sort_values(
        [
            "Included in search",
            "Reason for exclusion",
            "Matched with reference address",
        ],
        kind="stable",
    )
    return pd.concat([grp, total_row], ignore_index=True)


def results_summary_table_to_markdown(summary_df: pd.DataFrame) -> str:
    """
    Convert the summary table to a simple markdown table string for the GUI.
    """
    if summary_df is None or summary_df.empty:
        return "No summary available."

    df = summary_df.copy()
    # Keep column order stable (must match build_results_summary_table renames)
    preferred = [
        "Included in search",
        "Reason for exclusion",
        "Matched with reference address",
        "Count",
        "Percentage of eligible records (%)",
    ]
    cols = [c for c in preferred if c in df.columns]
    # Fallback if older column names are present
    if not cols:
        legacy = [
            "Included/Excluded",
            "Reason for exclusion",
            "Matched with reference address",
            "count",
            "pct_of_eligible",
        ]
        cols = [c for c in legacy if c in df.columns]
    df = df[cols]

    # Manual markdown for stability (avoid deps / pandas version differences)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = []
    for _, r in df.iterrows():
        rows.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    return "\n".join([header, sep] + rows)
