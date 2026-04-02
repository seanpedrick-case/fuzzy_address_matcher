import os
import numpy as np
import pandas as pd

from typing import Dict, List, Tuple, Type, Optional
import time
import re
import math
from datetime import datetime
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
import gradio as gr
from tqdm import tqdm

from tools.constants import (
    InitMatch,
    MatcherClass,
    batch_size,
    filter_to_lambeth_pcodes,
    max_predict_len,
    output_folder,
    ref_batch_size,
)

from tools.config import (
    USE_POSTCODE_BLOCKER,
    MAX_PARALLEL_WORKERS,
    RUN_BATCHES_IN_PARALLEL,
)

# Imports (must be module-level)
from tools.preparation import (
    prepare_search_address_string,
    prepare_search_address,
    extract_street_name,
    prepare_ref_address,
    remove_non_postal,
    check_no_number_addresses,
    extract_postcode,
)
from tools.fuzzy_match import (
    string_match_by_post_code_multiple,
    _create_fuzzy_match_results_output,
    create_results_df,
)
from tools.standardise import standardise_wrapper_func, remove_postcode

# Neural network functions
### Predict function for imported model
from tools.model_predict import (
    full_predict_func,
    full_predict_torch,
    post_predict_clean,
)
from tools.recordlinkage_funcs import score_based_match
from tools.helper_functions import initial_data_load, sum_numbers_before_seconds

# API functions
from tools.addressbase_api_funcs import places_api_query

# (max_predict_len, MatcherClass imported above)

# Type aliases / module globals (must come after imports)
PandasDataFrame = Type[pd.DataFrame]
PandasSeries = Type[pd.Series]
MatchedResults = Dict[str, Tuple[str, int]]
array = List[str]

today = datetime.now().strftime("%d%m%Y")
today_rev = datetime.now().strftime("%Y%m%d")
today_month_rev = datetime.now().strftime("%Y%m")

# Constants
run_fuzzy_match = True
run_nnet_match = True
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
    """Get the name of a file from a string, handling both Windows and Unix paths."""

    match = re.search(rf"{re.escape(os.sep)}(?!.*{re.escape(os.sep)})(.*)", in_name)
    if match:
        matched_result = match.group(1)
    else:
        matched_result = None

    return matched_result


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

    if key_col not in matched_results.columns:
        raise ValueError(f"{key_col} not a column in matched_results")

    if "full_match" not in matched_results.columns:
        raise ValueError("full_match not a column in matched_results")

    full_match_series = matched_results["full_match"].fillna(False).astype(bool)
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
    df: pd.DataFrame, new_join_col: List[str]
) -> pd.DataFrame:
    """
    If search data contains columns with the same names as `new_join_col`, rename them so
    merges and concat with reference-side values never keep the search copy by accident.
    """
    if df is None or df.empty or not new_join_col:
        return df
    rename_map = {c: f"__search_side_{c}" for c in new_join_col if c in df.columns}
    if rename_map:
        return df.rename(columns=rename_map)
    return df


def _column_has_usable_values(df: pd.DataFrame, col_name: str) -> bool:
    """Return True when the column exists and has at least one non-empty value."""
    if (
        (df is None)
        or (not isinstance(df, pd.DataFrame))
        or (col_name not in df.columns)
    ):
        return False

    cleaned_series = df[col_name].fillna("").astype(str).str.strip()
    return cleaned_series.ne("").any()


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
    )
    return batch_n, summary, batch_out


def query_addressbase_api(
    in_api_key: str,
    Matcher: MatcherClass,
    query_type: str,
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

                pd.concat(loop_list).to_parquet(
                    output_folder + api_ref_save_loc + ".parquet", index=False
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
                print("API call time in seconds: ", toc - tic)
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
            final_api_output_file_name_pq = (
                output_folder + api_ref_save_loc[:-5] + ".parquet"
            )
            final_api_output_file_name = output_folder + api_ref_save_loc[:-5] + ".csv"
            print("Saving reference file to: " + api_ref_save_loc[:-5] + ".parquet")
            Matcher.ref_df.to_parquet(
                output_folder + api_ref_save_loc + ".parquet", index=False
            )  # Save checkpoint as well
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
    progress=gr.Progress(track_tqdm=True),
):
    """
    Check for reference address data, do some preprocessing, and load in from the Addressbase API if required.
    """
    final_api_output_file_name = ""

    # Check if reference data loaded, bring in if already there
    if not ref_data_state.empty:
        Matcher.ref_df = ref_data_state
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
                    in_api_key, Matcher, query_type
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
        output_message, drop1, drop2, Matcher.search_df, results_data_state = (
            initial_data_load(in_file)
        )

        file_list = [string.name for string in in_file]
        data_file_names = [
            string for string in file_list if "results_" not in string.lower()
        ]

        Matcher.file_name = get_file_name(data_file_names[0])

        # search_df makes column to use as index
        Matcher.search_df["index"] = Matcher.search_df.index

    # Join previously created results file onto search_df if previous results file exists
    if not results_data_state.empty:

        print("Joining on previous results file")
        Matcher.results_on_orig_df = results_data_state.copy()
        Matcher.search_df = Matcher.search_df.merge(
            results_data_state, on="index", how="left"
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

    print("Shape of search_df before filtering: ", Matcher.search_df.shape)

    ### Filter addresses to those with length > 0
    zero_length_search_df = Matcher.search_df.copy()[Matcher.search_address_cols]
    zero_length_search_df = zero_length_search_df.fillna("").infer_objects(copy=False)
    Matcher.search_df["address_cols_joined"] = (
        zero_length_search_df.astype(str).sum(axis=1).str.strip()
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

            unique_ref_pcode_area = (
                Matcher.ref_df["postcode_search_area"][
                    Matcher.ref_df["postcode_search_area"].str.len() > 3
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

            print("Shape of ref_df before filtering is: ", Matcher.ref_df.shape)

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
            output_message, drop1, drop2, df, results_data_state = initial_data_load(
                in_file
            )

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
        Matcher.pre_filter_search_df, Matcher.new_join_col or []
    )
    Matcher.excluded_df = _rename_search_side_join_columns_overlap(
        Matcher.excluded_df, Matcher.new_join_col or []
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
        )

    print("Shape of ref_df after filtering is: ", Matcher.ref_df.shape)
    print("Shape of search_df after filtering is: ", Matcher.search_df.shape)

    Matcher.match_outputs_name = (
        output_folder + "diagnostics_initial_" + today_rev + ".csv"
    )
    Matcher.results_orig_df_name = (
        output_folder + "results_initial_" + today_rev + ".csv"
    )

    if "fuzzy_score" in Matcher.match_results_output.columns:
        Matcher.match_results_output["fuzzy_score"] = pd.to_numeric(
            Matcher.match_results_output["fuzzy_score"], errors="coerce"
        ).round(2)
    if "wratio_score" in Matcher.match_results_output.columns:
        Matcher.match_results_output["wratio_score"] = pd.to_numeric(
            Matcher.match_results_output["wratio_score"], errors="coerce"
        ).round(2)

    Matcher.match_results_output.to_csv(Matcher.match_outputs_name, index=None)
    Matcher.results_on_orig_df.to_csv(Matcher.results_orig_df_name, index=None)

    return Matcher, final_api_output_file_name


# Run whole matcher process
def run_matcher(
    in_text: str,
    in_file: str,
    in_ref: str,
    data_state: PandasDataFrame,
    results_data_state: PandasDataFrame,
    ref_data_state: PandasDataFrame,
    in_colnames: List[str],
    in_refcol: List[str],
    in_joincol: List[str],
    in_existing: List[str],
    in_api: str,
    in_api_key: str,
    use_postcode_blocker: bool = USE_POSTCODE_BLOCKER,
    run_batches_in_parallel: bool = RUN_BATCHES_IN_PARALLEL,
    max_parallel_workers: Optional[int] = MAX_PARALLEL_WORKERS,
    InitMatch: MatcherClass = InitMatch,
    progress=gr.Progress(track_tqdm=True),
):
    """
    Split search and reference data into batches. Loop and run through the match script for each batch of data.
    """
    output_files = []

    estimate_total_processing_time = 0.0

    overall_tic = time.perf_counter()

    # Load in initial data. This will filter to relevant addresses in the search and reference datasets that can potentially be matched, and will pull in API data if asked for.
    InitMatch, final_api_output_file_name = load_matcher_data(
        in_text,
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
        return out_message, output_files, estimate_total_processing_time

    # Run initial address preparation and standardisation processes
    # Prepare address format

    # Polars implementation not yet finalised
    # InitMatch.search_df = pl.from_pandas(InitMatch.search_df)
    # InitMatch.ref_df = pl.from_pandas(InitMatch.ref_df)

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

    # Initial preparation of reference addresses
    InitMatch.ref_df_cleaned = prepare_ref_address(
        InitMatch.ref_df, InitMatch.ref_address_cols, InitMatch.new_join_col
    )

    # Polars implementation - not finalised
    # InitMatch.search_df_cleaned = InitMatch.search_df_cleaned.to_pandas()
    # InitMatch.ref_df_cleaned = InitMatch.ref_df_cleaned.to_pandas()

    # Standardise addresses
    # Standardise - minimal

    tic = time.perf_counter()

    progress(0.1, desc="Performing minimal standardisation")

    InitMatch.search_df_after_stand, InitMatch.ref_df_after_stand = (
        standardise_wrapper_func(
            InitMatch.search_df_cleaned.copy(),
            InitMatch.ref_df_cleaned.copy(),
            standardise=False,
            filter_to_lambeth_pcodes=filter_to_lambeth_pcodes,
            match_task="fuzzy",
        )
    )  # InitMatch.search_df_after_stand_series, InitMatch.ref_df_after_stand_series

    toc = time.perf_counter()
    print(f"Performed the minimal standardisation step in {toc - tic:0.1f} seconds")

    progress(0.1, desc="Performing full standardisation")

    # Standardise - full
    tic = time.perf_counter()
    InitMatch.search_df_after_full_stand, InitMatch.ref_df_after_full_stand = (
        standardise_wrapper_func(
            InitMatch.search_df_cleaned.copy(),
            InitMatch.ref_df_cleaned.copy(),
            standardise=True,
            filter_to_lambeth_pcodes=filter_to_lambeth_pcodes,
            match_task="fuzzy",
        )
    )  # , InitMatch.search_df_after_stand_series_full_stand, InitMatch.ref_df_after_stand_series_full_stand

    toc = time.perf_counter()
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
        )
    else:
        range_df = create_street_batch_ranges(
            InitMatch.search_df_cleaned.copy(),
            InitMatch.ref_df_cleaned.copy(),
            batch_size,
        )

    print("Batches to run in this session: ", range_df)

    OutputMatch = copy.copy(InitMatch)

    number_of_batches = range_df.shape[0]
    batch_inputs = []
    for row in range(0, number_of_batches):
        search_range = range_df.iloc[row]["search_range"]
        ref_range = range_df.iloc[row]["ref_range"]

        BatchMatch = copy.copy(InitMatch)
        BatchMatch.search_df = BatchMatch.search_df[
            BatchMatch.search_df.index.isin(search_range)
        ].reset_index(drop=True)
        BatchMatch.search_df_not_matched = BatchMatch.search_df.copy()
        BatchMatch.search_df_cleaned = BatchMatch.search_df_cleaned[
            BatchMatch.search_df_cleaned.index.isin(search_range)
        ].reset_index(drop=True)

        BatchMatch.ref_df = BatchMatch.ref_df[
            BatchMatch.ref_df.index.isin(ref_range)
        ].reset_index(drop=True)
        BatchMatch.ref_df_cleaned = BatchMatch.ref_df_cleaned[
            BatchMatch.ref_df_cleaned.index.isin(ref_range)
        ].reset_index(drop=True)

        BatchMatch.search_df_after_stand = BatchMatch.search_df_after_stand[
            BatchMatch.search_df_after_stand.index.isin(search_range)
        ].reset_index(drop=True)
        BatchMatch.search_df_after_full_stand = BatchMatch.search_df_after_full_stand[
            BatchMatch.search_df_after_full_stand.index.isin(search_range)
        ].reset_index(drop=True)
        BatchMatch.ref_df_after_stand = BatchMatch.ref_df_after_stand[
            BatchMatch.ref_df_after_stand.index.isin(ref_range)
        ].reset_index(drop=True)
        BatchMatch.ref_df_after_full_stand = BatchMatch.ref_df_after_full_stand[
            BatchMatch.ref_df_after_full_stand.index.isin(ref_range)
        ].reset_index(drop=True)

        batch_inputs.append((row, BatchMatch))

    run_parallel = bool(run_batches_in_parallel) and number_of_batches > 1
    if run_parallel:
        try:
            worker_count = _resolve_parallel_worker_count(
                number_of_batches, max_parallel_workers, includes_nnet=run_nnet_match
            )
            print(f"Running batches in parallel with {worker_count} worker processes")
            future_results = []

            with ProcessPoolExecutor(max_workers=worker_count) as executor:
                submitted = [
                    executor.submit(
                        run_single_match_batch_worker,
                        batch_n,
                        batch_match,
                        number_of_batches,
                        use_postcode_blocker_effective,
                    )
                    for batch_n, batch_match in batch_inputs
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
                )

        except Exception as parallel_error:
            print(
                f"Parallel batch execution failed ({parallel_error}). Falling back to sequential batching."
            )
            run_parallel = False

    if not run_parallel:
        for batch_n, BatchMatch in progress.tqdm(
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
                BatchMatch_out.results_on_orig_df = pd.DataFrame(
                    data={
                        "index": BatchMatch.search_df.index,
                        "Excluded from search": False,
                        "Matched with reference address": False,
                    }
                )
            else:
                summary_of_summaries, BatchMatch_out = run_single_match_batch(
                    BatchMatch,
                    batch_n,
                    number_of_batches,
                    use_postcode_blocker=use_postcode_blocker_effective,
                )

            OutputMatch = combine_two_matches(
                OutputMatch,
                BatchMatch_out,
                "All up to and including batch " + str(batch_n + 1),
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

    # Remove any duplicates from reference df, prioritise successful matches
    OutputMatch.results_on_orig_df = OutputMatch.results_on_orig_df.sort_values(
        by=["index", "Matched with reference address"], ascending=[True, False]
    ).drop_duplicates(subset="index")

    # Ensure diagnostics contains exclusion info for all rows present,
    # and include explicit rows for pre-existing matches.
    if ("Excluded from search" in OutputMatch.results_on_orig_df.columns) and (
        OutputMatch.search_df_key_field in OutputMatch.results_on_orig_df.columns
    ):
        if "Excluded from search" not in OutputMatch.match_results_output.columns:
            # Ensure merge key dtypes match (can differ between pipelines)
            OutputMatch.match_results_output[OutputMatch.search_df_key_field] = (
                OutputMatch.match_results_output[
                    OutputMatch.search_df_key_field
                ].astype(str)
            )
            results_keyed = OutputMatch.results_on_orig_df[
                [OutputMatch.search_df_key_field, "Excluded from search"]
            ].drop_duplicates(subset=OutputMatch.search_df_key_field)
            results_keyed[OutputMatch.search_df_key_field] = results_keyed[
                OutputMatch.search_df_key_field
            ].astype(str)

            OutputMatch.match_results_output = OutputMatch.match_results_output.merge(
                results_keyed,
                on=OutputMatch.search_df_key_field,
                how="left",
            )

        pre_existing_rows = OutputMatch.results_on_orig_df[
            OutputMatch.results_on_orig_df["Excluded from search"]
            .fillna("")
            .astype(str)
            .eq("Previously matched")
        ].copy()

        if not pre_existing_rows.empty:
            diag_cols = list(OutputMatch.match_results_output.columns)
            diag_add = pd.DataFrame(columns=diag_cols)
            diag_add[OutputMatch.search_df_key_field] = pre_existing_rows[
                OutputMatch.search_df_key_field
            ].astype(str)

            if "Search data address" in pre_existing_rows.columns:
                diag_add["Search data address"] = pre_existing_rows[
                    "Search data address"
                ]
            if (
                "search_orig_address" in diag_add.columns
                and "Search data address" in diag_add.columns
            ):
                diag_add["search_orig_address"] = diag_add["Search data address"]

            if "match_method" in diag_add.columns:
                diag_add["match_method"] = "Pre-existing match"
            if "Excluded from search" in diag_add.columns:
                diag_add["Excluded from search"] = "Previously matched"
            if "fuzzy_score" in diag_add.columns:
                diag_add["fuzzy_score"] = 0.0
            if "wratio_score" in diag_add.columns:
                diag_add["wratio_score"] = 0.0
            if "full_match" in diag_add.columns:
                diag_add["full_match"] = False
            if "standardised_address" in diag_add.columns:
                diag_add["standardised_address"] = False

            OutputMatch.match_results_output = pd.concat(
                [OutputMatch.match_results_output, diag_add], axis=0, ignore_index=True
            )
            # Prefer real match rows over the placeholder "Pre-existing match" row
            if "match_method" in OutputMatch.match_results_output.columns:
                OutputMatch.match_results_output = (
                    OutputMatch.match_results_output.sort_values(
                        by=[OutputMatch.search_df_key_field, "match_method"],
                        ascending=[True, True],
                        kind="stable",
                    ).drop_duplicates(
                        subset=OutputMatch.search_df_key_field, keep="first"
                    )
                )

    overall_toc = time.perf_counter()
    time_out = (
        f"The overall match (all batches) took {overall_toc - overall_tic:0.1f} seconds"
    )

    print(OutputMatch.output_summary)

    if OutputMatch.output_summary == "":
        OutputMatch.output_summary = "No matches were found."

    fuzzy_not_std_output = OutputMatch.match_results_output.copy()
    fuzzy_not_std_output_mask = ~(
        fuzzy_not_std_output["match_method"].str.contains("Fuzzy match")
    ) | (fuzzy_not_std_output["standardised_address"])
    fuzzy_not_std_output.loc[fuzzy_not_std_output_mask, "full_match"] = False
    fuzzy_not_std_summary = create_match_summary(
        fuzzy_not_std_output, "Fuzzy not standardised"
    )

    fuzzy_std_output = OutputMatch.match_results_output.copy()
    fuzzy_std_output_mask = fuzzy_std_output["match_method"].str.contains("Fuzzy match")
    fuzzy_std_output.loc[~fuzzy_std_output_mask, "full_match"] = False
    fuzzy_std_summary = create_match_summary(fuzzy_std_output, "Fuzzy standardised")

    nnet_std_output = OutputMatch.match_results_output.copy()
    nnet_std_summary = create_match_summary(nnet_std_output, "Neural net standardised")

    final_summary = (
        fuzzy_not_std_summary
        + "\n"
        + fuzzy_std_summary
        + "\n"
        + nnet_std_summary
        + "\n"
        + time_out
    )

    estimate_total_processing_time = sum_numbers_before_seconds(time_out)
    print("Estimated total processing time:", str(estimate_total_processing_time))

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

    # Also attach the search-side in_existing value onto the diagnostics output so it can be
    # audited alongside match scores/methods. This is a left-join from the prepared search df.
    if _em_col and (
        OutputMatch.search_df_key_field in OutputMatch.search_df_cleaned.columns
    ):
        diag_output_col = f"{_em_col} (from search data)"
        if diag_output_col not in OutputMatch.match_results_output.columns:
            _src_col = None
            if _em_col in OutputMatch.search_df_cleaned.columns:
                _src_col = _em_col
            else:
                _alt = f"__search_side_{_em_col}"
                if _alt in OutputMatch.search_df_cleaned.columns:
                    _src_col = _alt

            if _src_col:
                _map_df = OutputMatch.search_df_cleaned[
                    [OutputMatch.search_df_key_field, _src_col]
                ].copy()
                _map_df = _map_df.rename(columns={_src_col: diag_output_col})
                _map_df[OutputMatch.search_df_key_field] = _map_df[
                    OutputMatch.search_df_key_field
                ].astype(str)
                if (
                    OutputMatch.search_df_key_field
                    in OutputMatch.match_results_output.columns
                ):
                    OutputMatch.match_results_output[
                        OutputMatch.search_df_key_field
                    ] = OutputMatch.match_results_output[
                        OutputMatch.search_df_key_field
                    ].astype(
                        str
                    )
                OutputMatch.match_results_output = (
                    OutputMatch.match_results_output.merge(
                        _map_df,
                        on=OutputMatch.search_df_key_field,
                        how="left",
                    )
                )

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

    OutputMatch.match_results_output.to_csv(OutputMatch.match_outputs_name, index=None)
    if essential_results_cols:
        OutputMatch.results_on_orig_df[essential_results_cols].to_csv(
            OutputMatch.results_orig_df_name, index=None
        )
    else:
        OutputMatch.results_on_orig_df.to_csv(
            OutputMatch.results_orig_df_name, index=None
        )

    output_files.extend(
        [OutputMatch.results_orig_df_name, OutputMatch.match_outputs_name]
    )

    return final_summary, output_files, estimate_total_processing_time


# Run a match run for a single batch
def create_simple_batch_ranges(
    df: PandasDataFrame, ref_df: PandasDataFrame, batch_size: int, ref_batch_size: int
):
    # print("Search df batch size: ", batch_size)
    # print("ref_df df batch size: ", ref_batch_size)

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
    search_indexes = df.index.tolist()
    ref_indexes = ref_df.index.tolist()

    if not search_indexes:
        return pd.DataFrame(
            data={
                "search_range": [[]],
                "ref_range": [ref_indexes],
                "batch_length": [0],
                "ref_length": [len(ref_indexes)],
            }
        )

    search_chunks = [
        search_indexes[i : i + batch_size]
        for i in range(0, len(search_indexes), batch_size)
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
):
    """
    Create batches of address indexes for search and reference dataframes based on shortened postcodes.
    """

    # If df sizes are smaller than the batch size limits, no need to run through everything
    if len(df) < batch_size and len(ref_df) < ref_batch_size:
        print(
            "Dataframe sizes are smaller than maximum batch sizes, no need to split data."
        )
        lengths_df = pd.DataFrame(
            data={
                "search_range": [df.index.tolist()],
                "ref_range": [ref_df.index.tolist()],
                "batch_length": len(df),
                "ref_length": len(ref_df),
            }
        )
        return lengths_df

    # df.index = df[search_postcode_col]

    df["index"] = df.index
    ref_df["index"] = ref_df.index

    # Remove the last character of postcode
    df["postcode_minus_last_character"] = (
        df[search_postcode_col]
        .str.lower()
        .str.strip()
        .str.replace(r"\s+", "", regex=True)
        .str[:-1]
    )
    ref_df["postcode_minus_last_character"] = (
        ref_df[ref_postcode_col]
        .str.lower()
        .str.strip()
        .str.replace(r"\s+", "", regex=True)
        .str[:-1]
    )

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
                print("Batch length reached - breaking")
                break

            try:
                current_postcode_search_data_add = df.loc[
                    [current_postcode]
                ]  # [df['postcode_minus_last_character'].isin(current_postcode)]
                current_postcode_ref_data_add = ref_df.loc[
                    [current_postcode]
                ]  # [ref_df['postcode_minus_last_character'].isin(current_postcode)]

                # print(current_postcode_search_data_add)

                if not current_postcode_search_data_add.empty:
                    current_batch.extend(current_postcode_search_data_add["index"])

                if not current_postcode_ref_data_add.empty:
                    current_ref_batch.extend(current_postcode_ref_data_add["index"])

            except Exception:
                # print("postcode not found: ", current_postcode)
                pass

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

    return lengths_df


def run_single_match_batch(
    InitialMatch: MatcherClass,
    batch_n: int,
    total_batches: int,
    use_postcode_blocker: bool = True,
    write_outputs: bool = True,
    show_progress: bool = True,
    progress=gr.Progress(track_tqdm=True),
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
        )

        if FuzzyNotStdMatch.abort_flag:
            message = "Nothing to match! Aborting address check."
            print(message)
            return message, InitialMatch

        FuzzyNotStdMatch = combine_two_matches(
            InitialMatch, FuzzyNotStdMatch, df_name, write_outputs=write_outputs
        )

        full_match_series = (
            FuzzyNotStdMatch.match_results_output["full_match"]
            .fillna(False)
            .astype(bool)
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
        )
        FuzzyStdMatch = combine_two_matches(
            FuzzyNotStdMatch, FuzzyStdMatch, df_name, write_outputs=write_outputs
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
        )
        FuzzyNNetNotStdMatch = combine_two_matches(
            FuzzyStdMatch, FuzzyNNetNotStdMatch, df_name, write_outputs=write_outputs
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
        )
        FuzzyNNetStdMatch = combine_two_matches(
            FuzzyNNetNotStdMatch,
            FuzzyNNetStdMatch,
            df_name,
            write_outputs=write_outputs,
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
):

    today_rev = datetime.now().strftime("%Y%m%d")

    # print(Matcher.standardise)
    Matcher.standardise = standardise

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
            existing_match_col=_existing_col,
        )
        if match_results_output.empty:
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
    Matcher.results_on_orig_df = results_on_orig_df
    Matcher.summary = summary
    Matcher.output_summary = create_match_summary(
        Matcher.match_results_output, df_name=df_name
    )

    Matcher.match_outputs_name = (
        output_folder + "diagnostics_" + file_stub + today_rev + ".csv"
    )
    Matcher.results_orig_df_name = (
        output_folder + "results_" + file_stub + today_rev + ".csv"
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
    existing_match_col: Optional[str] = None,
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

    # RUN WITH POSTCODE AS A BLOCKER #
    can_run_postcode_blocker = (
        use_postcode_blocker
        and _column_has_usable_values(search_df_after_stand, "postcode_search")
        and _column_has_usable_values(ref_df_after_stand, "postcode_search")
    )

    if can_run_postcode_blocker:
        # Exclude blank postcodes from postcode-blocked matching.
        # (They can still be considered later in street-blocked matching.)
        search_pc = (
            search_df_after_stand["postcode_search"].fillna("").astype(str).str.strip()
        )
        ref_pc = (
            ref_df_after_stand["postcode_search"].fillna("").astype(str).str.strip()
        )
        search_df_after_stand_pc = search_df_after_stand.loc[search_pc.ne("")].copy()
        ref_df_after_stand_pc = ref_df_after_stand.loc[ref_pc.ne("")].copy()

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
            print("Starting the fuzzy match")

            tic = time.perf_counter()
            results = string_match_by_post_code_multiple(
                match_address_series=search_df_after_stand_series.copy(),
                reference_address_series=ref_df_after_stand_series_checked,
                search_limit=fuzzy_search_addr_limit,
                scorer_name=fuzzy_scorer_used,
            )

            toc = time.perf_counter()
            print(f"Performed the fuzzy match in {toc - tic:0.1f} seconds")

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
    if (not match_results_output.empty) and (
        (sum(~match_results_output["full_match"]) == 0)
        | (
            sum(
                match_results_output[~match_results_output["full_match"]]["fuzzy_score"]
            )
            == 0
        )
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
            )
        else:
            results_on_orig_df = match_results_output

        print("results_on_orig_df in fuzzy_match shape: ", results_on_orig_df.shape)

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

    print("Starting the fuzzy match with street as blocker")

    tic = time.perf_counter()
    results_st = string_match_by_post_code_multiple(
        match_address_series=search_df_match_series_street.copy(),
        reference_address_series=ref_df_after_stand_series_street_checked.copy(),
        search_limit=fuzzy_search_addr_limit,
        scorer_name=fuzzy_scorer_used,
    )

    toc = time.perf_counter()

    print(f"Performed the fuzzy match in {toc - tic:0.1f} seconds")

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
        )
    else:
        results_on_orig_df = match_results_output

    print("results_on_orig_df in fuzzy_match shape: ", results_on_orig_df.shape)

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

    print(search_df_after_stand.shape[0])
    print(ref_df_after_stand.shape[0])

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

    summary_pc = create_match_summary(
        match_results_output_final_pc, df_name="NNet blocked by Postcode " + df_name
    )
    print(summary_pc)

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

    summary_street = create_match_summary(
        match_results_output_final_st, df_name="NNet blocked by Street " + df_name
    )
    print(summary_street)

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

    combined_std_not_matches = pd.concat(
        [orig_df, new_df], axis=0
    )  # , ignore_index=True)

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
    if np.issubdtype(m_series.dtype, np.number):
        m_bool = m_series.fillna(0).astype(bool)
    elif m_series.dtype == bool or str(m_series.dtype) == "boolean":
        m_bool = m_series.fillna(False).astype(bool)
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

    combined_std_not_matches_no_dups = combined_std_not_matches_no_dups.sort_index()

    return combined_std_not_matches_no_dups


def combine_two_matches(
    OrigMatchClass: MatcherClass,
    NewMatchClass: MatcherClass,
    df_name: str,
    write_outputs: bool = True,
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

    matched_mask = (
        NewMatchClass.results_on_orig_df["Matched with reference address"]
        .fillna(False)
        .astype(bool)
    )

    found_index = NewMatchClass.results_on_orig_df.loc[
        matched_mask,
        NewMatchClass.search_df_key_field,
    ].astype(int)

    key_field_values = NewMatchClass.search_df_not_matched[
        NewMatchClass.search_df_key_field
    ].astype(
        int
    )  # Assuming list conversion is suitable

    rows_to_drop = key_field_values[key_field_values.isin(found_index)].tolist()
    NewMatchClass.search_df_not_matched = NewMatchClass.search_df_not_matched.loc[
        ~NewMatchClass.search_df_not_matched[NewMatchClass.search_df_key_field].isin(
            rows_to_drop
        ),
        :,
    ]  # .drop(rows_to_drop, axis = 0)

    # Filter out rows from NewMatchClass.search_df_cleaned

    filtered_rows_to_keep = (
        NewMatchClass.search_df_cleaned[NewMatchClass.search_df_key_field]
        .astype(int)
        .isin(
            NewMatchClass.search_df_not_matched[
                NewMatchClass.search_df_key_field
            ].astype(int)
        )
        .to_list()
    )

    NewMatchClass.search_df_cleaned = NewMatchClass.search_df_cleaned.loc[
        filtered_rows_to_keep, :
    ]  # .drop(rows_to_drop, axis = 0)
    NewMatchClass.search_df_after_stand = NewMatchClass.search_df_after_stand.loc[
        filtered_rows_to_keep, :
    ]  # .drop(rows_to_drop)
    NewMatchClass.search_df_after_full_stand = (
        NewMatchClass.search_df_after_full_stand.loc[filtered_rows_to_keep, :]
    )  # .drop(rows_to_drop)

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

    match_results_output_match_score_is_0 = NewMatchClass.match_results_output[
        NewMatchClass.match_results_output["fuzzy_score"] == 0.0
    ][["index", "fuzzy_score"]].drop_duplicates(subset="index")
    match_results_output_match_score_is_0["index"] = (
        match_results_output_match_score_is_0["index"].astype(str)
    )
    # NewMatchClass.results_on_orig_df["index"] = NewMatchClass.results_on_orig_df["index"].astype(str)
    NewMatchClass.results_on_orig_df = NewMatchClass.results_on_orig_df.merge(
        match_results_output_match_score_is_0, on="index", how="left"
    )

    NewMatchClass.results_on_orig_df.loc[
        NewMatchClass.results_on_orig_df["fuzzy_score"] == 0.0, "Excluded from search"
    ] = "Match score is 0"
    NewMatchClass.results_on_orig_df = NewMatchClass.results_on_orig_df.drop(
        "fuzzy_score", axis=1
    )

    # Drop any duplicates, prioritise any matches
    NewMatchClass.results_on_orig_df["index"] = NewMatchClass.results_on_orig_df[
        "index"
    ].astype(int, errors="ignore")
    NewMatchClass.results_on_orig_df["ref_index"] = NewMatchClass.results_on_orig_df[
        "ref_index"
    ].astype(int, errors="ignore")

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
    print(NewMatchClass.output_summary)

    NewMatchClass.search_df_not_matched = filter_not_matched(
        NewMatchClass.match_results_output,
        NewMatchClass.search_df,
        NewMatchClass.search_df_key_field,
    )

    ### Rejoin the excluded matches onto the output file
    # NewMatchClass.results_on_orig_df = pd.concat([NewMatchClass.results_on_orig_df, NewMatchClass.excluded_df])

    NewMatchClass.match_outputs_name = (
        output_folder + "diagnostics_" + today_rev + ".csv"
    )  # + NewMatchClass.file_name + "_"
    NewMatchClass.results_orig_df_name = (
        output_folder + "results_" + today_rev + ".csv"
    )  # + NewMatchClass.file_name + "_"

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

    # Attach search-side in_existing value onto diagnostics output as well.
    if _em_col_btch and (
        NewMatchClass.search_df_key_field in NewMatchClass.search_df_cleaned.columns
    ):
        diag_output_col = f"{_em_col_btch} (from search data)"
        if diag_output_col not in NewMatchClass.match_results_output.columns:
            _src_col = None
            if _em_col_btch in NewMatchClass.search_df_cleaned.columns:
                _src_col = _em_col_btch
            else:
                _alt = f"__search_side_{_em_col_btch}"
                if _alt in NewMatchClass.search_df_cleaned.columns:
                    _src_col = _alt

            if _src_col:
                _map_df = NewMatchClass.search_df_cleaned[
                    [NewMatchClass.search_df_key_field, _src_col]
                ].copy()
                _map_df = _map_df.rename(columns={_src_col: diag_output_col})
                _map_df[NewMatchClass.search_df_key_field] = _map_df[
                    NewMatchClass.search_df_key_field
                ].astype(str)
                if (
                    NewMatchClass.search_df_key_field
                    in NewMatchClass.match_results_output.columns
                ):
                    NewMatchClass.match_results_output[
                        NewMatchClass.search_df_key_field
                    ] = NewMatchClass.match_results_output[
                        NewMatchClass.search_df_key_field
                    ].astype(
                        str
                    )
                NewMatchClass.match_results_output = (
                    NewMatchClass.match_results_output.merge(
                        _map_df,
                        on=NewMatchClass.search_df_key_field,
                        how="left",
                    )
                )

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

    full_match_series = match_results_output["full_match"].fillna(False).astype(bool)

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
