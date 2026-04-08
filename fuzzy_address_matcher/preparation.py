import os
import re
from datetime import datetime
from typing import Dict, List, Tuple, Type

import pandas as pd
import polars as pl
from gradio import Progress
from tqdm import tqdm

from fuzzy_address_matcher.standardise import remove_postcode

tqdm.pandas()  # Registers the progress_apply method

PandasDataFrame = Type[pd.DataFrame]
PandasSeries = Type[pd.Series]
MatchedResults = Dict[str, Tuple[str, int]]
array = List[str]

today = datetime.now().strftime("%d%m%Y")
today_rev = datetime.now().strftime("%Y%m%d")
VALID_PREPARATION_BACKENDS = {"pandas", "polars"}


def _get_preparation_backend() -> str:
    backend = os.environ.get("PREPARATION_BACKEND", "pandas").strip().lower()
    if backend not in VALID_PREPARATION_BACKENDS:
        return "pandas"
    return backend


def _use_polars_backend() -> bool:
    return _get_preparation_backend() == "polars"


def prepare_search_address_string(
    search_str: str,
) -> Tuple[pd.DataFrame, str, List[str], List[str]]:
    """Extracts address and postcode from search_str into new DataFrame"""

    # Validate input
    if not isinstance(search_str, str):
        raise TypeError("search_str must be a string")

    search_df = pd.DataFrame(data={"full_address": [search_str]})

    # print(search_df)

    # Extract postcode
    postcode_series = extract_postcode(search_df, "full_address").dropna(axis=1)[0]

    # Remove postcode from address
    address_series = remove_postcode(search_df, "full_address")

    # Construct output DataFrame
    search_df_out = pd.DataFrame()
    search_df_out["full_address"] = address_series
    search_df_out["postcode"] = postcode_series

    # Set key field for joining
    key_field = "index"

    # Reset index to use as key field
    search_df_out = search_df_out.reset_index()

    # Define column names to return
    address_cols = ["full_address"]
    postcode_col = ["postcode"]

    return search_df_out, key_field, address_cols, postcode_col


def prepare_search_address(
    search_df: pd.DataFrame,
    address_cols: list,
    postcode_col: list,
    key_col: str,
    progress=Progress(track_tqdm=True),
) -> Tuple[pd.DataFrame, str]:

    progress(0, "Preparing search address column")

    # Validate inputs
    if not isinstance(search_df, pd.DataFrame):
        raise TypeError("search_df must be a Pandas DataFrame")

    if not isinstance(address_cols, list):
        raise TypeError("address_cols must be a list")

    if not isinstance(postcode_col, list):
        raise TypeError("postcode_col must be a list")

    if not isinstance(key_col, str):
        raise TypeError("key_col must be a string")

    # If there is a full address and postcode column in the addresses, clean any postcodes from the first column
    if len(address_cols) == 2:
        # Remove postcode from address
        search_df[address_cols[0]] = remove_postcode(search_df, address_cols[0])

    # Join address columns into one
    full_addresses = _join_address(search_df, address_cols)

    # Clean address columns
    # search_df_polars = pl.from_dataframe(search_df)
    clean_addresses = _clean_columns(full_addresses, ["full_address"])

    # Add postcode column
    full_df = _add_postcode_column(clean_addresses, postcode_col)

    postcode_col_name = (
        postcode_col[0]
        if isinstance(postcode_col, list) and len(postcode_col) > 0
        else postcode_col
    )

    # For single-column input, split out postcode from the full address and then
    # recompose with exactly one space between address and postcode.
    if postcode_col_name == "full_address_postcode":
        full_df["full_address"] = remove_postcode(full_df, "full_address")
        full_df = _clean_columns(full_df, ["full_address"])

        full_df["postcode"] = (
            full_df["postcode"]
            .fillna("")
            .astype(str)
            .str.upper()
            .str.replace(r"\s{2,}", " ", regex=True)
            .str.strip()
        )
        has_postcode = full_df["postcode"].ne("")
        full_df.loc[has_postcode, "full_address"] = (
            full_df.loc[has_postcode, "full_address"].astype(str).str.strip()
            + " "
            + full_df.loc[has_postcode, "postcode"]
        ).str.strip()

    # Ensure index column
    final_df = _ensure_index(full_df, key_col)

    return final_df


# Helper functions
def _clean_columns(df: PandasDataFrame, cols: List[str]):
    if _use_polars_backend():
        return _clean_columns_polars(df, cols)

    # Cleaning logic
    def clean_col(col):
        return (
            col.astype(str)
            .fillna("")
            .infer_objects(copy=False)
            .str.replace("nan", "")
            .str.replace(r"\bNone\b", "", case=False, regex=True)
            .str.replace(r"\s{2,}", " ", regex=True)
            .str.replace(",", " ")
            .str.replace(r"[\r\n]+", " ", regex=True)  # Replace line breaks with spaces
            .str.strip()
            # Remove duplicate two words at the end if present
            .str.replace(r"(\b\w+\b\s+\b\w+\b)\s+\1$", r"\1", regex=True)
        )

    for col in tqdm(cols, desc="Cleaning columns"):
        df[col] = clean_col(df[col])

    return df


def _clean_columns_polars(df: PandasDataFrame, cols: List[str]):
    duplicate_two_words_pattern = re.compile(r"(\b\w+\b\s+\b\w+\b)\s+\1$")
    pldf = pl.from_pandas(df, include_index=False)
    exprs = []
    for col in cols:
        exprs.append(
            pl.col(col)
            .cast(pl.Utf8)
            .fill_null("")
            .str.replace_all("nan", "")
            .str.replace_all(r"(?i)\bnone\b", "")
            .str.replace_all(r"\s{2,}", " ")
            .str.replace_all(",", " ")
            .str.replace_all(r"[\r\n]+", " ")
            .str.strip_chars()
            .map_elements(
                lambda value: duplicate_two_words_pattern.sub(r"\1", value),
                return_dtype=pl.Utf8,
            )
            .alias(col)
        )
    updated = pldf.with_columns(exprs)
    for col in cols:
        df[col] = updated[col].to_numpy()
    return df


def _join_address(df: PandasDataFrame, cols: List[str]):
    if _use_polars_backend():
        pldf = pl.from_pandas(df[cols], include_index=False)
        cast_exprs = [
            pl.col(col).cast(pl.Utf8).fill_null("nan").alias(col) for col in cols
        ]
        updated = pldf.with_columns(cast_exprs).with_columns(
            pl.concat_str([pl.col(col) for col in cols], separator=" ")
            .str.replace_all(r"\s{2,}", " ")
            .str.strip_chars()
            .alias("full_address")
        )
        df["full_address"] = updated["full_address"].to_numpy()
        return df

    # Joining logic
    full_address = df[cols].apply(lambda row: " ".join(row.values.astype(str)), axis=1)
    df["full_address"] = full_address.str.replace(
        r"\s{2,}", " ", regex=True
    ).str.strip()

    return df


def _add_postcode_column(df: PandasDataFrame, postcodes: str):
    # Add postcode column
    if isinstance(postcodes, list):
        postcodes = postcodes[0]

    # Helper: normalise a postcode-like series to strings with blanks for missing
    def _norm_pc(s: pd.Series) -> pd.Series:
        return s.fillna("").astype(str).str.strip()

    if postcodes == "full_address_postcode":
        # Extract postcode from full address text into the same column name, then treat
        # it as the candidate postcode series.
        df["full_address_postcode"] = extract_postcode(df, "full_address_postcode")[0]
        candidate = _norm_pc(df["full_address_postcode"])
        source_col = "full_address_postcode"
    else:
        if postcodes not in df.columns:
            return df
        candidate = _norm_pc(df[postcodes])
        source_col = postcodes

    # Avoid creating duplicate column names ("postcode" already exists in some inputs).
    if "postcode" in df.columns:
        existing = _norm_pc(df["postcode"])
        use_existing = existing.ne("")
        df["postcode"] = existing.where(use_existing, candidate)
        # Drop the source column if it's not the canonical postcode column.
        if source_col != "postcode" and source_col in df.columns:
            df = df.drop(columns=[source_col])
    else:
        df["postcode"] = candidate
        if source_col != "postcode" and source_col in df.columns:
            df = df.drop(columns=[source_col])

    return df


def _ensure_index(df: PandasDataFrame, index_col: str):
    # Ensure index column exists
    if (index_col == "index") & ~("index" in df.columns):
        print("Resetting index in _ensure_index function")
        df = df.reset_index()

    df[index_col] = df[index_col].astype(str)

    return df


def create_full_address(df: PandasDataFrame):

    df = df.fillna("").infer_objects(copy=False)

    if "Organisation" not in df.columns:
        df["Organisation"] = ""

    df["full_address"] = (
        df["Organisation"]
        + " "
        + df["SaoText"]
        .str.replace(" - ", " REPL ")
        .str.replace("- ", " REPLEFT ")
        .str.replace(" -", " REPLRIGHT ")
        + " "
        + df["SaoStartNumber"].astype(str)
        + df["SaoStartSuffix"]
        + "-"
        + df["SaoEndNumber"].astype(str)
        + df["SaoEndSuffix"]
        + " "
        + df["PaoText"]
        .str.replace(" - ", " REPL ")
        .str.replace("- ", " REPLEFT ")
        .str.replace(" -", " REPLRIGHT ")
        + " "
        + df["PaoStartNumber"].astype(str)
        + df["PaoStartSuffix"]
        + "-"
        + df["PaoEndNumber"].astype(str)
        + df["PaoEndSuffix"]
        + " "
        + df["Street"]
        + " "
        + df["PostTown"]
        + " "
        + df["Postcode"]
    )

    # .str.replace(r'(?<=[a-zA-Z])-(?![a-zA-Z])|(?<![a-zA-Z])-(?=[a-zA-Z])', ' ', regex=True)\

    # .str.replace(".0","", regex=False)\

    df["full_address"] = (
        df["full_address"]
        .str.replace("-999", "")
        .str.replace(" -", " ")
        .str.replace("- ", " ")
        .str.replace(" REPL ", " - ")
        .str.replace(" REPLEFT ", "- ")
        .str.replace(" REPLRIGHT ", " -")
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    # .str.replace("  "," ")\

    return df["full_address"]


def prepare_ref_address(
    ref_df: PandasDataFrame,
    ref_address_cols: List[str],
    new_join_col=[],
    standard_cols=True,
    progress=Progress(track_tqdm=True),
):

    progress(0, "Preparing reference address")

    if ("SaoText" in ref_df.columns) | ("Secondary_Name_LPI" in ref_df.columns):
        standard_cols = True
    else:
        standard_cols = False

    ref_address_cols_uprn = ref_address_cols.copy()

    if new_join_col:
        ref_address_cols_uprn.extend(new_join_col)

    # Preserve any explicit street column if available so street-blocking can prefer it later.
    # (Users may include "Street" / "street" among `in_refcol`.)
    if ("Street" in ref_df.columns) and ("Street" not in ref_address_cols_uprn):
        ref_address_cols_uprn.append("Street")
    elif ("street" in ref_df.columns) and ("street" not in ref_address_cols_uprn):
        ref_address_cols_uprn.append("street")

    ref_address_cols_uprn_w_ref = ref_address_cols_uprn.copy()
    ref_address_cols_uprn_w_ref.extend(["Reference file"])

    ref_df_cleaned = ref_df.copy()

    ref_df_cleaned["ref_index"] = ref_df_cleaned.index

    # --- Postcode column normalisation ---
    # Prevent duplicate postcode column names from propagating downstream (which can
    # make df["Postcode"] return a DataFrame and break `.str` operations).
    #
    # Canonical reference postcode column name in this codebase is "Postcode".
    if (
        "Postcode" not in ref_df_cleaned.columns
        and "postcode" in ref_df_cleaned.columns
    ):
        ref_df_cleaned = ref_df_cleaned.rename(columns={"postcode": "Postcode"})

    # If both exist, prefer non-blank values in "Postcode" and fill from "postcode".
    if "Postcode" in ref_df_cleaned.columns and "postcode" in ref_df_cleaned.columns:
        _pc_main = ref_df_cleaned["Postcode"].fillna("").astype(str).str.strip()
        _pc_alt = ref_df_cleaned["postcode"].fillna("").astype(str).str.strip()
        ref_df_cleaned["Postcode"] = _pc_main.where(_pc_main.ne(""), _pc_alt)
        ref_df_cleaned = ref_df_cleaned.drop(columns=["postcode"])

    # If there are duplicated "Postcode" columns (possible after merges), keep the first.
    if (
        hasattr(ref_df_cleaned.columns, "duplicated")
        and ref_df_cleaned.columns.duplicated().any()
    ):
        dup_names = ref_df_cleaned.columns[ref_df_cleaned.columns.duplicated()].tolist()
        if "Postcode" in dup_names:
            ref_df_cleaned = ref_df_cleaned.loc[
                :, ~ref_df_cleaned.columns.duplicated()
            ].copy()

    # In on-prem LPI db street has been excluded, so put this back in
    if ("Street" not in ref_df_cleaned.columns) & (
        "Address_LPI" in ref_df_cleaned.columns
    ):
        ref_df_cleaned["Street"] = (
            ref_df_cleaned["Address_LPI"]
            .str.replace("\\n", " ", regex=True)
            .apply(extract_street_name)
        )  #

    if ("Organisation" not in ref_df_cleaned.columns) & (
        "SaoText" in ref_df_cleaned.columns
    ):
        ref_df_cleaned["Organisation"] = ""

    # Only keep columns that exist (e.g. UPRN may be absent in user-supplied ref data).
    ref_address_cols_uprn_w_ref_present = [
        col for col in ref_address_cols_uprn_w_ref if col in ref_df_cleaned.columns
    ]
    ref_df_cleaned = ref_df_cleaned[ref_address_cols_uprn_w_ref_present]

    ref_df_cleaned = ref_df_cleaned.fillna("").infer_objects(copy=False)
    all_columns = list(ref_df_cleaned)
    ref_df_cleaned[all_columns] = (
        ref_df_cleaned[all_columns].astype(str).replace("nan", "")
    )
    ref_df_cleaned = ref_df_cleaned.replace(r"\.0", "", regex=True)

    if standard_cols:
        ref_df_cleaned = (
            ref_df_cleaned[ref_address_cols_uprn_w_ref_present]
            .fillna("")
            .infer_objects(copy=False)
        )

        ref_df_cleaned["fulladdress"] = create_full_address(
            ref_df_cleaned[ref_address_cols_uprn_w_ref_present]
        )

    else:
        ref_df_cleaned = (
            ref_df_cleaned[ref_address_cols_uprn_w_ref_present]
            .fillna("")
            .infer_objects(copy=False)
        )

        # `ref_address_cols` can include optional fields (e.g. UPRN). Only use those present.
        ref_address_cols_present = [
            col for col in ref_address_cols if col in ref_df_cleaned.columns
        ]
        ref_df_cleaned = _join_address(ref_df_cleaned, ref_address_cols_present)
        ref_df_cleaned["fulladdress"] = ref_df_cleaned["full_address"]

    ref_df_cleaned = _clean_columns(ref_df_cleaned, ["fulladdress"])

    # Create a street column if it doesn't exist by extracting street from the full address

    if "Street" not in ref_df_cleaned.columns:
        ref_df_cleaned["Street"] = ref_df_cleaned["fulladdress"].apply(
            extract_street_name
        )

    # Add index column
    if "ref_index" not in ref_df_cleaned.columns:
        ref_df_cleaned["ref_index"] = ref_df_cleaned.index

    return ref_df_cleaned


def extract_postcode(df, col: str) -> PandasSeries:
    """
    Extract a postcode from a string column in a dataframe
    """
    postcode_series = (
        df[col]
        .str.upper()
        .str.extract(
            pat="(\\b(?:[A-Z][A-HJ-Y]?[0-9][0-9A-Z]? ?[0-9][A-Z]{2})|((GIR ?0A{2})\\b$)|(?:[A-Z][A-HJ-Y]?[0-9][0-9A-Z]? ?[0-9]{1}?)$)|(\\b(?:[A-Z][A-HJ-Y]?[0-9][0-9A-Z]?)\\b$)"
        )
    )

    return postcode_series


# Remove addresses with no numbers in at all - too high a risk of badly assigning an address
def check_no_number_addresses(
    df: PandasDataFrame, in_address_series: str
) -> PandasSeries:
    """
    Highlight addresses from a pandas df where there are no numbers in the address.
    """
    if _use_polars_backend():
        pldf = pl.DataFrame(
            {"text": df[in_address_series].astype(str).to_numpy()}
        ).with_columns(pl.col("text").cast(pl.Utf8).fill_null("").alias("text"))
        no_numbers_series = (~pldf["text"].str.contains(r"\d+")).to_numpy()
    else:
        df["in_address_series_temp"] = df[in_address_series].str.lower()
        no_numbers_series = df["in_address_series_temp"].str.contains(
            r"^(?!.*\d+).*$", regex=True
        )

    df.loc[no_numbers_series, "Excluded from search"] = (
        "Excluded - no numbers in address"
    )

    if "in_address_series_temp" in df.columns:
        df = df.drop("in_address_series_temp", axis=1)

    # print(df[["full_address", "Excluded from search"]])

    return df


def extract_street_name(address: str) -> str:
    """
    Extracts the street name from the given address.

    Args:
        address (str): The input address string.

    Returns:
        str: The extracted street name, or an empty string if no match is found.

    Examples:
        >>> address1 = "1 Ash Park Road SE54 3HB"
        >>> extract_street_name(address1)
        'Ash Park Road'

        >>> address2 = "Flat 14 1 Ash Park Road SE54 3HB"
        >>> extract_street_name(address2)
        'Ash Park Road'

        >>> address3 = "123 Main Blvd"
        >>> extract_street_name(address3)
        'Main Blvd'

        >>> address4 = "456 Maple AvEnUe"
        >>> extract_street_name(address4)
        'Maple AvEnUe'

        >>> address5 = "789 Oak Street"
        >>> extract_street_name(address5)
        'Oak Street'
    """

    street_types = [
        "Street",
        "St",
        "Boulevard",
        "Blvd",
        "Highway",
        "Hwy",
        "Broadway",
        "Freeway",
        "Causeway",
        "Cswy",
        "Expressway",
        "Way",
        "Walk",
        "Lane",
        "Ln",
        "Road",
        "Rd",
        "Avenue",
        "Ave",
        "Circle",
        "Cir",
        "Cove",
        "Cv",
        "Drive",
        "Dr",
        "Parkway",
        "Pkwy",
        "Park",
        "Court",
        "Ct",
        "Square",
        "Sq",
        "Loop",
        "Place",
        "Pl",
        "Parade",
        "Estate",
        "Alley",
        "Arcade",
        "Avenue",
        "Ave",
        "Bay",
        "Bend",
        "Brae",
        "Byway",
        "Close",
        "Corner",
        "Cove",
        "Crescent",
        "Cres",
        "Cul-de-sac",
        "Dell",
        "Drive",
        "Dr",
        "Esplanade",
        "Glen",
        "Green",
        "Grove",
        "Heights",
        "Hts",
        "Mews",
        "Parade",
        "Path",
        "Piazza",
        "Promenade",
        "Quay",
        "Ridge",
        "Row",
        "Terrace",
        "Ter",
        "Track",
        "Trail",
        "View",
        "Villas",
        "Marsh",
        "Embankment",
        "Cut",
        "Hill",
        "Passage",
        "Rise",
        "Vale",
        "Side",
    ]

    # Dynamically construct the regex pattern with all possible street types
    street_types_pattern = "|".join(
        rf"{re.escape(street_type)}" for street_type in street_types
    )

    # The overall regex pattern to capture the street name
    pattern = rf"(?:\d+\s+|\w+\s+\d+\s+|.*\d+[a-z]+\s+|.*\d+\s+)*(?P<street_name>[\w\s]+(?:{street_types_pattern}))"

    def replace_postcode(address):
        pattern = r"\b(?:[A-Z][A-HJ-Y]?[0-9][0-9A-Z]? ?[0-9][A-Z]{2}|GIR ?0A{2})\b$|(?:[A-Z][A-HJ-Y]?[0-9][0-9A-Z]? ?[0-9]{1}?)$|\b(?:[A-Z][A-HJ-Y]?[0-9][0-9A-Z]?)\b$"
        return re.sub(pattern, "", address)

    modified_address = replace_postcode(address.upper())
    # print(modified_address)
    # print(address)

    # Perform a case-insensitive search
    match = re.search(pattern, modified_address, re.IGNORECASE)

    if match:
        street_name = match.group("street_name")
        return street_name.strip()
    else:
        return ""

    # Exclude non-postal addresses


def remove_non_postal(df: PandasDataFrame, in_address_series: str):
    """
    Highlight non-postal addresses from a polars df where a string series that contain specific substrings
    indicating non-postal addresses like 'garage', 'parking', 'shed', etc.
    """
    if _use_polars_backend():
        pldf = pl.DataFrame(
            {"text": df[in_address_series].astype(str).to_numpy()}
        ).with_columns(pl.col("text").cast(pl.Utf8).fill_null("").alias("text"))
        non_postal_series = (
            pldf["text"]
            .str.to_lowercase()
            .str.contains(
                r"\bgarage\b|\bgarages\b|\bparking\b|\bshed\b|\bsheds\b|\bbike\b|\bbikes\b|\bbicycle store\b"
            )
        )
        non_postal_series = non_postal_series.to_numpy()
    else:
        df["in_address_series_temp"] = df[in_address_series].str.lower()

        garage_address_series = df["in_address_series_temp"].str.contains(
            "(?i)(?:\\bgarage\\b|\\bgarages\\b)", regex=True
        )
        parking_address_series = df["in_address_series_temp"].str.contains(
            "(?i)(?:\\bparking\\b)", regex=True
        )
        shed_address_series = df["in_address_series_temp"].str.contains(
            "(?i)(?:\\bshed\\b|\\bsheds\\b)", regex=True
        )
        bike_address_series = df["in_address_series_temp"].str.contains(
            "(?i)(?:\\bbike\\b|\\bbikes\\b)", regex=True
        )
        bicycle_store_address_series = df["in_address_series_temp"].str.contains(
            "(?i)(?:\\bbicycle store\\b|\\bbicycle store\\b)", regex=True
        )

        non_postal_series = (
            garage_address_series
            | parking_address_series
            | shed_address_series
            | bike_address_series
            | bicycle_store_address_series
        )

    df.loc[non_postal_series, "Excluded from search"] = "Excluded - non-postal address"

    if "in_address_series_temp" in df.columns:
        df = df.drop("in_address_series_temp", axis=1)

    return df
