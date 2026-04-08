import ast
import os
import time
import zipfile
from typing import List, Type

import pandas as pd
from pydantic import BaseModel

from fuzzy_address_matcher.config import (
    MATCHER_CUDA_VISIBLE_DEVICES,
    MODEL_EXTRACT_USE_PROJECT_ROOT,
    N_EPOCHS,
    USE_NNET_MODEL,
    data_sample_size,
    file_step_suffix,
    filter_to_lambeth_pcodes,
    fuzzy_match_limit,
    fuzzy_method,
    fuzzy_scorer_used,
    fuzzy_search_addr_limit,
    matching_variables,
    model_stub,
    model_type,
    model_version,
    output_folder,
    score_cut_off,
    standardise,
    text_columns,
    weights,
)

PandasDataFrame = Type[pd.DataFrame]
PandasSeries = Type[pd.Series]

# +
""" Fuzzywuzzy/Rapidfuzz scorer to use. Options are: ratio, partial_ratio, token_sort_ratio, partial_token_sort_ratio,
token_set_ratio, partial_token_set_ratio, QRatio, UQRatio, WRatio (default), UWRatio
details here: https://stackoverflow.com/questions/31806695/when-to-use-which-fuzz-function-to-compare-2-strings"""

if standardise:
    std = "_std"
else:
    std = "_not_std"

dataset_name = "data" + std

suffix_used = dataset_name + "_" + fuzzy_scorer_used

# https://stackoverflow.com/questions/59221557/tensorflow-v2-replacement-for-tf-contrib-predictor-from-saved-model


# Uncomment these lines for the tensorflow model
# model_type = "tf"
# model_stub = "addr_model_out_lon"
# model_version = "00000001"
# file_step_suffix = "550" # I add a suffix to output files to be able to separate comparisons of test data from the same model with different steps e.g. '350' indicates a model that has been through 350,000 steps of training

word_to_index = {}
cat_to_idx = {}
vocab = []
device = "cpu"

global labels_list
labels_list = []

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))


def _atomic_create_lock(lock_path: str) -> int | None:
    """
    Attempt to create a lock file atomically.
    Returns a file descriptor if lock acquired, otherwise None.
    """
    try:
        return os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
    except FileExistsError:
        return None


def _ensure_model_assets_extracted(model_zip_path: str, extract_dir: str) -> None:
    """
    Ensure model assets are extracted exactly-once across processes.

    This module is imported by multiprocessing spawned workers on Windows.
    Without a lock, concurrent `extractall()` calls can leave partially-written
    files that later fail to parse (e.g. SyntaxError when reading word_to_index).
    """
    os.makedirs(extract_dir, exist_ok=True)

    done_marker = os.path.join(extract_dir, ".model_extract.done")
    lock_path = os.path.join(extract_dir, ".model_extract.lock")

    # If we've already extracted successfully, nothing to do.
    if os.path.exists(done_marker):
        return

    # Fast path: if the expected files exist and are non-empty, mark as done.
    expected = [
        os.path.join(extract_dir, "vocab.txt"),
        os.path.join(extract_dir, "word_to_index.txt"),
        os.path.join(extract_dir, "cat_to_idx.txt"),
    ]
    if all(os.path.exists(p) and os.path.getsize(p) > 0 for p in expected):
        with open(done_marker, "w", encoding="utf-8") as f:
            f.write("ok\n")
        return

    # Acquire lock (spin with backoff) to avoid parallel extraction.
    fd = None
    for _ in range(60):  # up to ~30s
        fd = _atomic_create_lock(lock_path)
        if fd is not None:
            break
        # Another process is extracting; wait a bit and re-check.
        if os.path.exists(done_marker):
            return
        time.sleep(0.5)

    # If lock wasn't acquired, proceed without extraction but let caller fail loudly.
    if fd is None:
        return

    try:
        with zipfile.ZipFile(model_zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)

        # Create marker only after successful extraction.
        with open(done_marker, "w", encoding="utf-8") as f:
            f.write("ok\n")
    finally:
        try:
            os.close(fd)
        finally:
            # Best-effort cleanup. If deletion fails, marker still prevents work.
            try:
                os.remove(lock_path)
            except OSError:
                pass


def _read_literal(path: str):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    if not content:
        raise ValueError(f"File is empty: {path}")
    return ast.literal_eval(content)


# If using default relative output/, extract model next to project root; otherwise next to output path.
if MODEL_EXTRACT_USE_PROJECT_ROOT:
    out_model_dir = ROOT_DIR
    print(out_model_dir)
else:
    out_model_dir = output_folder.rstrip("/\\")
    print(out_model_dir)

model_dir_name = os.path.join(ROOT_DIR, "nnet_model", model_stub, model_version)

model_path = os.path.join(model_dir_name, "saved_model.zip")

exported_model = []

if USE_NNET_MODEL:
    if not os.path.exists(model_path):
        print(
            "USE_NNET_MODEL is enabled but no model zip at ",
            model_path,
            " — neural net matching disabled.",
        )
    else:
        try:
            import torch
        except ImportError:
            print(
                "USE_NNET_MODEL is enabled but torch is not installed — "
                "neural net matching disabled. Install the 'nnet' extra or torch."
            )
        else:
            from .pytorch_models import (
                LSTMTextClassifier,
                TextClassifier,
                TransformerClassifier,
            )

            os.environ["CUDA_VISIBLE_DEVICES"] = MATCHER_CUDA_VISIBLE_DEVICES
            device = "cpu"

            _ensure_model_assets_extracted(model_path, out_model_dir)

            if "pytorch" in model_stub:

                labels_list = [
                    "SaoText",  # 1
                    "SaoStartNumber",  # 2
                    "SaoStartSuffix",  # 3
                    "SaoEndNumber",  # 4
                    "SaoEndSuffix",  # 5
                    "PaoText",  # 6
                    "PaoStartNumber",  # 7
                    "PaoStartSuffix",  # 8
                    "PaoEndNumber",  # 9
                    "PaoEndSuffix",  # 10
                    "Street",  # 11
                    "PostTown",  # 12
                    "AdministrativeArea",  # 13
                    "Postcode",  # 14
                    "IGNORE",
                ]

                if (
                    (model_type == "transformer")
                    | (model_type == "gru")
                    | (model_type == "lstm")
                ):
                    # Load vocab and word_to_index
                    vocab = _read_literal(os.path.join(out_model_dir, "vocab.txt"))
                    word_to_index = _read_literal(
                        os.path.join(out_model_dir, "word_to_index.txt")
                    )
                    cat_to_idx = _read_literal(
                        os.path.join(out_model_dir, "cat_to_idx.txt")
                    )

                    VOCAB_SIZE = len(word_to_index)
                    OUTPUT_DIM = len(cat_to_idx) + 1  # Number of classes/categories
                    EMBEDDING_DIM = 48
                    DROPOUT = 0.1
                    PAD_TOKEN = 0

                    if model_type == "transformer":
                        NHEAD = 4
                        NUM_ENCODER_LAYERS = 1

                        exported_model = TransformerClassifier(
                            VOCAB_SIZE,
                            EMBEDDING_DIM,
                            NHEAD,
                            NUM_ENCODER_LAYERS,
                            OUTPUT_DIM,
                            DROPOUT,
                            PAD_TOKEN,
                        )

                    elif model_type == "gru":
                        N_LAYERS = 3
                        HIDDEN_DIM = 128
                        exported_model = TextClassifier(
                            VOCAB_SIZE,
                            EMBEDDING_DIM,
                            HIDDEN_DIM,
                            OUTPUT_DIM,
                            N_LAYERS,
                            DROPOUT,
                            PAD_TOKEN,
                        )

                    elif model_type == "lstm":
                        N_LAYERS = 3
                        HIDDEN_DIM = 128

                        exported_model = LSTMTextClassifier(
                            VOCAB_SIZE,
                            EMBEDDING_DIM,
                            HIDDEN_DIM,
                            OUTPUT_DIM,
                            N_LAYERS,
                            DROPOUT,
                            PAD_TOKEN,
                        )

                    out_model_file_name = (
                        "output_model_"
                        + str(data_sample_size)
                        + "_"
                        + str(N_EPOCHS)
                        + "_"
                        + model_type
                        + ".pth"
                    )

                    out_model_path = os.path.join(out_model_dir, out_model_file_name)
                    print("Model location: ", out_model_path)
                    exported_model.load_state_dict(
                        torch.load(
                            out_model_path,
                            map_location=torch.device("cpu"),
                            weights_only=False,
                        )
                    )
                    exported_model.eval()

                    device = "cpu"
                    exported_model.to(device)

            else:
                exported_model = []

run_nnet_match = bool(exported_model)

### ADDRESS MATCHING FUNCTIONS
# batch_size, ref_batch_size, fuzzy_method, score_cut_off, ref_address_cols, weights, etc. are set in fuzzy_address_matcher/config.py (env-driven).

# Creating Pydantic basemodel class


class MatcherClass(BaseModel):
    # Fuzzy/general attributes
    fuzzy_scorer_used: str
    fuzzy_match_limit: int
    fuzzy_search_addr_limit: int
    filter_to_lambeth_pcodes: bool
    standardise: bool
    suffix_used: str

    # Neural net attributes
    matching_variables: List[str]
    model_dir_name: str
    file_step_suffix: str
    exported_model: List

    fuzzy_method: str
    score_cut_off: float
    text_columns: List[str]
    weights: dict
    model_type: str
    labels_list: List[str]

    # These are variables that are added on later
    # Pytorch optional variables
    word_to_index: dict
    cat_to_idx: dict
    device: str
    vocab: List[str]

    # Join data
    file_name: str
    ref_name: str
    search_df: pd.DataFrame
    excluded_df: pd.DataFrame
    pre_filter_search_df: pd.DataFrame
    search_address_cols: List[str]
    search_postcode_col: List[str]
    search_df_key_field: str
    ref_df: pd.DataFrame
    ref_pre_filter: pd.DataFrame
    ref_address_cols: List[str]
    new_join_col: List[str]
    # in_joincol_list: List[str]
    existing_match_cols: List[str]
    standard_llpg_format: List[str]

    # Results attributes
    match_results_output: pd.DataFrame
    predict_df_nnet: pd.DataFrame

    # Other attributes generated during training
    compare_all_candidates: List[str]
    diag_shortlist: List[str]
    diag_best_match: List[str]

    results_on_orig_df: pd.DataFrame

    summary: str
    output_summary: str
    match_outputs_name: str
    results_orig_df_name: str

    search_df_after_stand: pd.DataFrame
    ref_df_after_stand: pd.DataFrame
    search_df_after_full_stand: pd.DataFrame
    ref_df_after_full_stand: pd.DataFrame

    search_df_after_stand_series: pd.Series
    ref_df_after_stand_series: pd.Series
    search_df_after_stand_series_full_stand: pd.Series
    ref_df_after_stand_series_full_stand: pd.Series

    # Abort flag if the matcher couldn't even get the results of the first match
    abort_flag: bool

    # This is to allow for Pandas DataFrame types as an argument
    class Config:
        # Allow for custom types such as Pandas DataFrames in the class
        arbitrary_types_allowed = True
        extra = "allow"
        # Disable protected namespaces to avoid conflicts
        protected_namespaces = ()


# Creating an instance of MatcherClass
InitMatch = MatcherClass(
    # Fuzzy/general attributes
    fuzzy_scorer_used=fuzzy_scorer_used,
    fuzzy_match_limit=fuzzy_match_limit,
    fuzzy_search_addr_limit=fuzzy_search_addr_limit,
    filter_to_lambeth_pcodes=filter_to_lambeth_pcodes,
    standardise=standardise,
    suffix_used=suffix_used,
    # Neural net attributes
    matching_variables=matching_variables,
    model_dir_name=model_dir_name,
    file_step_suffix=file_step_suffix,
    exported_model=[exported_model],
    fuzzy_method=fuzzy_method,
    score_cut_off=score_cut_off,
    text_columns=text_columns,
    weights=weights,
    model_type=model_type,
    labels_list=labels_list,
    # These are variables that are added on later
    # Pytorch optional variables
    word_to_index=word_to_index,
    cat_to_idx=cat_to_idx,
    device=device,
    vocab=vocab,
    # Join data
    file_name="",
    ref_name="",
    df_name="",
    search_df=pd.DataFrame(),
    excluded_df=pd.DataFrame(),
    pre_filter_search_df=pd.DataFrame(),
    search_df_not_matched=pd.DataFrame(),
    search_df_cleaned=pd.DataFrame(),
    search_address_cols=[],
    search_postcode_col=[],
    search_df_key_field="index",
    ref_df=pd.DataFrame(),
    ref_df_cleaned=pd.DataFrame(),
    ref_pre_filter=pd.DataFrame(),
    ref_address_cols=[],
    new_join_col=[],
    # in_joincol_list = [],
    existing_match_cols=[],
    standard_llpg_format=[],
    # Results attributes
    match_results_output=pd.DataFrame(),
    predict_df_nnet=pd.DataFrame(),
    # Other attributes generated during training
    compare_all_candidates=[],
    diag_shortlist=[],
    diag_best_match=[],
    results_on_orig_df=pd.DataFrame(),
    summary="",
    output_summary="",
    match_outputs_name="",
    results_orig_df_name="",
    # Post dataset preparation variables
    search_df_after_stand=pd.DataFrame(),
    ref_df_after_stand=pd.DataFrame(),
    search_df_after_stand_series=pd.Series(),
    ref_df_after_stand_series=pd.Series(),
    search_df_after_full_stand=pd.DataFrame(),
    ref_df_after_full_stand=pd.DataFrame(),
    search_df_after_stand_series_full_stand=pd.Series(),
    ref_df_after_stand_series_full_stand=pd.Series(),
    # Abort flag if the matcher couldn't even get the results of the first match
    abort_flag=False,
)
