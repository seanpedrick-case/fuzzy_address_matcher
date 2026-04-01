import os
import pandas as pd
import torch
import zipfile
from typing import List, Type
from pydantic import BaseModel

from .pytorch_models import TransformerClassifier, TextClassifier, LSTMTextClassifier

PandasDataFrame = Type[pd.DataFrame]
PandasSeries = Type[pd.Series]


def get_or_create_env_var(var_name, default_value):
    # Get the environment variable if it exists
    value = os.environ.get(var_name)

    # If it doesn't exist, set it to the default value
    if value is None:
        os.environ[var_name] = default_value
        value = default_value

    return value


# Retrieving or setting output folder
env_var_name = "GRADIO_OUTPUT_FOLDER"
default_value = "output/"

output_folder = get_or_create_env_var(env_var_name, default_value)
# print(f"The value of {env_var_name} is {output_folder}")

# +
""" Fuzzywuzzy/Rapidfuzz scorer to use. Options are: ratio, partial_ratio, token_sort_ratio, partial_token_sort_ratio,
token_set_ratio, partial_token_set_ratio, QRatio, UQRatio, WRatio (default), UWRatio
details here: https://stackoverflow.com/questions/31806695/when-to-use-which-fuzz-function-to-compare-2-strings"""

fuzzy_scorer_used = "token_set_ratio"

fuzzy_match_limit = 85
fuzzy_search_addr_limit = 20
filter_to_lambeth_pcodes = True
standardise = False

if standardise:
    std = "_std"
if not standardise:
    std = "_not_std"

dataset_name = "data" + std

suffix_used = dataset_name + "_" + fuzzy_scorer_used

# https://stackoverflow.com/questions/59221557/tensorflow-v2-replacement-for-tf-contrib-predictor-from-saved-model


# Uncomment these lines for the tensorflow model
# model_type = "tf"
# model_stub = "addr_model_out_lon"
# model_version = "00000001"
# file_step_suffix = "550" # I add a suffix to output files to be able to separate comparisons of test data from the same model with different steps e.g. '350' indicates a model that has been through 350,000 steps of training

# Uncomment these lines for the pytorch model
model_type = "lstm"
model_stub = "pytorch/lstm"
model_version = ""
file_step_suffix = ""
data_sample_size = 476887
N_EPOCHS = 10
max_predict_len = 12000

word_to_index = {}
cat_to_idx = {}
vocab = []
device = "cpu"

global labels_list
labels_list = []

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))

# If in a non-standard location (e.g. on AWS Lambda Function URL, then save model to tmp drive)
if output_folder == "output/":
    out_model_dir = ROOT_DIR
    print(out_model_dir)
else:
    out_model_dir = output_folder[:-1]
    print(out_model_dir)

model_dir_name = os.path.join(ROOT_DIR, "nnet_model", model_stub, model_version)

model_path = os.path.join(model_dir_name, "saved_model.zip")
print("Model zip path: ", model_path)

if os.path.exists(model_path):

    os.environ["CUDA_VISIBLE_DEVICES"] = (
        "-1"  # Better to go without GPU to avoid 'out of memory' issues
    )
    device = "cpu"

    ## The labels_list object defines the structure of the prediction outputs. It must be the same as what the model was originally trained on

    """ Load pre-trained model """

    with zipfile.ZipFile(model_path, "r") as zip_ref:
        zip_ref.extractall(out_model_dir)

    # if model_stub == "addr_model_out_lon":

    # import tensorflow as tf

    # tf.config.list_physical_devices('GPU')

    #     # Number of labels in total (+1 for the blank category)
    #     n_labels = len(labels_list) + 1

    #     # Allowable characters for the encoded representation
    #     vocab = list(string.digits + string.ascii_lowercase + string.punctuation + string.whitespace)

    #     #print("Loading TF model")

    #     exported_model = tf.saved_model.load(model_dir_name)

    #     labels_list = [
    #     'SaoText',  # 1
    #     'SaoStartNumber',  # 2
    #     'SaoStartSuffix',  # 3
    #     'SaoEndNumber',  # 4
    #     'SaoEndSuffix',  # 5
    #     'PaoText',  # 6
    #     'PaoStartNumber',  # 7
    #     'PaoStartSuffix',  # 8
    #     'PaoEndNumber',  # 9
    #     'PaoEndSuffix',  # 10
    #     'Street',  # 11
    #     'PostTown',  # 12
    #     'AdministrativeArea', #13
    #     'Postcode'  # 14
    #     ]

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
            with open(out_model_dir + "/vocab.txt", "r") as f:
                vocab = eval(f.read())
            with open(out_model_dir + "/word_to_index.txt", "r") as f:
                word_to_index = eval(f.read())
            with open(out_model_dir + "/cat_to_idx.txt", "r") as f:
                cat_to_idx = eval(f.read())

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
                    out_model_path, map_location=torch.device("cpu"), weights_only=False
                )
            )
            exported_model.eval()

            device = "cpu"
            # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            exported_model.to(device)

    else:
        exported_model = []  # tf.keras.models.load_model(model_dir_name, compile=False)
        # Compile the model with a loss function and an optimizer
        # exported_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['categorical_crossentropy'])

else:
    exported_model = []

### ADDRESS MATCHING FUNCTIONS
# Address matcher will try to match <batch_size> records in one go to avoid exceeding memory limits.
batch_size = 10000
ref_batch_size = 20000

### Fuzzy match method

""" https://recordlinkage.readthedocs.io/en/latest/ref_df-compare.html#recordlinkage.compare.String
 The Python Record Linkage Toolkit uses the jellyfish package for the Jaro, Jaro-Winkler, Levenshtein and Damerau- Levenshtein algorithms.
 Options are [‘jaro’, ‘jarowinkler’, ‘levenshtein’, ‘damerau_levenshtein’, ‘qgram’, ‘cosine’, ‘smith_waterman’, ‘lcs’]

 Comparison of some of the Jellyfish string comparison methods: https://manpages.debian.org/testing/python-jellyfish-doc/jellyfish.3.en.html """

fuzzy_method = "jarowinkler"

# Required overall match score for all columns to count as a match
score_cut_off = 98.7  # 97.5
# I set a higher score cut off for nnet street blocking based on empirical data. Under this match value I was seeing errors. This value was (.99238), but set here to .995 to be maximally stringent. It is set in 'recordlinkage_funcs.py', score_based_match function
score_cut_off_nnet_street = 99.5  # 99.238
# If there are no numbers in the address, then the matcher needs to get a perfect score (otherwise too many issues).
no_number_fuzzy_match_limit = 100

# Reference data 'official' column names
ref_address_cols = [
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

# Create a list of matching variables. Text columns will be fuzzy matched.
matching_variables = ref_address_cols
text_columns = ["Organisation", "PaoText", "Street", "PostTown", "Postcode"]

# Modify relative importance of columns (weights) for the recordlinkage part of the match. Modify weighting for scores - Town and AdministrativeArea are not very important as we have postcode. Street number and name are important
Organisation_weight = 0.1  # Organisation weight is very low just to resolve tie breakers for very similar addresses
PaoStartNumber_weight = 2
SaoStartNumber_weight = 2
Street_weight = 2
PostTown_weight = 0
Postcode_weight = 0.5
AdministrativeArea_weight = 0
# -

weight_vals = [1] * len(ref_address_cols)
weight_keys = ref_address_cols
weights = {weight_keys[i]: weight_vals[i] for i in range(len(weight_keys))}

# +
# Modify weighting for scores - Town and AdministrativeArea are not very important as we have postcode. Street number and name are important

weights["Organisation"] = Organisation_weight
weights["SaoStartNumber"] = SaoStartNumber_weight
weights["PaoStartNumber"] = PaoStartNumber_weight
weights["Street"] = Street_weight
weights["PostTown"] = PostTown_weight
weights["Postcode"] = Postcode_weight

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
