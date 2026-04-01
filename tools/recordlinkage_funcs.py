import pandas as pd
from typing import Type, Dict, List, Tuple
import recordlinkage
from datetime import datetime

from tools.constants import score_cut_off_nnet_street

PandasDataFrame = Type[pd.DataFrame]
PandasSeries = Type[pd.Series]
MatchedResults = Dict[str, Tuple[str, int]]
array = List[str]

today = datetime.now().strftime("%d%m%Y")
today_rev = datetime.now().strftime("%Y%m%d")


# ## Recordlinkage matching functions
def compute_match(
    predict_df_search,
    ref_search,
    orig_search_df,
    matching_variables,
    text_columns,
    blocker_column,
    weights,
    fuzzy_method,
):
    # Use the merge command to match group1 and group2
    predict_df_search[matching_variables] = predict_df_search[
        matching_variables
    ].astype(str)
    ref_search[matching_variables] = (
        ref_search[matching_variables].astype(str).replace("-999", "")
    )

    # SaoText needs to be exactly the same to get a 'full' match. So I moved that to the exact match group
    exact_columns = list(set(matching_variables) - set(text_columns))

    # Replace all blanks with a space, so they can be included in the fuzzy match searches
    for column in text_columns:
        predict_df_search.loc[predict_df_search[column] == "", column] = " "
        ref_search.loc[ref_search[column] == "", column] = " "

    # Score based match functions

    # Create an index of all pairs
    indexer = recordlinkage.Index()

    # Block on selected blocker column

    ## Remove all NAs from predict_df blocker column
    if blocker_column[0] == "PaoStartNumber":
        predict_df_search = predict_df_search[
            ~(predict_df_search[blocker_column[0]].isna())
            & ~(predict_df_search[blocker_column[0]] == "")
            & ~(predict_df_search[blocker_column[0]].str.contains(r"^\s*$", na=False))
        ]

    indexer.block(blocker_column)  # matchkey.block(["Postcode", "PaoStartNumber"])

    # Generate candidate pairs

    pairsSBM = indexer.index(predict_df_search, ref_search)

    print(
        "Running with " + blocker_column[0] + " as blocker has created",
        len(pairsSBM),
        "pairs.",
    )

    # If no pairs are found, break
    if len(pairsSBM) == 0:
        return pd.DataFrame()

    # Call the compare class from the toolkit
    compareSBM = recordlinkage.Compare()

    # Assign variables to matching technique - exact
    for columns in exact_columns:
        compareSBM.exact(columns, columns, label=columns, missing_value=0)

    # Assign variables to matching technique - fuzzy
    for columns in text_columns:
        if columns == "Postcode":
            compareSBM.string(
                columns, columns, label=columns, missing_value=0, method="levenshtein"
            )
        else:
            compareSBM.string(
                columns, columns, label=columns, missing_value=0, method=fuzzy_method
            )

    ## Run the match - compare each column within the blocks according to exact or fuzzy matching (defined in cells above)

    scoresSBM = compareSBM.compute(
        pairs=pairsSBM, x=predict_df_search, x_link=ref_search
    )

    return scoresSBM


def calc_final_nnet_scores(scoresSBM, weights, matching_variables):
    # Modify the output scores by the weights set at the start of the code
    scoresSBM_w = scoresSBM * weights

    ### Determine matched roles that score above a threshold

    # Sum all columns
    scoresSBM_r = scoresSBM_w

    scoresSBM_r["score"] = scoresSBM_r[matching_variables].sum(axis=1)
    scoresSBM_r["score_max"] = sum(
        weights.values()
    )  # + 2 for the additional scoring from the weighted variables a couple of cells above
    scoresSBM_r["score_perc"] = (scoresSBM_r["score"] / scoresSBM_r["score_max"]) * 100

    scoresSBM_r = scoresSBM_r.reset_index()

    # Rename the index if misnamed
    scoresSBM_r = scoresSBM_r.rename(columns={"index": "level_1"}, errors="ignore")

    # Sort all comparisons by score in descending order
    scoresSBM_r = scoresSBM_r.sort_values(by=["level_0", "score_perc"], ascending=False)

    # Within each search address, remove anything below the max
    scoresSBM_g = scoresSBM_r.reset_index()

    # Get maximum score to join on
    scoresSBM_g = (
        scoresSBM_g.groupby("level_0")
        .max("score_perc")
        .reset_index()[["level_0", "score_perc"]]
    )
    scoresSBM_g = scoresSBM_g.rename(columns={"score_perc": "score_perc_max"})
    scoresSBM_search = scoresSBM_r.merge(scoresSBM_g, on="level_0", how="left")

    scoresSBM_search["score_perc"] = round(scoresSBM_search["score_perc"], 1).astype(
        float
    )
    scoresSBM_search["score_perc_max"] = round(
        scoresSBM_search["score_perc_max"], 1
    ).astype(float)

    return scoresSBM_search


def join_on_pred_ref_details(scoresSBM_search_m, ref_search, predict_df_search):
    ## Join back search and ref_df address details onto matching df
    scoresSBM_search_m_j = scoresSBM_search_m.merge(
        ref_search,
        left_on="level_1",
        right_index=True,
        how="left",
        suffixes=("", "_ref"),
    )

    scoresSBM_search_m_j = scoresSBM_search_m_j.merge(
        predict_df_search,
        left_on="level_0",
        right_index=True,
        how="left",
        suffixes=("", "_pred"),
    )

    scoresSBM_search_m_j = scoresSBM_search_m_j.reindex(
        sorted(scoresSBM_search_m_j.columns), axis=1
    )

    return scoresSBM_search_m_j


def rearrange_columns(
    scoresSBM_search_m_j, new_join_col, search_df_key_field, blocker_column, standardise
):

    start_columns = new_join_col.copy()

    start_columns.extend(
        [
            "address",
            "fulladdress",
            "level_0",
            "level_1",
            "score",
            "score_max",
            "score_perc",
            "score_perc_max",
        ]
    )

    other_columns = list(set(scoresSBM_search_m_j.columns) - set(start_columns))

    all_columns_order = start_columns.copy()
    all_columns_order.extend(sorted(other_columns))

    # Place important columns at start

    scoresSBM_search_m_j = scoresSBM_search_m_j.reindex(all_columns_order, axis=1)

    scoresSBM_search_m_j = scoresSBM_search_m_j.rename(
        columns={
            "address": "address_pred",
            "fulladdress": "address_ref",
            "level_0": "index_pred",
            "level_1": "index_ref",
            "score": "match_score",
            "score_max": "max_possible_score",
            "score_perc": "perc_weighted_columns_matched",
            "score_perc_max": "perc_weighted_columns_matched_max_for_pred_address",
        }
    )

    scoresSBM_search_m_j = scoresSBM_search_m_j.sort_values(
        "index_pred", ascending=True
    )

    # ref_index is just a duplicate of index_ref, needed for outputs
    scoresSBM_search_m_j["ref_index"] = scoresSBM_search_m_j["index_ref"]

    # search_df_j = orig_search_df[["full_address_search", search_df_key_field]]

    # scoresSBM_out = scoresSBM_search_m_j.merge(search_df_j, left_on = "address_pred", right_on = "full_address_search", how = "left")

    final_cols = new_join_col.copy()
    final_cols.extend(
        [
            search_df_key_field,
            "full_match_score_based",
            "address_pred",
            "address_ref",
            "match_score",
            "max_possible_score",
            "perc_weighted_columns_matched",
            "perc_weighted_columns_matched_max_for_pred_address",
            "Organisation",
            "Organisation_ref",
            "Organisation_pred",
            "SaoText",
            "SaoText_ref",
            "SaoText_pred",
            "SaoStartNumber",
            "SaoStartNumber_ref",
            "SaoStartNumber_pred",
            "SaoStartSuffix",
            "SaoStartSuffix_ref",
            "SaoStartSuffix_pred",
            "SaoEndNumber",
            "SaoEndNumber_ref",
            "SaoEndNumber_pred",
            "SaoEndSuffix",
            "SaoEndSuffix_ref",
            "SaoEndSuffix_pred",
            "PaoStartNumber",
            "PaoStartNumber_ref",
            "PaoStartNumber_pred",
            "PaoStartSuffix",
            "PaoStartSuffix_ref",
            "PaoStartSuffix_pred",
            "PaoEndNumber",
            "PaoEndNumber_ref",
            "PaoEndNumber_pred",
            "PaoEndSuffix",
            "PaoEndSuffix_ref",
            "PaoEndSuffix_pred",
            "PaoText",
            "PaoText_ref",
            "PaoText_pred",
            "Street",
            "Street_ref",
            "Street_pred",
            "PostTown",
            "PostTown_ref",
            "PostTown_pred",
            "Postcode",
            "Postcode_ref",
            "Postcode_pred",
            "Postcode_predict",
            "index_pred",
            "index_ref",
            "Reference file",
        ]
    )

    scoresSBM_out = scoresSBM_search_m_j[final_cols]

    return scoresSBM_out, start_columns


def create_matched_results_nnet(
    scoresSBM_best,
    search_df_key_field,
    orig_search_df,
    new_join_col,
    standardise,
    ref_search,
    blocker_column,
    score_cut_off,
):

    ### Make the final 'matched output' file
    scoresSBM_best_pred_cols = scoresSBM_best.filter(regex="_pred$").iloc[:, 1:-1]
    scoresSBM_best["search_orig_address"] = (
        (scoresSBM_best_pred_cols.agg(" ".join, axis=1))
        .str.strip()
        .str.replace(r"\s{2,}", " ", regex=True)
    )

    scoresSBM_best_ref_cols = scoresSBM_best.filter(regex="_ref$").iloc[:, 1:-1]
    scoresSBM_best["reference_mod_address"] = (
        (scoresSBM_best_ref_cols.agg(" ".join, axis=1))
        .str.strip()
        .str.replace(r"\s{2,}", " ", regex=True)
    )

    ## Create matched output df
    matched_output_SBM = (
        orig_search_df[
            [
                search_df_key_field,
                "full_address",
                "postcode",
                "property_number",
                "prop_number",
                "flat_number",
                "apart_number",
                "block_number",
                "unit_number",
                "room_number",
                "house_court_name",
            ]
        ]
        .replace(r"\bnan\b", "", regex=True)
        .infer_objects(copy=False)
    )
    matched_output_SBM[search_df_key_field] = matched_output_SBM[
        search_df_key_field
    ].astype(str)

    ###
    matched_output_SBM = matched_output_SBM.merge(
        scoresSBM_best[
            [
                search_df_key_field,
                "index_ref",
                "address_ref",
                "full_match_score_based",
                "Reference file",
            ]
        ],
        on=search_df_key_field,
        how="left",
    ).rename(columns={"full_address": "search_orig_address"})

    if "index" not in ref_search.columns:
        ref_search["ref_index"] = ref_search.index

    matched_output_SBM = matched_output_SBM.merge(
        ref_search.drop_duplicates("fulladdress")[
            [
                "ref_index",
                "fulladdress",
                "Postcode",
                "property_number",
                "prop_number",
                "flat_number",
                "apart_number",
                "block_number",
                "unit_number",
                "room_number",
                "house_court_name",
                "ref_address_stand",
            ]
        ],
        left_on="address_ref",
        right_on="fulladdress",
        how="left",
        suffixes=("_search", "_reference"),
    ).rename(
        columns={
            "fulladdress": "reference_orig_address",
            "ref_address_stand": "reference_list_address",
        }
    )

    # To replace with number check

    matched_output_SBM = matched_output_SBM.rename(
        columns={"full_match_score_based": "full_match"}
    )

    matched_output_SBM["property_number_match"] = matched_output_SBM["full_match"]

    scores_SBM_best_cols = [
        search_df_key_field,
        "full_match_score_based",
        "perc_weighted_columns_matched",
        "address_pred",
    ]  # , "reference_mod_address"]
    scores_SBM_best_cols.extend(new_join_col)

    matched_output_SBM_b = scoresSBM_best[scores_SBM_best_cols]

    matched_output_SBM = matched_output_SBM.merge(
        matched_output_SBM_b.drop_duplicates(search_df_key_field),
        on=search_df_key_field,
        how="left",
    )

    from tools.fuzzy_match import create_diag_shortlist

    matched_output_SBM = create_diag_shortlist(
        matched_output_SBM,
        "search_orig_address",
        score_cut_off,
        blocker_column,
        fuzzy_col="perc_weighted_columns_matched",
        search_mod_address="address_pred",
        resolve_tie_breaks=False,
    )

    matched_output_SBM["standardised_address"] = standardise

    matched_output_SBM = matched_output_SBM.rename(
        columns={
            "address_pred": "search_mod_address",
            "perc_weighted_columns_matched": "fuzzy_score",
        }
    )

    matched_output_SBM_cols = [
        search_df_key_field,
        "search_orig_address",
        "reference_orig_address",
        "full_match",
        "full_number_match",
        "flat_number_match",
        "room_number_match",
        "block_number_match",
        "property_number_match",
        "close_postcode_match",
        "house_court_name_match",
        "fuzzy_score_match",
        "fuzzy_score",
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
        "Postcode",
        "postcode",
        "ref_index",
        "Reference file",
    ]

    matched_output_SBM_cols.extend(new_join_col)
    matched_output_SBM_cols.extend(["standardised_address"])
    matched_output_SBM = matched_output_SBM[matched_output_SBM_cols]

    matched_output_SBM = matched_output_SBM.sort_values(
        search_df_key_field, ascending=True
    )

    return matched_output_SBM


def score_based_match(
    predict_df_search,
    ref_search,
    orig_search_df,
    matching_variables,
    text_columns,
    blocker_column,
    weights,
    fuzzy_method,
    score_cut_off,
    search_df_key_field,
    standardise,
    new_join_col,
    score_cut_off_nnet_street=score_cut_off_nnet_street,
):

    scoresSBM = compute_match(
        predict_df_search,
        ref_search,
        orig_search_df,
        matching_variables,
        text_columns,
        blocker_column,
        weights,
        fuzzy_method,
    )

    if scoresSBM.empty:
        # If no pairs are found, break
        return pd.DataFrame(), pd.DataFrame()

    scoresSBM_search = calc_final_nnet_scores(scoresSBM, weights, matching_variables)

    # Filter potential matched address scores to those with highest scores only
    scoresSBM_search_m = scoresSBM_search[
        scoresSBM_search["score_perc"] == scoresSBM_search["score_perc_max"]
    ]

    scoresSBM_search_m_j = join_on_pred_ref_details(
        scoresSBM_search_m, ref_search, predict_df_search
    )

    # When blocking by street, may to have an increased threshold as this is more prone to making mistakes
    if blocker_column[0] == "Street":
        scoresSBM_search_m_j["full_match_score_based"] = (
            scoresSBM_search_m_j["score_perc"] >= score_cut_off_nnet_street
        )

    else:
        scoresSBM_search_m_j["full_match_score_based"] = (
            scoresSBM_search_m_j["score_perc"] >= score_cut_off
        )

    ### Reorder some columns
    scoresSBM_out, start_columns = rearrange_columns(
        scoresSBM_search_m_j,
        new_join_col,
        search_df_key_field,
        blocker_column,
        standardise,
    )

    matched_output_SBM = create_matched_results_nnet(
        scoresSBM_out,
        search_df_key_field,
        orig_search_df,
        new_join_col,
        standardise,
        ref_search,
        blocker_column,
        score_cut_off,
    )

    matched_output_SBM_best = matched_output_SBM.sort_values(
        [search_df_key_field, "full_match"], ascending=[True, False]
    ).drop_duplicates(search_df_key_field)

    scoresSBM_best = scoresSBM_out[
        scoresSBM_out[search_df_key_field].isin(
            matched_output_SBM_best[search_df_key_field]
        )
    ]

    return scoresSBM_best, matched_output_SBM_best


def check_matches_against_fuzzy(match_results, scoresSBM, search_df_key_field):

    if not match_results.empty:

        if "fuzz_full_match" not in match_results.columns:
            match_results["fuzz_full_match"] = False

        match_results = match_results.add_prefix("fuzz_").rename(
            columns={"fuzz_" + search_df_key_field: search_df_key_field}
        )

        # Merge fuzzy match full matches onto model data

        scoresSBM_m = scoresSBM.merge(
            match_results.drop_duplicates(search_df_key_field),
            on=search_df_key_field,
            how="left",
        )

    else:
        scoresSBM_m = scoresSBM
        scoresSBM_m["fuzz_full_match"] = False
        scoresSBM_m["fuzz_fuzzy_score_match"] = False
        scoresSBM_m["fuzz_property_number_match"] = False
        scoresSBM_m["fuzz_fuzzy_score"] = 0
        scoresSBM_m["fuzz_reference_orig_address"] = ""

    scoresSBM_t = scoresSBM[scoresSBM["full_match_score_based"]]

    ### Create a df of matches the model finds that the fuzzy matching work did not

    scoresSBM_m_model_add_matches = scoresSBM_m[
        (scoresSBM_m["full_match_score_based"]) & (not scoresSBM_m["fuzz_full_match"])
    ]

    # Drop some irrelevant columns

    first_cols = [
        "UPRN",
        search_df_key_field,
        "full_match_score_based",
        "fuzz_full_match",
        "fuzz_fuzzy_score_match",
        "fuzz_property_number_match",
        "fuzz_fuzzy_score",
        "match_score",
        "max_possible_score",
        "perc_weighted_columns_matched",
        "perc_weighted_columns_matched_max_for_pred_address",
        "address_pred",
        "address_ref",
        "fuzz_reference_orig_address",
    ]

    last_cols = [
        col for col in scoresSBM_m_model_add_matches.columns if col not in first_cols
    ]

    scoresSBM_m_model_add_matches = scoresSBM_m_model_add_matches[
        first_cols + last_cols
    ].drop(
        [
            "fuzz_search_mod_address",
            "fuzz_reference_mod_address",
            "fuzz_fulladdress",
            "fuzz_UPRN",
        ],
        axis=1,
        errors="ignore",
    )

    ### Create a df for matches the fuzzy matching found that the neural net model does not

    if not match_results.empty:
        scoresSBM_t_model_failed = match_results[
            (~match_results[search_df_key_field].isin(scoresSBM_t[search_df_key_field]))
            & (match_results["fuzz_full_match"])
        ]

        scoresSBM_t_model_failed = scoresSBM_t_model_failed.merge(
            scoresSBM.drop_duplicates(search_df_key_field),
            on=search_df_key_field,
            how="left",
        )

        scoresSBM_t_model_failed = scoresSBM_t_model_failed[
            first_cols + last_cols
        ].drop(
            [
                "fuzz_search_mod_address",
                "fuzz_reference_mod_address",
                "fuzz_fulladdress",
                "fuzz_UPRN",
            ],
            axis=1,
            errors="ignore",
        )
    else:
        scoresSBM_t_model_failed = pd.DataFrame()

    ## Join back onto original results file and export

    scoresSBM_new_matches_from_model = scoresSBM_m_model_add_matches.drop_duplicates(
        search_df_key_field
    )

    if not match_results.empty:
        match_results_out = match_results.merge(
            scoresSBM_new_matches_from_model[
                [
                    search_df_key_field,
                    "full_match_score_based",
                    "address_pred",
                    "address_ref",
                ]
            ],
            on=search_df_key_field,
            how="left",
        )

        match_results_out.loc[
            match_results_out["full_match_score_based"].isna(), "full_match_score_based"
        ] = False

        # match_results_out['full_match_score_based'].value_counts()

        match_results_out["full_match_fuzzy_or_score_based"] = (
            match_results_out["fuzz_full_match"]
        ) | (match_results_out["full_match_score_based"])
    else:
        match_results_out = match_results

    return scoresSBM_m_model_add_matches, scoresSBM_t_model_failed, match_results_out
