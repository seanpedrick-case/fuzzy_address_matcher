import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Type
from datetime import datetime
from rapidfuzz import fuzz, process
import gradio as gr
from tqdm import tqdm

PandasDataFrame = Type[pd.DataFrame]
PandasSeries = Type[pd.Series]
MatchedResults = Dict[str,Tuple[str,int]]
array = List[str]

today = datetime.now().strftime("%d%m%Y")
today_rev = datetime.now().strftime("%Y%m%d")

from tools.constants import no_number_fuzzy_match_limit, fuzzy_match_limit

def string_match_array(to_match:array, choices:array,
                      index_name:str, matched_name:str) -> PandasDataFrame:
    
    temp = {name: process.extractOne(name,choices) 
            for name in to_match}
    
    return _create_frame(matched_results=temp, index_name=index_name,
                        matched_name=matched_name)

# Fuzzy match algorithm
def create_fuzzy_matched_col(df:PandasDataFrame, orig_match_address_series:PandasSeries, pred_match_address_series:PandasSeries, fuzzy_method:"WRatio", match_score=95):

    results = []

    for orig_index, orig_string in df[orig_match_address_series].items():
        
        predict_string = df[pred_match_address_series][orig_index] 
        
        if (orig_string == '') and (predict_string == ''):
            results.append(np.nan)
            
        else:
            fuzz_score = process.extract(orig_string, [predict_string], scorer= getattr(fuzz, fuzzy_method))
            results.append(fuzz_score[0][1])

    new_result_col_score = (orig_match_address_series + "_fuzz_score")
    new_result_col_match = (orig_match_address_series + "_fuzz_match") 

    df[new_result_col_score] = results
    df[new_result_col_match] = df[new_result_col_score] >= match_score
    #df[new_result_col_match][df[new_result_col_score].isna()] = np.nan
    df.loc[df[new_result_col_score].isna(), new_result_col_match] = np.nan
    
    return df

def string_match_by_post_code_multiple(match_address_series:PandasSeries, reference_address_series:PandasSeries,
                              search_limit=100, scorer_name="token_set_ratio", progress=gr.Progress(track_tqdm=True))-> MatchedResults:
    '''
    Matches by Series values; for example idx is post code and 
    values address. Search field is reduced by comparing same post codes address reference_address_series.
    
    Default scorer is fuzz.Wratio. This tries to weight the different algorithms
    to give the best score.
    Choice of ratio type seems to make a big difference. Looking at this link:
    https://chairnerd.seatgeek.com/fuzzywuzzy-fuzzy-string-matching-in-python/
    and this one: 
    https://stackoverflow.com/questions/31806695/when-to-use-which-fuzz-function-to-compare-2-strings    

    '''

    def do_one_match(reference_addresses: pd.Series, scorer: callable, search_limit: int, postcode_match: str, search_addresses: pd.Series) -> MatchedResults:

        def _prepare_results(search_addresses, reference_addresses, matched, postcode_match):

            # Create a list to store the results
            results = []

            # Iterate through the matched dataframe and store results in the list
            for i, search_address in enumerate(search_addresses):
                for j, reference_address in enumerate(reference_addresses):
                    score = matched[i][j]
                    results.append((postcode_match, search_address, reference_address, score))

            # Create a dataframe from the results list
            matched_out = pd.DataFrame(results, columns=['postcode_search', 'fuzzy_match_search_address', 'fuzzy_match_reference_address', 'fuzzy_score'])

            return matched_out

        try:
            if isinstance(reference_addresses, str):  # reference_addresses can be a str-> 1 address per postcode
                matched = process.cdist(search_addresses.values, [reference_addresses], scorer=scorer, score_cutoff=fuzzy_match_limit, workers=-1)

                # Transform results into a dataframe
                matched_out = _prepare_results(search_addresses, reference_addresses, matched, postcode_match)

            else:  # 1+ addresses
                matched = process.cdist(search_addresses.values, reference_addresses.values, scorer=scorer, score_cutoff=fuzzy_match_limit, workers=-1) 

                # Transform results into a dataframe
                matched_out = _prepare_results(search_addresses, reference_addresses, matched, postcode_match)

            # Sort the matched results by score in descending order
            matched_out = matched_out.sort_values(by='fuzzy_score', ascending=False)

            # Keep only the top search_limit number of results - doesn't work anymore when working with multiple results
            #matched_out = matched_out.head(search_limit)

        except KeyError:
            matched_out = pd.DataFrame()

        return matched_out
 
    def apply_fuzzy_matching(postcode_match:str, search_addresses:PandasSeries, reference_addresses:PandasSeries, scorer:callable, search_limit:int)-> tuple:
        
        try:
            matched = do_one_match(reference_addresses, scorer, search_limit, postcode_match, search_addresses)
            return matched
        except KeyError:
            matched = pd.DataFrame() #[("NA", 0)] # for _ in range(1, search_limit + 1)]
            return matched

    print("Fuzzy match column length: ", len(match_address_series))
    print("Fuzzy Reference column length: ", len(reference_address_series))

    match_address_series = match_address_series.rename_axis('postcode_search')
    match_address_df = pd.DataFrame(match_address_series.reset_index())
    match_address_df['index'] = list(range(0,len(match_address_df)))

    reference_address_series = reference_address_series.rename_axis('postcode_search')
    reference_address_df = pd.DataFrame(reference_address_series.reset_index())
    reference_address_df['index'] = list(range(0,len(reference_address_df)))

    
    # Apply the match functions to each address
    scorer = getattr(fuzz, scorer_name)                  
    results = {}
    #counter = 0

    index_list = []
    match_list = []
    search_addresses_list = []
    reference_addresses_list = []

    unique_postcodes = pd.unique(match_address_df['postcode_search'])

    for postcode_match in tqdm(unique_postcodes, desc="Fuzzy matching", unit="fuzzy matched postcodes"):

        postcode_match_list = [postcode_match]
        search_indexes = pd.Series()
        search_addresses = pd.Series()
        reference_addresses = pd.Series()

        try:
            search_indexes = match_address_df.loc[match_address_df["postcode_search"].isin(postcode_match_list), "index"]
            search_addresses = match_address_df.loc[match_address_df["postcode_search"].isin(postcode_match_list), "search_address_stand"]
            reference_addresses = reference_address_df.loc[reference_address_df["postcode_search"].isin(postcode_match_list), "ref_address_stand"]

            if isinstance(reference_addresses, str):  # reference_addresses can be a str-> 1 address per postcode
                reference_addresses = pd.Series(reference_addresses)
        except KeyError:
            reference_addresses = pd.Series("NA")

        matched = apply_fuzzy_matching(postcode_match, search_addresses, reference_addresses, scorer, search_limit)

        # Write to output lists
        match_list.extend([matched])
        index_list.extend(search_indexes.tolist())
        search_addresses_list.extend(search_addresses.tolist())
        reference_addresses_list.extend(reference_addresses.tolist())

    out_frame = pd.concat(match_list)

    return out_frame

def _create_fuzzy_match_results_output(results:PandasDataFrame, search_df_after_stand:PandasDataFrame, ref_df_cleaned:PandasDataFrame, ref_df_after_stand:PandasDataFrame, fuzzy_match_limit:int, search_df_cleaned:PandasDataFrame, search_df_key_field:str, new_join_col:str, standardise:bool, blocker_col:str):

        '''
        Take fuzzy match outputs, create shortlist dataframes, rearrange, return diagnostics and shortlist dataframes for export
        '''

        ## Diagnostics

        diag_shortlist, diag_best_match = create_diagnostic_results(results_df=results,\
                                      matched_df = search_df_after_stand, ref_list_df = ref_df_after_stand,
                                      fuzzy_match_limit = fuzzy_match_limit, blocker_col=blocker_col)
        
        ## Fuzzy search results
        match_results_cols = ['search_orig_address','reference_orig_address', 'ref_index',
        'full_match',
        'full_number_match',
        'flat_number_match',
        'room_number_match',
        'block_number_match',
        'unit_number_match',
        'property_number_match',
        'close_postcode_match',
        'house_court_name_match',
        'fuzzy_score_match',
        "fuzzy_score",
        "wratio_score",
        'property_number_search', 'property_number_reference',  
        'flat_number_search', 'flat_number_reference', 
        'room_number_search', 'room_number_reference',
        'unit_number_search', 'unit_number_reference',
        'block_number_search', 'block_number_reference',
        'house_court_name_search', 'house_court_name_reference',
        "search_mod_address", 'reference_mod_address','Postcode']

        # Join results data onto the original housing list to create the full output
        search_df_cleaned_join_cols = [search_df_key_field, "full_address","postcode"]

        match_results_output = search_df_cleaned[search_df_cleaned_join_cols].merge(
            diag_best_match[match_results_cols], how = "left", left_on = "full_address", right_on = "search_orig_address")
        
        match_results_output = match_results_output.drop(["postcode", "search_orig_address"], axis = 1).rename(columns={"full_address":"search_orig_address"})
        
        # Join UPRN back onto the data from reference data
        joined_ref_cols = ["fulladdress", "Reference file"]
        joined_ref_cols.extend(new_join_col)

        #print("joined_ref_cols: ", joined_ref_cols)
        # Keep only columns that exist in reference dataset
        joined_ref_cols = [col for col in joined_ref_cols if col in ref_df_cleaned.columns]

        match_results_output = pd.merge(match_results_output,ref_df_cleaned[joined_ref_cols].drop_duplicates("fulladdress"), how = "left", left_on = "reference_orig_address",right_on = "fulladdress").drop("fulladdress", axis = 1)

        # Convert long keys to string to avoid data loss
        match_results_output[search_df_key_field] = match_results_output[search_df_key_field].astype("str")
        match_results_output[new_join_col] = match_results_output[new_join_col].astype("string")
        match_results_output["standardised_address"] = standardise
    
        match_results_output = match_results_output.sort_values(search_df_key_field, ascending = True)
                
        return match_results_output, diag_shortlist, diag_best_match

def create_diag_shortlist(results_df:PandasDataFrame, matched_col:str, fuzzy_match_limit:int, blocker_col:str, fuzzy_col:str="fuzzy_score", search_mod_address:str = "search_mod_address", resolve_tie_breaks:bool=True, no_number_fuzzy_match_limit:int=no_number_fuzzy_match_limit) -> PandasDataFrame:
    '''
    Create a shortlist of the best matches from a list of suggested matches
    '''

    ## Calculate highest fuzzy score from all candidates, keep all candidates with matching highest fuzzy score
    results_max_fuzzy_score = results_df.groupby(matched_col)[fuzzy_col].max().reset_index().rename(columns={fuzzy_col: "max_fuzzy_score"}).drop_duplicates(subset=matched_col)

    results_df = pd.merge(results_df, results_max_fuzzy_score, how = "left", on = matched_col)

    diag_shortlist = results_df[(results_df[fuzzy_col] == results_df["max_fuzzy_score"])]

    # Fuzzy match limit for records with no numbers in it is 0.95 or the provided fuzzy_match_limit, whichever is higher
    #diag_shortlist["fuzzy_score_match"] = diag_shortlist[fuzzy_col] >= fuzzy_match_limit
    diag_shortlist.loc[diag_shortlist[fuzzy_col] >= fuzzy_match_limit, "fuzzy_score_match"] = True

    ### Count number of numbers in search string
    # Using .loc
    diag_shortlist.loc[:, "number_count_search_string"] = diag_shortlist.loc[:, search_mod_address].str.count(r'\d')
    diag_shortlist.loc[:, "no_numbers_in_search_string"] = (diag_shortlist.loc[:, "number_count_search_string"] == 0)


    # Replace fuzzy_score_match values for addresses with no numbers in them
    diag_shortlist.loc[(diag_shortlist["no_numbers_in_search_string"]==True) & (diag_shortlist[fuzzy_col] >= no_number_fuzzy_match_limit), "fuzzy_score_match"] = True
    diag_shortlist.loc[(diag_shortlist["no_numbers_in_search_string"]==True) & (diag_shortlist[fuzzy_col] < no_number_fuzzy_match_limit), "fuzzy_score_match"] = False

    # If blocking on street, don't match addresses with 0 numbers in. There are too many options and the matches are rarely good
    if blocker_col == "Street":
        diag_shortlist.loc[(diag_shortlist["no_numbers_in_search_string"]==True), "fuzzy_score_match"] = False
                            
    diag_shortlist = diag_shortlist.fillna("").infer_objects(copy=False).drop(["number_count_search_string", "no_numbers_in_search_string"], axis = 1)

    # Following considers full matches to be those that match on property number and flat number, and the postcode is relatively close.
    #print(diag_shortlist.columns) 
    diag_shortlist["property_number_match"] = (diag_shortlist["property_number_search"] == diag_shortlist["property_number_reference"])
    diag_shortlist["flat_number_match"] = (diag_shortlist['flat_number_search'] == diag_shortlist['flat_number_reference'])
    diag_shortlist["room_number_match"] = (diag_shortlist['room_number_search'] == diag_shortlist['room_number_reference'])
    diag_shortlist["block_number_match"] = (diag_shortlist['block_number_search'] == diag_shortlist['block_number_reference'])
    diag_shortlist["unit_number_match"] = (diag_shortlist['unit_number_search'] == diag_shortlist['unit_number_reference'])
    diag_shortlist["house_court_name_match"] = (diag_shortlist['house_court_name_search'] == diag_shortlist['house_court_name_reference'])

    # Full number match is currently considered only a match between property number and flat number
                            
    diag_shortlist['full_number_match'] = (diag_shortlist["property_number_match"] == True) &\
        (diag_shortlist["flat_number_match"] == True) &\
        (diag_shortlist["room_number_match"] == True) &\
        (diag_shortlist["block_number_match"] == True) &\
        (diag_shortlist["unit_number_match"] == True) &\
        (diag_shortlist["house_court_name_match"] == True)

    
    ### Postcodes need to be close together, so all the characters should match apart from the last two 
    diag_shortlist['close_postcode_match'] = diag_shortlist['postcode'].str.lower().str.replace(" ","").str[:-2] == diag_shortlist['Postcode'].str.lower().str.replace(" ","").str[:-2]
        
    
    diag_shortlist["full_match"] = (diag_shortlist["fuzzy_score_match"] == True) &\
        (diag_shortlist['full_number_match'] == True) &\
        (diag_shortlist['close_postcode_match'] == True)
    
    diag_shortlist = diag_shortlist.rename(columns = {"reference_list_address":"reference_mod_address"})

    ### Dealing with tie breaks ##
    # Do a backup simple Wratio search on the open text to act as a tie breaker when the fuzzy scores are identical
    # fuzz.WRatio
    if resolve_tie_breaks == True:
        def compare_strings_wratio(row, scorer = fuzz.ratio, fuzzy_col = fuzzy_col):
            search_score = process.cdist([row[search_mod_address]], [row["reference_mod_address"]], scorer=scorer)
            return search_score[0][0]

        diag_shortlist_dups = diag_shortlist[diag_shortlist['full_number_match'] == True]
        diag_shortlist_dups = diag_shortlist_dups.loc[diag_shortlist_dups.duplicated(subset= [search_mod_address, 'full_number_match', "room_number_search", fuzzy_col], keep=False)]

        if not diag_shortlist_dups.empty:
            diag_shortlist_dups["wratio_score"] = diag_shortlist_dups.apply(compare_strings_wratio, axis=1)
                                
            diag_shortlist = diag_shortlist.merge(diag_shortlist_dups[["wratio_score"]], left_index=True, right_index=True, how = "left")

    if 'wratio_score' not in diag_shortlist.columns:
        diag_shortlist['wratio_score'] = None 

    # Order by best score
    diag_shortlist = diag_shortlist.sort_values([
        search_mod_address, 'full_match', 'full_number_match', fuzzy_col, "wratio_score"],
        ascending = [True, False, False, False, False])          

    return diag_shortlist

def create_diagnostic_results(results_df:PandasDataFrame, 
                           matched_df:PandasDataFrame,
                           ref_list_df:PandasDataFrame,
                           matched_col="fuzzy_match_search_address",
                           ref_list_col="fuzzy_match_reference_address",
                           final_matched_address_col="search_address_stand",
                           final_ref_address_col="ref_address_stand",
                           orig_matched_address_col = "full_address",
                           orig_ref_address_col = "fulladdress",
                           fuzzy_match_limit=fuzzy_match_limit,
                           blocker_col="Postcode") -> PandasDataFrame:
    '''
    This function takes a result file from the fuzzy search, then refines the 'matched results' according
    the score limit specified by the user and exports results list, matched and unmatched files.
    '''
       
    # Rename score column
    results_df = results_df.rename(columns = {"score":"fuzzy_score"})
          
    # Remove empty addresses
    results_df = results_df[results_df[matched_col] !=0 ]

    ### Join property number and flat/room number etc. onto results_df
    if 'ref_index' not in ref_list_df.columns:
        print("Existing ref_index column not found")
        ref_list_df["ref_index"] = ref_list_df.index

    ref_join_cols = ["ref_index", final_ref_address_col, "property_number","flat_number","room_number","block_number", "unit_number", 'house_court_name', orig_ref_address_col,"Postcode"]
    ref_list_df = ref_list_df[ref_join_cols].rename(columns={orig_ref_address_col: "reference_orig_address", final_ref_address_col:'reference_list_address'})

    results_df = results_df.merge(ref_list_df, how = "left", left_on = ref_list_col, right_on = "reference_list_address")


    ### Join on relevant details from the standardised match dataframe
    matched_df_cols = [final_matched_address_col,"property_number","flat_number","room_number", "block_number", "unit_number", 'house_court_name', orig_matched_address_col, "postcode"]
    matched_df = matched_df[matched_df_cols].rename(columns={orig_matched_address_col:"search_orig_address",final_matched_address_col:'search_mod_address'})
    
    results_df = results_df.merge(matched_df, how = "left", left_on = matched_col, right_on = "search_mod_address", suffixes=("_reference", "_search"))   
    
    # Choose your best matches from the list of options
    diag_shortlist = create_diag_shortlist(results_df, matched_col, fuzzy_match_limit, blocker_col)

    ### Create matched results output ###
    # Columns for the output match_results file in order
    match_results_cols = ['search_orig_address','reference_orig_address', 'ref_index',
        'full_match',
        'full_number_match',
        'flat_number_match',
        'room_number_match',
        'block_number_match',
        'unit_number_match',
        'house_court_name_match',
        'property_number_match',
        'close_postcode_match',
        'fuzzy_score_match',
        "fuzzy_score",
        "wratio_score",
        'property_number_search', 'property_number_reference',  
        'flat_number_search', 'flat_number_reference', 
        'room_number_search', 'room_number_reference',
        'block_number_search', 'block_number_reference',
        'unit_number_search', 'unit_number_reference',
        'house_court_name_search', 'house_court_name_reference',
        "search_mod_address", 'reference_mod_address', 'postcode','Postcode']

    diag_shortlist = diag_shortlist[match_results_cols]

    diag_shortlist["ref_index"] = diag_shortlist["ref_index"].astype(int, errors="ignore")
    diag_shortlist["wratio_score"] = diag_shortlist["wratio_score"].astype(float, errors="ignore")

    # Choose best match from the shortlist that has been ordered according to score descending
    diag_best_match = diag_shortlist[match_results_cols].drop_duplicates("search_mod_address")
   
    return diag_shortlist, diag_best_match

def create_results_df(match_results_output:PandasDataFrame, search_df:PandasDataFrame, search_df_key_field:str, new_join_col:List[str]) -> PandasDataFrame:
    ''' 
    Following the fuzzy match, join the match results back to the original search dataframe to create a results dataframe.
    '''
    match_results_output_success = match_results_output[match_results_output["full_match"]==True]

    # If you're joining to the original df on index you will need to recreate the index again 

    match_results_output_success = match_results_output_success.rename(columns={
                                        "reference_orig_address":"Reference matched address",
                                        "full_match":"Matched with reference address",
                                        'uprn':'UPRN'                                                                             
                                     }, errors="ignore")
    
    ref_df_after_stand_cols = ["ref_index", "Reference matched address","Matched with reference address", "Reference file", search_df_key_field]
    ref_df_after_stand_cols.extend(new_join_col)
    
    if (search_df_key_field == "index"):
        # Check index is int
        print("Search df key field is index")
        #match_results_output_success[search_df_key_field] = match_results_output_success[search_df_key_field].astype(float).astype(int)     
        results_for_orig_df_join = search_df.merge(match_results_output_success[ref_df_after_stand_cols], on = search_df_key_field, how = "left", suffixes = ('', '_y'))  
    else:
        results_for_orig_df_join = search_df.merge(match_results_output_success[ref_df_after_stand_cols],how = "left", on = search_df_key_field, suffixes = ('', '_y'))

    # If the join columns already exist in the search_df, then use the new column to fill in the NAs in the original column, then delete the new column

    if "Reference matched address_y" in results_for_orig_df_join.columns: 
        results_for_orig_df_join['Reference matched address'] = results_for_orig_df_join['Reference matched address'].fillna(results_for_orig_df_join['Reference matched address_y']).infer_objects(copy=False)

    if "Matched with reference address_y" in results_for_orig_df_join.columns: 
        results_for_orig_df_join['Matched with reference address'] = pd.Series(np.where(results_for_orig_df_join['Matched with reference address_y'].notna(), results_for_orig_df_join['Matched with reference address_y'], results_for_orig_df_join['Matched with reference address']))

    if "Reference file_y" in results_for_orig_df_join.columns: 
        results_for_orig_df_join['Reference file'] = results_for_orig_df_join['Reference file'].fillna(results_for_orig_df_join['Reference file_y']).infer_objects(copy=False)

    if "UPRN_y" in results_for_orig_df_join.columns: 
        results_for_orig_df_join['UPRN'] = results_for_orig_df_join['UPRN'].fillna(results_for_orig_df_join['UPRN_y']).infer_objects(copy=False)

    # Drop columns that aren't useful
    results_for_orig_df_join = results_for_orig_df_join.drop(['Reference matched address_y', 'Matched with reference address_y', 'Reference file_y', 'search_df_key_field_y', 'UPRN_y', 'index_y', "full_address_search","postcode_search", "full_address_1", "full_address_2", 
                                   "address_stand", "property_number","prop_number" "flat_number" "apart_number" "first_sec_number" "room_number"], axis = 1, errors = "ignore")
    
    results_for_orig_df_join.rename(columns={"full_address":"Search data address"}, inplace = True)

    results_for_orig_df_join["index"] = results_for_orig_df_join["index"].astype(int, errors="ignore")
    results_for_orig_df_join["ref_index"] = results_for_orig_df_join["ref_index"].astype(int, errors="ignore")

    # Replace blanks with NA, fix UPRNs
    results_for_orig_df_join = results_for_orig_df_join.replace(r'^\s*$', np.nan, regex=True)   

    results_for_orig_df_join[new_join_col] = results_for_orig_df_join[new_join_col].astype(str).replace(".0","", regex=False).replace("nan","", regex=False)
    
    # Replace cells with only 'nan' with blank
    results_for_orig_df_join = results_for_orig_df_join.replace(r'^nan$', "", regex=True)

    results_for_orig_df_join.to_csv("output/results_for_orig_df_join.csv")
    
    return results_for_orig_df_join