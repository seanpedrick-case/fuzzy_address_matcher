import pandas as pd
import numpy as np
import re
from typing import Type, Dict, List, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings("ignore", 'This pattern is interpreted as a regular expression')

PandasDataFrame = Type[pd.DataFrame]
PandasSeries = Type[pd.Series]
MatchedResults = Dict[str,Tuple[str,int]]
array = List[str]

today = datetime.now().strftime("%d%m%Y")
today_rev = datetime.now().strftime("%Y%m%d")

# # Standardisation functions

def standardise_wrapper_func(search_df_cleaned:PandasDataFrame, ref_df_cleaned:PandasDataFrame,\
                               standardise = False, filter_to_lambeth_pcodes = True, match_task = "fuzzy"):
    '''
    Initial standardisation of search and reference dataframes before passing addresses and postcodes to the main standardisation function
    '''
    
    ## Search df - lower case addresses, replace spaces in postcode
    search_df_cleaned["full_address_search"] = search_df_cleaned["full_address"].str.lower().str.strip()
    search_df_cleaned['postcode_search'] = search_df_cleaned['postcode'].str.lower().str.strip().str.replace("\s+", "", regex=True)

    # Filter out records where 'Excluded from search' is not a postal address by making the postcode blank
    search_df_cleaned.loc[search_df_cleaned['Excluded from search'] == "Excluded - non-postal address", 'postcode_search'] = ""

    # Remove nulls from ref_df postcode
    ref_df_cleaned = ref_df_cleaned[ref_df_cleaned['Postcode'].notna()]
    
    ref_df_cleaned["full_address_search"] = ref_df_cleaned["fulladdress"].str.lower().str.strip()
    ref_df_cleaned['postcode_search'] = ref_df_cleaned['Postcode'].str.lower().str.strip().str.replace("\s+", "", regex=True)
    
    # Block only on first 5 characters of postcode string - Doesn't give more matches and makes everything a bit slower
    #search_df_cleaned['postcode_search'] = search_df_cleaned['postcode_search'].str[:-1]
    #ref_df_cleaned['postcode_search'] = ref_df_cleaned['postcode_search'].str[:-1]


    ### Use standardise function

    ### Remove 'non-housing' places from the list - not included as want to check all
    #search_df_after_stand = remove_non_housing(search_df_cleaned, 'full_address_search')
    search_df_after_stand = standardise_address(search_df_cleaned, "full_address_search", "search_address_stand", standardise = standardise, out_london = True)

    ## Standardise ref_df addresses

    if match_task == "fuzzy":
        ref_df_after_stand = standardise_address(ref_df_cleaned, "full_address_search", "ref_address_stand", standardise = standardise, out_london = True)
    else:
        # For the neural net matching, I didn't find that standardising the reference addresses helped at all, in fact it made things worse. So reference addresses are not standardised at this step.
        ref_df_after_stand = standardise_address(ref_df_cleaned, "full_address_search", "ref_address_stand", standardise = False, out_london = True)


    return search_df_after_stand, ref_df_after_stand#, search_df_after_stand_series, ref_df_after_stand_series

def standardise_address(df:PandasDataFrame, col:str, out_col:str, standardise:bool = True, out_london = True) -> PandasDataFrame:
    
    ''' 
    This function takes a 'full address' column and then standardises so that extraneous
    information is removed (i.e. postcodes & London, as this algorithm is used for London
    addresses only), and so that room/flat/property numbers can be accurately extracted. The
    standardised addresses can then be used for the fuzzy matching functions later in this 
    notebook.
    
    The function does the following:
    
    - Removes the post code and 'london' (if not dealing with addresses outside of london)
      from the address to reduce the text the algorithm has to search.
      Postcode removal uses regex to extract a UK postcode.
      
    - Remove the word 'flat' or 'apartment' from an address that has only one number in it
    
    - Add 'flat' to the start of any address that contains 'house' or 'court' (which are generally housing association buildings)
      This is because in the housing list, these addresses never have the word flat in front of them

    - Replace any addresses that don't have a space between the comma and the next word or double spaces
    
    - Replace 'number / number' and 'number-number' with 'number' (the first number in pair)
    
    - Add 'flat' to the start of addresses that include ground floor/first floor etc. flat 
      in the text. Replace with flat a,b,c etc.
    
    - Pull out property, flat, and room numbers from the address text
    
    - Return the data frame with the new columns included
    
    '''
   
    df_copy = df.copy(deep=True)
    
    # Trim the address to remove leading and tailing spaces
    df_copy[col] = df_copy[col].str.strip()
    
    ''' Remove the post code and 'london' from the address to reduce the text the algorithm has to search
    Using a regex to extract a UK postcode. I got the regex from the following. Need to replace their \b in the solution with \\b
    https://stackoverflow.com/questions/51828712/r-regular-expression-for-extracting-uk-postcode-from-an-address-is-not-ordered
        
    The following will pick up whole postcodes, postcodes with just the first part, and postcodes with the first
    part and the first number of the second half
    '''
    
    
    df_copy['add_no_pcode'] = remove_postcode(df_copy, col)
    
    if out_london == False:
        df_copy['add_no_pcode'] = df_copy['add_no_pcode'].str.replace("london","").str.replace(r",,|, ,","", regex=True)
    
    # If the user wants to standardise the address
    if standardise:
        
        df_copy['add_no_pcode'] = df_copy['add_no_pcode'].str.lower()

        # If there are dates at the start of the address, change this
        df_copy['add_no_pcode'] = replace_mistaken_dates(df_copy, 'add_no_pcode')

        # Replace flat name variations with flat, abbreviations with full name of item (e.g. rd to road)
        df_copy['add_no_pcode'] = df_copy['add_no_pcode'].str.replace(r"\brd\b","road", regex=True).\
                                                str.replace(r"\bst\b","street", regex=True).\
                                                str.replace(r"\bave\b","avenue", regex=True).\
                                                str.replace("'", "", regex=False).\
                                                str.replace(r"\bat\b ", " ",regex=True).\
                                                str.replace("apartment", "flat",regex=False).\
                                                str.replace("studio flat", "flat",regex=False).\
                                                str.replace("cluster flat", "flats",regex=False).\
                                                str.replace(r"\bflr\b", "floor", regex=True).\
                                                str.replace(r"\bflrs\b", "floors", regex=True).\
                                                str.replace(r"\blwr\b", "lower", regex=True).\
                                                str.replace(r"\bgnd\b", "ground", regex=True).\
                                                str.replace(r"\blgnd\b", "lower ground", regex=True).\
                                                str.replace(r"\bgrd\b", "ground", regex=True).\
                                                str.replace(r"\bmais\b", "flat", regex=True).\
                                                str.replace(r"\bmaisonette\b", "flat", regex=True).\
                                                str.replace(r"\bpt\b", "penthouse", regex=True).\
                                                str.replace(r"\bbst\b","basement", regex=True).\
                                                str.replace(r"\bbsmt\b","basement", regex=True).\
                                                str.replace(r"\s{2,}", " ", regex=True).\
                                                str.strip()
        
        df_copy["add_no_pcode_house"] = move_flat_house_court(df_copy)
    
        # Replace any addresses that don't have a space between the comma and the next word. and double spaces # df_copy['add_no_pcode_house']
        df_copy['add_no_pcode_house_comma'] = df_copy['add_no_pcode_house'].str.replace(r',(\w)', r', \1', regex=True).str.replace('  ', ' ', regex=False)

        # Replace number / number and number-number with number
        df_copy['add_no_pcode_house_comma_no'] = df_copy['add_no_pcode_house_comma'].str.replace(r'(\d+)\/(\d+)', r'\1', regex=True\
                        ).str.replace(r'(\d+)-(\d+)', r'\1', regex=True\
                        ).str.replace(r'(\d+) - (\d+)', r'\1', regex=True)

        # Add 'flat' to the start of addresses that include ground/first/second etc. floor flat in the text
        df_copy['floor_replacement'] = replace_floor_flat(df_copy, 'add_no_pcode_house_comma_no')
        df_copy['flat_added_to_start_addresses_begin_letter'] = add_flat_addresses_start_with_letter(df_copy, 'floor_replacement')

        df_copy[out_col] = merge_series(df_copy['add_no_pcode_house_comma_no'], df_copy['flat_added_to_start_addresses_begin_letter'])

        # Write stuff back to the original df
        df[out_col] = df_copy[out_col]
    
    else:
        df_copy[out_col] = df_copy['add_no_pcode']
        df[out_col] = df_copy['add_no_pcode']

    ## POST STANDARDISATION CLEANING AND INFORMATION EXTRACTION  
    # Remove trailing spaces
    df[out_col] = df[out_col].str.strip()
    
    # Pull out property, flat, and room numbers from the address text
    df['property_number'] = extract_prop_no(df_copy, out_col)

    # Extract flat, apartment numbers
    df = extract_flat_and_other_no(df, out_col)

    df['flat_number'] = merge_series(df['flat_number'], df['apart_number'])
    df['flat_number'] = merge_series(df['flat_number'], df['prop_number'])
    df['flat_number'] = merge_series(df['flat_number'], df['first_sec_number'])
    df['flat_number'] = merge_series(df['flat_number'], df['first_letter_flat_number'])
    df['flat_number'] = merge_series(df['flat_number'], df['first_letter_no_more_numbers'])
    
    # Extract room numbers
    df['room_number'] = extract_room_no(df, out_col)

    # Extract block and unit names
    df = extract_block_and_unit_name(df, out_col)

    # Extract house or court name
    df['house_court_name'] = extract_house_or_court_name(df, out_col)

    return df

def move_flat_house_court(df:PandasDataFrame):
            ''' Remove 'flat' from any address that contains 'house' or 'court'
            From the df address, remove the word 'flat' from any address that contains the word 'house' or 'court'
            This is because in the housing list, these addresses never have the word flat in front of them
            '''
            
            # Remove the word flat or apartment from addresses that have only one number in it. 'Flat' will be re-added later to relevant addresses 
            # that need it (replace_floor_flat)
            df['flat_removed'] = remove_flat_one_number_address(df, 'add_no_pcode')
        
        
            
            remove_flat_house = df['flat_removed'].str.lower().str.contains(r"\bhouse\b")#(?=\bhouse\b)(?!.*house road)")
            remove_flat_court = df['flat_removed'].str.lower().str.contains(r"\bcourt\b")#(?=\bcourt\b)(?!.*court road)")
            remove_flat_terrace = df['flat_removed'].str.lower().str.contains(r"\bterrace\b")#(?=\bterrace\b)(?!.*terrace road)")
            remove_flat_house_or_court = (remove_flat_house | remove_flat_court | remove_flat_terrace == 1)

            df['remove_flat_house_or_court'] = remove_flat_house_or_court

            # Assuming 'df' is your DataFrame
            df = df[~df.index.duplicated(keep='first')]

            df['house_court_replacement'] = "flat " + df.loc[df['remove_flat_house_or_court'] == True, 'flat_removed'].str.replace(r"\bflat\b","", regex=True).str.strip().map(str)

            #df["add_no_pcode_house"] = merge_columns(df, "add_no_pcode_house", 'flat_removed', "house_court_replacement")

            #merge_columns(df, "new_col", col1, 'letter_after_number')
            df["add_no_pcode_house"] = merge_series(df['flat_removed'], df["house_court_replacement"])

            return df["add_no_pcode_house"]

def extract_street_name(address:str) -> str:
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
        'Street', 'St', 'Boulevard', 'Blvd', 'Highway', 'Hwy', 'Broadway', 'Freeway',
        'Causeway', 'Cswy', 'Expressway', 'Way', 'Walk', 'Lane', 'Ln', 'Road', 'Rd',
        'Avenue', 'Ave', 'Circle', 'Cir', 'Cove', 'Cv', 'Drive', 'Dr', 'Parkway', 'Pkwy',
        'Park', 'Court', 'Ct', 'Square', 'Sq', 'Loop', 'Place', 'Pl', 'Parade', 'Estate',
        'Alley', 'Arcade','Avenue', 'Ave','Bay','Bend','Brae','Byway','Close','Corner','Cove',
        'Crescent', 'Cres','Cul-de-sac','Dell','Drive', 'Dr','Esplanade','Glen','Green','Grove','Heights', 'Hts',
        'Mews','Parade','Path','Piazza','Promenade','Quay','Ridge','Row','Terrace', 'Ter','Track','Trail','View','Villas',
        'Marsh', 'Embankment', 'Cut', 'Hill', 'Passage', 'Rise', 'Vale', 'Side'
    ]

    # Dynamically construct the regex pattern with all possible street types
    street_types_pattern = '|'.join(rf"{re.escape(street_type)}" for street_type in street_types)

    # The overall regex pattern to capture the street name
    pattern = rf'(?:\d+\s+|\w+\s+\d+\s+|.*\d+[a-z]+\s+|.*\d+\s+)*(?P<street_name>[\w\s]+(?:{street_types_pattern}))'

    def replace_postcode(address):
        pattern = r'\b(?:[A-Z][A-HJ-Y]?[0-9][0-9A-Z]? ?[0-9][A-Z]{2}|GIR ?0A{2})\b$|(?:[A-Z][A-HJ-Y]?[0-9][0-9A-Z]? ?[0-9]{1}?)$|\b(?:[A-Z][A-HJ-Y]?[0-9][0-9A-Z]?)\b$'
        return re.sub(pattern, "", address)

    
    modified_address = replace_postcode(address.upper())
    #print(modified_address)
    #print(address)
       
    # Perform a case-insensitive search
    match = re.search(pattern, modified_address, re.IGNORECASE)

    if match:
        street_name = match.group('street_name')
        return street_name.strip()
    else:
        return ""

def remove_flat_one_number_address(df:PandasDataFrame, col1:PandasSeries) -> PandasSeries:

    '''
    If there is only one number in the address, and there is no letter after the number,
    remove the word flat from the address
    '''
    
    df['contains_letter_after_number'] = df[col1].str.lower().str.contains(r"\d+(?:[a-z]|[A-Z])(?!.*\d+)", regex = True)
    df['contains_single_letter_before_number'] = df[col1].str.lower().str.contains(r'\b[A-Za-z]\b[^\d]* \d', regex = True)
    df['two_numbers_in_address'] =  df[col1].str.lower().str.contains(r"(?:\d+.*?)[^a-zA-Z0-9_].*?\d+", regex = True)
    df['contains_apartment'] = df[col1].str.lower().str.contains(r"\bapartment\b \w+|\bapartments\b \w+", "", regex = True)
    df['contains_flat'] = df[col1].str.lower().str.contains(r"\bflat\b \w+|\bflats\b \w+", "", regex = True)
    df['contains_room'] = df[col1].str.lower().str.contains(r"\broom\b \w+|\brooms\b \w+", "", regex = True)
      
    
    df['selected_rows'] = (df['contains_letter_after_number'] == False) &\
                          (df['two_numbers_in_address'] == False) &\
                            (df['contains_single_letter_before_number'] == False) &\
                             ((df['contains_flat'] == True) |\
                             (df['contains_apartment'] == True) |\
                             (df['contains_room'] == True))
        
    df['one_number_no_flat'] =  df[df['selected_rows'] == True][col1]
    df['one_number_no_flat'] =  df['one_number_no_flat'].str.replace(r"(\bapartment\b)|(\bapartments\b)", "", regex=True).str.replace(r"(\bflat\b)|(\bflats\b)", "", regex=True).str.replace(r"(\broom\b)|(\brooms\b)", "", regex=True)

    df["new_col"] = merge_series(df[col1], df["one_number_no_flat"])
    
    return df['new_col']

def add_flat_addresses_start_with_letter(df:PandasDataFrame, col1:PandasSeries) -> PandasSeries:
    '''
    Add the word flat to addresses that start with a letter.
    '''
    
    df['contains_single_letter_at_start_before_number'] = df[col1].str.lower().str.contains(r'^\b[A-Za-z]\b[^\d]* \d', regex = True)

    df['selected_rows'] = (df['contains_single_letter_at_start_before_number'] == True)
    df['flat_added_to_string_start'] =  "flat " + df[df['selected_rows'] == True][col1]
    
    #merge_columns(df, "new_col", col1, 'flat_added_to_string_start')
    df["new_col"] = merge_series(df[col1], df['flat_added_to_string_start'])
    
    
    return df['new_col']

def extract_letter_one_number_address(df:PandasDataFrame, col1:PandasSeries) -> PandasSeries:
    '''
    This function looks for addresses that have a letter after a number, but ONLY one number
    in the string, and doesn't already have a flat, apartment, or room number. 
        
    It then extracts this letter and returns this.
    
    This is for addresses such as '2b sycamore road', changes it to
    flat b 2 sycamore road so that 'b' is selected as the flat number

    
    '''
    
    df['contains_no_numbers_without_letter'] = df[col1].str.lower().str.contains(r"^(?:(?!\d+ ).)*$")
    df['contains_letter_after_number'] = df[col1].str.lower().str.contains(r"\d+(?:[a-z]|[A-Z])(?!.*\d+)")      
    df['contains_apartment'] = df[col1].str.lower().str.contains(r"\bapartment\b \w+|\bapartments\b \w+", "")
    df['contains_flat'] = df[col1].str.lower().str.contains(r"\bflat\b \w+|\bflats\b \w+", "")
    df['contains_room'] = df[col1].str.lower().str.contains(r"\broom\b \w+|\brooms\b \w+", "")
        
    df['selected_rows'] = (df['contains_no_numbers_without_letter'] == True) &\
                             (df['contains_letter_after_number'] == True) &\
                             (df['contains_flat'] == False) &\
                             (df['contains_apartment'] == False) &\
                             (df['contains_room'] == False)
            
    df['extract_letter'] =  df[(df['selected_rows'] == True)\
                                  ][col1].str.extract(r"\d+([a-z]|[A-Z])")
    
    df['extract_number'] =  df[(df['selected_rows'] == True)\
                                  ][col1].str.extract(r"(\d+)[a-z]|[A-Z]")
    

    df['letter_after_number'] = "flat " +\
                                df[(df['selected_rows'] == True)\
                                  ]['extract_letter'] +\
                                " " +\
                                df[(df['selected_rows'] == True)\
                                  ]['extract_number'] +\
                                " " +\
                                df[(df['selected_rows'])\
                                  ][col1].str.replace(r"\bflat\b","", regex=True).str.replace(r"\d+([a-z]|[A-Z])","", regex=True).map(str)

    #merge_columns(df, "new_col", col1, 'letter_after_number')
    df["new_col"] = merge_series(df[col1], df['letter_after_number'])
    
    return df['new_col']

# def extract_letter_one_number_address(df:PandasDataFrame, col1:PandasSeries) -> PandasSeries:
#     '''
#     This function extracts a letter after a single number in an address, excluding cases with existing flat, apartment, or room numbers.
#     It transforms addresses like '2b sycamore road' to 'flat b 2 sycamore road' to designate 'b' as the flat number.
#     '''
    
#     df['selected_rows'] = (df[col1].str.lower().str.contains(r"^(?:(?!\d+ ).)*$") & \
#                            df[col1].str.lower().str.contains(r"\d+(?:[a-z]|[A-Z])(?!.*\d+)") & \
#                            ~df[col1].str.lower().str.contains(r"\bflat\b \w+|\bflats\b \w+|\bapartment\b \w+|\bapartments\b \w+|\broom\b \w+|\brooms\b \w+"))
    
#     df['extract_letter'] = df.loc[df['selected_rows'], col1].str.extract(r"\d+([a-z]|[A-Z])")
#     df['extract_number'] = df.loc[df['selected_rows'], col1].str.extract(r"(\d+)[a-z]|[A-Z]")
    
#     df['letter_after_number'] = "flat " + df['extract_letter'] + " " + df['extract_number'] + " " + \
#                                 df.loc[df['selected_rows'], col1].str.replace(r"\bflat\b", "", regex=True).str.replace(r"\d+([a-z]|[A-Z])", "", regex=True).map(str)
    
#     df["new_col"] = df[col1].copy()
#     df.loc[df['selected_rows'], "new_col"] = df['letter_after_number']
    
#     return df['new_col']

def replace_floor_flat(df:PandasDataFrame, col1:PandasSeries) -> PandasSeries:
    ''' In references to basement, ground floor, first floor, second floor, and top floor
    flats, this function moves the word 'flat' to the front of the address. This is so that the
    following word (e.g. basement, ground floor) is recognised as the flat number in the 'extract_flat_and_other_no' function
    '''
    
    df['letter_after_number'] = extract_letter_one_number_address(df, col1)
       
   
    df['basement'] = "flat basement" + df[df[col1].str.lower().str.contains(r"basement"\
                                )][col1].str.replace(r"\bflat\b","", regex=True).str.replace(r"\bbasement\b","", regex=True).map(str)
    

    df['ground_floor'] = "flat a " + df[df[col1].str.lower().str.contains(r"\bground floor\b"\
                                )][col1].str.replace(r"\bflat\b","", regex=True).str.replace(r"\bground floor\b","", regex=True).map(str)

    df['first_floor'] = "flat b " + df[df[col1].str.lower().str.contains(r"\bfirst floor\b"\
                                )][col1].str.replace(r"\bflat\b","", regex=True).str.replace(r"\bfirst floor\b","", regex=True).map(str)

    df['ground_and_first_floor'] = "flat ab " + df[df[col1].str.lower().str.contains(r"\bground and first floor\b"\
                                )][col1].str.replace(r"\bflat\b","", regex=True).str.replace(r"\bground and first floor\b","", regex=True).map(str)

    df['basement_ground_and_first_floor'] = "flat basementab " + df[df[col1].str.lower().str.contains(r"\bbasement ground and first floors\b"\
                                )][col1].str.replace(r"\bflat\b","", regex=True).str.replace(r"\bbasement and ground and first floors\b","", regex=True).map(str)

    df['basement_ground_and_first_floor2'] = "flat basementab " + df[df[col1].str.lower().str.contains(r"\bbasement ground and first floors\b"\
                                )][col1].str.replace(r"\bflat\b","", regex=True).str.replace(r"\bbasement ground and first floors\b","", regex=True).map(str)

    df['second_floor'] = "flat c " + df[df[col1].str.lower().str.contains(r"\bsecond floor\b"\
                                )][col1].str.replace(r"\bflat\b","", regex=True).str.replace(r"\bsecond floor\b","", regex=True).map(str)

    df['first_and_second_floor'] = "flat bc " + df[df[col1].str.lower().str.contains(r"\bfirst and second floor\b"\
                                )][col1].str.replace(r"\bflat\b","", regex=True).str.replace(r"\bfirst and second floor\b","", regex=True).map(str)

    df['first1_floor'] = "flat b " + df[df[col1].str.lower().str.contains(r"\b1st floor\b"\
                                )][col1].str.replace(r"\bflat\b","", regex=True).str.replace(r"\b1st floor\b","", regex=True).map(str)

    df['second2_floor'] = "flat c " + df[df[col1].str.lower().str.contains(r"\b2nd floor\b"\
                                )][col1].str.replace(r"\bflat\b","", regex=True).str.replace(r"\b2nd floor\b","", regex=True).map(str)

    df['ground_first_second_floor'] = "flat abc " + df[df[col1].str.lower().str.contains(r"\bground and first and second floor\b"\
                                )][col1].str.replace(r"\bflat\b","", regex=True).str.replace(r"\bground and first and second floor\b","", regex=True).map(str)

    df['third_floor'] = "flat d " + df[df[col1].str.lower().str.contains(r"\bthird floor\b"\
                                )][col1].str.replace(r"\bflat\b","", regex=True).str.replace(r"\bthird floor\b","", regex=True).map(str)

    df['third3_floor'] = "flat d " + df[df[col1].str.lower().str.contains(r"\b3rd floor\b"\
                                )][col1].str.replace(r"\bflat\b","", regex=True).str.replace(r"\b3rd floor\b","", regex=True).map(str)   

    df['top_floor'] = "flat top " + df[df[col1].str.lower().str.contains(r"\btop floor\b"\
                                )][col1].str.replace(r"\bflat\b","", regex=True).str.replace(r"\btop floor\b","", regex=True).map(str)
    
    #merge_columns(df, "new_col", col1, 'letter_after_number')
    df["new_col"] = merge_series(df[col1], df['letter_after_number'])
    df["new_col"] = merge_series(df["new_col"], df['basement'])
    df["new_col"] = merge_series(df["new_col"], df['ground_floor'])
    df["new_col"] = merge_series(df["new_col"], df['first_floor'])
    df["new_col"] = merge_series(df["new_col"], df['first1_floor'])
    df["new_col"] = merge_series(df["new_col"], df['ground_and_first_floor'])
    df["new_col"] = merge_series(df["new_col"], df['basement_ground_and_first_floor'])
    df["new_col"] = merge_series(df["new_col"], df['basement_ground_and_first_floor2'])
    df["new_col"] = merge_series(df["new_col"], df['second_floor'])
    df["new_col"] = merge_series(df["new_col"], df['second2_floor'])
    df["new_col"] = merge_series(df["new_col"], df['first_and_second_floor'])
    df["new_col"] = merge_series(df["new_col"], df['ground_first_second_floor'])
    df["new_col"] = merge_series(df["new_col"], df['third_floor'])
    df["new_col"] = merge_series(df["new_col"], df['third3_floor'])
    df["new_col"] = merge_series(df["new_col"], df['top_floor'])
       
    return df['new_col']

# def replace_floor_flat(df:PandasDataFrame, col1:PandasSeries) -> PandasSeries:
#     '''Moves the word 'flat' to the front of addresses with floor references like basement, ground floor, etc.'''
    
#     floor_mapping = {
#         'basement': 'basement',
#         'ground floor': 'a',
#         'first floor': 'b',
#         'ground and first floor': 'ab',
#         'basement ground and first floors': 'basementab',
#         'second floor': 'c',
#         'first and second floor': 'bc',
#         '1st floor': 'b',
#         '2nd floor': 'c',
#         'ground and first and second floor': 'abc',
#         'third floor': 'd',
#         '3rd floor': 'd',
#         'top floor': 'top'
#     }
    
#     for key, value in floor_mapping.items():
#         df[key] = f"flat {value} " + df[df[col1].str.lower().str.contains(fr"\b{key}\b")][col1].str.replace(r"\bflat\b", "", regex=True).str.replace(fr"\b{key}\b", "", regex=True).map(str)
    
#     df["new_col"] = df[col1].copy()
    
#     for key in floor_mapping.keys():
#         df["new_col"] = merge_series(df["new_col"], df[key])
    
#     return df["new_col"]


def remove_non_housing(df:PandasDataFrame, col1:PandasSeries) -> PandasDataFrame:
    '''
    Remove items from the housing list that are not housing. Includes addresses including
    the text 'parking', 'garage', 'store', 'visitor bay', 'visitors room', and 'bike rack',
    'yard', 'workshop'
    '''
    df_copy = df.copy()[~df[col1].str.lower().str.contains(\
    r"parking|garage|\bstore\b|\bstores\b|\bvisitor bay\b|visitors room|\bbike rack\b|\byard\b|\bworkshop\b")]
                                                                 
    return df_copy

def extract_prop_no(df:PandasDataFrame, col1:PandasSeries) -> PandasSeries:
    '''
    Extract property number from an address. Remove flat/apartment/room numbers, 
    then extract the last number/number + letter in the string.
    '''
    try:
        prop_no = df[col1].str.replace(r"(^\bapartment\b \w+)|(^\bapartments\b \w+)", "", regex=True\
                                  ).str.replace(r"(^\bflat\b \w+)|(^\bflats\b \w+)", "", regex=True\
                                               ).str.replace(r"(^\broom\b \w+)|(^\brooms\b \w+)", "", regex=True\
                                                            ).str.replace(",", "", regex=True\
                                                              ).str.extract(r"(\d+\w+|\d+)(?!.*\d+)") #"(\d+\w+|\d+)(?!.*\d+)" 
    except:
        prop_no = np.nan
        
    return prop_no

def extract_room_no(df:PandasDataFrame, col1:PandasSeries) -> PandasSeries:
    '''
    Extract room number from an address. Find rows where the address contains 'room', then extract
    the next word after 'room' in the string.
    '''
    try:
        room_no = df[df[col1].str.lower().str.contains(r"\broom\b|\brooms\b",regex=True\
                                )][col1].str.replace("no.","").str.extract(r'room. (\w+)',regex=True\
                                                        ).rename(columns = {0:"room_number"})
    except:
        room_no = np.nan
        
    return room_no

def extract_flat_and_other_no(df:PandasDataFrame, col1:PandasSeries) -> PandasSeries:
    '''
    Extract flat number from an address. 
    It looks for letters after a property number IF THERE ARE NO MORE NUMBERS IN THE STRING,
    the words following the words 'flat' or 'apartment', or
    the last regex selects all characters in a word containing a digit if there are two numbers in the address
    # ^\d+([a-z]|[A-Z])
    '''  
    
    #  the regex  essentially matches strings that satisfy any of the following conditions:

    # Start with a number followed by a single letter (either lowercase or uppercase) and not followed by any other number.
    # Contain the word "flat" or "apartment".
    # Start with a number, followed by any characters that are not alphanumeric (denoted by [^a-zA-Z0-9_]), and then another number.

    replaced_series = df[df[col1].str.lower().str.replace(r"^\bflats\b","flat", regex=True).\
                         str.contains(\
            r"^\d+([a-z]|[A-Z])(?!.*\d+)|^([a-z] |[A-Z] )(?!.*\d+)|\bflat\b|\bapartment\b|(\d+.*?)[^a-zA-Z0-9_].*?\d+")][col1].str.replace("no.","", regex=True)
    
    extracted_series = replaced_series.str.extract(r'^\d+([a-z]|[A-Z])(?!.*\d+)')[0]

    extracted_series = extracted_series[~extracted_series.index.duplicated()]
    df = df[~df.index.duplicated(keep='first')]

    df["prop_number"] = extracted_series

    extracted_series = replaced_series.str.extract(r'(?i)(?:flat|flats) (\w+)')
    if 1 in  extracted_series.columns:  
        df["flat_number"] = extracted_series[0].fillna(extracted_series[1]).infer_objects(copy=False)
    else: 
        df["flat_number"] = extracted_series[0]

    extracted_series = replaced_series.str.extract(r'(?i)(?:apartment|apartments) (\w+)')
    if 1 in  extracted_series.columns:  
        df["apart_number"] = extracted_series[0].fillna(extracted_series[1]).infer_objects(copy=False)
    else: 
        df["apart_number"] = extracted_series[0]
    
    df["first_sec_number"] = replaced_series.str.extract(r'(\d+.*?)[^a-zA-Z0-9_].*?\d+')
    df["first_letter_flat_number"] = replaced_series.str.extract(r'\b([A-Za-z])\b[^\d]* \d')
    df["first_letter_no_more_numbers"] = replaced_series.str.extract(r'^([a-z] |[A-Z] )(?!.*\d+)')
        
    return df

def extract_house_or_court_name(df:PandasDataFrame, col1:PandasSeries) -> PandasSeries:
    '''
    Extract house or court name. Extended to include estate, buildings, and mansions
    '''
    extracted_series = df[col1].str.extract(r"(\w+)\s+(house|court|estate|buildings|mansions)")
    if 1 in  extracted_series.columns:  
        df["house_court_name"] = extracted_series[0].fillna(extracted_series[1]).infer_objects(copy=False)
    else: 
        df["house_court_name"] = extracted_series[0]

    return df["house_court_name"]

def extract_block_and_unit_name(df:PandasDataFrame, col1:PandasSeries) -> PandasSeries:
    '''
    Extract house or court name. Extended to include estate, buildings, and mansions
    '''

    extracted_series = df[col1].str.extract(r'(?i)(?:block|blocks) (\w+)')
    if 1 in  extracted_series.columns:  
        df["block_number"] = extracted_series[0].fillna(extracted_series[1]).infer_objects(copy=False)
    else: 
        df["block_number"] = extracted_series[0]

    extracted_series = df[col1].str.extract(r'(?i)(?:unit|units) (\w+)')
    if 1 in  extracted_series.columns:  
        df["unit_number"] = extracted_series[0].fillna(extracted_series[1]).infer_objects(copy=False)
    else: 
        df["unit_number"] = extracted_series[0]

    return df

def extract_postcode(df:PandasDataFrame, col:str) -> PandasSeries:
    '''
    Extract a postcode from a string column in a dataframe
    '''
    postcode_series = df[col].str.upper().str.extract(pat = \
    "(\\b(?:[A-Z][A-HJ-Y]?[0-9][0-9A-Z]? ?[0-9][A-Z]{2})|((GIR ?0A{2})\\b$)|(?:[A-Z][A-HJ-Y]?[0-9][0-9A-Z]? ?[0-9]{1}?)$)|(\\b(?:[A-Z][A-HJ-Y]?[0-9][0-9A-Z]?)\\b$)")
    
    return postcode_series

def remove_postcode(df:PandasDataFrame, col:str) -> PandasSeries:
    '''
    Remove a postcode from a string column in a dataframe
    '''

    
    address_series_no_pcode = df[col].str.upper().str.replace(\
    "\\b(?:[A-Z][A-HJ-Y]?[0-9][0-9A-Z]? ?[0-9][A-Z]{2}|GIR ?0A{2})\\b$|(?:[A-Z][A-HJ-Y]?[0-9][0-9A-Z]? ?[0-9]{1}?)$|\\b(?:[A-Z][A-HJ-Y]?[0-9][0-9A-Z]?)\\b$","",\
                                                              regex=True
                                                               ).str.lower()
    
    return address_series_no_pcode

# Remove addresses with no numbers in at all - too high a risk of badly assigning an address
def check_no_number_addresses(df:PandasDataFrame, in_address_series:PandasSeries) -> PandasSeries:
    '''
    Highlight addresses from a pandas df where there are no numbers in the address.
    '''
    df["in_address_series_temp"] = df[in_address_series].str.lower()

    no_numbers_series = df["in_address_series_temp"].str.contains("^(?!.*\d+).*$", regex=True)

    df.loc[no_numbers_series == True, 'Excluded from search'] = "Excluded - no numbers in address"

    df = df.drop("in_address_series_temp", axis = 1)

    print(df[["full_address", "Excluded from search"]])

    return df

# Exclude non-postal addresses
def remove_non_postal(df, in_address_series):
    '''
    Highlight non-postal addresses from a pandas df where a string series that contain specific substrings
    indicating non-postal addresses like 'garage', 'parking', 'shed', etc.
    '''
    df["in_address_series_temp"] = df[in_address_series].str.lower()

    garage_address_series = df["in_address_series_temp"].str.contains("(?i)(?:\\bgarage\\b|\\bgarages\\b)", regex=True)
    parking_address_series = df["in_address_series_temp"].str.contains("(?i)(?:\\bparking\\b)", regex=True)
    shed_address_series = df["in_address_series_temp"].str.contains("(?i)(?:\\bshed\\b|\\bsheds\\b)", regex=True)
    bike_address_series = df["in_address_series_temp"].str.contains("(?i)(?:\\bbike\\b|\\bbikes\\b)", regex=True)
    bicycle_store_address_series = df["in_address_series_temp"].str.contains("(?i)(?:\\bbicycle store\\b|\\bbicycle store\\b)", regex=True)

    non_postal_series = (garage_address_series | parking_address_series | shed_address_series | bike_address_series | bicycle_store_address_series)
    
    df.loc[non_postal_series == True, 'Excluded from search'] = "Excluded - non-postal address"

    df = df.drop("in_address_series_temp", axis = 1)

    return df

def replace_mistaken_dates(df:PandasDataFrame, col:str) -> PandasSeries:
    '''
    Identify addresses that mistakenly have dates in them and replace these dates with number values
    '''
    # Regex pattern to identify the date-month format
    pattern = r'(\d{2})-(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)'
    
    # Dictionary to map month abbreviations to numbers
    month_map = {
        'jan': '1', 'feb': '2', 'mar': '3', 'apr': '4', 'may': '5', 'jun': '6',
        'jul': '7', 'aug': '8', 'sep': '9', 'oct': '10', 'nov': '11', 'dec': '12'
    }
    
    # Custom replacement function
    def replace_month(match):
        day = match.group(1).lstrip('0')  # Get the day and remove leading zeros
        month = month_map[match.group(2)]  # Convert month abbreviation to number
        return f"{day}-{month}"
    
    # Apply the regex replacement
    corrected_addresses = df[col].str.replace(pattern, replace_month, regex = True)

    return corrected_addresses

def merge_series(full_series: pd.Series, partially_filled_series: pd.Series) -> pd.Series:
    '''
    Merge two series. The 'full_series' is the series you want to replace values in
    'partially_filled_series' is the replacer series.
    '''
    replacer_series_is_null = partially_filled_series.isnull()

    # Start with full_series values
    merged_series = full_series.copy()

    # Replace values in merged_series where partially_filled_series is not null
    merged_series[~replacer_series_is_null] = partially_filled_series.dropna()

    return merged_series

def clean_cols(col:str) -> str:
    return col.lower().strip().replace(r" ", "_").strip()