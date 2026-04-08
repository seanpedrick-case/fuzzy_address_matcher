# Installation

Requires Python 3.10 or newer.

## Installing from pypi

Install the latest release from PyPI:

```bash
pip install fuzzy_address_matcher
```

This installation supports both Python-script usage and the GUI console command.

## Use in a Python script

Import the matcher function:

```python
from tools.matcher_funcs import fuzzy_address_match
```

### 1) Match using external CSV files

Pass file paths for your search dataset and reference dataset.

```python
from tools.matcher_funcs import fuzzy_address_match

final_summary, output_files, estimated_seconds, summary_table_md = fuzzy_address_match(
    in_file="example_data/search_addresses_london.csv",
    in_ref="example_data/reference_addresses_london.csv",
    in_colnames=["address_line_1", "address_line_2", "postcode"],
    in_refcol=["addr1", "addr2", "addr3", "addr4", "postcode"],
    in_joincol=None,
    output_folder="outputs",
)

print(final_summary)
print(output_files)
print(summary_table_md)
```

### 2) Match using DataFrames already loaded in Python

If your data is already in memory, pass DataFrames directly with `search_df` and `ref_df`.

```python
from tools.matcher_funcs import fuzzy_address_match

# Assume search_df and ref_df already exist in your Python session.
final_summary, output_files, estimated_seconds, summary_table_md = fuzzy_address_match(
    search_df=search_df,
    ref_df=ref_df,
    in_colnames=["address_line_1", "address_line_2", "postcode"],
    in_refcol=["addr1", "addr2", "addr3", "addr4", "postcode"],
    in_joincol=None,
    output_folder="outputs",
)

print(final_summary)
print(output_files)
print(summary_table_md)
```

## Run the GUI app

If you installed from PyPI, you can run the Gradio GUI via the console script:

```bash
fuzzy-address-matcher
```

Or, to run from source, clone the repo and run it from the project root:

```bash
git clone https://github.com/seanpedrick-case/fuzzy_address_matcher.git
cd fuzzy_address_matcher
pip install -e .
python app.py
```

Further details on use can be found in the [User guide (GitHub Pages)](https://seanpedrick-case.github.io/fuzzy_address_matcher/src/user_guide.html) (source: [`src/user_guide.qmd`](src/user_guide.qmd)).

# Introduction 
Match single or multiple addresses to a reference / canonical dataset. The tool can accept CSV, XLSX (with one sheet), and Parquet files. After you have chosen a reference file, an address match file, and specified its address columns, click 'Match addresses' to run the tool.
    
Fuzzy matching should work on any address columns. If you have a postcode column, place this at the end of the list of address columns. If a postcode is not present in the address, the app will use street-only blocking. Ensure to untick the 'Use postcode blocker' checkbox to use street-only blocking. The final files will appear in the relevant output boxes, which you can download. Note that this app is based on UK address data.

Note that this app is based on UK address data. Matching is unlikely to be 100% accurate, so outputs should be checked by a human before further use. 

## Method

Address columns are concatenated together to form a single string address. Important details are extracted by regex (e.g. flat, house numbers, postcodes). Addresses may be 'standardised' in a number of ways; e.g. variations of words used for 'ground floor' such as 'grd' or 'grnd' are replaced with 'ground floor' to give a more consistent address wording. This has been found to increase match rates. Then the two datasets are compared with fuzzy matching. The closest fuzzy matches are selected, and then a post hoc test compares flat/property numbers to ensure a 'full match'.

## Important note
I suggest that this app should be used in conjunction with the excellent [uk_address_matcher package](https://github.com/moj-analytical-services/uk_address_matcher). I am finding that this package is great for ~95% of matches with uk addresses. However, the repo here (fuzzy_address_matcher) uses slightly different methods for matching (address standardisation, fuzzy matching), and so, as of April 2026, it can still pick up some new matches.

My suggested workflow would be:

1. Match your datasets with the uk_address_matcher package, then 
2. Run the output file through this app for further address matches that can be picked up by the standardisation / fuzzy matching

Further details on use can be found in the [User guide](src/user_guide.qmd).





