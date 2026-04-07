---
title: Fuzzy address matching
emoji: 🌍
colorFrom: purple
colorTo: gray
sdk: gradio
sdk_version: 6.11.0
app_file: app.py
pinned: false
license: apache-2.0
---

## Installation

Requires Python 3.10 or newer.

Install the project locally:

```bash
pip install -e .
```

Install with optional extras:

```bash
# Neural-net matching (PyTorch)
pip install -e ".[nnet]"

# Development tools
pip install -e ".[dev]"

# Test dependencies
pip install -e ".[test]"
```

Run the app:

```bash
python app.py
```
# Important note
I suggest that this app should be used in conjunction with the excellent [uk_address_matcher package](https://github.com/moj-analytical-services/uk_address_matcher). I am finding that this package is great for ~95% of matches with uk addresses. However, the repo here (fuzzy_address_matcher) uses slightly different methods for matching (address standardisation, fuzzy matching), and so, as of April 2026, it can still pick up some new matches.

My suggested workflow would be:

1. Match your datasets with the uk_address_matcher package, then 
2. Run the output file through this app for further address matches that can be picked up by the standardisation / fuzzy matching

Further details below and in the [User guide](src/user_guide.qmd)

# Introduction 
Fuzzy match a dataset with an LLPG dataset in the LPI format (with columns SaoText, SaoStartNumber etc.). Address columns are concatenated together to form a single string address. Important details are extracted by regex (e.g. flat, house numbers, postcodes). Addresses may be 'standardised' in a number of ways; e.g. variations of words used for 'ground floor' such as 'grd' or 'grnd' are replaced with 'ground floor' to give a more consistent address wording. This has been found to increase match rates.

Then the two datasets are compared with fuzzy matching. The closest fuzzy matches are selected, and then a post hoc test compares flat/property numbers to ensure a 'full match'.

If the LLPG reference file is in the standard LPI format, the neural net model should then initialise. This will break down the addresses to match into a list of sub address fields in the LLPG LPI format. It will then do exact or fuzzy comparisons of each address to the LLPG dataset to find closest matches. The neural net is capable of blocking on postcode and on street name, which is where most of the new matches are found according to testing.

The final files will appear in the relevant output boxes, which you can download. Note that this app is based on UK address data. Matching is unlikely to be 100% accurate, so outputs should be checked by a human before further use.




