---
title: Fuzzy address matching
emoji: 🌍
colorFrom: purple
colorTo: gray
sdk: gradio
sdk_version: 6.10.0
app_file: app.py
pinned: false
license: apache-2.0
---

# Introduction 
Fuzzy match a dataset with an LLPG dataset in the LPI format (with columns SaoText, SaoStartNumber etc.). Address columns are concatenated together to form a single string address. Important details are extracted by regex (e.g. flat, house numbers, postcodes). Addresses may be 'standardised' in a number of ways; e.g. variations of words used for 'ground floor' such as 'grd' or 'grnd' are replaced with 'ground floor' to give a more consistent address wording. This has been found to increase match rates.

Then the two datasets are compared with fuzzy matching. The closest fuzzy matches are selected, and then a post hoc test compares flat/property numbers to ensure a 'full match'.

If the LLPG reference file is in the standard LPI format, the neural net model should then initialise. This will break down the addresses to match into a list of sub address fields in the LLPG LPI format. It will then do exact or fuzzy comparisons of each address to the LLPG dataset to find closest matches. The neural net is capable of blocking on postcode and on street name, which is where most of the new matches are found according to testing.

The final files will appear in the relevant output boxes, which you can download.


