# Fuzzy Address Match Test Fixtures

This folder contains two synthetic London datasets for testing `fuzzy_address_match`:

- `search_addresses_london.csv` (10 rows): columns `address_line_1,address_line_2,postcode`
- `reference_addresses_london.csv` (20 rows): columns `addr1,addr2,addr3,addr4,postcode`

Expected overlap:

- 8 search addresses are present in reference data
- 2 search addresses are intentionally not present (for non-match testing)

## Example usage with file paths

```python
from tools.matcher_funcs import fuzzy_address_match

msg, output_files, est, _ = fuzzy_address_match(
    in_file="example_data/search_addresses_london.csv",
    in_ref="example_data/reference_addresses_london.csv",
    in_colnames=["address_line_1", "address_line_2", "postcode"],
    in_refcol=["addr1", "addr2", "addr3", "addr4", "postcode"],
)
```

## Example usage with dataframes

```python
import pandas as pd
from tools.matcher_funcs import fuzzy_address_match

search_df = pd.read_csv("example_data/search_addresses_london.csv")
ref_df = pd.read_csv("example_data/reference_addresses_london.csv")

msg, output_files, est, _ = fuzzy_address_match(
    search_df=search_df,
    ref_df=ref_df,
    in_colnames=["address_line_1", "address_line_2", "postcode"],
    in_refcol=["addr1", "addr2", "addr3", "addr4", "postcode"],
)
```
