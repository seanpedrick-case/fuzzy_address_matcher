# %%
import urllib
from datetime import datetime
import pandas as pd
import time
import requests

today_rev = datetime.now().strftime("%Y%m%d")


# url = 'https://api.os.uk/search/places/v1/uprn?%s'
# params = urllib.parse.urlencode({'uprn':<UPRN>,'dataset':'LPI', 'key':os.environ["ADDRESSBASE_API_KEY"]})

# Places API
# Technical guide: https://osdatahub.os.uk/docs/places/technicalSpecification


def places_api_query(query, api_key, query_type):

    def make_api_call(url):
        max_retries = 3
        retries = 0

        while retries < max_retries:
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    # If successful response, return the response
                    return response
                elif response.status_code == 429:
                    # If rate limited, wait for 5 seconds before retrying
                    print("Rate limited. Retrying in 5 seconds...")
                    time.sleep(3)
                    retries += 1
                else:
                    # For other errors, return the response
                    return response
            except Exception as e:
                print("Error:", str(e))
                retries += 1

        # If maximum retries reached, return None
        return None

    if api_key:

        overall_tic = time.perf_counter()

        # filter_code_lsc = "LOGICAL_STATUS_CODE:1"
        filter_code_lpi_lsc = "LPI_LOGICAL_STATUS_CODE:1"
        concat_results = []

        if query_type == "Address":
            url = "https://api.os.uk/search/places/v1/find?%s"
            params = urllib.parse.urlencode(
                {
                    "query": query,
                    "dataset": "LPI",
                    "key": api_key,
                    "maxresults": 20,
                    "minmatch": 0.70,  # This includes partial matches
                    "matchprecision": 2,
                    "fq": filter_code_lpi_lsc,
                    "lr": "EN",
                }
            )

            try:
                request_text = url % params
                # print(request_text)
                response = make_api_call(request_text)
            except Exception as e:
                print(str(e))

            if response is not None:
                if response.status_code == 200:
                    # Process the response
                    print("Successful response")
                    # print("Successful response:", response.json())
                else:
                    print("Error:", response.status_code)

            else:
                print("Maximum retries reached. Error occurred.")
                return pd.DataFrame()  # Return blank dataframe

            # Load JSON response
            response_data = response.json()

            # Extract 'results' part
            try:
                results = response_data["results"]
                concat_results.extend(results)

            except Exception as e:
                print(str(e))
                return pd.DataFrame()  # Return blank dataframe

        # If querying postcode, need to use pagination and postcode API
        elif query_type == "Postcode":

            max_results_requested = 100
            remaining_calls = 1
            totalresults = max_results_requested
            call_number = 1

            while remaining_calls > 0 and call_number <= 10:

                offset = (call_number - 1) * max_results_requested

                # print("Remaining to query:", remaining_calls)

                url = "https://api.os.uk/search/places/v1/postcode?%s"
                params = urllib.parse.urlencode(
                    {
                        "postcode": query,
                        "dataset": "LPI",
                        "key": api_key,
                        "maxresults": max_results_requested,
                        "offset": offset,
                        #'fq':filter_code_lsc,
                        "fq": filter_code_lpi_lsc,
                        "lr": "EN",
                    }
                )

                try:
                    request_text = url % params
                    # print(request_text)
                    response = make_api_call(request_text)
                except Exception as e:
                    print(str(e))

                if response is not None:
                    if response.status_code == 200:
                        totalresults = response.json()["header"]["totalresults"]

                        print("Successful response")
                        print("Total results:", totalresults)

                        remaining_calls = totalresults - (
                            max_results_requested * call_number
                        )

                        call_number += 1

                        # Concat results together
                        try:
                            results = response.json()["results"]
                            concat_results.extend(results)
                        except Exception as e:
                            print("Result concat failed with error: ", str(e))
                            concat_results.append(
                                {"invalid_request": True, "POSTCODE_LOCATOR": query}
                            )

                    else:
                        print(
                            "Error:",
                            response.status_code,
                            "For postcode: ",
                            query,
                            " With query: ",
                            request_text,
                        )
                        concat_results.append(
                            {"invalid_request": True, "POSTCODE_LOCATOR": query}
                        )
                        return pd.DataFrame(
                            data={
                                "invalid_request": [True],
                                "POSTCODE_LOCATOR": [query],
                            },
                            index=[0],
                        )  # Return blank dataframe
                else:
                    print("Maximum retries reached. Error occurred.")
                    return pd.DataFrame()  # Return blank dataframe

    else:
        print("No API key provided.")
        return pd.DataFrame()  # Return blank dataframe

    # Convert 'results' to DataFrame

    # Check if 'LPI' sub-branch exists in the JSON response
    # print(concat_results)

    if "LPI" in concat_results[-1]:
        # print("LPI in result columns")
        df = pd.json_normalize(concat_results)
        df.rename(columns=lambda x: x.replace("LPI.", ""), inplace=True)
    else:
        # Normalize the entire JSON data if 'LPI' sub-branch doesn't exist
        df = pd.json_normalize(concat_results)

    # Ensure df is a DataFrame, even if it has a single row
    if isinstance(df, pd.Series):
        print("This is a series!")
        df = df.to_frame().T  # Convert the Series to a DataFrame with a single row

    overall_toc = time.perf_counter()
    time_out = f"The API call took {overall_toc - overall_tic:0.1f} seconds"
    print(time_out)

    return df
