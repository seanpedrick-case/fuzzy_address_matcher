import os
import re

import boto3
import gradio as gr
import pandas as pd
from botocore.exceptions import (
    BotoCoreError,
    ClientError,
    NoCredentialsError,
    PartialCredentialsError,
)

from fuzzy_address_matcher.config import (
    AWS_USER_POOL_ID,
    CUSTOM_HEADER,
    CUSTOM_HEADER_VALUE,
    OUTPUT_FOLDER,
    SESSION_OUTPUT_FOLDER,
    convert_string_to_boolean,
)


def detect_file_type(filename):
    """Detect the file type based on its extension."""
    if (
        (filename.endswith(".csv"))
        | (filename.endswith(".csv.gz"))
        | (filename.endswith(".zip"))
    ):
        return "csv"
    elif filename.endswith(".xlsx"):
        return "xlsx"
    elif filename.endswith(".parquet"):
        return "parquet"
    else:
        raise ValueError("Unsupported file type.")


def read_file(filename):
    """Read the file based on its detected type."""
    file_type = detect_file_type(filename)

    if file_type == "csv":
        return pd.read_csv(filename, low_memory=False)
    elif file_type == "xlsx":
        return pd.read_excel(filename)
    elif file_type == "parquet":
        return pd.read_parquet(filename)


def initial_data_load(in_file):
    new_choices = []
    concat_choices = []
    output_message = ""
    data_file_names_end = ""
    results_df = pd.DataFrame()
    df = pd.DataFrame()

    if not in_file:
        return (
            "No files provided.",
            gr.update(choices=[]),
            gr.update(choices=[]),
            df,
            results_df,
            data_file_names_end,
        )

    file_list = [string.name for string in in_file]

    # Keep this predicate aligned with matcher_funcs.load_match_data_and_filter.
    data_file_names = [
        string for string in file_list if "results_" not in string.lower()
    ]
    # Get the list of file names after last slash in paths
    data_file_names_end = (
        os.path.basename(data_file_names[0]) if data_file_names else ""
    )

    if data_file_names:
        df = read_file(data_file_names[0])
    else:
        error_message = "No data file found."
        return (
            error_message,
            gr.update(choices=concat_choices),
            gr.update(choices=concat_choices),
            df,
            results_df,
            data_file_names_end,
        )

    results_file_names = [
        string for string in file_list if "results_" in string.lower()
    ]
    if results_file_names:
        results_df = read_file(results_file_names[0])

    new_choices = list(df.columns)
    concat_choices.extend(new_choices)

    output_message = "Data successfully loaded"

    return (
        output_message,
        gr.update(choices=concat_choices),
        gr.update(choices=concat_choices),
        df,
        results_df,
        data_file_names_end,
    )


def ensure_output_folder_exists(output_folder):
    """Checks if the output folder exists, creates it if not."""

    folder_name = output_folder

    if not os.path.exists(folder_name):
        # Create the folder if it doesn't exist
        os.makedirs(folder_name)
        print("Created the output folder:", folder_name)
    else:
        print("The output folder already exists:", folder_name)


def dummy_function(in_colnames):
    """
    A dummy function that exists just so that dropdown updates work correctly.
    """
    return None


# Upon running a process, the feedback buttons are revealed
def reveal_feedback_buttons():
    return (
        gr.Radio(visible=True),
        gr.Textbox(visible=True),
        gr.Button(visible=True),
        gr.Markdown(visible=True),
    )


def clear_inputs(in_file, in_ref, in_text):
    return gr.File(value=[]), gr.File(value=[]), gr.Textbox(value="")


## Get final processing time for logs:
def sum_numbers_before_seconds(string):
    """Extracts numbers that precede the word 'seconds' from a string and adds them up.

    Args:
        string: The input string.

    Returns:
        The sum of all numbers before 'seconds' in the string.
    """

    # Extract numbers before 'seconds' using regular expression
    numbers = re.findall(r"(\d+\.\d+)?\s*seconds", string)

    # Extract the numbers from the matches
    numbers = [float(num.split()[0]) for num in numbers]

    # Sum up the extracted numbers
    sum_of_numbers = round(sum(numbers), 1)

    return sum_of_numbers


# async def get_connection_params(request: gr.Request):
#     base_folder = ""

#     if request:
#         # print("request user:", request.username)

#         # request_data = await request.json()  # Parse JSON body
#         # print("All request data:", request_data)
#         # context_value = request_data.get('context')
#         # if 'context' in request_data:
#         #     print("Request context dictionary:", request_data['context'])

#         # print("Request headers dictionary:", request.headers)
#         # print("All host elements", request.client)
#         # print("IP address:", request.client.host)
#         # print("Query parameters:", dict(request.query_params))
#         # To get the underlying FastAPI items you would need to use await and some fancy @ stuff for a live query: https://fastapi.tiangolo.com/vi/reference/request/
#         # print("Request dictionary to object:", request.request.body())
#         # print("Session hash:", request.session_hash)

#         # Retrieving or setting CUSTOM_CLOUDFRONT_HEADER
#         CUSTOM_CLOUDFRONT_HEADER_var = get_or_create_env_var(
#             "CUSTOM_CLOUDFRONT_HEADER", ""
#         )
#         # print(f'The value of CUSTOM_CLOUDFRONT_HEADER is {CUSTOM_CLOUDFRONT_HEADER_var}')

#         # Retrieving or setting CUSTOM_CLOUDFRONT_HEADER_VALUE
#         CUSTOM_CLOUDFRONT_HEADER_VALUE_var = get_or_create_env_var(
#             "CUSTOM_CLOUDFRONT_HEADER_VALUE", ""
#         )
#         # print(f'The value of CUSTOM_CLOUDFRONT_HEADER_VALUE_var is {CUSTOM_CLOUDFRONT_HEADER_VALUE_var}')

#         if CUSTOM_CLOUDFRONT_HEADER_var and CUSTOM_CLOUDFRONT_HEADER_VALUE_var:
#             if CUSTOM_CLOUDFRONT_HEADER_var in request.headers:
#                 supplied_cloudfront_custom_value = request.headers[
#                     CUSTOM_CLOUDFRONT_HEADER_var
#                 ]
#                 if (
#                     supplied_cloudfront_custom_value
#                     == CUSTOM_CLOUDFRONT_HEADER_VALUE_var
#                 ):
#                     print(
#                         "Custom Cloudfront header found:",
#                         supplied_cloudfront_custom_value,
#                     )
#                 else:
#                     raise (
#                         ValueError,
#                         "Custom Cloudfront header value does not match expected value.",
#                     )

#         # Get output save folder from 1 - username passed in from direct Cognito login, 2 - Cognito ID header passed through a Lambda authenticator, 3 - the session hash.

#         if request.username:
#             out_session_hash = request.username
#             base_folder = "user-files/"
#             print("Request username found:", out_session_hash)

#         elif "x-cognito-id" in request.headers:
#             out_session_hash = request.headers["x-cognito-id"]
#             base_folder = "user-files/"
#             print("Cognito ID found:", out_session_hash)

#         else:
#             out_session_hash = request.session_hash
#             base_folder = "temp-files/"
#             # print("Cognito ID not found. Using session hash as save folder:", out_session_hash)

#         output_folder = base_folder + out_session_hash + "/"
#         # if bucket_name:
#         #    print("S3 output folder is: " + "s3://" + bucket_name + "/" + output_folder)

#         return out_session_hash, output_folder, out_session_hash
#     else:
#         print("No session parameters found.")
#         return "", ""


async def get_connection_params(
    request: gr.Request,
    output_folder_textbox: str = OUTPUT_FOLDER,
    session_output_folder: bool = SESSION_OUTPUT_FOLDER,
):
    # Convert session_output_folder to boolean if it's a string (from Gradio Textbox)
    if isinstance(session_output_folder, str):
        session_output_folder = convert_string_to_boolean(session_output_folder)

    if CUSTOM_HEADER and CUSTOM_HEADER_VALUE:
        if CUSTOM_HEADER in request.headers:
            supplied_custom_header_value = request.headers[CUSTOM_HEADER]
            if supplied_custom_header_value == CUSTOM_HEADER_VALUE:
                print("Custom header supplied and matches CUSTOM_HEADER_VALUE")
            else:
                print("Custom header value does not match expected value.")
                raise ValueError("Custom header value does not match expected value.")
        else:
            print("Custom header value not found.")
            raise ValueError("Custom header value not found.")

    # Get output save folder from 1 - username passed in from direct Cognito login, 2 - Cognito ID header passed through a Lambda authenticator, 3 - the session hash.

    if request.username:
        out_session_hash = request.username
        # print("Request username found:", out_session_hash)

    elif "x-cognito-id" in request.headers:
        out_session_hash = request.headers["x-cognito-id"]
        print("Cognito ID found:", out_session_hash)

    elif "x-amzn-oidc-identity" in request.headers:
        out_session_hash = request.headers["x-amzn-oidc-identity"]

        if AWS_USER_POOL_ID:
            try:
                # Fetch email address using Cognito client
                cognito_client = boto3.client("cognito-idp")

                response = cognito_client.admin_get_user(
                    UserPoolId=AWS_USER_POOL_ID,  # Replace with your User Pool ID
                    Username=out_session_hash,
                )
                email = next(
                    attr["Value"]
                    for attr in response["UserAttributes"]
                    if attr["Name"] == "email"
                )
                print("Cognito email address found, will be used as session hash")

                out_session_hash = email
            except (
                ClientError,
                NoCredentialsError,
                PartialCredentialsError,
                BotoCoreError,
            ) as e:
                print(f"Error fetching Cognito user details: {e}")
                print("Falling back to using AWS ID as session hash")
                # out_session_hash already set to the AWS ID from header, so no need to change it
            except Exception as e:
                print(f"Unexpected error when fetching Cognito user details: {e}")
                print("Falling back to using AWS ID as session hash")
                # out_session_hash already set to the AWS ID from header, so no need to change it

        print("AWS ID found, will be used as username for session:", out_session_hash)

    else:
        out_session_hash = request.session_hash

    if session_output_folder:
        output_folder = output_folder_textbox + out_session_hash + "/"

    else:
        output_folder = output_folder_textbox

    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    return (
        out_session_hash,
        output_folder,
        out_session_hash,
    )
