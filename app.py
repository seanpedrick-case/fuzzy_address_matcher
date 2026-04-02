import os
import socket
import warnings
from datetime import datetime
from pathlib import Path

import gradio as gr
import pandas as pd

from tools.auth import authenticate_user
from tools.aws_functions import upload_file_to_s3
from tools.config import (
    ACCESS_LOG_DYNAMODB_TABLE_NAME,
    ADDRESSBASE_API_KEY,
    COGNITO_AUTH,
    CSV_ACCESS_LOG_HEADERS,
    CSV_FEEDBACK_LOG_HEADERS,
    CSV_USAGE_LOG_HEADERS,
    DYNAMODB_ACCESS_LOG_HEADERS,
    DYNAMODB_FEEDBACK_LOG_HEADERS,
    DYNAMODB_USAGE_LOG_HEADERS,
    FEEDBACK_LOG_DYNAMODB_TABLE_NAME,
    FEEDBACK_LOG_FILE_NAME,
    LOG_FILE_NAME,
    SAVE_LOGS_TO_CSV,
    SAVE_LOGS_TO_DYNAMODB,
    SHOW_FEEDBACK,
    USAGE_LOG_DYNAMODB_TABLE_NAME,
    USAGE_LOG_FILE_NAME,
    output_folder,
)
from tools.custom_csvlogger import CSVLogger_custom
from tools.helper_functions import (
    ensure_output_folder_exists,
    get_connection_params,
    initial_data_load,
    reveal_feedback_buttons,
)
from tools.matcher_funcs import run_matcher

# Remove warnings from print statements
warnings.filterwarnings("ignore", "This pattern is interpreted as a regular expression")
warnings.filterwarnings("ignore", "Downcasting behavior")
warnings.filterwarnings(
    "ignore", "A value is trying to be set on a copy of a slice from a DataFrame"
)
warnings.filterwarnings("ignore")

today = datetime.now().strftime("%d%m%Y")
today_rev = datetime.now().strftime("%Y%m%d")

# Base folder is where the code file is stored
base_folder = Path(os.getcwd())
# output_folder comes from tools/config.py (GRADIO_OUTPUT_FOLDER)

ensure_output_folder_exists(output_folder)

host_name = socket.gethostname()

feedback_logs_folder = "feedback/" + today_rev + "/" + host_name + "/"
access_logs_folder = "logs/" + today_rev + "/" + host_name + "/"
usage_logs_folder = "usage/" + today_rev + "/" + host_name + "/"

# Launch the Gradio app
# Create the gradio interface
block = gr.Blocks(
    analytics_enabled=False,
    title="Fuzzy address matcher",
    delete_cache=(43200, 43200),  # Temporary file cache deleted every 12 hours
    fill_width=False,
)

with block:

    data_state = gr.State(pd.DataFrame())
    ref_data_state = gr.State(pd.DataFrame())
    results_data_state = gr.State(pd.DataFrame())
    ref_results_data_state = gr.State(pd.DataFrame())

    session_hash_state = gr.State()
    s3_output_folder_state = gr.State()

    # Logging state
    feedback_logs_state = gr.State(feedback_logs_folder + "log.csv")
    feedback_s3_logs_loc_state = gr.State(feedback_logs_folder)
    access_logs_state = gr.State(access_logs_folder + "log.csv")
    access_s3_logs_loc_state = gr.State(access_logs_folder)
    usage_logs_state = gr.State(usage_logs_folder + "log.csv")
    usage_s3_logs_loc_state = gr.State(usage_logs_folder)

    s3_logs_output_textbox = gr.State()
    estimated_time_taken_number = gr.State()
    session_hash_textbox = gr.State()

    data_file_names_end = gr.State()
    ref_data_file_names_end = gr.State()

    gr.Markdown(
        """# Fuzzy address matcher
    Match single or multiple addresses to a reference / canonical dataset. The tool can accept CSV, XLSX (with one sheet), and Parquet files.

    After you have chosen a reference file, an address match file, and specified its address columns, click 'Match addresses' to run the tool.
    
    Fuzzy matching should work on any address columns. If you have a postcode column, place this at the end of the list of address columns. If a postcode is not present in the address, the app will use street-only blocking. Ensure to untick the 'Use postcode blocker' checkbox to use street-only blocking.

    Note that this app is based on UK address data, and all output should be checked by a human before further use. """
    )

    with gr.Tab("Match addresses"):

        with gr.Accordion(
            "Match multiple addresses in a CSV/XLSX/Parquet file", open=True
        ):
            in_file = gr.File(label="Input addresses from file", file_count="multiple")
            in_colnames = gr.Dropdown(
                value=[],
                choices=[],
                multiselect=True,
                label="Select columns that make up the address. If you provide a postcode column, place this at the end of the list of address columns.",
            )
            in_existing = gr.Dropdown(
                value=[],
                choices=[],
                multiselect=False,
                label="Select columns that indicate existing matches.",
            )
            use_postcode_blocker = gr.Checkbox(
                label="Use postcode as blocker (untick to use street-only blocking). Advised to untick only if you don't have a postcode column.",
                value=True,
            )

        with gr.Accordion("Single address input", open=False):
            in_text = gr.Textbox(label="Input a single address as text")

        gr.Markdown("""
        ## Choose reference file / call API
        Upload a reference file to match against, or alternatively call the Addressbase API (requires API key). Fuzzy matching will work on any address format, but the neural network will only work with the LLPG LPI format, e.g. with columns SaoText, SaoStartNumber etc.. This joins on the UPRN column. If any of these are different for you, open 'Custom reference file format or join columns' below.
        """)

        with gr.Accordion(
            "Use Addressbase API (instead of reference file)", open=False
        ):
            in_api = gr.Dropdown(
                label="Choose API type",
                multiselect=False,
                value=None,
                choices=["Postcode"],
            )  # ["Postcode", "UPRN"]) #choices=["Address", "Postcode", "UPRN"])
            in_api_key = gr.Textbox(
                label="Addressbase API key", type="password", value=ADDRESSBASE_API_KEY
            )

        with gr.Accordion(
            "Match against reference list of addresses in a CSV/XLSX/Parquet file",
            open=True,
        ):
            in_ref = gr.File(
                label="Input reference addresses from file", file_count="multiple"
            )

        with gr.Accordion(
            "Custom reference file format or join columns (if not LLPG/Addressbase format with columns SaoText, SaoStartNumber etc.)",
            open=False,
        ):
            in_refcol = gr.Dropdown(
                value=[],
                choices=[],
                multiselect=True,
                label="Select columns that make up the reference address. Make sure postcode is at the end",
            )
            in_joincol = gr.Dropdown(
                value=[],
                choices=[],
                multiselect=True,
                label="Select columns you want to join on to the search dataset",
            )

        match_btn = gr.Button("Match addresses", variant="primary")

        with gr.Row(equal_height=True):
            with gr.Column(scale=2):
                output_summary = gr.Markdown(
                    value="Output summary will appear here", container=True, buttons=["copy"]
                )
            with gr.Column(scale=3):
                output_file = gr.File(label="Output file")

        output_summary_table_md = gr.Markdown(
            value="Match summary table will appear here", buttons=["copy"]
        )

        feedback_accordion = gr.Accordion(
            label="Please give feedback", open=False, visible=SHOW_FEEDBACK
        )
        with feedback_accordion:
            feedback_title = gr.Markdown(value="## Please give feedback", visible=False)
            feedback_radio = gr.Radio(
                choices=["The results were good", "The results were not good"],
                visible=False,
            )
            further_details_text = gr.Textbox(
                label="Please give more detailed feedback about the results:",
                visible=False,
            )
            submit_feedback_btn = gr.Button(value="Submit feedback", visible=False)

    # Updates to components
    in_file.change(
        fn=initial_data_load,
        inputs=[in_file],
        outputs=[
            output_summary,
            in_colnames,
            in_existing,
            data_state,
            results_data_state,
            data_file_names_end,
        ],
    )
    in_ref.change(
        fn=initial_data_load,
        inputs=[in_ref],
        outputs=[
            output_summary,
            in_refcol,
            in_joincol,
            ref_data_state,
            ref_results_data_state,
            ref_data_file_names_end,
        ],
    )

    match_btn.click(
        fn=run_matcher,
        inputs=[
            in_text,
            in_file,
            in_ref,
            data_state,
            results_data_state,
            ref_data_state,
            in_colnames,
            in_refcol,
            in_joincol,
            in_existing,
            in_api,
            in_api_key,
            use_postcode_blocker,
        ],
        outputs=[
            output_summary,
            output_file,
            estimated_time_taken_number,
            output_summary_table_md,
        ],
        api_name="address",
        show_progress_on=[output_summary],
    ).then(
        fn=reveal_feedback_buttons,
        outputs=[
            feedback_radio,
            further_details_text,
            submit_feedback_btn,
            feedback_title,
        ],
    )

    # Get connection details on app load
    block.load(
        get_connection_params,
        inputs=None,
        outputs=[session_hash_state, s3_output_folder_state, session_hash_textbox],
    )

    # Log usernames and times of access to file (to know who is using the app when running on AWS)
    access_callback = CSVLogger_custom(dataset_file_name=LOG_FILE_NAME)
    access_callback.setup([session_hash_textbox], access_logs_folder)
    session_hash_textbox.change(
        fn=lambda *args: access_callback.flag(
            list(args),
            save_to_csv=SAVE_LOGS_TO_CSV,
            save_to_dynamodb=SAVE_LOGS_TO_DYNAMODB,
            dynamodb_table_name=ACCESS_LOG_DYNAMODB_TABLE_NAME,
            dynamodb_headers=DYNAMODB_ACCESS_LOG_HEADERS,
            replacement_headers=CSV_ACCESS_LOG_HEADERS,
        ),
        inputs=[session_hash_textbox],
        outputs=None,
    ).then(
        fn=upload_file_to_s3,
        inputs=[access_logs_state, access_s3_logs_loc_state],
        outputs=[s3_logs_output_textbox],
    )

    # User submitted feedback for pdf redactions
    feedback_callback = CSVLogger_custom(dataset_file_name=FEEDBACK_LOG_FILE_NAME)
    feedback_callback.setup(
        [feedback_radio, further_details_text, in_file], feedback_logs_folder
    )
    submit_feedback_btn.click(
        fn=lambda *args: feedback_callback.flag(
            list(args),
            save_to_csv=SAVE_LOGS_TO_CSV,
            save_to_dynamodb=SAVE_LOGS_TO_DYNAMODB,
            dynamodb_table_name=FEEDBACK_LOG_DYNAMODB_TABLE_NAME,
            dynamodb_headers=DYNAMODB_FEEDBACK_LOG_HEADERS,
            replacement_headers=CSV_FEEDBACK_LOG_HEADERS,
        ),
        inputs=[feedback_radio, further_details_text, in_file],
        outputs=None,
    ).then(
        fn=upload_file_to_s3,
        inputs=[feedback_logs_state, feedback_s3_logs_loc_state],
        outputs=[further_details_text],
    )

    # Log processing time/token usage when making a query
    usage_callback = CSVLogger_custom(dataset_file_name=USAGE_LOG_FILE_NAME)
    usage_callback.setup(
        [
            session_hash_textbox,
            data_file_names_end,
            ref_data_file_names_end,
            estimated_time_taken_number,
        ],
        usage_logs_folder,
    )
    estimated_time_taken_number.change(
        fn=lambda *args: usage_callback.flag(
            list(args),
            save_to_csv=SAVE_LOGS_TO_CSV,
            save_to_dynamodb=SAVE_LOGS_TO_DYNAMODB,
            dynamodb_table_name=USAGE_LOG_DYNAMODB_TABLE_NAME,
            dynamodb_headers=DYNAMODB_USAGE_LOG_HEADERS,
            replacement_headers=CSV_USAGE_LOG_HEADERS,
        ),
        inputs=[
            session_hash_textbox,
            data_file_names_end,
            ref_data_file_names_end,
            estimated_time_taken_number,
        ],
        outputs=None,
    ).then(
        fn=upload_file_to_s3,
        inputs=[usage_logs_state, usage_s3_logs_loc_state],
        outputs=[s3_logs_output_textbox],
    )

# Launch the Gradio app

# print(f"The value of COGNITO_AUTH is {COGNITO_AUTH}")

if __name__ == "__main__":
    if COGNITO_AUTH:
        block.queue().launch(
            show_error=True,
            auth=authenticate_user,
            max_file_size="50mb",
            theme=gr.themes.Base(),
        )
    else:
        block.queue().launch(
            show_error=True,
            inbrowser=True,
            max_file_size="50mb",
            theme=gr.themes.Base(),
        )
