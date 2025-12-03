import os
from datetime import datetime
from pathlib import Path
import gradio as gr
import pandas as pd
import socket

from tools.matcher_funcs import run_matcher
from tools.helper_functions import initial_data_load, ensure_output_folder_exists, get_connection_params, get_or_create_env_var, reveal_feedback_buttons
from tools.aws_functions import load_data_from_aws, upload_file_to_s3
from tools.constants import output_folder
from tools.auth import authenticate_user

import warnings
# Remove warnings from print statements
warnings.filterwarnings("ignore", 'This pattern is interpreted as a regular expression')
warnings.filterwarnings("ignore", 'Downcasting behavior')
warnings.filterwarnings("ignore", 'A value is trying to be set on a copy of a slice from a DataFrame')
warnings.filterwarnings("ignore")

today = datetime.now().strftime("%d%m%Y")
today_rev = datetime.now().strftime("%Y%m%d")

# Base folder is where the code file is stored
base_folder = Path(os.getcwd())
# output_folder = "output/" # This is now defined in constants

ensure_output_folder_exists(output_folder)

host_name = socket.gethostname()

feedback_logs_folder = 'feedback/' + today_rev + '/' + host_name + '/'
access_logs_folder = 'logs/' + today_rev + '/' + host_name + '/'
usage_logs_folder = 'usage/' + today_rev + '/' + host_name + '/'

# Launch the Gradio app
ADDRESSBASE_API_KEY = get_or_create_env_var('ADDRESSBASE_API_KEY', '')

# Create the gradio interface
block = gr.Blocks()

with block:

    data_state = gr.State(pd.DataFrame())
    ref_data_state = gr.State(pd.DataFrame())
    results_data_state = gr.State(pd.DataFrame())
    ref_results_data_state =gr.State(pd.DataFrame())

    session_hash_state = gr.State()
    s3_output_folder_state = gr.State()

    # Logging state
    feedback_logs_state = gr.State(feedback_logs_folder + 'log.csv')
    feedback_s3_logs_loc_state = gr.State(feedback_logs_folder)
    access_logs_state = gr.State(access_logs_folder + 'log.csv')
    access_s3_logs_loc_state = gr.State(access_logs_folder)
    usage_logs_state = gr.State(usage_logs_folder + 'log.csv')
    usage_s3_logs_loc_state = gr.State(usage_logs_folder)   

    gr.Markdown(
    """
    # Address matcher
    Match single or multiple addresses to the reference address file of your choice. *Please note that a postcode column is required for matching*. Fuzzy matching should work on any address columns as long as you specify the postcode column at the end. The neural network component only activates with the in-house neural network model - contact me for details if you have access to AddressBase already. The neural network component works with LLPG files in the LPI format.
    
    The tool can accept csv, xlsx (with one sheet), and parquet files. You need to specify the address columns of the file to match specifically in the address column area with postcode at the end. 
    
    Use the 'New Column' button to create a new cell for each column name. After you have chosen a reference file, an address match file, and specified its address columns (plus postcode), you can press 'Match addresses' to run the tool.""")

    with gr.Tab("Match addresses"):
    
        with gr.Accordion("I have multiple addresses in a CSV/XLSX/Parquet file", open = True):
            in_file = gr.File(label="Input addresses from file", file_count= "multiple")
            in_colnames = gr.Dropdown(value=[], choices=[], multiselect=True, label="Select columns that make up the address. Make sure postcode is at the end")
            in_existing = gr.Dropdown(value=[], choices=[], multiselect=False, label="Select columns that indicate existing matches.")

        with gr.Accordion("Quick check - single address", open = False):
            in_text = gr.Textbox(label="Input a single address as text")
        
        
        gr.Markdown(
        """
        ## Choose reference file / call API
        Upload a reference file to match against, or alternatively call the Addressbase API (requires API key). Fuzzy matching will work on any address format, but the neural network will only work with the LLPG LPI format, e.g. with columns SaoText, SaoStartNumber etc.. This joins on the UPRN column. If any of these are different for you, open 'Custom reference file format or join columns' below.
        """)

        with gr.Accordion("Use Addressbase API (instead of reference file)", open = False):
            in_api = gr.Dropdown(label="Choose API type", multiselect=False, value=None, choices=["Postcode"])#["Postcode", "UPRN"]) #choices=["Address", "Postcode", "UPRN"])
            in_api_key = gr.Textbox(label="Addressbase API key", type='password', value = ADDRESSBASE_API_KEY)


        with gr.Accordion("Match against reference list of addresses in a CSV/XLSX/Parquet file", open = True):
            in_ref = gr.File(label="Input reference addresses from file", file_count= "multiple")
        
        with gr.Accordion("Custom reference file format or join columns (if not LLPG/Addressbase format with columns SaoText, SaoStartNumber etc.)", open = False):
            in_refcol = gr.Dropdown(value=[], choices=[], multiselect=True, label="Select columns that make up the reference address. Make sure postcode is at the end")
            in_joincol = gr.Dropdown(value=[], choices=[], multiselect=True, label="Select columns you want to join on to the search dataset")
        
        match_btn = gr.Button("Match addresses", variant="primary")
        
        with gr.Row():
            output_summary = gr.Textbox(label="Output summary")
            output_file = gr.File(label="Output file")

        feedback_title = gr.Markdown(value="## Please give feedback", visible=False)
        feedback_radio = gr.Radio(choices=["The results were good", "The results were not good"], visible=False)
        further_details_text = gr.Textbox(label="Please give more detailed feedback about the results:", visible=False)
        submit_feedback_btn = gr.Button(value="Submit feedback", visible=False)

        with gr.Row():
            s3_logs_output_textbox = gr.Textbox(label="Feedback submission logs", visible=False)
            # This keeps track of the time taken to match files for logging purposes.
            estimated_time_taken_number = gr.Number(value=0.0, precision=1, visible=False)
            # Invisible text box to hold the session hash/username just for logging purposes
            session_hash_textbox = gr.Textbox(value="", visible=False)

    with gr.Tab(label="Advanced options"):
        with gr.Accordion(label = "AWS data access", open = False):
                aws_password_box = gr.Textbox(label="Password for AWS data access (ask the Data team if you don't have this)")
                with gr.Row():
                    in_aws_file = gr.Dropdown(label="Choose keyword file to load from AWS (only valid for API Gateway app)", choices=["None", "Lambeth address data example file"])
                    load_aws_data_button = gr.Button(value="Load keyword data from AWS", variant="secondary")
                    
                aws_log_box = gr.Textbox(label="AWS data load status")

    ### Loading AWS data ###
    load_aws_data_button.click(fn=load_data_from_aws, inputs=[in_aws_file, aws_password_box], outputs=[in_ref, aws_log_box])

    # Updates to components
    in_file.change(fn = initial_data_load, inputs=[in_file], outputs=[output_summary, in_colnames, in_existing, data_state, results_data_state])
    in_ref.change(fn = initial_data_load, inputs=[in_ref], outputs=[output_summary, in_refcol, in_joincol, ref_data_state, ref_results_data_state])      

    match_btn.click(fn = run_matcher, inputs=[in_text, in_file, in_ref, data_state, results_data_state, ref_data_state, in_colnames, in_refcol, in_joincol, in_existing, in_api, in_api_key],
                    outputs=[output_summary, output_file, estimated_time_taken_number], api_name="address").\
    then(fn = reveal_feedback_buttons, outputs=[feedback_radio, further_details_text, submit_feedback_btn, feedback_title])
    

    # Get connection details on app load
    block.load(get_connection_params, inputs=None, outputs=[session_hash_state, s3_output_folder_state, session_hash_textbox])

    # Log usernames and times of access to file (to know who is using the app when running on AWS)
    access_callback = gr.CSVLogger()
    access_callback.setup([session_hash_textbox], access_logs_folder)
    session_hash_textbox.change(lambda *args: access_callback.flag(list(args)), [session_hash_textbox], None, preprocess=False).\
    then(fn = upload_file_to_s3, inputs=[access_logs_state, access_s3_logs_loc_state], outputs=[s3_logs_output_textbox])

    # User submitted feedback for pdf redactions
    feedback_callback = gr.CSVLogger()
    feedback_callback.setup([feedback_radio, further_details_text, in_file], feedback_logs_folder)
    submit_feedback_btn.click(lambda *args: feedback_callback.flag(list(args)), [feedback_radio, further_details_text, in_file], None, preprocess=False).\
    then(fn = upload_file_to_s3, inputs=[feedback_logs_state, feedback_s3_logs_loc_state], outputs=[further_details_text])

    # Log processing time/token usage when making a query
    usage_callback = gr.CSVLogger()
    usage_callback.setup([session_hash_textbox, in_file, estimated_time_taken_number], usage_logs_folder)
    estimated_time_taken_number.change(lambda *args: usage_callback.flag(list(args)), [session_hash_textbox, in_file, estimated_time_taken_number], None, preprocess=False).\
    then(fn = upload_file_to_s3, inputs=[usage_logs_state, usage_s3_logs_loc_state], outputs=[s3_logs_output_textbox])

# Launch the Gradio app
COGNITO_AUTH = get_or_create_env_var('COGNITO_AUTH', '0')
print(f'The value of COGNITO_AUTH is {COGNITO_AUTH}')

if __name__ == "__main__":
    if os.environ['COGNITO_AUTH'] == "1":
        block.queue().launch(show_error=True, auth=authenticate_user, max_file_size='50mb', theme = gr.themes.Base())
    else:
        block.queue().launch(show_error=True, inbrowser=True, max_file_size='50mb', theme = gr.themes.Base())

