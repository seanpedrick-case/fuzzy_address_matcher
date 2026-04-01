# import tensorflow as tf # Tensorflow use deprecated
import torch
import pandas as pd
import numpy as np
from typing import Type, Dict, List, Tuple
from datetime import datetime

PandasDataFrame = Type[pd.DataFrame]
PandasSeries = Type[pd.Series]
MatchedResults = Dict[str, Tuple[str, int]]
array = List[str]

today = datetime.now().strftime("%d%m%Y")
today_rev = datetime.now().strftime("%Y%m%d")

# # Neural net functions


def vocab_lookup(characters: str, vocab) -> (int, np.ndarray):
    """
    Taken from the function from the addressnet package by Jason Rigby

    Converts a string into a list of vocab indices
    :param characters: the string to convert
    :param training: if True, artificial typos will be introduced
    :return: the string length and an array of vocab indices
    """
    result = list()
    for c in characters.lower():
        try:
            result.append(vocab.index(c) + 1)
        except ValueError:
            result.append(0)
    return len(characters), np.array(result, dtype=np.int64)


# ## Neural net predictor functions


def text_to_model_input_local(in_text, vocab, model_type="estimator"):
    addresses_out = []
    model_input_out = []
    encoded_text = []

    # Calculate longest string length
    import heapq

    # get the index of the largest element in the list
    index = heapq.nlargest(1, range(len(in_text)), key=lambda x: len(in_text[x]))[0]

    # use the index to get the corresponding string
    len(in_text[index])

    # print("Longest string is: " + str(longest_string))

    for x in range(0, len(in_text)):

        out = vocab_lookup(in_text[x], vocab)
        addresses_out.append(out)

        # print(out)

        # Tensorflow model use deprecated
        # if model_type == "estimator":
        #     model_input_add= tf.train.Example(features=tf.train.Features(feature={
        #     'lengths': tf.train.Feature(int64_list=tf.train.Int64List(value=[out[0]])),
        #     'encoded_text': tf.train.Feature(int64_list=tf.train.Int64List(value=out[1].tolist()))
        #     })).SerializeToString()

        #     model_input_out.append(model_input_add)

        if model_type == "keras":
            encoded_text.append(out[1])

    # Tensorflow model use deprecated
    # if model_type == "keras":
    #     # Pad out the strings so they're all the same length. 69 seems to be the value for spaces
    #     model_input_out = tf.keras.utils.pad_sequences(encoded_text, maxlen=longest_string, padding="post", truncating="post", value=0)#69)

    return addresses_out, model_input_out


def reformat_predictions_local(predict_out):

    predictions_list_reformat = []

    for x in range(0, len(predict_out["pred_output_classes"])):

        new_entry = {
            "class_ids": predict_out["pred_output_classes"][x],
            "probabilities": predict_out["probabilities"][x],
        }
        predictions_list_reformat.append(new_entry)

    return predictions_list_reformat


def predict_serve_conv_local(
    in_text: List[str], labels_list, predictions
) -> List[Dict[str, str]]:

    class_names = [label.replace("_code", "") for label in labels_list]
    class_names = [label.replace("_abbreviation", "") for label in class_names]

    # print(input_text)

    # print(list(zip(input_text, predictions)))

    for addr, res in zip(in_text, predictions):

        # print(zip(input_text, predictions))

        mappings = dict()

        # print(addr.upper())
        # print(res['class_ids'])

        for char, class_id in zip(addr.upper(), res["class_ids"]):
            # print(char)
            if class_id == 0:
                continue
            cls = class_names[class_id - 1]
            mappings[cls] = mappings.get(cls, "") + char

        # print(mappings)
        yield mappings
        # return mappings


def prep_predict_export(prediction_outputs, in_text):

    out_list = list(prediction_outputs)

    df_out = pd.DataFrame(out_list)

    # print(in_text)
    # print(df_out)

    df_out["address"] = in_text

    return out_list, df_out


def full_predict_func(list_to_predict, model, vocab, labels_list):

    if hasattr(
        model, "summary"
    ):  # Indicates this is a keras model rather than an estimator
        model_type = "keras"
    else:
        model_type = "estimator"

    list_to_predict = [x.upper() for x in list_to_predict]

    addresses_out, model_input = text_to_model_input_local(
        list_to_predict, vocab, model_type
    )

    if hasattr(model, "summary"):
        probs = model.predict(model_input, use_multiprocessing=True)

        classes = probs.argmax(axis=-1)

        predictions = {"pred_output_classes": classes, "probabilities": probs}

    else:
        print("Tensorflow use deprecated")
        # predictions = model.signatures["predict_output"](predictor_inputs=tf.constant(model_input)) # This was for when using the contrib module
        # predictions = model.signatures["serving_default"](predictor_inputs=tf.constant(model_input))

    predictions_list_reformat = reformat_predictions_local(predictions)

    #### Final output as list or dataframe

    output = predict_serve_conv_local(
        list(list_to_predict), labels_list, predictions_list_reformat
    )

    list_out, predict_df = prep_predict_export(output, list_to_predict)

    # Add organisation as a column if it doesn't already exist
    if "Organisation" not in predict_df.columns:
        predict_df["Organisation"] = ""

    return list_out, predict_df


# -


def predict_torch(model, model_type, input_text, word_to_index, device):
    # print(device)

    # Convert input_text to tensor of character indices
    indexed_texts = [
        [word_to_index.get(char, word_to_index["<UNK>"]) for char in text]
        for text in input_text
    ]

    # Calculate max_len based on indexed_texts
    max_len = max(len(text) for text in indexed_texts)

    # Pad sequences and convert to tensor
    padded_texts = torch.tensor(
        [
            text + [word_to_index["<pad>"]] * (max_len - len(text))
            for text in indexed_texts
        ]
    )

    with torch.no_grad():
        texts = padded_texts.to(device)

        if (model_type == "lstm") | (model_type == "gru"):
            text_lengths = texts.ne(word_to_index["<pad>"]).sum(dim=1)
            predictions = model(texts, text_lengths)

        if model_type == "transformer":
            # Call model with texts and pad_idx
            predictions = model(texts, word_to_index["<pad>"])

    # Convert predictions to most likely category indices
    _, predicted_indices = predictions.max(2)
    return predicted_indices


def torch_predictions_to_dicts(input_text, predicted_indices, index_to_category):
    results = []
    for i, text in enumerate(input_text):
        # Treat each character in the input text as a "token"
        tokens = list(text)  # Convert string to a list of characters

        # Create a dictionary for the current text
        curr_dict = {}

        # Iterate over the predicted categories and the tokens together
        for category_index, token in zip(predicted_indices[i], tokens):
            # Convert the category index to its name
            category_name = index_to_category[category_index.item()]

            # Append the token to the category in the dictionary (or create the category if it doesn't exist)
            if category_name in curr_dict:
                curr_dict[category_name] += token  # No space needed between characters
            else:
                curr_dict[category_name] = token

        results.append(curr_dict)

    return results


def torch_prep_predict_export(prediction_outputs, in_text):

    # out_list = list(prediction_outputs)

    df_out = pd.DataFrame(prediction_outputs).drop("IGNORE", axis=1)

    # print(in_text)
    # print(df_out)

    df_out["address"] = in_text

    return df_out


def full_predict_torch(
    model, model_type, input_text, word_to_index, cat_to_idx, device
):

    input_text = [x.upper() for x in input_text]

    predicted_indices = predict_torch(
        model, model_type, input_text, word_to_index, device
    )

    index_to_category = {v: k for k, v in cat_to_idx.items()}

    results_dict = torch_predictions_to_dicts(
        input_text, predicted_indices, index_to_category
    )

    df_out = torch_prep_predict_export(results_dict, input_text)

    return results_dict, df_out


def post_predict_clean(
    predict_df, orig_search_df, ref_address_cols, search_df_key_field
):

    # Add address to ref_address_cols
    ref_address_cols_add = ref_address_cols.copy()
    ref_address_cols_add.extend(["address"])

    # Create column if it doesn't exist
    for x in ref_address_cols:

        predict_df[x] = predict_df.get(x, np.nan)

    predict_df = predict_df[ref_address_cols_add]

    # Columns that are in the ref and model, but are not matched in this instance, need to be filled in with blanks

    predict_cols_match = list(predict_df.drop(["address"], axis=1).columns)
    predict_cols_match_uprn = predict_cols_match.copy()
    predict_cols_match_uprn.append("UPRN")

    pred_output_missing_cols = list(set(ref_address_cols) - set(predict_cols_match))
    predict_df[pred_output_missing_cols] = np.nan
    predict_df = predict_df.fillna("").infer_objects(copy=False)

    # Convert all columns to string

    all_columns = list(predict_df)  # Creates list of all column headers
    predict_df[all_columns] = predict_df[all_columns].astype(str)

    predict_df = predict_df.replace(r"\.0", "", regex=True)

    # When comparing with ref, the postcode existing in the data will be used to compare rather than the postcode predicted by the model. This is to minimise errors in matching

    predict_df = predict_df.rename(columns={"Postcode": "Postcode_predict"})

    orig_search_df_pc = (
        orig_search_df[[search_df_key_field, "postcode"]]
        .rename(columns={"postcode": "Postcode"})
        .reset_index(drop=True)
    )
    predict_df = predict_df.merge(
        orig_search_df_pc, left_index=True, right_index=True, how="left"
    )

    predict_df[search_df_key_field] = predict_df[search_df_key_field].astype(str)

    return predict_df
