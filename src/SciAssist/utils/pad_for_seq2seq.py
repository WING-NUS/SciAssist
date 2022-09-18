import nltk
import numpy as np

from SciAssist.models.components.bart_tokenizer import BartTokenizer, MAX_SOURCE_LENGTH, MAX_TARGET_LENGTH


def tokenize_and_align_labels(examples, inputs_column="text", labels_column="summary"):
    """
    Args:
        examples: Dataset, {"text":[s1,s2..],"summary":[l1,l2..]}
        inputs(str): The name of input column
        labels(str): The name of target column

    Returns:
        Dict{
            "input_ids":,
            "attention_mask":,
            "labels":,
        }
    """
    # Select input column
    inputs = examples[inputs_column]

    # Setup the tokenizer for inputs
    model_inputs = BartTokenizer(inputs, max_length=MAX_SOURCE_LENGTH, padding="max_length", truncation=True)

    # Select target column
    if labels_column in examples.keys():
        labels = examples[labels_column]
        # Setup the tokenizer for targets
        with BartTokenizer.as_target_tokenizer():
            labels = BartTokenizer(labels, max_length=MAX_TARGET_LENGTH, padding="max_length", truncation=True)
            # Ignore padding in the loss
            labels["input_ids"] = [
                [(l if l != BartTokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]
        model_inputs["labels"] = labels["input_ids"]

    return model_inputs

def postprocess(preds, labels):


    decoded_preds = BartTokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.array(labels.to("cpu"))
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, BartTokenizer.pad_token_id)

    decoded_labels = BartTokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]

    # rougeLSum expects newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label)) for label in decoded_labels]

    return decoded_preds, decoded_labels