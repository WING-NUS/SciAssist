import nltk
import numpy as np

from src.models.components.bart_tokenizer import bart_tokenizer, MAX_SOURCE_LENGTH, MAX_TARGET_LENGTH


def tokenize_and_align_labels(examples, inputs_column="text", labels_column="summary"):
    """
    Args:
        examples: Dataset, {"text":[[s1],[s2]..],"summary":[[l1],[l2]..]}
        inputs(str): The name of input column
        labels(str): The name of target column

    Returns:
        Dict{
            "input_ids":,
            "attention_mask":,
            "labels":,
        }
    """
    # Select input column and target column

    inputs = examples[inputs_column]
    labels = examples[labels_column]

    # Setup the tokenizer for inputs
    model_inputs = bart_tokenizer(inputs, max_length=MAX_SOURCE_LENGTH, padding="max_length", truncation=True)

    # Setup the tokenizer for targets
    with bart_tokenizer.as_target_tokenizer():
        labels = bart_tokenizer(labels, max_length=MAX_TARGET_LENGTH, padding="max_length", truncation=True)
        # Ignore padding in the loss
        labels["input_ids"] = [
            [(l if l != bart_tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]
    model_inputs["labels"] = labels["input_ids"]

    return model_inputs

def postprocess(preds, labels):


    decoded_preds = bart_tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.array(labels.to("cpu"))
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, bart_tokenizer.pad_token_id)

    decoded_labels = bart_tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]

    # rougeLSum expects newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label)) for label in decoded_labels]

    return decoded_preds, decoded_labels