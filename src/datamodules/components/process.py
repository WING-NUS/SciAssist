import bs4
import torch
import numpy as np
from bs4 import BeautifulSoup
from collections import Counter
from src.models.components.bert_tokenizer import bert_tokenizer


def preprocess(examples):
    all_ref_strings = examples["content"]
    processed_ref_strings = []
    new_labels = []

    for raw_string in all_ref_strings:
        # Removing white spaces in between the strings
        raw_string = raw_string.replace("> <", "><")
        soup = BeautifulSoup(raw_string, 'html.parser')

        processed_ref_string_splitted = []
        processed_ref_string_label = []
        for child in soup.children:
            # If the child is not a tag instance, skip
            if not isinstance(child, bs4.element.Tag):
                continue
            # If the child has no content, skip
            if len(str(child.contents)) == 0:
                continue

            # Get the content and label of this pair
            label = child.name
            data = str(child.contents[0])

            # If nested tag
            while isinstance(data, bs4.element.Tag):
                label = data.name
                data = str(data.contents[0])

            # Have the tokens splitted by white spaces
            tokens = data.split()

            for token in tokens:
                processed_ref_string_splitted.append(token)
                processed_ref_string_label.append(label)

        processed_ref_strings.append(processed_ref_string_splitted)
        new_labels.append(processed_ref_string_label)

    examples["tokens"] = processed_ref_strings
    examples["labels"] = new_labels
    return examples


def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(-100)
        else:
            # Same word as previous token
            label = labels[word_id]
            new_labels.append(label)

    return new_labels


def convert_to_list(batch):
    res = []
    for i in batch:
        input_ids = i["input_ids"]
        token_type_ids = i["token_type_ids"]
        attn_mask = i["attention_mask"]
        word_ids = i["word_ids"]
        labels = i["labels"]
        res.append([input_ids, token_type_ids, attn_mask, word_ids, labels])
    return res


def pad(batch):
    # Pads to the longest sample
    batch = convert_to_list(batch)
    get_element = lambda x: [sample[x] for sample in batch]
    seq_len = [len(tokens) for tokens in get_element(0)]
    maxlen = np.array(seq_len).max()

    do_pad = lambda x, seqlen: [sample[x] + [0] * (seqlen - len(sample[x])) for sample in batch]  # 0: <pad>
    do_word_ids_pad = lambda x, seqlen: [sample[x] + [-1] * (seqlen - len(sample[x])) for sample in batch]
    do_labels_pad = lambda x, seqlen: [sample[x] + [-100] * (seqlen - len(sample[x])) for sample in batch]
    input_ids = do_pad(0, maxlen)
    token_type_ids = do_pad(1, maxlen)
    attn_mask = do_pad(2, maxlen)
    word_ids = do_word_ids_pad(3, maxlen)
    labels = do_labels_pad(4, maxlen)
    LT = torch.LongTensor

    token_ids = get_element(0)
    token_ids_len = torch.LongTensor(list(map(len, token_ids)))
    _, sorted_idx = token_ids_len.sort(0, descending=True)

    input_ids = LT(input_ids)[sorted_idx]
    attn_mask = LT(attn_mask)[sorted_idx]
    token_type_ids = LT(token_type_ids)[sorted_idx]
    word_ids = LT(word_ids)[sorted_idx]
    labels = LT(labels)[sorted_idx]

    return {
        "input_ids": input_ids,
        "token_type_ids": token_type_ids,
        "attention_mask": attn_mask,
        "word_ids": word_ids,
        "labels": labels
    }


def tokenize_and_align_labels(examples, label2id):
    tokenized_inputs = bert_tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True
    )

    all_labels = examples["labels"]
    new_labels = []
    all_word_ids = []
    for i, labels in enumerate(all_labels):
        numerical_labels = list(map(lambda l: int(label2id[l]), labels))
        word_ids = tokenized_inputs.word_ids(i)
        # Replace None with -1 in order to form tensors
        new_labels.append(align_labels_with_tokens(numerical_labels, word_ids))
        word_ids = [word_id if word_id is not None else -1 for word_id in word_ids]
        all_word_ids.append(word_ids)

    tokenized_inputs["labels"] = new_labels
    tokenized_inputs["word_ids"] = all_word_ids

    return tokenized_inputs


def postprocess(word_ids, predictions, labels, label_names):
    label2id = {label: str(i) for i, label in enumerate(label_names)}
    true_word_ids = [[id for id in word_id if id != -1] for word_id in word_ids]
    true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    grouped_true_labels = list()
    grouped_true_predictions = list()

    for word_id, true_label, true_prediction in zip(true_word_ids, true_labels, true_predictions):
        grouped_true_label = list()
        grouped_true_prediction = list()

        current_group_labels = list()
        current_group_predictions = list()

        for i in range(len(word_id)):
            current_group_labels.append(true_label[i])
            current_group_predictions.append(true_prediction[i])

            if i + 1 == len(word_id):
                grouped_true_label.append(current_group_labels)
                grouped_true_prediction.append(current_group_predictions)

            elif word_id[i] != word_id[i + 1]:
                grouped_true_label.append(current_group_labels)
                grouped_true_prediction.append(current_group_predictions)

                current_group_labels = list()
                current_group_predictions = list()

        grouped_true_labels.append(grouped_true_label)
        grouped_true_predictions.append(grouped_true_prediction)

    merged_true_labels = list()
    merged_true_predictions = list()
    for grouped_true_label, grouped_true_prediction in zip(grouped_true_labels, grouped_true_predictions):
        merged_true_label = list(map(lambda l: int(label2id[Counter(l).most_common(1)[0][0]]), grouped_true_label))
        merged_true_prediction = list(map(lambda l: int(label2id[Counter(l).most_common(1)[0][0]]), grouped_true_prediction))
        
        merged_true_labels.append(merged_true_label)
        merged_true_predictions.append(merged_true_prediction)

    max_cols = max([len(lst) for lst in merged_true_labels])

    padded_true_preds = [lst + [len(label_names)] * (max_cols - len(lst)) for lst in merged_true_predictions]
    padded_true_labels = [lst + [len(label_names)] * (max_cols - len(lst)) for lst in merged_true_labels]
    
    true_preds = torch.LongTensor(padded_true_preds)
    true_labels = torch.LongTensor(padded_true_labels)
    return true_preds, true_labels

