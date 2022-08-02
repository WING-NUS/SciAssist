from typing import Dict, List

import torch
import numpy as np
from src.models.components.bert_tokenizer import bert_tokenizer


def tokenize_and_align_labels(examples, label2id=None):
    '''

    Prepare the dataset for input.
    For token-level task, construct h_mapping to obtain token based BERT representation from subtoken based one.

    Args:
        examples: Dataset, {"tokens":[[s1],[s2]..],"labels":[[l1],[l2]..]}
        label2id: Map label to label_id

    Returns:
        Dict{
            "input_ids":,
            "token_type_ids":,
            "attention_mask":,
            "h_mapping":,
            "labels":,
        }
    '''

    #Get input_ids, token_type_ids, attention_mask
    tokenized_inputs = bert_tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True
    )
    if "labels" in examples.keys():
        raw_labels = examples["labels"]
        #label2id
        labels = [[int(label2id[l]) for l in label] for label in raw_labels]
        tokenized_inputs["labels"] = labels
        #Map sub-token to token
    tokenized_inputs["word_ids"] = []
    for i in range(len(examples["tokens"])):
        tokenized_inputs["word_ids"].append(tokenized_inputs.word_ids(i))

    # Prepare h_mapping for obtaining token based BERT representeation
    # Construct a subtoken to token mapping matrix h_mapping mapping [bsize, max_tok_len, max_subtok_len].
    # For example, in sent i, token j include subtokens[s:t), then mapping[i, j, s:t] = 1 / (t - s)
    # after obtaining subtoken based BERT representation `subtoken_context`[bsize, max_subtok_len, 768], use torch.matmul()
    # to obtain token based BERT representation
    # token_context = torch.matmul(h_mapping, subtoken_context)
    h_mappings = []
    for tokens, word_ids in zip(examples["tokens"], tokenized_inputs["word_ids"]):
        current_tok = 0

        #len(subtok_count) == the length of tokens for input, maybe smaller than origin ones
        #calculate the number of subtokens of a token
        subtok_count = [0]
        for tok_id in word_ids:
            if tok_id == None:
                continue
            if tok_id == current_tok:
                subtok_count[current_tok] += 1
            else:
                current_tok += 1
                subtok_count.append(1)
        #construct h_mapping
        h_mapping =  []
        for i in range(len(subtok_count)):
            h_mapping.append([])
            for j in range(len(word_ids)):
                h_mapping[i].append(0)

        for subtok_id, tok_id in enumerate(word_ids):
            if tok_id == None:
                continue
            h_mapping[tok_id][subtok_id] = 1/subtok_count[tok_id]

        h_mappings.append(h_mapping)

    tokenized_inputs["h_mapping"] = h_mappings
    return tokenized_inputs


def convert_to_list(batch):
    res = []
    for i in batch:
        input_ids = i["input_ids"]
        token_type_ids = i["token_type_ids"]
        attn_mask = i["attention_mask"]
        h_mapping = i["h_mapping"]
        if "labels" in i.keys():
            labels = i["labels"]
            res.append([input_ids, token_type_ids, attn_mask, h_mapping, labels])
        else:
            res.append([input_ids, token_type_ids, attn_mask, h_mapping])
    return res


def pad(batch: List[Dict]):
    # Pads to the longest sample
    batch = convert_to_list(batch)
    get_element = lambda x: [sample[x] for sample in batch]
    #subtoken length
    subtok_len = [len(tokens) for tokens in get_element(0)]
    max_subtok_len = np.array(subtok_len).max()
    #origin token length
    tok_len = [len(tokens) for tokens in get_element(3)]
    max_tok_len = np.array(tok_len).max()

    do_pad = lambda x, seqlen: [sample[x] + [0] * (seqlen - len(sample[x])) for sample in batch]  # 0: <pad>
    do_labels_pad = lambda x, seqlen: [sample[x] + [-100] * (seqlen - len(sample[x])) for sample in batch]

    #pad for origin tokens
    do_map_pad1 = lambda x, seqlen: [sample[x] + [[0]] * (seqlen - len(sample[x])) for sample in batch]
    #pad for subtokens
    do_map_pad2 = lambda batch, seqlen: [[subtoks + [0] * (seqlen - len(subtoks)) for subtoks in sample] for sample in batch]

    input_ids = do_pad(0, max_subtok_len)
    token_type_ids = do_pad(1, max_subtok_len)
    attn_mask = do_pad(2, max_subtok_len)
    h_mapping = do_map_pad1(3, max_tok_len)
    h_mapping = do_map_pad2(h_mapping, max_subtok_len) #[batch_size, max_tok_len, max_subtok_len]

    LT = torch.LongTensor

    input_ids = LT(input_ids)
    attn_mask = LT(attn_mask)
    token_type_ids = LT(token_type_ids)
    h_mapping = torch.Tensor(h_mapping)
    if len(batch[0]) == 5:
        labels = do_labels_pad(4, max_tok_len)
        labels = LT(labels)
    else:
        return {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attn_mask,
            "h_mapping": h_mapping
        }

    return {
        "input_ids": input_ids,
        "token_type_ids": token_type_ids,
        "attention_mask": attn_mask,
        "labels": labels,
        "h_mapping": h_mapping
    }

def postprocess(preds, labels, label_names):
    '''
    Remove `-100` label and mask the padded labels with len(label_names).
    '''
    preds = preds.tolist()
    labels = labels.tolist()
    do_pad = lambda x, seqlen: [x + [len(label_names)] * (seqlen - len(x))]
    true_preds, true_labels = [], []
    for pred, label in zip(preds, labels):
        true_len = 0
        for l in label:
            if l == -100:
                break
            else:
                true_len += 1

        true_preds.append(do_pad(pred[:true_len], len(label)))
        true_labels.append(do_pad(label[:true_len], len(label)))
    true_labels = torch.LongTensor(true_labels)
    true_preds = torch.LongTensor(true_preds)
    return true_preds, true_labels