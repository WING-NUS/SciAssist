from typing import List, Dict

import nltk
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers import DataCollatorForSeq2Seq

from SciAssist import BASE_CACHE_DIR
from SciAssist.datamodules.components.cora_label import label2id as cora_label2id
from SciAssist.models.components.bart_summarization import BartForSummarization


class DataUtilsForSeq2Seq():
    """

    Args:
        tokenizer (`PretrainedTokenizer`, default to None):
            The tokenizer for tokenization.
        checkpoint (`str`):
            The checkpoint from which the tokenizer is loaded.
        model_max_length (`int`, *optional*): The max sequence length the model accepts.
        max_source_length (`int`, *optional*): The max length of the input text.
        max_target_length (`int`, *optional*): The max length of the generated summary.
    """


    def __init__(self, tokenizer = None, model_class = BartForSummarization,
                 checkpoint = "facebook/bart-large-cnn",
                 model_max_length = 1024,
                 max_source_length = 1024,
                 max_target_length = 128,
                 ):

        self.checkpoint = checkpoint
        self.model_max_length = model_max_length
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.model_class = model_class

        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.checkpoint,
                model_max_length = self.model_max_length,
                cache_dir=BASE_CACHE_DIR,
                use_fast=True
            )
        else:
            self.tokenizer = tokenizer


    def tokenize_and_align_labels(self, examples, inputs_column="text", labels_column="summary"):

        """

        Process the dataset for model input, for example, do tokenization and prepare label_ids.

        Args:
            examples (`Dataset`): { "text": [s1, s2, ...], "summary": [l1, l2, ...]}
            inputs (`str`): The name of input column
            labels (`str`): The name of target column

        Returns:
            `Dict`: {"input_ids": input_ids, "attention_mask": attention_mask, "labels": label_ids }

        """

        # Select input column
        inputs = examples[inputs_column]

        # Setup the tokenizer for inputs
        model_inputs = self.tokenizer(inputs, max_length=self.max_source_length, padding="max_length", truncation=True)

        # Select target column
        if labels_column in examples.keys():
            labels = examples[labels_column]
            # Setup the tokenizer for targets
            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(labels, max_length=self.max_target_length, padding="max_length", truncation=True)
                # Ignore padding in the loss
                labels["input_ids"] = [
                    [(l if l != self.tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
                ]
            model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    def collator(self):

        """

        The collating function.

        Returns:
            `function`: A collating function.

            For example, **DataCollatorForSeq2Seq(...)**.

            You can also custom a collating function, but remember that `collator()` needs to return a **function**.
        """

        from SciAssist.models.components.bart_summarization import BartForSummarization

        return DataCollatorForSeq2Seq(self.tokenizer, model=BartForSummarization, pad_to_multiple_of=8)

    def postprocess(self, preds, labels):

        """
        Process model's outputs and get the final results rather than simple ids.

        Args:
            preds (Tensor): Prediction labels, the output of the model.
            labels (Tensor): True labels

        Returns:
            `(LongTensor, LongTensor)`: decoded_preds, decoded_labels

        """

        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)

        labels = np.array(labels.to("cpu"))
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)

        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]

        # rougeLSum expects newline after each sentence
        decoded_preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in decoded_preds]
        decoded_labels = ["\n".join(nltk.sent_tokenize(label)) for label in decoded_labels]

        return decoded_preds, decoded_labels

    def get_dataloader(self, dataset, inputs_column="text", labels_column="summary"):

        """
        Generate DataLoader for a dataset.

        Args:
            dataset (`Dataset`): The raw dataset.
            inputs_column (`str`): Column name of the inputs.
            labels_column (`str`): Column name of the labels.

        Returns:
            `DataLoader`: A dataloader for the dataset. Will be used for inference.
        """

        tokenized_example = dataset.map(
            lambda x: self.tokenize_and_align_labels(x, inputs_column=inputs_column, labels_column=labels_column),
            batched=True,
            remove_columns=dataset.column_names
        )
        dataloader = DataLoader(
            dataset=tokenized_example,
            batch_size=8,
            collate_fn=self.collator(),
        )

        return dataloader


class DataUtilsForTokenClassification():

    def __init__(self, tokenizer = None,
                 checkpoint ="allenai/scibert_scivocab_uncased",
                 model_max_length = 512,
                 ):

        self.checkpoint = checkpoint
        self.model_max_length = model_max_length

        if tokenizer == None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.checkpoint,
                model_max_length=self.model_max_length,
                cache_dir=BASE_CACHE_DIR
            )
        else:
            self.tokenizer = tokenizer

    def tokenize_and_align_labels(self, examples, label2id=None):
        '''

        Prepare the dataset for input.
        For token-level task, construct token_mapping to obtain token based BERT representation from subtoken based one.

        Args:
            examples: Dataset, {"tokens":[[s1],[s2]..],"labels":[[l1],[l2]..]}
            label2id: Map label to label_id

        Returns:
            Dict{
                "input_ids":,
                "token_type_ids":,
                "attention_mask":,
                "token_mapping":,
                "labels":,
            }
        '''

        # Get input_ids, token_type_ids, attention_mask
        tokenized_inputs = self.tokenizer(
            examples["tokens"], truncation=True, is_split_into_words=True
        )
        if "labels" in examples.keys():
            raw_labels = examples["labels"]
            # label2id
            labels = [[int(label2id[l]) for l in label] for label in raw_labels]
            tokenized_inputs["labels"] = labels
            # Map sub-token to token
        tokenized_inputs["word_ids"] = []
        for i in range(len(examples["tokens"])):
            tokenized_inputs["word_ids"].append(tokenized_inputs.word_ids(i))

        # Prepare token_mapping for obtaining token based BERT representeation
        # Construct a subtoken to token mapping matrix token_mapping mapping [bsize, max_tok_len, max_subtok_len].
        # For example, in sent i, token j include subtokens[s:t), then mapping[i, j, s:t] = 1 / (t - s)
        # after obtaining subtoken based BERT representation `subtoken_context`[bsize, max_subtok_len, 768], use torch.matmul()
        # to obtain token based BERT representation
        # token_context = torch.matmul(token_mapping, subtoken_context)
        token_mappings = []
        for tokens, word_ids in zip(examples["tokens"], tokenized_inputs["word_ids"]):
            current_tok = 0

            # len(subtok_count) == the length of tokens for input, maybe smaller than origin ones
            # calculate the number of subtokens of a token
            subtok_count = [0]
            for tok_id in word_ids:
                if tok_id == None:
                    continue
                if tok_id == current_tok:
                    subtok_count[current_tok] += 1
                else:
                    current_tok += 1
                    subtok_count.append(1)
            # construct token_mapping
            token_mapping = []
            for i in range(len(subtok_count)):
                token_mapping.append([])
                for j in range(len(word_ids)):
                    token_mapping[i].append(0)

            for subtok_id, tok_id in enumerate(word_ids):
                if tok_id == None:
                    continue
                token_mapping[tok_id][subtok_id] = 1 / subtok_count[tok_id]

            token_mappings.append(token_mapping)

        tokenized_inputs["token_mapping"] = token_mappings
        return tokenized_inputs

    def convert_to_list(self, batch):
        res = []
        for i in batch:
            input_ids = i["input_ids"]
            token_type_ids = i["token_type_ids"]
            attn_mask = i["attention_mask"]
            token_mapping = i["token_mapping"]
            if "labels" in i.keys():
                labels = i["labels"]
                res.append([input_ids, token_type_ids, attn_mask, token_mapping, labels])
            else:
                res.append([input_ids, token_type_ids, attn_mask, token_mapping])
        return res

    def pad(self, batch: List[Dict]):
        # Pads to the longest sample
        batch = self.convert_to_list(batch)
        get_element = lambda x: [sample[x] for sample in batch]
        # subtoken length
        subtok_len = [len(tokens) for tokens in get_element(0)]
        max_subtok_len = np.array(subtok_len).max()
        # origin token length
        tok_len = [len(tokens) for tokens in get_element(3)]
        max_tok_len = np.array(tok_len).max()

        do_pad = lambda x, seqlen: [sample[x] + [0] * (seqlen - len(sample[x])) for sample in batch]  # 0: <pad>
        do_labels_pad = lambda x, seqlen: [sample[x] + [-100] * (seqlen - len(sample[x])) for sample in batch]

        # pad for origin tokens
        do_map_pad1 = lambda x, seqlen: [sample[x] + [[0]] * (seqlen - len(sample[x])) for sample in batch]
        # pad for subtokens
        do_map_pad2 = lambda batch, seqlen: [[subtoks + [0] * (seqlen - len(subtoks)) for subtoks in sample] for sample in batch]

        input_ids = do_pad(0, max_subtok_len)
        token_type_ids = do_pad(1, max_subtok_len)
        attn_mask = do_pad(2, max_subtok_len)
        token_mapping = do_map_pad1(3, max_tok_len)
        token_mapping = do_map_pad2(token_mapping, max_subtok_len)  # [batch_size, max_tok_len, max_subtok_len]

        LT = torch.LongTensor

        input_ids = LT(input_ids)
        attn_mask = LT(attn_mask)
        token_type_ids = LT(token_type_ids)
        token_mapping = torch.Tensor(token_mapping)
        if len(batch[0]) == 5:
            labels = do_labels_pad(4, max_tok_len)
            labels = LT(labels)
        else:
            return {
                "input_ids": input_ids,
                "token_type_ids": token_type_ids,
                "attention_mask": attn_mask,
                "token_mapping": token_mapping
            }

        return {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attn_mask,
            "labels": labels,
            "token_mapping": token_mapping
        }

    def collator(self):
        return self.pad

    def postprocess(self, preds, labels, label_names):
        '''

        Remove `-100` label and mask the padded labels with len(label_names).
        Args:
            preds (Tensor): Prediction labels
            labels (Tensor): True labels
            label_names (List): Label types

        Returns:
            (LongTensor, LongTensor):

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

    def get_dataloader(self, dataset, label2id = cora_label2id):

        tokenized_example = dataset.map(
            lambda x: self.tokenize_and_align_labels(x, label2id),
            batched=True,
            remove_columns=dataset.column_names
        )
        dataloader = DataLoader(
            dataset=tokenized_example,
            batch_size=8,
            collate_fn=self.collator(),
        )

        return dataloader