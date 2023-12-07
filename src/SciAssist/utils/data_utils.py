import nltk
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from transformers import DataCollatorForSeq2Seq
from typing import List, Dict

from SciAssist import BASE_CACHE_DIR
from SciAssist.datamodules.components.cora_label import label2id as cora_label2id
from SciAssist.models.components.bart_summarization import BartForSummarization
from SciAssist.models.components.flant5_summarization import FlanT5ForSummarization


class MyDatasetExtraction(Dataset):
    def __init__(self, tokenizer, dataset, token_pad_idx = 0, tag_pad_idx = -1):
        self.batch_size = 32
        self.max_len = 128
        self.token_pad_idx = token_pad_idx
        self.tag_pad_idx = tag_pad_idx
        self.tag2idx = {'B-DATA': 0, 'I-DATA': 1, 'O': 2}
        self.idx2tag = {0: 'B-DATA', 1: 'I-DATA', 2: 'O'}
        self.tokenizer = tokenizer
        self.dataset = self.preprocess(dataset)

     
    def __len__(self):
        """get dataset size"""
        return self.dataset['size']
 

    def __getitem__(self, idx):
        """sample data to get batch"""
        sentences = self.dataset['data'][idx]

        if 'labels' in self.dataset.keys():
            labels = self.dataset['labels'][idx]
            tags = self.dataset['tags'][idx]
            return [sentences, labels, tags]
        else:
            return [sentences]


    def preprocess(self, dataset):
        """Loads sentences and tags from their corresponding files. 
            Maps tokens and tags to their indices and stores them in the provided dict d.
        """
        processed_dataset = {}
        sentences = []
        if 'labels' in dataset.keys():
            labels = []
            tags = []

        for line in dataset['data']['text']:
            # replace each token by its index
            tokens = line.strip().split(' ')
            # print(tokens)
            subwords = list(map(self.tokenizer.tokenize, tokens)) # 每个词切分成子词
            # print(subwords)
            subword_lengths = list(map(len, subwords)) # 记录子词的长度，用于对齐tag
            # print(subword_lengths)
            subwords = ['[CLS]'] + [item for indices in subwords for item in indices]
            # subwords = ['<s>'] + [item for indices in subwords for item in indices] # 组成输入 token
            # print(subwords)
            token_start_idxs = 1 + np.cumsum([0] + subword_lengths[:-1]) # 记录每个token开始的位置
            # print(token_start_idxs)
            # print(self.tokenizer.convert_tokens_to_ids(subwords), token_start_idxs)
            sentences.append((self.tokenizer.convert_tokens_to_ids(subwords), token_start_idxs))

        if 'labels' in dataset.keys():
            for line in dataset['labels']['text']:
                labels.append(int(line.strip()))
            assert len(sentences) == len(labels)
            processed_dataset['labels'] = labels

            for line in dataset['tags']['text']:
                # replace each tag by its index
                tag_seq = [self.tag2idx.get(tag) for tag in line.strip().split(' ')]
                tags.append(tag_seq)
            # checks to ensure there is a tag for each token
            assert len(sentences) == len(tags)
            for i in range(len(sentences)):
                assert len(tags[i]) == len(sentences[i][-1])
            processed_dataset['tags'] = tags

        processed_dataset['data'] = sentences
        processed_dataset['size'] = len(sentences)

        return processed_dataset


    def collate_fn(self, batch):
        sentences = [x[0] for x in batch]
        processed_batch = {}
        if len(batch[0]) == 3:
            labels = [x[1] for x in batch]
            tags = [x[2] for x in batch]

        # batch length
        batch_len = len(sentences)  # batch size
        batch_max_subwords_len = max([len(s[0]) for s in sentences])
        max_subword_len = min(batch_max_subwords_len, self.max_len)
        max_token_len = 0
 
        # padding data 初始化
        batch_data = self.token_pad_idx * np.ones((batch_len, max_subword_len))
        batch_token_starts = []
 
        # padding and aligning
        for j in range(batch_len):
            cur_subwords_len = len(sentences[j][0])  # word_id list
            if cur_subwords_len <= max_subword_len:
                batch_data[j][:cur_subwords_len] = sentences[j][0]
            else:
                batch_data[j] = sentences[j][0][:max_subword_len]
            token_start_ids = sentences[j][-1]
            token_starts = np.zeros(max_subword_len)
            token_starts[[idx for idx in token_start_ids if idx < max_subword_len]] = 1
            batch_token_starts.append(token_starts)
            max_token_len = max(int(sum(token_starts)), max_token_len)

        processed_batch['input_subwords'] = torch.tensor(batch_data, dtype = torch.long)
        processed_batch['input_token_start_indexs'] = torch.tensor(np.array(batch_token_starts), dtype = torch.long)
        # processed_batch['attention_mask'] = (processed_batch['input_subwords'] != 1)
        processed_batch['attention_mask'] = processed_batch['input_subwords'].gt(0)

        if len(batch[0]) == 3:
            batch_tags = self.tag_pad_idx * np.ones((batch_len, max_token_len))
            batch_labels = np.ones((batch_len, ))
            for j in range(batch_len):
                batch_labels[j] = labels[j]
                cur_tags_len = len(tags[j])
                if cur_tags_len <= max_token_len:
                    batch_tags[j][:cur_tags_len] = tags[j]
                else:
                    batch_tags[j] = tags[j][:max_token_len]
            processed_batch['ner_tags'] = torch.tensor(np.array(batch_tags), dtype = torch.long)
            processed_batch['cls_labels'] = torch.tensor(np.array(batch_labels), dtype = torch.long)

        return processed_batch


class DataUtilsForDatasetExtraction():
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
    def __init__(self, tokenizer = None,
                 checkpoint = "allenai/scibert_scivocab_uncased",
                 model_max_length = 128
                 ):

        self.checkpoint = checkpoint
        self.model_max_length = model_max_length

        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.checkpoint,
                model_max_length = self.model_max_length,
                cache_dir=BASE_CACHE_DIR,
                use_fast=True
            )
        else:
            self.tokenizer = tokenizer

        self.tag2idx = {'B-DATA': 0, 'I-DATA': 1, 'O': 2}
        self.idx2tag = {0: 'B-DATA', 1: 'I-DATA', 2: 'O'}


    def tokenize_and_align_labels(self, dataset):

        """

        Process the dataset for model input, for example, do tokenization and prepare label_ids.

        Args:
            dataset (`Dataset`): { "text": [s1, s2, ...], "summary": [l1, l2, ...]}
            inputs (`str`): The name of input column
            labels (`str`): The name of target column

        Returns:
            `Dict`: {"input_ids": input_ids, "attention_mask": attention_mask, "labels": label_ids }

        """
        
        """Loads sentences and tags from their corresponding files.
            Maps tokens and tags to their indices and stores them in the provided dict d.
        """
        processed_dataset = MyDatasetExtraction(self.tokenizer, dataset)

        return processed_dataset


    def postprocess(self, ner_output, cls_output, batch_tags, batch_labels):

        """
        Process model's outputs and get the final results rather than simple ids.

        Args:
            preds (Tensor): Prediction labels, the output of the model.
            labels (Tensor): True labels

        Returns:
            `(LongTensor, LongTensor)`: decoded_preds, decoded_labels

        """
        pred_tags = []
        true_tags = []
        pred_labels = []
        true_labels = []

        ner_output = ner_output.detach().cpu().numpy()
        cls_output = cls_output.detach().cpu().numpy()
        batch_tags = batch_tags.to('cpu').numpy()
        batch_labels = batch_labels.to('cpu').numpy()

        pred_tags.extend([[self.idx2tag.get(idx) for idx in indices] for indices in np.argmax(ner_output, axis=2)])
        true_tags.extend([[self.idx2tag.get(idx) if idx != -1 else 'O' for idx in indices] for indices in batch_tags])
        true_labels.extend(batch_labels)

        pred_labels.extend(cls_output)
        pred_labels = np.argmax(pred_labels, axis=1)

        assert len(pred_tags) == len(true_tags)

        return pred_tags, true_tags, pred_labels, true_labels


    def get_dataloader(self, dataset):

        """
        Generate DataLoader for a dataset.

        Args:
            dataset (`Dataset`): The raw dataset.
            inputs_column (`str`): Column name of the inputs.
            labels_column (`str`): Column name of the labels.

        Returns:
            `DataLoader`: A dataloader for the dataset. Will be used for inference.
        """
        tokenized_dataset = self.tokenize_and_align_labels(dataset)

        dataloader = DataLoader(
            dataset=tokenized_dataset,
            batch_size=32,
            collate_fn=tokenized_dataset.collate_fn,
        )

        return dataloader


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


        return DataCollatorForSeq2Seq(self.tokenizer, model=self.model_class, pad_to_multiple_of=8)

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



class DataUtilsForT5():
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


    def __init__(self, tokenizer = None,
                 model_class = FlanT5ForSummarization,
                 checkpoint = "facebook/bart-large-cnn",
                 prompt = "Please give a summary of the following text: ",
                 model_max_length = 1024,
                 max_source_length = 1024,
                 max_target_length = 128,
                 ):

        self.checkpoint = checkpoint
        self.model_max_length = model_max_length
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.model_class = model_class
        self.prompt = prompt

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
        inputs = [self.prompt + raw_text for raw_text in inputs]
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


        return DataCollatorForSeq2Seq(self.tokenizer, model=self.model_class, pad_to_multiple_of=8)

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


class DataUtilsForFlanT5():
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


    def __init__(self, tokenizer = None,
                 model_class = FlanT5ForSummarization,
                 checkpoint = "google/flan-t5-base",
                 prompt = "Please give a summary of the following text ",
                 model_max_length = 1024,
                 max_source_length = 1024,
                 max_target_length = 500,
                 ):

        self.checkpoint = checkpoint
        self.model_max_length = model_max_length
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.model_class = model_class
        self.prompt = prompt



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

        inputs = examples[inputs_column]
        prompts = [ self.prompt for i in inputs ]

        # kw_instructions = ["{}"]
        def kw(prompt, keyword):
            # i = randint(1,10)
            if keyword is not None and keyword != [""]:
                return "Keywords: [ " + str(", ".join(keyword)) + " ]. " + prompt + "based on these keywords " if keyword is not None else prompt
            return prompt

        def leng(prompt, length):
            if length is not None:
                return prompt + ", which has less than " + str(length) + " words " if length is not None else prompt
            return prompt

        if "keywords" in examples.keys():
            keywords = examples["keywords"]
            if keywords is not None:
                prompts = [ kw(prompt,keyword) for (prompt,keyword) in zip(prompts,keywords) ]

        if "length" in examples.keys():
            if examples["length"] is not None:
                lengths = examples["length"]
                prompts = [ leng(prompt,length) for (prompt,length) in zip(prompts,lengths)]

        inputs = [ prompt + ": " + raw_text for (prompt,raw_text) in zip(prompts, inputs) ]
        

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

        #when test
        # if "id" in examples.keys():
        #     model_inputs["id"] = examples["id"]
        # if "length" in examples.keys() and examples["length"] is not None:
        #     if examples["length"][0] is not None:
        #         model_inputs["length"] = examples["length"]
        return model_inputs

    def collator(self):

        """

        The collating function.

        Returns:
            `function`: A collating function.

            For example, **DataCollatorForSeq2Seq(...)**.

            You can also custom a collating function, but remember that `collator()` needs to return a **function**.
        """


        return DataCollatorForSeq2Seq(self.tokenizer, model=self.model_class, pad_to_multiple_of=8)

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


# class DataUtilsForFiD():
#     """
#
#     Args:
#         tokenizer (`PretrainedTokenizer`, default to None):
#             The tokenizer for tokenization.
#         checkpoint (`str`):
#             The checkpoint from which the tokenizer is loaded.
#         model_max_length (`int`, *optional*): The max sequence length the model accepts.
#         max_source_length (`int`, *optional*): The max length of the input text.
#         max_target_length (`int`, *optional*): The max length of the generated summary.
#     """
#
#     def __init__(self, tokenizer = None, model_class = FiDT5,
#                  checkpoint = "google/flan-t5-large",
#                  model_max_length = 64,
#                  max_source_length = 64,
#                  max_target_length = 128,
#                  ):
#
#         self.checkpoint = checkpoint
#         self.model_max_length = model_max_length
#         self.max_source_length = max_source_length
#         self.max_target_length = max_target_length
#         self.model_class = model_class
#
#         if tokenizer is None:
#             self.tokenizer = AutoTokenizer.from_pretrained(
#                 self.checkpoint,
#                 model_max_length = self.model_max_length,
#                 cache_dir=BASE_CACHE_DIR,
#             )
#         else:
#             self.tokenizer = tokenizer
#
#
#     def tokenize_and_align_labels(self, examples, inputs_column="text", labels_column="summary", token_per_paragraph=50):
#
#         """
#
#         Process the dataset for model input, for example, do tokenization and prepare label_ids.
#
#         Args:
#             examples (`Dataset`): { "text": [s1, s2, ...], "summary": [l1, l2, ...]}
#             inputs (`str`): The name of input column
#             labels (`str`): The name of target column
#
#         Returns:
#             `Dict`: {"input_ids": input_ids, "attention_mask": attention_mask, "labels": label_ids }
#
#         """
#
#         # Select input column
#         inputs = examples[inputs_column]
#         dataset = {"paragraphs": []}
#
#         for input in inputs:
#             texts = ["Please give a summary of the following text: "]
#             tokens = input.split(" ")
#             index = 0
#             while index+token_per_paragraph < min(len(tokens),120*token_per_paragraph):
#                 p = " ".join(tokens[index:index+token_per_paragraph])
#                 texts.append(p)
#                 index += token_per_paragraph
#             texts.append(" ".join(tokens[index:index+token_per_paragraph]))
#             dataset["paragraphs"].append(texts)
#
#
#         # Select target column
#         if labels_column in examples.keys():
#             labels = examples[labels_column]
#             dataset["labels"] = labels
#
#         return dataset
#
#     def collator(self):
#
#         """
#
#         The collating function.
#
#         Returns:
#             `function`: A collating function.
#
#             For example, **DataCollatorForSeq2Seq(...)**.
#
#             You can also custom a collating function, but remember that `collator()` needs to return a **function**.
#         """
#
#         from SciAssist.utils.collators.CollatorForFid import DataCollatorForFid
#
#         return DataCollatorForFid(self.max_source_length, self.tokenizer, self.max_target_length)
#
#     def postprocess(self, preds, labels):
#
#         """
#         Process model's outputs and get the final results rather than simple ids.
#
#         Args:
#             preds (Tensor): Prediction labels, the output of the model.
#             labels (Tensor): True labels
#
#         Returns:
#             `(LongTensor, LongTensor)`: decoded_preds, decoded_labels
#
#         """
#
#         decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
#
#         labels = np.array(labels.to("cpu"))
#         # Replace -100 in the labels as we can't decode them.
#         labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
#
#         decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
#
#         decoded_preds = [pred.strip() for pred in decoded_preds]
#         decoded_labels = [label.strip() for label in decoded_labels]
#
#         # rougeLSum expects newline after each sentence
#         decoded_preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in decoded_preds]
#         decoded_labels = ["\n".join(nltk.sent_tokenize(label)) for label in decoded_labels]
#
#         return decoded_preds, decoded_labels
#
#     def get_dataloader(self, dataset, inputs_column="text", labels_column="summary"):
#
#         """
#         Generate DataLoader for a dataset.
#
#         Args:
#             dataset (`Dataset`): The raw dataset.
#             inputs_column (`str`): Column name of the inputs.
#             labels_column (`str`): Column name of the labels.
#
#         Returns:
#             `DataLoader`: A dataloader for the dataset. Will be used for inference.
#         """
#
#         tokenized_example = dataset.map(
#             lambda x: self.tokenize_and_align_labels(x, inputs_column=inputs_column, labels_column=labels_column),
#             batched=True,
#             remove_columns=dataset.column_names
#         )
#         dataloader = DataLoader(
#             dataset=tokenized_example,
#             batch_size=8,
#             collate_fn=self.collator(),
#         )
#
#         return dataloader
#
#
#
# class DataUtilsForFrost():
#     """
#
#     Args:
#         tokenizer (`PretrainedTokenizer`, default to None):
#             The tokenizer for tokenization.
#         checkpoint (`str`):
#             The checkpoint from which the tokenizer is loaded.
#         model_max_length (`int`, *optional*): The max sequence length the model accepts.
#         max_source_length (`int`, *optional*): The max length of the input text.
#         max_target_length (`int`, *optional*): The max length of the generated summary.
#     """
#
#
#     def __init__(self, tokenizer = None, model_class = FrostForSummarization,
#                  checkpoint = "pegasus/frost",
#                  model_max_length = 1024,
#                  max_source_length = 1024,
#                  max_target_length = 128,
#                  ):
#
#         self.checkpoint = checkpoint
#         self.model_max_length = model_max_length
#         self.max_source_length = max_source_length
#         self.max_target_length = max_target_length
#         self.model_class = model_class
#
#         if tokenizer is None:
#             self.tokenizer = PegasusTokenizer.from_pretrained(
#                 self.checkpoint,
#                 cache_dir=BASE_CACHE_DIR,
#                 model_max_length=self.model_max_length,
#             )
#         else:
#             self.tokenizer = tokenizer
#
#         # FROST Constants
#         self.ENTITYCHAIN_START_TOKEN = "[CONTENT]"
#         self.SUMMARY_START_TOKEN = "[SUMMARY]"
#         self.ENTITY_SEPARATOR = " | "
#         self.ENTITY_SENTENCE_SEPARATOR = " ||| "
#
#         # Prepare Spacy processor
#         self.SPACY_MODEL_OR_PATH = "en_core_web_sm"
#         self.SPACY_PROCESSOR = spacy.load(self.SPACY_MODEL_OR_PATH)
#
#     def get_frost_labels(self, text):
#         """Gets Spacy Frost processor."""
#         entity_plans = []
#         for text_sent in self.SPACY_PROCESSOR(text.replace("\n", " ")).sents:
#             entity_plans.append(
#                 self.ENTITY_SEPARATOR.join(
#                     [entity.text for entity in self.SPACY_PROCESSOR(text_sent.text).ents]))
#         text_with_entityplans = (
#                 self.ENTITYCHAIN_START_TOKEN + " " +
#                 self.ENTITY_SENTENCE_SEPARATOR.join(entity_plans) + " " +
#                 self.SUMMARY_START_TOKEN + " " + text)
#         return text_with_entityplans
#
#
#     def tokenize_and_align_labels(self, examples, inputs_column="text", labels_column="summary"):
#
#         """
#
#         Process the dataset for model input, for example, do tokenization and prepare label_ids.
#
#         Args:
#             examples (`Dataset`): { "text": [s1, s2, ...], "summary": [l1, l2, ...]}
#             inputs (`str`): The name of input column
#             labels (`str`): The name of target column
#
#         Returns:
#             `Dict`: {"input_ids": input_ids, "attention_mask": attention_mask, "labels": label_ids }
#
#         """
#
#         # Select input column
#         inputs = examples[inputs_column]
#
#         # Setup the tokenizer for inputs
#         model_inputs = self.tokenizer(inputs, max_length=self.max_target_length, padding="max_length", truncation=True)
#
#         # Select target column
#         if labels_column in examples.keys():
#             labels = examples[labels_column]
#             labels = [self.get_frost_labels(label) for label in labels]
#
#             # Setup the tokenizer for targets
#             with self.tokenizer.as_target_tokenizer():
#                 labels = self.tokenizer(labels, max_length=self.max_target_length, padding="max_length", truncation=True)
#                 # Ignore padding in the loss
#                 labels["input_ids"] = [
#                     [(l if l != self.tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
#                 ]
#
#             model_inputs["labels"] = labels["input_ids"]
#
#         return model_inputs
#
#     def collator(self):
#
#         """
#
#         The collating function.
#
#         Returns:
#             `function`: A collating function.
#
#             For example, **DataCollatorForSeq2Seq(...)**.
#
#             You can also custom a collating function, but remember that `collator()` needs to return a **function**.
#         """
#
#
#         return DataCollatorForSeq2Seq(self.tokenizer, model=self.model_class, pad_to_multiple_of=8)
#
#     def postprocess(self, preds, labels):
#
#         """
#         Process model's outputs and get the final results rather than simple ids.
#
#         Args:
#             preds (Tensor): Prediction labels, the output of the model.
#             labels (Tensor): True labels
#
#         Returns:
#             `(LongTensor, LongTensor)`: decoded_preds, decoded_labels
#
#         """
#
#         decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
#
#         labels = np.array(labels.to("cpu"))
#         # Replace -100 in the labels as we can't decode them.
#         labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
#
#         decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=False)
#
#         decoded_preds = [pred.strip() for pred in decoded_preds]
#         decoded_labels = [label.strip() for label in decoded_labels]
#
#         # rougeLSum expects newline after each sentence
#         decoded_preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in decoded_preds]
#         decoded_labels = ["\n".join(nltk.sent_tokenize(label)) for label in decoded_labels]
#
#         return decoded_preds, decoded_labels
#
#     def get_dataloader(self, dataset, inputs_column="text", labels_column="summary"):
#
#         """
#         Generate DataLoader for a dataset.
#
#         Args:
#             dataset (`Dataset`): The raw dataset.
#             inputs_column (`str`): Column name of the inputs.
#             labels_column (`str`): Column name of the labels.
#
#         Returns:
#             `DataLoader`: A dataloader for the dataset. Will be used for inference.
#         """
#
#         tokenized_example = dataset.map(
#             lambda x: self.tokenize_and_align_labels(x, inputs_column=inputs_column, labels_column=labels_column),
#             batched=True,
#             remove_columns=dataset.column_names
#         )
#         dataloader = DataLoader(
#             dataset=tokenized_example,
#             batch_size=8,
#             collate_fn=self.collator(),
#         )
#
#         return dataloader

class DataUtilsForT5():
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


    def __init__(self, tokenizer = None,
                 model_class = BartForSummarization,
                 checkpoint = "facebook/bart-large-cnn",
                 prompt = "Please give a summary of the following text: ",
                 model_max_length = 1024,
                 max_source_length = 1024,
                 max_target_length = 128,
                 ):

        self.checkpoint = checkpoint
        self.model_max_length = model_max_length
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.model_class = model_class
        self.prompt = prompt

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
        inputs = [self.prompt + raw_text for raw_text in inputs]
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


        return DataCollatorForSeq2Seq(self.tokenizer, model=self.model_class, pad_to_multiple_of=8)

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



class DataUtilsForBart():
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


    def __init__(self, tokenizer = None,
                 model_class = FlanT5ForSummarization,
                 checkpoint = "google/flan-t5-base",
                 prompt = "Please give a summary of the following text ",
                 model_max_length = 1024,
                 max_source_length = 1024,
                 max_target_length = 500,
                 ):

        self.checkpoint = checkpoint
        self.model_max_length = model_max_length
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.model_class = model_class
        self.prompt = prompt



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

        inputs = examples[inputs_column]
        prompts = [ "" for i in inputs ]

        # kw_instructions = ["{}"]
        def kw(prompt, keyword):
            # i = randint(1,10)
            if keyword is not None and keyword != [""]:
                return str(" | ".join(keyword)) if keyword is not None else prompt
            return prompt

        def leng(prompt, length):
            if length is not None:
                return prompt + ", which has less than " + str(length) + " words " if length is not None else prompt
            return prompt

        if "keywords" in examples.keys():
            keywords = examples["keywords"]
            if keywords is not None:
                prompts = [ kw(prompt,keyword) for (prompt,keyword) in zip(prompts,keywords) ]

        if "length" in examples.keys():
            if examples["length"] is not None:
                lengths = examples["length"]
                prompts = [ leng(prompt,length) for (prompt,length) in zip(prompts,lengths)]

        inputs = [ prompt + ": " + raw_text for (prompt,raw_text) in zip(prompts, inputs) ]

        print(inputs[0][:200])
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

        #when test
        if "id" in examples.keys():
            model_inputs["id"] = examples["id"]
        if "length" in examples.keys() and examples["length"] is not None:
            if examples["length"][0] is not None:
                model_inputs["length"] = examples["length"]
        return model_inputs

    def collator(self):

        """

        The collating function.

        Returns:
            `function`: A collating function.

            For example, **DataCollatorForSeq2Seq(...)**.

            You can also custom a collating function, but remember that `collator()` needs to return a **function**.
        """


        return DataCollatorForSeq2Seq(self.tokenizer, model=self.model_class, pad_to_multiple_of=8)

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