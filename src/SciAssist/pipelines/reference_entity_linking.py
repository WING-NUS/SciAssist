import json
import os
from typing import List, Tuple, Optional, Union, Dict

from datasets import Dataset
from transformers import PreTrainedTokenizer

from SciAssist import BASE_OUTPUT_DIR, BASE_TEMP_DIR, BASE_CACHE_DIR
from SciAssist.datamodules.components.cora_label import LABEL_NAMES
from SciAssist.datamodules.components.cora_label import label2id
from SciAssist.pipelines.pipeline import Pipeline
from SciAssist.utils.pdf2text import process_pdf_file
from SciAssist.utils import windows_pdf2text

from SciAssist import ReferenceStringParsing
import enchant
import datetime
import re
import time
import requests



class ReferenceEntityLinkingLocal(Pipeline):
    """
    The pipeline for reference entity linking (Local Version, No Internet Connection Required).

    Args:
        model_name (`str`, *optional*):
            A string, the *model name* of a pretrained model provided for this task.
        device (`str`, *optional*):
            A string, `cpu` or `gpu`.
        cache_dir (`str` or `os.PathLike`, *optional*):
            Path to a directory in which a downloaded pretrained model should be
            cached if the standard cache should not be used.
        output_dir (`str` or `os.PathLike`, *optional*):
            Path to a directory in which the predicted results files should be stored.
        temp_dir (`str` or `os.PathLike`, *optional*):
            Path to a directory which holds temporary files such as `.tei.xml`.
        tokenizer (PreTrainedTokenizer, *optional*):
            A specific tokenizer.
        checkpoint (`str` or `os.PathLike`, *optional*):
            A checkpoint for the tokenizer. You can also specify the `checkpoint` while
            using the default tokenizer.
            Can be either:

                - A string, the *model id* of a predefined tokenizer hosted inside a model repo on huggingface.co.
                  Valid model ids can be located at the root-level, like `bert-base-uncased`, or namespaced under a
                  user or organization name, like `allenai/scibert_scivocab_uncased`.
                - A path to a *directory* containing vocabulary files required by the tokenizer, for instance saved
                  using the [`~PreTrainedTokenizer.save_pretrained`] method, e.g., `./my_model_directory/`.
                - A path or url to a single saved vocabulary file if and only if the tokenizer only requires a
                  single vocabulary file (like Bert or XLNet), e.g.: `./my_model_directory/vocab.txt`. (Not
                  applicable to all derived classes)

        model_max_length (`int`, *optional*): The max sequence length the model accepts.
    """

    def __init__(
            self,
            model_name: Optional[str] = "default",
            device: Optional[str] = "gpu",
            cache_dir = None,
            output_dir = None,
            temp_dir = None,
            tokenizer = None,
            checkpoint="allenai/scibert_scivocab_uncased",
            model_max_length=512,
            os_name=None,
    ):
        self.model_name = model_name
        self.device = device
        self.cache_dir = cache_dir if cache_dir is not None else BASE_CACHE_DIR
        self.output_dir = output_dir if output_dir is not None else BASE_OUTPUT_DIR
        self.temp_dir = temp_dir if temp_dir is not None else BASE_TEMP_DIR
        self.tokenizer = tokenizer
        self.checkpoint = checkpoint
        self.model_max_length = model_max_length
        self.os_name = os_name if os_name != None else os.name
        self.utils = Utils()

    def predict(
            self, files_dir,
            output_dir=None,
            temp_dir=None,
            save_results=True,
    ):

        """

        Args:
            files_dir (`str` or `os.PathLike`):
                Path to a directory in which the PDF files should be stored.
            output_dir (`str` or `os.PathLike`, *optional*):
                Path to a directory in which the predicted results files should be stored.
                If not provided, it will use the `output_dir` set for the pipeline.
            temp_dir (`str` or `os.PathLike`, *optional*):
                Path to a directory which holds temporary files such as `.tei.xml`.
                If not provided, it will use the `temp_dir` set for the pipeline.
            save_results (`bool`, default to `True`):
                Whether to save the results in a *.json* file.

        Returns:
            `List[Dict]`: [{"reference_id|reference_title": [[reference_id, reference_title], [reference_id, reference_title], ... ]}, ...]


        Examples:

            >>> from SciAssist import ReferenceEntityLinkingLocal
            >>> pipeline = ReferenceEntityLinkingLocal()
            >>> pipeline.predict(
            ...     './sample_papers'
            ... )

            [
                {
                    "reference_key": "0_6|bert: pre-training of deep bidirectional transformers for language understanding",
                    "cited_references": [
                        ["0_6", "bert: pre-training of deep bidirectional transformers for language understanding"],
                        ["2_11", "bert: pretraining of deep bidirectional transformers for lan-guage understanding"]
                        ]
                    "cited_number": cited_number
                },
                // ... Another JSON objects
            ]

        """

        if output_dir is None:
            output_dir = self.output_dir
        if temp_dir is None:
            temp_dir = self.temp_dir
        ref_lines = self._batch_pdf_to_referencess(files_dir)
        results = self._ref_clustering(ref_lines)

        # Save predicted results as a text file
        if save_results:
            output_path = os.path.join(output_dir, f"local_reference_entity_linking_{datetime.datetime.now()}.txt")
            self.utils.json_list_to_file(results, output_path)

        return results

    def _batch_pdf_to_referencess(self, files_path):
        """
        Parse references from multiple PDF files stored in the files_path.

        Args:
            files_path (`str`): Path to a directory in which the PDF files should be stored.
        Returns:
            `List[Dict]`: Reference information, where each element represents a reference in the form of a Dict.
        """

        def pdf_to_references(file_path, ref_parser):
            """
            Invoke ReferenceStringParsing to parse references for an individual PDF-formatted paper and process them into block format.

            Returns:
                `List[List[str,str]]`: List of reference information, where each element represents details such as authors, date, title, etc. for each reference.
                Example:
                    [
                        [["Federico Bianchi, Debora Nozza, and Dirk Hovy.", "author"], ["2022.", "date"], ["XLM-EMO: Multilingual Emotion Prediction in So- cial Media Text.", "title"], ["In Proceedings of the 12th Work- shop on Computational Approaches to Subjectivity, Sentiment and Social Media Analysis.", "booktitle"], ["Association for Computational Linguistics.", "publisher"]],
                        [["Bianchi, F.; Nozza, D.; Hovy, D.", "author"], ["XLM-EMO: Multilingual emotion prediction in social media text.", "title"], ["In Proceedings of the 12th Workshop on Computational Approaches to Subjectivity, Sentiment & Social Media Analysis,", "booktitle"], ["Dublin, Ireland,", "location"], ["26 May 2022;", "date"]]
                    ]
            """
            ref_strings = []
            try:
                ref_list = ref_parser.predict(file_path)
            except Exception as e:
                print(e)
                ref_list = []
            for data in ref_list:
                ref_strings.append(data)
            output = self._output_process(ref_strings)
            return output

        # Process the parsed reference information, construct a reference ID, and consider each parsed reference title as a separate reference.
        def blocks_to_reference_titles(blocks, article_id, article_name):
            result = []
            id = 0
            for i, block in enumerate(blocks):
                for entity in block:
                    if entity[1] == 'title':
                        result.append({"reference_id": f'{article_id}_{id}', "reference_title": entity[0], "citing_paper_id": article_id,
                                       "citing_paper": article_name, "reference_detail": block})
                        id += 1
            return result

        ref_parser = ReferenceStringParsing(
            model_name=self.model_name,
            device=self.device,
            cache_dir=self.cache_dir,
            output_dir=self.output_dir,
            temp_dir=self.temp_dir,
            tokenizer=self.tokenizer,
            checkpoint=self.checkpoint,
            model_max_length=self.model_max_length,
            os_name=self.os_name
        )

        res = []
        files = os.listdir(files_path)
        for i, file in enumerate(files):
            if file.split(".")[-1] == 'pdf':
                path = os.path.join(files_path, file)
                ref_blocks = pdf_to_references(path, ref_parser)
                ref_lines = blocks_to_reference_titles(ref_blocks, i, file)
                res.extend(ref_lines)
        return res

    def _ref_clustering(self, reference_data: List[Dict]):
        """
        Cluster based on the parsed reference information.
        Primarily, perform clustering similar to DBCAN based on the similarity of reference titles.


        Args:
            reference_data (`List[Dict]`): Reference information, where each element represents a reference in the form of a Dict.
        Returns:
            `List[Dict]`: Reference information, where similar references are merged based on the similarity of reference titles.

        """

        start_time = datetime.datetime.now()

        # Basic handling of titles
        reference_list = [{"reference_id": data["reference_id"], "reference_title": self.utils.text_process(data["reference_title"])}
                      for data in reference_data]
        # end_time = datetime.datetime.now()
        # print(f'time cost for titles processing: {end_time - start_time}')

        # Compute the similarity matrix.
        start_time = datetime.datetime.now()
        similarity_matrix = self._cal_similarity_matrix(reference_list)
        # end_time = datetime.datetime.now()
        # print(f'finish similarity matrix, time cost:{end_time - start_time}')

        # Clustering using a method similar to DBSCAN.
        reference_group = self._dbscan_grouping(reference_list, similarity_matrix)
        # end_time = datetime.datetime.now()
        # print(f'finish clustering since begin, time cost:{end_time - start_time}')

        group_data = [{'reference_key': key, 'cited_references': reference_group[key], "cited_number": len(reference_group[key])} for key in reference_group]
        group_data.sort(key=lambda x: x['cited_number'], reverse=True)
        return group_data

    def _cal_similarity_matrix(self, reference_list):
        """
        Calculate the pairwise similarity between reference titles.

        Args:
            reference_list (`List[Dict]`): List of reference information
        Returns:
            `List[List[float]]`: Matrix storing the pairwise similarity scores between reference titles,
                where each element represents the dissimilarity score between titles.

        """
        n = len(reference_list)
        similarity_matrix = [[0 for _ in range(n)] for _ in range(n)]
        for i in range(n):
            for j in range(i, n):
                difference = self.utils.title_similarity(reference_list[i]["reference_title"], reference_list[j]["reference_title"])
                similarity_matrix[i][j] = difference
                similarity_matrix[j][i] = difference
        return similarity_matrix

    def _dbscan_grouping(self, reference_list, similarity_matrix):
        """
        Parse title strings from a text and save the result as a text file.

        Args:
            reference_list (`List[Dict]`): List of reference information
            similarity_matrix (`List[List[float]]`): Matrix storing the pairwise similarity scores between reference titles,
                where each element represents the dissimilarity score between titles.
        Returns:
            `Dict`: Clusters of reference titles, where each element is a dictionary storing cluster members.
            Example:
                {
                    "0_6|bert: pre-training of deep bidirectional transformers for language understanding":[
                        ["0_6", "bert: pre-training of deep bidirectional transformers for language understanding"],
                        ["2_11", "bert: pretraining of deep bidirectional transformers for lan-guage understanding"]
                    ]
                }


        """
        threshold = 0.2
        group = {}
        used = set()
        for i in range(len(reference_list)):

            idx = reference_list[i]["reference_id"]
            title = reference_list[i]["reference_title"]
            group_key = "|".join([idx, title])

            if i not in used:
                used.add(i)
                next_option = [i]
                group[group_key] = [[idx, title]]
                option = []
                while next_option != []:
                    for k in next_option:
                        for j in range(len(reference_list)):
                            if j != k and j not in used:
                                difference = similarity_matrix[i][j]
                                if difference < threshold:
                                    group[group_key].append([reference_list[j]["reference_id"], reference_list[j]["reference_title"]])
                                    option.append(j)
                                    used.add(j)
                    next_option = option
                    option = []
        return group

    def _output_process(self, output):
        """
        Convert all parsed reference formats from tags to block format.
        """
        re = []
        for parsed in output:
            tokens = parsed['tokens']
            tags = parsed['tags']
            re += [self._tags_to_blocks(tokens, tags)]
        return re

    def _tags_to_blocks(self, tokens, tags):
        """
        Convert a single parsed reference format from tags to block format.

        Args:
            tokens (`List[str]`): List of tokenized reference strings.
            tags (`List[str]`): Tags corresponding to each token, such as author, date, title, etc.
        Returns:
             `List[str,str]`: List of reference information, where each element represents details such as authors, date, title, etc. for each reference.
            Example:
                [["Federico Bianchi, Debora Nozza, and Dirk Hovy.", "author"], ["2022.", "date"], ["XLM-EMO: Multilingual Emotion Prediction in So- cial Media Text.", "title"], ["In Proceedings of the 12th Work- shop on Computational Approaches to Subjectivity, Sentiment and Social Media Analysis.", "booktitle"], ["Association for Computational Linguistics.", "publisher"]],

        """
        if len(tokens) == 1:
            return [tokens[0], tags[0]]
        result = []
        p1, p2 = 0, 1
        tmp = tags[0]
        length = min(len(tokens), len(tags))
        while p2 < length:
            if tags[p2] == tags[p2 - 1]:
                p2 += 1
            else:
                result.append([' '.join(tokens[p1:p2]), tmp])
                p1 = p2
                tmp = tags[p2]
                p2 += 1
        result.append([' '.join(tokens[p1::]), tmp])
        return result

class ReferenceEntityLinkingOnline(Pipeline):
    """
    The pipeline for reference entity linking (Online Version, Requires Network for Retrieving Corresponding Reference Articles).

    Args:
        model_name (`str`, *optional*):
            A string, the *model name* of a pretrained model provided for this task.
        device (`str`, *optional*):
            A string, `cpu` or `gpu`.
        cache_dir (`str` or `os.PathLike`, *optional*):
            Path to a directory in which a downloaded pretrained model should be
            cached if the standard cache should not be used.
        output_dir (`str` or `os.PathLike`, *optional*):
            Path to a directory in which the predicted results files should be stored.
        temp_dir (`str` or `os.PathLike`, *optional*):
            Path to a directory which holds temporary files such as `.tei.xml`.
        tokenizer (PreTrainedTokenizer, *optional*):
            A specific tokenizer.
        checkpoint (`str` or `os.PathLike`, *optional*):
            A checkpoint for the tokenizer. You can also specify the `checkpoint` while
            using the default tokenizer.
            Can be either:

                - A string, the *model id* of a predefined tokenizer hosted inside a model repo on huggingface.co.
                  Valid model ids can be located at the root-level, like `bert-base-uncased`, or namespaced under a
                  user or organization name, like `allenai/scibert_scivocab_uncased`.
                - A path to a *directory* containing vocabulary files required by the tokenizer, for instance saved
                  using the [`~PreTrainedTokenizer.save_pretrained`] method, e.g., `./my_model_directory/`.
                - A path or url to a single saved vocabulary file if and only if the tokenizer only requires a
                  single vocabulary file (like Bert or XLNet), e.g.: `./my_model_directory/vocab.txt`. (Not
                  applicable to all derived classes)

        model_max_length (`int`, *optional*): The max sequence length the model accepts.
    """

    def __init__(
            self,
            device: Optional[str] = "gpu",
            cache_dir = None,
            output_dir = None,
            temp_dir = None,
            os_name=None,
    ):
        self.device = device
        self.cache_dir = cache_dir if cache_dir is not None else BASE_CACHE_DIR
        self.output_dir = output_dir if output_dir is not None else BASE_OUTPUT_DIR
        self.temp_dir = temp_dir if temp_dir is not None else BASE_TEMP_DIR
        self.os_name = os_name if os_name != None else os.name
        self.utils = Utils()

    def predict(
            self, files_dir,
            output_dir=None,
            temp_dir=None,
            save_results=True,
    ):

        """

        Args:
            files_dir (`str`):
                Path to a directory in which the PDF files should be stored.
            output_dir (`str` or `os.PathLike`, *optional*):
                Path to a directory in which the predicted results files should be stored.
                If not provided, it will use the `output_dir` set for the pipeline.
            temp_dir (`str` or `os.PathLike`, *optional*):
                Path to a directory which holds temporary files such as `.tei.xml`.
                If not provided, it will use the `temp_dir` set for the pipeline.
            save_results (`bool`, default to `True`):
                Whether to save the results in a *.json* file.

        Returns:
            `List[Dict]`: [{"reference_id": reference_id, "detail": detail} , ... ]

        Examples:
            >>> from SciAssist import ReferenceEntityLinkingOnline
            >>> pipeline = ReferenceEntityLinkingOnline()
            >>> pipeline.predict(
            ...     './sample_papers'
            ... )

            [
                {
                    "reference_id": "3f144a0d7ad2d92589f61cf466acc921c6d6a123",
                    "detail":{
                        "paperId": "3f144a0d7ad2d92589f61cf466acc921c6d6a123",
                        "title": "Automatic Generation of Related Work Sections in Scientific Papers: An Optimization Approach",
                        "isOpenAccess": true,
                        "openAccessPdf": {"url": "https://aclanthology.org/D14-1170.pdf", "status": null},
                        "citationStyles": {"bibtex": "@Misc{None,\n author = {Yue Hu and Xiaojun Wan},\n title = {Automatic Generation of Related Work Sections in Scientific Papers: An Optimization Approach}\n}\n"},
                        "authors": [{"authorId": "2108954687", "name": "Yue Hu"}, {"authorId": "145078589", "name": "Xiaojun Wan"}],
                        "cited_number": 3
                        }
                },
                // ... Another JSON objects
            ]

        """

        if output_dir is None:
            output_dir = self.output_dir
        if temp_dir is None:
            temp_dir = self.temp_dir

        title_list = self.batch_pdf_to_title(files_dir)
        results = self.batch_title_to_references(title_list)
        if save_results:
            output_path = os.path.join(output_dir, f"online_reference_entity_linking_{datetime.datetime.now()}.txt")
            self.utils.json_list_to_file(results, output_path)
        return results

    def batch_title_to_references(self, title_list):
        """
        Search online for the corresponding paper and its reference information based on the title of each paper and merge identical references.

        Args:
            title_list (`List[str]`): List of paper titles.
        Returns:
            `List[Dict]`: All reference information, where each element in the list is in the form of a dictionary representing reference details.
                The list is sorted in descending order based on the number of citations.

        Examples:
            >>> from SciAssist import ReferenceEntityLinkingOnline
            >>> pipeline = ReferenceEntityLinkingOnline()
            >>> pipeline.batch_title_to_references(
            ...     [
            ...            'Automatic generation of related work sections in scientific papers: an Optimization Approach',
            ...            'Automatic Generation of Citation Texts in Scholarly Papers: A Pilot Study',
            ...             #... Another Title Strings
            ...       ]
            ... )

            [
                {
                    "reference_id": "3f144a0d7ad2d92589f61cf466acc921c6d6a123",
                    "detail":{
                        "paperId": "3f144a0d7ad2d92589f61cf466acc921c6d6a123",
                        "title": "Automatic Generation of Related Work Sections in Scientific Papers: An Optimization Approach",
                        "isOpenAccess": true,
                        "openAccessPdf": {"url": "https://aclanthology.org/D14-1170.pdf", "status": null},
                        "citationStyles": {"bibtex": "@Misc{None,\n author = {Yue Hu and Xiaojun Wan},\n title = {Automatic Generation of Related Work Sections in Scientific Papers: An Optimization Approach}\n}\n"},
                        "authors": [{"authorId": "2108954687", "name": "Yue Hu"}, {"authorId": "145078589", "name": "Xiaojun Wan"}],
                        "cited_number": 3
                        }
                },
                // ... Another JSON objects
            ]

        """

        def ref_id_group(ref_list):
            result = {}
            for ref in ref_list:
                ref_detail = ref['citedPaper']
                ref_id = ref_detail['paperId']
                if ref_id not in result:
                    result[ref_id] = ref_detail
                    ref_detail['cited_number'] = 1
                else:
                    result[ref_id]['cited_number'] += 1
            return result

        ref_list = []
        for title in title_list:
            refs = self.request_reference_from_title(title)
            ref_list.extend(refs)
            time.sleep(2)
        ref_group = ref_id_group(ref_list)
        group_data = [{'reference_id': key, 'detail': ref_group[key]} for key in ref_group]
        group_data.sort(key=lambda x: x['detail']['cited_number'], reverse=True)
        return group_data

    def batch_pdf_to_title(self, files_path):
        """
        Parse the titles of papers from multiple PDF files stored in the files_path.

        Args:
            files_dir (`str`): Path to a directory in which the PDF files should be stored.

        Returns:
            `List[str]`: List of paper titles.

        """

        files = os.listdir(files_path)
        title_list = []
        for i, file in enumerate(files):
            if file.split(".")[-1] == 'pdf':
                pdf_file = os.path.join(files_path,file)
                if self.os_name == "posix":
                    json_file = process_pdf_file(input_file=pdf_file, temp_dir=BASE_TEMP_DIR, output_dir=BASE_TEMP_DIR)
                    title = self._get_titles(json_file)
                elif self.os_name == "posix":
                    josn_file = windows_pdf2text.process_pdf(pdf_file)
                    title = josn_file["title"]
                else:
                    continue
                title_list.append(title)
        return title_list

    def _get_titles(self, json_file: str, output_dir: str = BASE_TEMP_DIR):
        """
        Read the titles of papers from the JSON file generated by parsing PDFs.
        """

        os.makedirs(output_dir, exist_ok=True)
        assert json_file[-4:] == "json"
        with open(json_file, 'r') as f:
            data = json.load(f)
            title = data["title"]
        return title

    def request_reference_from_title(self, paper_title):
        """
        Invoke the API interface to search for the corresponding paper and its reference information based on the title of each paper.

        Args:
            paper_title (`str`): Title of the paper
        Returns:
            `List[Dict]`: Each element in the list represents a reference information, including the reference paper's ID, title, authors, PDF download link, and citation format.

        """
        response = self.utils.search_real_title(paper_title)
        if response:
            paperId, search_real_title = response
            url = f'https://api.semanticscholar.org/graph/v1/paper/{paperId}/references?' \
                  f'fields=paperId,title,authors,isOpenAccess,openAccessPdf,citationStyles'
            content = self.utils.request_url(url)
            return content.get("data", [])
        else:
            return []

class Utils:
    def __init__(self):
        self.dictionary = enchant.Dict("en_US")

    def text_to_query(self, text):
        text = text.lower()
        text = text.strip(",.'\" ")
        text = self.text_process(text)
        text = re.sub('-', ' ', text)
        text = re.sub('[:,.!?\'\"]', '', text)
        print('query_title:', text)
        text = re.sub(' ', '+', text)
        return text

    def text_process(self, text):
        text = text.lower()
        text = text.strip(",.'\" ")
        w_list = text.split(' ')
        out = []
        count = 0
        length = len(w_list)
        for i in range(length - 1):
            if w_list[i][-1] == '-':
                count += 1
                word = w_list[i][0:-1] + w_list[i + 1]
                if self.dictionary.check(word):
                    w_list[i + 1] = word
                else:
                    w_list[i + 1] = w_list[i] + w_list[i + 1]
            else:
                out.append(w_list[i])
        out.append(w_list[-1])
        return ' '.join(out)

    def tokenize(self, strings):
        res = re.split('[^a-zA-Z0-9]', strings)
        return res

    def file_to_json_list(self, path):
        res = []
        with open(path, "r", encoding="utf=8") as f:
            for line in f:
                data = json.loads(line.strip())
                res.append(data)
        return res

    def json_list_to_file(self, ref_lines, path):
        with open(path, "w", encoding="utf=8") as f:
            for line in ref_lines:
                json.dump(line, f, ensure_ascii=False)
                f.write("\n")

    def title_similarity(self, title1, title2):
        """
        Compare the titles using a method similar to the minimum edit distance.

        Args:
            title1 (`str`): The first title to compare.
            title2 (`str`): The second title to compare.
        Returns:
            `float`: A similarity score ranging from 0 to 1 indicates the degree of difference between two titles.
                A score of 0 signifies complete similarity, while 1 indicates complete dissimilarity.

        """
        replace_weight = 2.5 # Will assign additional weight to replace operations.
        title1 = title1.lower()
        title2 = title2.lower()
        tokens1 = self.tokenize(title1)
        tokens2 = self.tokenize(title2)
        len1, len2 = len(tokens1), len(tokens2)

        dp = [[0] * (len2 + 1) for i in range(len1 + 1)]
        rp = [[0] * (len2 + 1) for i in range(len1 + 1)]
        for i in range(len1 + 1):
            dp[i][0] = i
            rp[i][0] = 0
        for j in range(len2 + 1):
            dp[0][j] = j
            rp[0][j] = 0

        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                tmp = dp[i - 1][j - 1]
                if tokens1[i - 1] != tokens2[j - 1]:
                    tmp += 2
                dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, tmp)
                # Calculate the number of replace operations.
                if dp[i][j] == dp[i - 1][j] + 1:
                    rp[i][j] = max(rp[i - 1][j], rp[i][j])
                if dp[i][j] == dp[i][j - 1] + 1:
                    rp[i][j] = max(rp[i][j - 1], rp[i][j])
                if dp[i][j] == tmp and tmp == dp[i - 1][j - 1] + 2:
                    rp[i][j] = max(rp[i - 1][j - 1] + 1, rp[i][j])
                if dp[i][j] == tmp and tmp == dp[i - 1][j - 1]:
                    rp[i][j] = rp[i - 1][j - 1]

        min_edit = dp[len1][len2]

        score = (min_edit + rp[-1][-1] * replace_weight) / (len1 + len2)
        return score


    def search_real_title(self, parsed_title, title_limit=10):
        """
        compare query_title with the titles in the searching result.

        """

        query_str = self.text_to_query(parsed_title)
        url = f'https://api.semanticscholar.org/graph/v1/paper/search?query={query_str}&offset=0&limit={title_limit}'
        content = self.request_url(url)
        if content.get("data", None):
            for data in content['data']:
                score = self.title_similarity(parsed_title, data['title'])
                if score <= 0.2:
                    title = data['title']
                    id = data['paperId']
                    print(f'search successfully: {title}')
                    return [id, title]
                else:
                    print("No Matched Paper")
        else:
            print("Request Error")

    def request_url(self, url, try_limit=2, retry_interval=5):
        for try_n in range(try_limit):
            try:
                req = requests.get(url)
                req.raise_for_status()  # Raise HTTPError for bad responses
                return req.json()
            except requests.RequestException as e:
                print(f'Request failed on attempt {try_n + 1}/{try_limit}')
                print(f'Error details: {e}')
                time.sleep(retry_interval)

        print(f'Failed to retrieve data from {url} after {try_limit} attempts')
        return {'error': 'Max retries exceeded'}

