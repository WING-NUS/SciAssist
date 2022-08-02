
<div align="center">
   
# BERT ParsCit

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/paper/2020)

</div>

##  Set up PDF parsing engine s2orc-doc2json

The current `doc2json` tool uses Grobid to first process each PDF into XML, then extracts paper components from the XML.
If you fail to install Doc2Json or Grobid with `bin/doc2json/scripts/run.sh` , try to execute the following command:
```bash
cd bin/doc2json
python setup.py develop
cd ../..
```
This will setup Doc2Json.

### Install Grobid

You will need to have Java installed on your machine. Then, you can install your own version of Grobid and get it running, or you can run the following script:

```console
bash bin/doc2json/scripts/setup_grobid.sh
```

This will setup Grobid, currently hard-coded as version 0.6.1. Then run:

```console
bash bin/doc2json/scripts/run_grobid.sh
```

to start the Grobid server. Don't worry if it gets stuck at 87%; this is normal and means Grobid is ready to process PDFs.


## How to Parse Reference Strings

To parse reference strings from **a PDF file**, try:

```python
from src.pipelines.bert_parscit import predict_for_pdf

results, tokens, tags = predict_for_pdf(filename, output_dir, temp_dir)
```
This will generate a text file of reference strings in the specified `output_dir`.
And the JSON format of the origin PDF will be saved in the specified `temp_dir`. 
The default `output_dir` is `result/` from your path and the default `temp_dir` is `temp/` from your path.
The output `results` is a list of tagged strings, which seems like:
```
['<author>Waleed</author> <author>Ammar,</author> <author>Matthew</author> <author>E.</author> <author>Peters,</author> <author>Chandra</author> <author>Bhagavat-</author> <author>ula,</author> <author>and</author> <author>Russell</author> <author>Power.</author> <date>2017.</date> <title>The</title> <title>ai2</title> <title>system</title> <title>at</title> <title>semeval-2017</title> <title>task</title> <title>10</title> <title>(scienceie):</title> <title>semi-supervised</title> <title>end-to-end</title> <title>entity</title> <title>and</title> <title>relation</title> <title>extraction.</title> <booktitle>In</booktitle> <booktitle>ACL</booktitle> <booktitle>workshop</booktitle> <booktitle>(SemEval).</booktitle>']
```

`tokens` is a list of origin tokens of the input file and `tags` is the list of predicted tags corresponding to the input:
tokens:
```
[['Waleed', 'Ammar,', 'Matthew', 'E.', 'Peters,', 'Chandra', 'Bhagavat-', 'ula,', 'and', 'Russell', 'Power.', '2017.', 'The', 'ai2', 'system', 'at', 'semeval-2017', 'task', '10', '(scienceie):', 'semi-supervised', 'end-to-end', 'entity', 'and', 'relation', 'extraction.', 'In', 'ACL', 'workshop', '(SemEval).']]
```
tags:
```
[['author', 'author', 'author', 'author', 'author', 'author', 'author', 'author', 'author', 'author', 'author', 'date', 'title', 'title', 'title', 'title', 'title', 'title', 'title', 'title', 'title', 'title', 'title', 'title', 'title', 'title', 'booktitle', 'booktitle', 'booktitle', 'booktitle']]
```

You can also process **a single string** or parse strings from **a TEXT file**:

```python
from src.pipelines.bert_parscit import predict_for_string, predict_for_text

str_results, str_tokens, str_tags = predict_for_string(
    "Waleed Ammar, Matthew E. Peters, Chandra Bhagavat- ula, and Russell Power. 2017. The ai2 system at semeval-2017 task 10 (scienceie): semi-supervised end-to-end entity and relation extraction. In ACL workshop (SemEval).")
text_results, text_tokens, text_tags = predict_for_text(filename)
```

### To extract strings from a PDF
You can extract strings you need with the script. 
For example, to get reference strings, try:
```console
python pdf2text.py --input_file file_path --reference --output_dir output/ --temp_dir temp/
```
 
