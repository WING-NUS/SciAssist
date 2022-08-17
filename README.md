

<div align="center">
   
# BERT ParsCit

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/paper/2020)

</div>

## Description

This is the repository of BERT ParsCit and is under active development at National University of Singapore (NUS), Singapore. The project was built upon a [template by ashleve](https://github.com/ashleve/lightning-hydra-template).
BERT ParsCit is a BERT version of [Neural ParsCit](https://github.com/WING-NUS/Neural-ParsCit) built by researchers under [WING@NUS](https://wing.comp.nus.edu.sg/).

## Installation

```bash
# clone project
git clone https://github.com/ljhgabe/BERT-ParsCit
cd BERT-ParsCit

# [OPTIONAL] create conda environment
conda create -n myenv python=3.8
conda activate myenv

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

### Set up PDF parsing engine s2orc-doc2json
The current `doc2json` tool is used to convert PDF to JSON. It uses Grobid to first process each PDF into XML, then extracts paper components from the XML.
To setup Doc2Json, you should run:
```bash
sh bin/doc2json/scripts/run.sh
```
This will setup Doc2Json and Grobid. And after installation, it starts the Grobid server in the background by default.



## Example usage

```python
from src.pipelines.bert_parscit import predict_for_string, predict_for_text, predict_for_pdf

str_result = predict_for_string(
    "Calzolari, N. (1982) Towards the organization of lexical definitions on a database structure. In E. Hajicova (Ed.), COLING '82 Abstracts, Charles University, Prague, pp.61-64.")
text_result = predict_for_text("test.txt")
pdf_result = predict_for_pdf("test.pdf")
```


## How to train

Train model with default configuration

```bash
# train on CPU

python train.py trainer=cpu

# train on GPU
python train.py trainer=gpu 

```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python train.py experiment=experiment_name.yaml
```

You can override any parameter from command line like this

```bash
python train.py trainer.max_epochs=20 datamodule.batch_size=64
```

To show the full stack trace for error occurred during training or testing

```bash
HYDRA_FULL_ERROR=1 python train.py
```


