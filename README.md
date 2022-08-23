

<div align="center">
   
# SciAssist
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/paper/2020)
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
  <br> <br>
  <a href="#about">About</a> •
  <a href="#installation">Installation</a> •
  <a href="#usage">Usage</a> •
  <a href="#develop">Develop</a> 
  <br> <br>
</div>

## About

This is the repository of SciAssist, which is a toolkit to assist scientists' research. SciAssist currently supports reference string parsing, more functions are under active development by [WING@NUS](https://wing.comp.nus.edu.sg/), Singapore. The project was built upon an open-sourced [template by ashleve](https://github.com/ashleve/lightning-hydra-template).

## Installation

```bash
# clone project
git clone https://github.com/WING-NUS/SciAssist
cd SciAssist

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



## Usage

Here are some example usages.

_Reference string parsing:_
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

## Develop
Here's a simple introduction about how to incorporate a new task into SciAssist.
For details and examples, see [REAMDE_details.md](README_details.md).
### How to train on a new task
For a new task, the most important things are the dataset and the model to be used.
#### To prepare your dataset
Basically, you can create a **DataModule** in [src/datamodules/](src/datamodules/) to prepare your dataloader.
For example, we have [cora_datamodule.py](src/datamodules/cora_datamodule.py) for Cora dataset. 

Then, create a _.yaml_ in [configs/datamodule](configs/datamodule) to instantiate your datamodule. 


#### To add a model
All the components of a model should be included in [src/models/components](src/models/components), including the model structure or a customed tokenizer and so on. 

Next, define the logic of training, validation and test for your model in a **LightningModule**.
Also, create a config file in [configs/model](configs/model).

#### To create a Trainer and train
Actually there have been a perfect train_pipeline in our project, so there's no need to write a train pipeline yourself. 
To prepare the LightningDataModule and LightningModule is all you need to do. 

You can learn the procedure in [REAMDE_details.md](README_details.md).

___
Finally, choose your config files and train your model with the command line:
```bash
python train.py trainer=gpu datamodule=dataconfig model=modelconfig
```
### How to build a pipeline for a new task
As SciAssist aims to serve users, you need to write a pipeline easy to use.
The pipelines are stored in [src/pipelines](src/pipelines). 

For convenience, we don't use hydra in a pipeline. 
So simply create a _xx.py_ file, in which you load a model and define functions which can be directly used.
We hope your function can be imported with:
```python
from src.pipelines.xx import predict
res = predict(...)
```

