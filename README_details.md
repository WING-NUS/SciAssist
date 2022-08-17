
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
This is the repository of SciAssist and is under active development at National University of Singapore (NUS), Singapore. The project was built upon a [template by ashleve](https://github.com/ashleve/lightning-hydra-template).

SciAssist is a toolkit for scientific document processing. Now it has included reference string parsing.
## Installation
### Set up PDF parsing engine s2orc-doc2json

The current `doc2json` tool uses Grobid to first process each PDF into XML, then extracts paper components from the XML.
If you fail to install Doc2Json or Grobid with `bin/doc2json/scripts/run.sh` , try to execute the following command:
```bash
cd bin/doc2json
python setup.py develop
cd ../..
```
This will setup Doc2Json.

#### Install Grobid

You will need to have Java installed on your machine. Then, you can install your own version of Grobid and get it running, or you can run the following script:

```console
bash bin/doc2json/scripts/setup_grobid.sh
```

This will setup Grobid, currently hard-coded as version 0.6.1. Then run:

```console
bash bin/doc2json/scripts/run_grobid.sh
```

to start the Grobid server. Don't worry if it gets stuck at 87%; this is normal and means Grobid is ready to process PDFs.

## Usage
### How to Parse Reference Strings

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

#### To extract strings from a PDF
You can extract strings you need with the script. 
For example, to get reference strings, try:
```console
python pdf2text.py --input_file file_path --reference --output_dir output/ --temp_dir temp/
```
 

## Develop
### How to train on a new task
For a new task, the most important things are the dataset and the model to be used.
#### To prepare your dataset
Basically, you can create a **DataModule** in [src/datamodules/](src/datamodules/) to prepare your dataloader.
For example, we have [cora_datamodule.py](src/datamodules/cora_datamodule.py) for Cora dataset. 

A **DataModule** standardizes the training, val, test splits, data preparation and transforms. 
_A datamodule looks like this:_
```python
from pytorch_lightning import LightningDataModule

class MyDataModule(LightningDataModule):
    def __init__(self):
        super().__init__()
    def prepare_data(self):
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed
    def setup(self, stage):
        # make assignments here (val/train/test split)
        # called on every process in DDP
    def train_dataloader(self):
        train_split = Dataset(...)
        return DataLoader(train_split)
    def val_dataloader(self):
        val_split = Dataset(...)
        return DataLoader(val_split)
    def test_dataloader(self):
        test_split = Dataset(...)
        return DataLoader(test_split)
    #def teardown(self):
        # clean up after fit or test
        # called on every process in DDP
```
They are actually hook functions, so you can simply overwrite them as you like.

In [datamodules/components](src/datamodules/components), you can save some fixed properties such as the label set.
 
There should be some customed functions for preprocessing which can be shared in several tasks. For example, the procedures for tokenization and padding of different sequence labeling tasks remain consistent. It will be good if you define them as an utility in [src/utils](src/utils), which may facilitates others' work. 

Then, create a _.yaml_ in [configs/datamodule](configs/datamodule) to instantiate your datamodule. 
_A data config file looks like this:_ 
```yaml
# The target class of the following configs 
_target_: src.datamodules.my_datamodule.MyDataModule

# Pass constructor parameters to the target class
data_repo: "myvision/cora-dataset-final"
train_batch_size: 8
num_workers: 0
pin_memory: False
data_cache_dir:  ${paths.data_dir}/new_task/
```


#### To add a model
All the components of a model should be included in [src/models/components](src/models/components), including the model structure or a customed tokenizer and so on. 

Next, define the logic of training, validation and test for your model in a **LightningModule**.
Same as a LightningDataModule, a **LightningModule** provides some hook functions to simplify the procedure. _Usually it looks like:_ 
```python
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F


class LitModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(28 * 28, 10)
        # Define computations here
        # You can easily use multiple components in `models/components` 

    def forward(self, x):
        # Use for inference only (separate from training_step)
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_idx):
        # the complete training loop
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def validation_step(self, batch: Any, batch_idx: int):
        # the complete validation loop
        return loss
    
    def test_step(self, batch: Any, batch_idx: int):
        # the complete test loop
        return loss
      
    def configure_optimizers(self):
        # define optimizers and LR schedulers
        return torch.optim.Adam(self.parameters(), lr=0.02)
```
The **LightningModule** has many convenience methods, and here are the core ones.
Check [https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html](https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html) for further information.

Also, create a config file in [configs/model](configs/model):
```yaml
# The target Class
_target_: src.models.cora_module.LitModule
lr: 2e-5

# Parameters can be nested
# When instantiating the LitModule, the following model will be automatically constructed.  
model:
  _target_: src.models.components.bert_token_classifier.BertTokenClassifier
  model_checkpoint: "allenai/scibert_scivocab_uncased"
  output_size: 13
  cache_dir: ${paths.root_dir}/.cache/
  save_name: ${model_name}
  model_dir: ${paths.model_dir}
```

#### To create a Trainer and train
**NOTICE**: Actually there have been a perfect train_pipeline in our project, so there's no need to write a train pipeline yourself. 
To prepare the LightningDataModule and LightningModule is all you need to do.

But here's an introduction to this procedure in case of any unknown problem.
___
The last step before starting training is to prepare a trainer config:
```yaml
_target_: pytorch_lightning.Trainer

accelerator: 'gpu'
devices: 1
min_epochs: 1
max_epochs: 5

# ckpt path
resume_from_checkpoint: null
``` 

And then you can create a Pytorch lightning Trainer to manage the whole training process:
```python
import hydra
from omegaconf import DictConfig
from pytorch_lightning import (
    LightningDataModule,
    LightningModule,
    Trainer,
)   

# To introduce hydra config files
@hydra.main(version_base="1.2", config_path="configs/", config_name="train.yaml")
def train(config: DictConfig):
    # Init datamodule
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)

    # Init lightning model
    model: LightningModule = hydra.utils.instantiate(config.model)
    
    # Init Trainer
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer, callbacks=callbacks, logger=logger, _convert_="partial"
    )
    
    # To train the model
    trainer.fit(model=model, datamodule=datamodule)

```

___
Finally, you can choose your config files and train your model with the command line:
```bash
python train.py trainer=gpu datamodule=dataconfig model=modelconfig
```
### How to build a pipeline for a new task
As SciAssist aims to serve users, you need to write a pipeline easy to use.
The pipelines are stored in [src/pipelines](src/pipelines). 

For convenience, we don't use hydra in a pipeline. 
So simply create a _xx.py_ file, in which you load a model and define functions which can be directly used:
```python
model = BertTokenClassifier(
    model_checkpoint="allenai/scibert_scivocab_uncased",
    output_size=13,
    cache_dir=BASE_CACHE_DIR
)

model.load_state_dict(torch.load("models/default/scibert-uncased.pt"))
model.eval()

def predict(...):
    return results
```  
And in this example we hope it can be imported with:
```python
from src.pipelines.xx import predict
res = predict(...)
```

