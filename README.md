

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

``` bash
pip install SciAssist
```
### Setup Grobid for pdf processing
After you install the package, you can simply setup grobid with the CLI:
```bash
setup_grobid
```
This will setup Grobid. And after installation, starts the Grobid server with:
```bash
run_grobid
```




## Usage

Here are some example usages.

_Reference string parsing:_
```python
from SciAssist import ReferenceStringParsing

pipeline = ReferenceStringParsing()
# For string
tagged_result, tokens, tags = pipeline.predict(
    """Calzolari, N. (1982) Towards the organization of lexical definitions on a 
    database structure. In E. Hajicova (Ed.), COLING '82 Abstracts, Charles 
    University, Prague, pp.61-64.""", type="str")
# For text
tagged_result, tokens, tags  = pipeline.predict("test.txt", type="txt")
# For pdf
tagged_result, tokens, tags = pipeline.predict("test.pdf")
```
_Summarizarion for single document:_
```python
from SciAssist import SingleSummarization

pipeline = SingleSummarization()
text = """1 INTRODUCTION . Statistical learning theory studies the learning 
properties of machine learning algorithms , and more fundamentally , the conditions
 under which learning from finite data is possible . In this context , classical 
learning theory focuses on the size of the hypothesis space in terms of different 
complexity measures , such as combinatorial dimensions , covering numbers and 
Rademacher/Gaussian complexities ( Shalev-Shwartz & Ben-David , 2014 ; Boucheron 
et al. , 2005 ) . Another more recent approach is based on defining suitable notions 
of stability with respect to perturbation of the data ( Bousquet & Elisseeff , 2001 ; 
Kutin & Niyogi , 2002 ) . In this view , the continuity of the process that maps 
data to estimators is crucial , rather than the complexity of the hypothesis space . 
Different notions of stability can be considered , depending on the data perturbation
 and metric considered ( Kutin & Niyogi , 2002 ) . Interestingly , the stability and
 complexity approaches to characterizing the learnability of problems are not at odds 
with each other , and can be shown to be equivalent as shown in Poggio et al . 
( 2004 ) and Shalev-Shwartz et al . ( 2010 ) . In modern machine learning 
overparameterized models , with a larger number of parameters than the size of the 
training data , have become common . The ability of these models to generalize is well
 explained by classical statistical learning theory as long as some form of 
regularization is used in the training process ( Bühlmann & Van De Geer , 2011 ; 
Steinwart & Christmann , 2008 ) . However , it was recently shown - first for deep 
networks ( Zhang et al. , 2017 ) , and more recently for kernel methods ( Belkin et 
al. , 2019 ) - that learning is possible in the absence of regularization , i.e. , 
when perfectly fitting/interpolating the data . Much recent work in statistical 
learning theory has tried to find theoretical ground for this empirical finding . 
Since learning using models that interpolate is not exclusive to deep neural networks
 , we study generalization in the presence of interpolation in the case of kernel 
methods . We study both linear and kernel least squares problems in this paper . Our 
Contributions : • We characterize the generalization properties of interpolating 
solutions for linear and kernel least squares problems using a stability approach . 
While the ( uniform ) stability properties of regularized kernel methods are well 
known ( Bousquet & Elisseeff , 2001 ) , we study interpolating solutions of the 
unregularized ( `` ridgeless '' ) regression problems . • We obtain an upper bound 
on the stability of interpolating solutions , and show that this upper bound is 
minimized by the minimum norm interpolating solution . This also means that among 
all interpolating solutions , the minimum norm solution has the best test error . 
In particular , the same conclusion is also true for gradient descent , since it 
converges to the minimum norm solution in the setting we consider , see e.g . Rosasco 
& Villa ( 2015 ) . • Our stability bounds show that the average stability of the 
minimum norm solution is controlled by the condition number of the empirical kernel 
matrix . It is well known that the numerical stability of the least squares solution 
is governed by the condition number of the associated kernel matrix ( see the 
discussion of why overparametrization is “ good ” in Poggio et al . ( 2019 ) ) . Our 
results show that the condition number also controls stability ( and hence , test 
error ) in a statistical sense . Organization : In section 2 , we introduce basic 
ideas in statistical learning and empirical risk minimization , as well as the 
notation used in the rest of the paper . In section 3 , we briefly recall some 
definitions of stability . In section 4 , we study the stability of interpolating 
solutions to kernel least squares and show that the minimum norm solutions minimize 
an upper bound on the stability . In section 5 we discuss our results in the context 
of recent work on high dimensional regression . """

# For string
source_text, summ = pipeline.predict(text, type="str")
# For text
source_text, summ = pipeline.predict("bodytext.txt", type="txt")
# For pdf
source_text, summ = pipeline.predict("raw.pdf")

```

## Develop
Here's a simple introduction about how to incorporate a new task into SciAssist.
For details and examples, see [REAMDE_details.md](README_details.md).
### How to train on a new task
For a new task, the most important things are the dataset and the model to be used.
#### To prepare your dataset
Basically, you can create a **DataModule** in [datamodules/](src/SciAssist/datamodules/) to prepare your dataloader.
For example, we have [cora_datamodule.py](src/SciAssist/datamodules/cora_datamodule.py) for Cora dataset. 

Then, create a _.yaml_ in [configs/datamodule](src/SciAssist/configs/datamodule) to instantiate your datamodule. 


#### To add a model
All the components of a model should be included in [models/components](src/SciAssist/models/components), including the model structure or a customed tokenizer and so on. 

Next, define the logic of training, validation and test for your model in a **LightningModule**.
Also, create a config file in [configs/model](src/SciAssist/configs/model).

#### To create a Trainer
Actually there have been a perfect train_pipeline in our project, so there's no need to write a train pipeline yourself. 
To prepare the LightningDataModule and LightningModule is all you need to do. 

You can learn the procedure in [REAMDE_details.md](README_details.md).

#### To train
Finally, specify your data and model, and train your model with the command line:
```bash
python train.py trainer=gpu datamodule=dataconfig model=modelconfig
```
You can change other configs in this way too. For example:

Train model with default configuration:

```bash
# train on CPU

python train.py trainer=cpu

# train on GPU
python train.py trainer=gpu 

```

Train model with chosen experiment configuration from [configs/experiment/](src/SciAssist/configs/experiment/):

```bash
python train.py experiment=experiment_name.yaml
```

You can override any parameter from command line like this:

```bash
python train.py trainer.max_epochs=20 datamodule.batch_size=64
```

To show the full stack trace for error occurred during training or testing:

```bash
HYDRA_FULL_ERROR=1 python train.py
```
 


### How to build a pipeline for a new task
As SciAssist aims to serve users, you need to write a pipeline easy to use.
The pipelines are stored in [pipelines](src/SciAssist/pipelines). 

For convenience, we don't use hydra in a pipeline. 
So simply create a _xx.py_ file, we'd like to encapsulate involved functions 
in a class derived from `Pipeline`, where we can specify a default model and allow
users to choose one at the same time.
We list all provided tasks and available models in [pipelines/__init__.py](src/SciAssist/pipelines/__init__.py).
We recommend to name the pipeline class by the task, just like `ReferenceStringParsing`.

The most important function is `predict(..)`, which we hope users can easily invoke 
in the way described in [Usage](#Usage). 

