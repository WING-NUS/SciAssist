

<div align="center">
   
# SciAssist
[![PyPI Status](https://badge.fury.io/py/sciassist.svg)](https://badge.fury.io/py/sciassist)
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![ReadTheDocs](https://readthedocs.org/projects/wing-sciassist/badge/)](https://wing-sciassist.readthedocs.io/en/latest/)
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

**Reference string parsing:**
```python
from SciAssist import ReferenceStringParsing

# Set device="cpu" if you want to use only CPU. The default device is "gpu".
# pipleine = ReferenceStringParsing(device="cpu")
pipeline = ReferenceStringParsing(device="gpu")

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

**Summarizarion for single document:**
```python
from SciAssist import SingleSummarization

# Set device="cpu" if you want to use only CPU. The default device is "gpu".
# pipleine = SingleSummarization(device="cpu")
pipeline = SingleSummarization(device="gpu")

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

## Contribution
Here's a simple introduction about how to incorporate a new task into SciAssist. 
Generally, to add a new task, you will need to:

    1. Git clone this repo and prepare the virtual environment.
    2. Install Grobid Server.
    3. Create a LightningModule and a DataLightningModule.
    4. Train a model.
    5. Provide a pipeline for users.
    
We provide a step-by-step contribution guide, see [SciAssist’s documentation](https://wing-sciassist.readthedocs.io/en/latest/Contribution.html#).

