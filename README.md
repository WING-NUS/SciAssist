

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
from SciAssist.pipelines.bert_parscit import predict_for_string, predict_for_text, predict_for_pdf

str_result = predict_for_string(
    "Calzolari, N. (1982) Towards the organization of lexical definitions on a database structure. In E. Hajicova (Ed.), COLING '82 Abstracts, Charles University, Prague, pp.61-64.")
text_result = predict_for_text("test.txt")
pdf_result = predict_for_pdf("test.pdf")
```
_Summarizarion:_
```python
from SciAssist.pipelines.bert_parscit import summarize_for_string, summarize_for_text, summarize_for_pdf

string = "1 INTRODUCTION . Statistical learning theory studies the learning properties of machine learning algorithms , and more fundamentally , the conditions under which learning from finite data is possible . In this context , classical learning theory focuses on the size of the hypothesis space in terms of different complexity measures , such as combinatorial dimensions , covering numbers and Rademacher/Gaussian complexities ( Shalev-Shwartz & Ben-David , 2014 ; Boucheron et al. , 2005 ) . Another more recent approach is based on defining suitable notions of stability with respect to perturbation of the data ( Bousquet & Elisseeff , 2001 ; Kutin & Niyogi , 2002 ) . In this view , the continuity of the process that maps data to estimators is crucial , rather than the complexity of the hypothesis space . Different notions of stability can be considered , depending on the data perturbation and metric considered ( Kutin & Niyogi , 2002 ) . Interestingly , the stability and complexity approaches to characterizing the learnability of problems are not at odds with each other , and can be shown to be equivalent as shown in Poggio et al . ( 2004 ) and Shalev-Shwartz et al . ( 2010 ) . In modern machine learning overparameterized models , with a larger number of parameters than the size of the training data , have become common . The ability of these models to generalize is well explained by classical statistical learning theory as long as some form of regularization is used in the training process ( Bühlmann & Van De Geer , 2011 ; Steinwart & Christmann , 2008 ) . However , it was recently shown - first for deep networks ( Zhang et al. , 2017 ) , and more recently for kernel methods ( Belkin et al. , 2019 ) - that learning is possible in the absence of regularization , i.e. , when perfectly fitting/interpolating the data . Much recent work in statistical learning theory has tried to find theoretical ground for this empirical finding . Since learning using models that interpolate is not exclusive to deep neural networks , we study generalization in the presence of interpolation in the case of kernel methods . We study both linear and kernel least squares problems in this paper . Our Contributions : • We characterize the generalization properties of interpolating solutions for linear and kernel least squares problems using a stability approach . While the ( uniform ) stability properties of regularized kernel methods are well known ( Bousquet & Elisseeff , 2001 ) , we study interpolating solutions of the unregularized ( `` ridgeless '' ) regression problems . • We obtain an upper bound on the stability of interpolating solutions , and show that this upper bound is minimized by the minimum norm interpolating solution . This also means that among all interpolating solutions , the minimum norm solution has the best test error . In particular , the same conclusion is also true for gradient descent , since it converges to the minimum norm solution in the setting we consider , see e.g . Rosasco & Villa ( 2015 ) . • Our stability bounds show that the average stability of the minimum norm solution is controlled by the condition number of the empirical kernel matrix . It is well known that the numerical stability of the least squares solution is governed by the condition number of the associated kernel matrix ( see the discussion of why overparametrization is “ good ” in Poggio et al . ( 2019 ) ) . Our results show that the condition number also controls stability ( and hence , test error ) in a statistical sense . Organization : In section 2 , we introduce basic ideas in statistical learning and empirical risk minimization , as well as the notation used in the rest of the paper . In section 3 , we briefly recall some definitions of stability . In section 4 , we study the stability of interpolating solutions to kernel least squares and show that the minimum norm solutions minimize an upper bound on the stability . In section 5 we discuss our results in the context of recent work on high dimensional regression . We conclude in section 6 . 2 STATISTICAL LEARNING AND EMPIRICAL RISK MINIMIZATION . We begin by recalling the basic ideas in statistical learning theory . In this setting , X is the space of features , Y is the space of targets or labels , and there is an unknown probability distribution µ on the product space Z = X × Y . In the following , we consider X = Rd and Y = R. The distribution µ is fixed but unknown , and we are given a training set S consisting of n samples ( thus |S| = n ) drawn i.i.d . from the probability distribution on Zn , S = ( zi ) ni=1 = ( xi , yi ) n i=1 . Intuitively , the goal of supervised learning is to use the training set S to “ learn ” a function fS that evaluated at a new value xnew should predict the associated value of ynew , i.e . ynew ≈ fS ( xnew ) . The loss is a function V : F × Z → [ 0 , ∞ ) , where F is the space of measurable functions from X to Y , that measures how well a function performs on a data point . We define a hypothesis space H ⊆ F where algorithms search for solutions . With the above notation , the expected risk of f is defined as I [ f ] = EzV ( f , z ) which is the expected loss on a new sample drawn according to the data distribution µ . In this setting , statistical learning can be seen as the problem of finding an approximate minimizer of the expected risk given a training set S. A classical approach to derive an approximate solution is empirical risk minimization ( ERM ) where we minimize the empirical risk IS [ f ] = 1 n ∑n i=1 V ( f , zi ) . A natural error measure for our ERM solution fS is the expected excess risk ES [ I [ fS ] −minf∈H I [ f ] ] . Another common error measure is the expected generalization error/gap given by ES [ I [ fS ] − IS [ fS ] ] . These two error measures are closely related since , the expected excess risk is easily bounded by the expected generalization error ( see Lemma 5 ) . 2.1 KERNEL LEAST SQUARES AND MINIMUM NORM SOLUTION . The focus in this paper is on the kernel least squares problem . We assume the loss function V is the square loss , that is , V ( f , z ) = ( y − f ( x ) ) 2 . The hypothesis space is assumed to be a reproducing kernel Hilbert space , defined by a positive definite kernel K : X ×X → R or an associated feature map Φ : X → H , such that K ( x , x′ ) = 〈Φ ( x ) , Φ ( x′ ) 〉H for all x , x′ ∈ X , where 〈· , ·〉H is the inner product in H. In this setting , functions are linearly parameterized , that is there exists w ∈ H such that f ( x ) = 〈w , Φ ( x ) 〉H for all x ∈ X . The ERM problem typically has multiple solutions , one of which is the minimum norm solution : f†S = arg min f∈M ‖f‖H , M = arg min f∈H 1 n n∑ i=1 ( f ( xi ) − yi ) 2 . ( 1 ) Here ‖·‖H is the norm onH induced by the inner product . The minimum norm solution can be shown to be unique and satisfy a representer theorem , that is for all x ∈ X : f†S ( x ) = n∑ i=1 K ( x , xi ) cS [ i ] , cS = K †y ( 2 ) where cS = ( cS [ 1 ] , . . . , cS [ n ] ) , y = ( y1 . . . yn ) ∈ Rn , K is the n by n matrix with entries Kij = K ( xi , xj ) , i , j = 1 , . . . , n , and K† is the Moore-Penrose pseudoinverse of K. If we assume n ≤ d and that we have n linearly independent data features , that is the rank of X is n , then it is possible to show that for many kernels one can replace K† by K−1 ( see Remark 2 ) . Note that invertibility is necessary and sufficient for interpolation . That is , if K is invertible , f†S ( xi ) = yi for all i = 1 , . . . , n , in which case the training error in ( 1 ) is zero . Remark 1 ( Pseudoinverse for underdetermined linear systems ) A simple yet relevant example are linear functions f ( x ) = w > x , that correspond toH = Rd and Φ the identity map . If the rank of X ∈ Rd×n is n , then any interpolating solution wS satisfies w > S xi = yi for all i = 1 , . . . , n , and the minimum norm solution , also called Moore-Penrose solution , is given by ( w†S ) > = y > X† where the pseudoinverse X† takes the form X† = X > ( XX > ) −1 . Remark 2 ( Invertibility of translation invariant kernels ) Translation invariant kernels are a family of kernel functions given by K ( x1 , x2 ) = k ( x1 − x2 ) where k is an even function on Rd . Translation invariant kernels are Mercer kernels ( positive semidefinite ) if the Fourier transform of k ( · ) is non-negative . For Radial Basis Function kernels ( K ( x1 , x2 ) = k ( ||x1 − x2|| ) ) we have the additional property due to Theorem 2.3 of Micchelli ( 1986 ) that for distinct points x1 , x2 , . . . , xn ∈ Rd the kernel matrix K is non-singular and thus invertible . The above discussion is directly related to regularization approaches . Remark 3 ( Stability and Tikhonov regularization ) Tikhonov regularization is used to prevent potential unstable behaviors . In the above setting , it corresponds to replacing Problem ( 1 ) by minf∈H 1 n ∑n i=1 ( f ( xi ) − yi ) 2 + λ ‖f‖ 2 H where the corresponding unique solution is given by fλS ( x ) = ∑n i=1K ( x , xi ) c [ i ] , c = ( K + λIn ) −1y . In contrast to ERM solutions , the above approach prevents interpolation . The properties of the corresponding estimator are well known . In this paper , we complement these results focusing on the case λ→ 0 . Finally , we end by recalling the connection between minimum norm and the gradient descent . Remark 4 ( Minimum norm and gradient descent ) In our setting , it is well known that both batch and stochastic gradient iterations converge exactly to the minimum norm solution when multiple solutions exist , see e.g . Rosasco & Villa ( 2015 ) . Thus , a study of the properties of the minimum norm solution explains the properties of the solution to which gradient descent converges . In particular , when ERM has multiple interpolating solutions , gradient descent converges to a solution that minimizes a bound on stability , as we show in this paper ."
source_text, summ = summarize_for_string(string) 
source_text, summ = summarize_for_text("bodytext.txt")
source_text, summ = summarize_for_pdf("raw.pdf")

```

## How to train

Train model with default configuration

```bash
# train on CPU

python train.py trainer=cpu

# train on GPU
python train.py trainer=gpu 

```

Train model with chosen experiment configuration from [configs/experiment/](src/SciAssist/configs/experiment/)

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
Basically, you can create a **DataModule** in [src/datamodules/](src/SciAssist/datamodules/) to prepare your dataloader.
For example, we have [cora_datamodule.py](src/SciAssist/datamodules/cora_datamodule.py) for Cora dataset. 

Then, create a _.yaml_ in [configs/datamodule](src/SciAssist/configs/datamodule) to instantiate your datamodule. 


#### To add a model
All the components of a model should be included in [src/models/components](src/SciAssist/models/components), including the model structure or a customed tokenizer and so on. 

Next, define the logic of training, validation and test for your model in a **LightningModule**.
Also, create a config file in [configs/model](src/SciAssist/configs/model).

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
The pipelines are stored in [src/pipelines](src/SciAssist/pipelines). 

For convenience, we don't use hydra in a pipeline. 
So simply create a _xx.py_ file, in which you load a model and define functions which can be directly used.
We hope your function can be imported with:
```python
from SciAssist.pipelines.xx import predict
res = predict(...)
```

