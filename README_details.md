
<div align="center">
   
# SciAssist
  <br> <br>
  <a href="#about">About</a> •
  <a href="#installation">Installation</a> •
  <a href="#usage">Usage</a> •
  <a href="#develop">Develop</a> 
  <br> <br>
</div>

## About
This document contains more detailed instructions for some important steps in [README.md](README.md), including installation, usage and further development.
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
from SciAssist.pipelines.bert_parscit import predict_for_pdf

results, tokens, tags = predict_for_pdf(filename, output_dir, temp_dir)
```
This will generate a text file of reference strings in the specified `output_dir`.
And the JSON format of the origin PDF will be saved in the specified `temp_dir`. 
The default `output_dir` is `output/result/` from your path and the default `output/.temp_dir` is `temp/` from your path.
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
from SciAssist.pipelines.bert_parscit import predict_for_string, predict_for_text

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
### How to Summarize
To do summarization for a **a PDF file**, try:
```python
from SciAssist.pipelines.summarization import summarize_for_pdf

text, summ = summarize_for_pdf(filename, output_dir, temp_dir)
```
This will generate a text file of bodytext of the input pdf in the specified `output_dir`.
And the JSON format of the origin PDF will be saved in the specified `temp_dir`. 
The default `output_dir` is `output/result/` from your path and the default `temp_dir` is `output/.temp/` from your path.
The output includes the source text `text` and the predicted summary `summ`. 


You can also process **a single string** or summarize **a TEXT file**:

```python
from SciAssist.pipelines.summarization import summarize_for_string, summarize_for_text

text, summ = summarize_for_string(
    "1 INTRODUCTION . Statistical learning theory studies the learning properties of machine learning algorithms , and more fundamentally , the conditions under which learning from finite data is possible . In this context , classical learning theory focuses on the size of the hypothesis space in terms of different complexity measures , such as combinatorial dimensions , covering numbers and Rademacher/Gaussian complexities ( Shalev-Shwartz & Ben-David , 2014 ; Boucheron et al. , 2005 ) . Another more recent approach is based on defining suitable notions of stability with respect to perturbation of the data ( Bousquet & Elisseeff , 2001 ; Kutin & Niyogi , 2002 ) . In this view , the continuity of the process that maps data to estimators is crucial , rather than the complexity of the hypothesis space . Different notions of stability can be considered , depending on the data perturbation and metric considered ( Kutin & Niyogi , 2002 ) . Interestingly , the stability and complexity approaches to characterizing the learnability of problems are not at odds with each other , and can be shown to be equivalent as shown in Poggio et al . ( 2004 ) and Shalev-Shwartz et al . ( 2010 ) . In modern machine learning overparameterized models , with a larger number of parameters than the size of the training data , have become common . The ability of these models to generalize is well explained by classical statistical learning theory as long as some form of regularization is used in the training process ( Bühlmann & Van De Geer , 2011 ; Steinwart & Christmann , 2008 ) . However , it was recently shown - first for deep networks ( Zhang et al. , 2017 ) , and more recently for kernel methods ( Belkin et al. , 2019 ) - that learning is possible in the absence of regularization , i.e. , when perfectly fitting/interpolating the data . Much recent work in statistical learning theory has tried to find theoretical ground for this empirical finding . Since learning using models that interpolate is not exclusive to deep neural networks , we study generalization in the presence of interpolation in the case of kernel methods . We study both linear and kernel least squares problems in this paper . Our Contributions : • We characterize the generalization properties of interpolating solutions for linear and kernel least squares problems using a stability approach . While the ( uniform ) stability properties of regularized kernel methods are well known ( Bousquet & Elisseeff , 2001 ) , we study interpolating solutions of the unregularized ( `` ridgeless '' ) regression problems . • We obtain an upper bound on the stability of interpolating solutions , and show that this upper bound is minimized by the minimum norm interpolating solution . This also means that among all interpolating solutions , the minimum norm solution has the best test error . In particular , the same conclusion is also true for gradient descent , since it converges to the minimum norm solution in the setting we consider , see e.g . Rosasco & Villa ( 2015 ) . • Our stability bounds show that the average stability of the minimum norm solution is controlled by the condition number of the empirical kernel matrix . It is well known that the numerical stability of the least squares solution is governed by the condition number of the associated kernel matrix ( see the discussion of why overparametrization is “ good ” in Poggio et al . ( 2019 ) ) . Our results show that the condition number also controls stability ( and hence , test error ) in a statistical sense . Organization : In section 2 , we introduce basic ideas in statistical learning and empirical risk minimization , as well as the notation used in the rest of the paper . In section 3 , we briefly recall some definitions of stability . In section 4 , we study the stability of interpolating solutions to kernel least squares and show that the minimum norm solutions minimize an upper bound on the stability . In section 5 we discuss our results in the context of recent work on high dimensional regression . We conclude in section 6 . 2 STATISTICAL LEARNING AND EMPIRICAL RISK MINIMIZATION . We begin by recalling the basic ideas in statistical learning theory . In this setting , X is the space of features , Y is the space of targets or labels , and there is an unknown probability distribution µ on the product space Z = X × Y . In the following , we consider X = Rd and Y = R. The distribution µ is fixed but unknown , and we are given a training set S consisting of n samples ( thus |S| = n ) drawn i.i.d . from the probability distribution on Zn , S = ( zi ) ni=1 = ( xi , yi ) n i=1 . Intuitively , the goal of supervised learning is to use the training set S to “ learn ” a function fS that evaluated at a new value xnew should predict the associated value of ynew , i.e . ynew ≈ fS ( xnew ) . The loss is a function V : F × Z → [ 0 , ∞ ) , where F is the space of measurable functions from X to Y , that measures how well a function performs on a data point . We define a hypothesis space H ⊆ F where algorithms search for solutions . With the above notation , the expected risk of f is defined as I [ f ] = EzV ( f , z ) which is the expected loss on a new sample drawn according to the data distribution µ . In this setting , statistical learning can be seen as the problem of finding an approximate minimizer of the expected risk given a training set S. A classical approach to derive an approximate solution is empirical risk minimization ( ERM ) where we minimize the empirical risk IS [ f ] = 1 n ∑n i=1 V ( f , zi ) . A natural error measure for our ERM solution fS is the expected excess risk ES [ I [ fS ] −minf∈H I [ f ] ] . Another common error measure is the expected generalization error/gap given by ES [ I [ fS ] − IS [ fS ] ] . These two error measures are closely related since , the expected excess risk is easily bounded by the expected generalization error ( see Lemma 5 ) . 2.1 KERNEL LEAST SQUARES AND MINIMUM NORM SOLUTION . The focus in this paper is on the kernel least squares problem . We assume the loss function V is the square loss , that is , V ( f , z ) = ( y − f ( x ) ) 2 . The hypothesis space is assumed to be a reproducing kernel Hilbert space , defined by a positive definite kernel K : X ×X → R or an associated feature map Φ : X → H , such that K ( x , x′ ) = 〈Φ ( x ) , Φ ( x′ ) 〉H for all x , x′ ∈ X , where 〈· , ·〉H is the inner product in H. In this setting , functions are linearly parameterized , that is there exists w ∈ H such that f ( x ) = 〈w , Φ ( x ) 〉H for all x ∈ X . The ERM problem typically has multiple solutions , one of which is the minimum norm solution : f†S = arg min f∈M ‖f‖H , M = arg min f∈H 1 n n∑ i=1 ( f ( xi ) − yi ) 2 . ( 1 ) Here ‖·‖H is the norm onH induced by the inner product . The minimum norm solution can be shown to be unique and satisfy a representer theorem , that is for all x ∈ X : f†S ( x ) = n∑ i=1 K ( x , xi ) cS [ i ] , cS = K †y ( 2 ) where cS = ( cS [ 1 ] , . . . , cS [ n ] ) , y = ( y1 . . . yn ) ∈ Rn , K is the n by n matrix with entries Kij = K ( xi , xj ) , i , j = 1 , . . . , n , and K† is the Moore-Penrose pseudoinverse of K. If we assume n ≤ d and that we have n linearly independent data features , that is the rank of X is n , then it is possible to show that for many kernels one can replace K† by K−1 ( see Remark 2 ) . Note that invertibility is necessary and sufficient for interpolation . That is , if K is invertible , f†S ( xi ) = yi for all i = 1 , . . . , n , in which case the training error in ( 1 ) is zero . Remark 1 ( Pseudoinverse for underdetermined linear systems ) A simple yet relevant example are linear functions f ( x ) = w > x , that correspond toH = Rd and Φ the identity map . If the rank of X ∈ Rd×n is n , then any interpolating solution wS satisfies w > S xi = yi for all i = 1 , . . . , n , and the minimum norm solution , also called Moore-Penrose solution , is given by ( w†S ) > = y > X† where the pseudoinverse X† takes the form X† = X > ( XX > ) −1 . Remark 2 ( Invertibility of translation invariant kernels ) Translation invariant kernels are a family of kernel functions given by K ( x1 , x2 ) = k ( x1 − x2 ) where k is an even function on Rd . Translation invariant kernels are Mercer kernels ( positive semidefinite ) if the Fourier transform of k ( · ) is non-negative . For Radial Basis Function kernels ( K ( x1 , x2 ) = k ( ||x1 − x2|| ) ) we have the additional property due to Theorem 2.3 of Micchelli ( 1986 ) that for distinct points x1 , x2 , . . . , xn ∈ Rd the kernel matrix K is non-singular and thus invertible . The above discussion is directly related to regularization approaches . Remark 3 ( Stability and Tikhonov regularization ) Tikhonov regularization is used to prevent potential unstable behaviors . In the above setting , it corresponds to replacing Problem ( 1 ) by minf∈H 1 n ∑n i=1 ( f ( xi ) − yi ) 2 + λ ‖f‖ 2 H where the corresponding unique solution is given by fλS ( x ) = ∑n i=1K ( x , xi ) c [ i ] , c = ( K + λIn ) −1y . In contrast to ERM solutions , the above approach prevents interpolation . The properties of the corresponding estimator are well known . In this paper , we complement these results focusing on the case λ→ 0 . Finally , we end by recalling the connection between minimum norm and the gradient descent . Remark 4 ( Minimum norm and gradient descent ) In our setting , it is well known that both batch and stochastic gradient iterations converge exactly to the minimum norm solution when multiple solutions exist , see e.g . Rosasco & Villa ( 2015 ) . Thus , a study of the properties of the minimum norm solution explains the properties of the solution to which gradient descent converges . In particular , when ERM has multiple interpolating solutions , gradient descent converges to a solution that minimizes a bound on stability , as we show in this paper ."
text, summ = summarize_for_text("bodytext.txt")
```
#### To extract bodytext from a PDF
You can extract bodytext you need with the script. Try:
```console
python pdf2text.py --input_file file_path --bodytext --output_dir output/ --temp_dir temp/
```

## Develop
### How to train on a new task
For a new task, the most important things are the dataset and the model to be used.
#### To prepare your dataset
Basically, you can create a **DataModule** in [src/datamodules/](src/SciAssist/datamodules/) to prepare your dataloader.
For example, we have [cora_datamodule.py](src/SciAssist/datamodules/cora_datamodule.py) for Cora dataset. 

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

In [datamodules/components](src/SciAssist/datamodules/components), you can save some fixed properties such as the label set.
 
There should be some customed functions for preprocessing which can be shared in several tasks. For example, the procedures for tokenization and padding of different sequence labeling tasks remain consistent. It will be good if you define them as an utility in [src/utils](src/SciAssist/utils), which may facilitates others' work. 

Then, create a _.yaml_ in [configs/datamodule](src/SciAssist/configs/datamodule) to instantiate your datamodule. 
_A data config file looks like this:_ 
```yaml
# The target class of the following configs 
_target_: SciAssist.datamodules.my_datamodule.MyDataModule

# Pass constructor parameters to the target class
data_repo: "myvision/cora-dataset-final"
train_batch_size: 8
num_workers: 0
pin_memory: False
data_cache_dir:  ${paths.data_dir}/new_task/
```


#### To add a model
All the components of a model should be included in [src/models/components](src/SciAssist/models/components), including the model structure or a customed tokenizer and so on. 

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

Also, create a config file in [configs/model](src/SciAssist/configs/model):
```yaml
# The target Class
_target_: SciAssist.models.cora_module.LitModule
lr: 2e-5

# Parameters can be nested
# When instantiating the LitModule, the following model will be automatically constructed.  
model:
  _target_: SciAssist.models.components.bert_token_classifier.BertTokenClassifier
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
@hydra.main(version_base="1.2", config_path="src/SciAssist/configs/", config_name="train.yaml")
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
The pipelines are stored in [src/pipelines](src/SciAssist/pipelines). 

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
from SciAssist.pipelines import predict
res = predict(...)
```

### Other points
#### Default directories
For convenient management, we set some default value as follows.
* src/: all source codes
* configs/: hydra config files
* bin/: third-party toolkits 
* data/: datasets
* models/: models or checkpoints we trained
* .cache/: cached files such as models loaded from huggingface
* logs/: experiment logs
* scripts/: quickstart 

Some files such as experimental logs and checkpoints don't need to be commited to the repo. 

### (Other standards and regulations are to be added here)