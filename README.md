
<div align="center">

# CocoSciSum
[![PyPI Status](https://badge.fury.io/py/sciassist.svg)](https://badge.fury.io/py/sciassist)
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![ReadTheDocs](https://readthedocs.org/projects/wing-sciassist/badge/)](https://wing-sciassist.readthedocs.io/en/docs/Usage.html#controlled-summarization-cocoscisum)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/wing-nus/SciAssist)
  <br> <br>
  <a href="#about">About</a> •
  <a href="#installation">Installation</a> •
  <a href="#usage">Usage</a> •
  <a href="#development">Development</a> 
  <br> <br>
</div>

## About

This is the repository of CocoScisum, a toolkit for controlled summarization of scientific documents, 
designed for the specific needs of the scientific community. 
CocoScisum is featuring its support for compositional controlled summarization, 
which allows it to handle multiple control attributes simultaneously. 
Currently, we support keywords and length control and offer a tutorial for developers to explore more attributes and models.

## Installation

``` bash
pip install SciAssist
```
#### Setup Grobid for pdf processing
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


```python
from SciAssist import Summarization

# Set device="cpu" if you want to use only CPU. The default device is "gpu".

# summerizer = Summarization(device="cpu")
summerizer = Summarization(device="gpu")

text = """Language model pre-training has been shown to be effective for improving many natural language processing tasks ( Dai and Le , 2015 ; Peters et al. , 2018a ; Radford et al. , 2018 ; Howard and Ruder , 2018 ) . 
These include sentence-level tasks such as natural language inference ( Bowman et al. , 2015 ; Williams et al. , 2018 ) and paraphrasing ( Dolan and Brockett , 2005 ) , which aim to predict the relationships between 
sentences by analyzing them holistically , as well as token-level tasks such as named entity recognition and question answering , where models are required to produce fine-grained output at the token level ( Tjong Kim 
Sang and De Meulder , 2003 ; Rajpurkar et al. , 2016 ) . There are two existing strategies for applying pre-trained language representations to downstream tasks : feature-based and fine-tuning . The feature-based 
approach , such as ELMo ( Peters et al. , 2018a ) , uses task-specific architectures that include the pre-trained representations as additional features . The fine-tuning approach , such as the Generative Pre-trained 
Transformer ( OpenAI GPT ) ( Radford et al. , 2018 ) , introduces minimal task-specific parameters , and is trained on the downstream tasks by simply fine-tuning all pretrained parameters . The two approaches share 
the same objective function during pre-training , where they use unidirectional language models to learn general language representations . We argue that current techniques restrict the power of the pre-trained 
representations , especially for the fine-tuning approaches . The major limitation is that standard language models are unidirectional , and this limits the choice of architectures that can be used during pre-training .
For example , in OpenAI GPT , the authors use a left-toright architecture , where every token can only attend to previous tokens in the self-attention layers of the Transformer ( Vaswani et al. , 2017 ) . Such 
restrictions are sub-optimal for sentence-level tasks , and could be very harmful when applying finetuning based approaches to token-level tasks such as question answering , where it is crucial to incorporate context 
from both directions . In this paper , we improve the fine-tuning based approaches by proposing BERT : Bidirectional Encoder Representations from Transformers . BERT alleviates the previously mentioned unidirectionality 
constraint by using a `` masked language model '' ( MLM ) pre-training objective , inspired by the Cloze task ( Taylor , 1953 ) . The masked language model randomly masks some of the tokens from the input , and the 
objective is to predict the original vocabulary id of the masked arXiv:1810.04805v2 [ cs.CL ] 24 May 2019 word based only on its context . Unlike left-toright language model pre-training , the MLM objective enables the
 representation to fuse the left and the right context , which allows us to pretrain a deep bidirectional Transformer . In addition to the masked language model , we also use a `` next sentence prediction '' task that
jointly pretrains text-pair representations . The contributions of our paper are as follows : • We demonstrate the importance of bidirectional pre-training for language representations . Unlike Radford et al . ( 2018 ) ,
which uses unidirectional language models for pre-training , BERT uses masked language models to enable pretrained deep bidirectional representations . This is also in contrast to Peters et al . ( 2018a ) , which uses
a shallow concatenation of independently trained left-to-right and right-to-left LMs . • We show that pre-trained representations reduce the need for many heavily-engineered taskspecific architectures . BERT is the 
first finetuning based representation model that achieves state-of-the-art performance on a large suite of sentence-level and token-level tasks , outperforming many task-specific architectures . • BERT advances the 
state of the art for eleven NLP tasks . The code and pre-trained models are available at https : //github.com/ google-research/bert .  """

# For string
res = summerizer.predict(text, type="str", length=50, keywords=["Cloze task"])
# For text
res = summerizer.predict("bert_bodytext.txt", type="txt", length=50, keywords=["Cloze task"])
# For pdf
res = summerizer.predict('bert_paper.pdf', type="pdf", length=50, keywords=["Cloze task"])

>>> res["summary"]
['This paper proposes a bidirectional pre-training method for language representations. The method is inspired by the Cloze task. The method is evaluated on a large suite of sentence-level and token-level tasks.']


```






## Development

Here's a brief introduction about how to incorporate a new dataset and a new model into CocoSciSum.
Generally, you will need to:

    1. Git clone this repo and prepare the virtual environment.
    2. Install Grobid Server.
    3. Create a LightningModule and a DataLightningModule.
    4. Train a model.
    5. Provide a pipeline for users.

    - To explore more attributes, you can simply create a DataLightningModule for a new dataset containing instances for different attributes control and train a model.

We also provide a step-by-step contribution guide to develop based on our toolkit, see [SciAssist’s documentation](https://wing-sciassist.readthedocs.io/en/latest/Contribution.html#).

## LICENSE
This toolkit is licensed under the `Attribution-NonCommercial-ShareAlike 4.0 International`.
Read [LICENSE](https://github.com/WING-NUS/SciAssist/blob/main/LICENSE) for more information.
