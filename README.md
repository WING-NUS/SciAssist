
### **CHANGE LOG

**Source Code**
- Rename `SingleSummarization` to `Summarization`.
- Change the format of output files from `.txt` to `.json`.

**Documentation**
- Move the definition of `Pipeline` class from `Usage` to `Contribution Guide`.
- Add catalog for Contribution Guide.
- Add examples for choosing devices in `Usage`.

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
  <a href="#contribution">Contribution</a> 
  <br> <br>
</div>

## About

This is the repository of SciAssist, which is a toolkit to assist scientists' research. SciAssist currently supports reference string parsing, more functions are under active development by [WING@NUS](https://wing.comp.nus.edu.sg/), Singapore. The project was built upon an open-sourced [template by ashleve](https://github.com/ashleve/lightning-hydra-template).

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

**Reference string parsing:**

```python
from SciAssist import ReferenceStringParsing

# Set device="cpu" if you want to use only CPU. The default device is "gpu".
# ref_parser = ReferenceStringParsing(device="cpu")
ref_parser = ReferenceStringParsing(device="gpu")

# For string
res = ref_parser.predict(
    """Calzolari, N. (1982) Towards the organization of lexical definitions on a 
    database structure. In E. Hajicova (Ed.), COLING '82 Abstracts, Charles 
    University, Prague, pp.61-64.""", type="str")
# For text
res  = ref_parser.predict("test.txt", type="txt")
# For pdf
res = ref_parser.predict("test.pdf")
```

**Summarizarion for single document:**

```python
from SciAssist import Summarization

# Set device="cpu" if you want to use only CPU. The default device is "gpu".
# pipleine = Summarization(device="cpu")
summerizer = Summarization(device="gpu")

text = """1 INTRODUCTION . Statistical learning theory studies the learning properties of machine learning algorithms , and more fundamentally , the conditions under which learning from finite data is possible . 
In this context , classical learning theory focuses on the size of the hypothesis space in terms of different complexity measures , such as combinatorial dimensions , covering numbers and Rademacher/Gaussian complexities ( Shalev-Shwartz & Ben-David , 2014 ; Boucheron et al. , 2005 ) . 
Another more recent approach is based on defining suitable notions of stability with respect to perturbation of the data ( Bousquet & Elisseeff , 2001 ; Kutin & Niyogi , 2002 ) . 
In this view , the continuity of the process that maps data to estimators is crucial , rather than the complexity of the hypothesis space . 
Different notions of stability can be considered , depending on the data perturbation and metric considered ( Kutin & Niyogi , 2002 ) . 
Interestingly , the stability and complexity approaches to characterizing the learnability of problems are not at odds with each other , and can be shown to be equivalent as shown in Poggio et al . ( 2004 ) and Shalev-Shwartz et al . ( 2010 ) . 
In modern machine learning overparameterized models , with a larger number of parameters than the size of the training data , have become common . 
The ability of these models to generalize is well explained by classical statistical learning theory as long as some form of regularization is used in the training process ( Bühlmann & Van De Geer , 2011 ; Steinwart & Christmann , 2008 ) . 
However , it was recently shown - first for deep networks ( Zhang et al. , 2017 ) , and more recently for kernel methods ( Belkin et al. , 2019 ) - that learning is possible in the absence of regularization , i.e. , when perfectly fitting/interpolating the data . 
Much recent work in statistical learning theory has tried to find theoretical ground for this empirical finding . 
Since learning using models that interpolate is not exclusive to deep neural networks , we study generalization in the presence of interpolation in the case of kernel methods . 
We study both linear and kernel least squares problems in this paper . """

# For string
res = summerizer.predict(text, type="str")
# For text
res = summerizer.predict("bodytext.txt", type="txt")
# For pdf
res = summerizer.predict("raw.pdf")

```

**Dataset mention extraction:**

```python
from SciAssist import DatasetExtraction

# Set device="cpu" if you want to use only CPU. The default device is "gpu".
# ref_parser = DatasetExtraction(device="cpu")
extractor = DatasetExtraction(device="gpu")

# For string
res = extractor.extract("The impact of gender identity on emotions was examined by researchers using a subsample from the National Longitudinal Study of Adolescent Health. The study aimed to investigate the direct effects of gender identity on emotional experiences and expression. By focusing on a subsample of the larger study, the researchers were able to hone in on the specific relationship between gender identity and emotions. Through their analysis, the researchers sought to determine whether gender identity could have a significant and direct impact on emotional well-being. The findings of the study have important implications for our understanding of the complex interplay between gender identity and emotional experiences, and may help to inform future interventions and support for individuals who experience gender-related emotional distress.", type="str")
# For text: please input the path of your .txt file
res = extractor.extract("test.txt", type="txt")
# For pdf: please input the path of your .pdf file
res = extractor.predict("test.pdf", type="pdf")
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

## LICENSE
This toolkit is licensed under the `Attribution-NonCommercial-ShareAlike 4.0 International`.
Read [LICENSE](https://github.com/WING-NUS/SciAssist/blob/main/LICENSE) for more information.