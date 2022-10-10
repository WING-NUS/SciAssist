Contributing Guide
===================

.. _Contribution:

1. Fork the SciAssist repository
--------------------------------

First, make a fork of the `SciAssist <https://github.com/WING-NUS/SciAssist>`_ repository. 
Clone the forked repository and install the package dependencies in ``requirements.txt`` into your virtual environment.

.. code-block:: bash

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


2. Install Grobid Server for pdf processing
-------------------------------------------

The current ``doc2json`` tool uses Grobid to first process each PDF into XML, then extracts paper components from the XML.
You will need to have Java installed on your machine. Then, you can install 
your own version of Grobid and get it running.
To setup Grobid conveniently, you can use the CLIs provided by SciAssist.

.. code-block:: bash

    # setup CLIs
    python setup.py develop

    # setup Grobid
    setup_grobid

    # run Grobid
    run_grobid

3. Train a model on a new task
---------------------------------

General overview
'''''''''''''''''

Generally, to train on a new task, you will need to:

    - Enter ``src/`` directory.
    - Choose or create a datautils class in ``src/SciAssist/utils/data_utils.py``.
    - Prepare a datamodule(LightningDataModule) in ``src/SciAssist/datamodules`` and create datamodule config in ``src/SciAssist/configs/datamodule``
    - Prepare a model(LightningModule) in ``src/SciAssist/models`` and create model config in ``src/SciAssist/configs/model``
    - Specify the configs and train your model in command line:
  
    .. code-block:: bash

        python SciAssist/train.py --datamodule=dataconfig --model=modelconfig

You may want more information about `Hydra <https://hydra.cc/docs/intro/>`_ to understand the config files better.   

Step-by-step recipe to train on a new task
''''''''''''''''''''''''''''''''''''''''''

The ``src/`` directory is considered the of SciAssist's source code, so enter ``src/`` before the next steps.

.. code-block:: bash

    cd src

.. _DataUtils:

Create datautils
"""""""""""""""""""

There should be some customed functions for data processing, 
you will need to create a ``DataUtils`` class for them for easy reuse in both training and inference.
For example, this is a datautil for Seq2Seq task:

.. autoclass:: SciAssist.utils.data_utils.DataUtilsForSeq2Seq
    :members:
    :member-order: bysource


Prepare your datamodule
""""""""""""""""""""""""""

Basically, you can create a **DataModule** in ``datamodules/`` to prepare your dataloader.
For example, we have ``cora_datamodule.py`` for Cora dataset.
In ``datamodules/components``, you can save some fixed properties such as the label set.

A **DataModule** standardizes the training, val, test splits, data preparation and transforms.
A DataModule looks like this:

.. code-block:: python

    from pytorch_lightning import LightningDataModule
    from SciAssist.datamodules.components.cora_label import label2id
    from SciAssist.utils.data_utils import DataUtilsForTokenClassification

    class MyDataModule(LightningDataModule):
        
        def __init__(self, data_repo: str = "myvision/cora-dataset-final", data_utils=DataUtilsForTokenClassification):
            super().__init__()
            
            # use parameters by self.hparams.xx
            self.save_hyperparameters(logger=False)
            
            self.data_utils = self.hparams.data_utils
            self.data_train: Optional[Dataset] = None
            self.data_val: Optional[Dataset] = None
            self.data_test: Optional[Dataset] = None

        def prepare_data(self):
            # download, split, etc...
            # only called on 1 GPU/TPU in distributed
            raw_datasets = datasets.load_dataset(
                self.hparams.data_repo,
            )
        return raw_datasets


        def setup(self, stage):
            # make assignments here (val/train/test split)
            # called on every process in DDP
            if not self.data_train and not self.data_val and not self.data_test:
                processed_datasets = self.prepare_data()
                tokenized_datasets = processed_datasets.map(
                    lambda x: self.data_utils.tokenize_and_align_labels(x, label2id),
                    batched=True,
                    remove_columns=processed_datasets["train"].column_names,
                    load_from_cache_file=True
                )
                self.data_train = tokenized_datasets["train"]
                self.data_val = tokenized_datasets["val"]
                self.data_test = tokenized_datasets["test"]

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

They are actually hook functions, so you can simply fill in them as you like.

Then, create a ``.yaml`` in ``configs/datamodule`` to instantiate your datamodule.
A data config file looks like this:

.. code-block:: yaml

    # The target class of the following configs
    _target_: SciAssist.datamodules.my_datamodule.MyDataModule

    # Pass constructor parameters to the target class
    data_repo: "myvision/cora-dataset-final"
    data_utils: 
      _target_: SciAssist.utils.data_utils.DataUtilsForTokenClassification

For more details about **DataModule**, refer `DataLightningModule <https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html>`_.

Prepare your model
""""""""""""""""""

All the components of a model should be included in ``SciAssist/models/components``, including model structure, tokenizers and so on.

Next, define the logic of training, validation and test for your model in a **LightningModule**.
Same as a **LightningDataModule**, a **LightningModule** provides some hook functions to simplify the procedure.
For example:

.. code-block:: python

    from pytorch_lightning import LightningModule
    from torchmetrics.classification.accuracy import Accuracy


    from SciAssist.datamodules.components.cora_label import num_labels, LABEL_NAMES
    from SciAssist.models.components.bert_token_classifier import BertForTokenClassifier
    from SciAssist.utils.data_utils import DataUtilsForTokenClassification


    class LitModel(pl.LightningModule):
        def __init__(self, model: BertForTokenClassifier, 
                    data_utils: DataUtilsForTokenClassification, 
                    lr: float = 2e-5):

            # Define computations here
            # You can easily use multiple components in `models/components`
            super().__init__()
            self.save_hyperparameters(logger=False)
            self.data_utils = data_utils # or self.hparams.data_utils
            self.model = model # or self.hparams.model
            
            # num_classes + 1 to account for the extra class used for padding
            self.val_acc = Accuracy(num_classes=num_labels+1, ignore_index=num_labels)
            self.test_acc = Accuracy(num_classes=num_labels+1, ignore_index=num_labels)


        def forward(self, input):
            # Use for inference only (separate from training_step)
            return self.hparams.model(**inputs)


        def step(self, batch):
            inputs, labels = batch, batch["labels"]
            outputs = self.forward(inputs)
            loss = outputs.loss
            preds = outputs.logits.argmax(dim=-1)
            return loss, preds, labels


        def training_step(self, batch, batch_idx):
            # the complete training loop
            loss, preds, labels = self.step(batch)
            self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
            return {"loss": loss}

        def validation_step(self, batch: Any, batch_idx: int):
            # the complete validation loop
            return {"loss": loss, "preds": true_preds, "labels": true_labels} 

        def test_step(self, batch: Any, batch_idx: int):
            # the complete test loop
            return {"loss": loss, "preds": true_preds, "labels": true_labels}

        def configure_optimizers(self):
            # define optimizers and LR schedulers
           return torch.optim.AdamW(
                params=self.hparams.model.parameters(), lr=self.hparams.lr
            )

The **LightningModule** has many convenience methods, and here are the core ones.
Check `LightningModule <https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html>` for further information.

Also, create a config file in ``configs/model``:

.. code-block:: yaml

    # The target Class
    _target_: SciAssist.models.cora_module.CoraLitModule
    lr: 2e-5
    data_utils:
        _target_: SciAssist.utils.data_utils.DataUtilsForTokenClassification


    # Parameters can be nested
    # When instantiating the LitModule, the following model will be automatically constructed.
    model:
        _target_: SciAssist.models.components.bert_token_classifier.BertForTokenClassifier
        model_checkpoint: "allenai/scibert_scivocab_uncased"
        output_size: 13
        cache_dir: ${paths.root_dir}/.cache/
        save_name: ${model_name}
        model_dir: ${paths.model_dir}

Create a Trainer and start training
"""""""""""""""""""""""""""""""""""

Create a trainer
::::::::::::::::

.. note::

    Actually there have been a perfect ``train_pipeline.py`` in SciAssist, 
    so there's no need to write a train pipeline yourself. 
    After preparing the **LightningDataModule** and **LightningModule**, you can
    train the model with `SciAssist/train.py` as shown above.
    But here's an introduction to this procedure in case of any unknown problem. 
    If you are not interested, skip to the next part :ref:`Train`.

The last step before starting training is to prepare a trainer config:

.. code-block:: yaml

    _target_: pytorch_lightning.Trainer

    accelerator: 'gpu'
    devices: 1
    min_epochs: 1
    max_epochs: 5

    # ckpt path
    resume_from_checkpoint: null

And then you can create a Pytorch lightning Trainer to manage the whole training process:

.. code-block:: python

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

More details are provided in `Trainer <https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#basic-use>`_ .

.. _Train:

Start training
:::::::::::::::

Finally, you can choose your config files and train your model with the command line:

.. code-block:: bash

    python SciAssist/train.py trainer=gpu datamodule=dataconfig model=modelconfig



4. Build a pipeline for the new task
--------------------------------------

General overview
''''''''''''''''
As SciAssist aims to serve users, you will need to write a pipeline easy to use.
The pipelines are stored in ``SciAssist/pipelines``. Generally, you need to:
    
    - Add the model configs in ``SciAssist/pipelines/__init__.py``.
    - Create a task-specific pipeline class inherited from ``Pipeline`` class.
    - Implement ``predict()`` function in the class.
    - Import the pipeline class in ``SciAssist/__init__.py``.  


Step-by-step recipe to add a pipeline
'''''''''''''''''''''''''''''''''''''''

Add the model configs
""""""""""""""""""""""

After you have a new model, add its corresponding configs to the dict ``Tasks`` in ``SciAssist/pipelines/__init__.py``.

- **model**: A ModelClass in ``models/components``.
- **model_dict_url**: URL to download the model dict. 
  ``Pipeline`` will load model weights from the `.pt` according to the URL.  
- **data_utils**: A DataUtils Class.

.. code-block:: python

    TASKS = {
        "new-task": {
            "new-model":  {
                "model": ModelClass,
                "model_dict_url": url,
                "data_utils": DataUtilsForNewTask
            },
            "default": {
                "model": ModelClass,
                "model_dict_url": url,
                "data_utils": DataUtilsForNewTask,
            },
        },

Create a task-specific pipeline class
""""""""""""""""""""""""""""""""""""""
A task-specific pipeline class should be inherited from ``Pipeline`` class, 
which loads a model according to `task_name` and `model_name`. 

.. autoclass:: SciAssist.pipelines.pipeline.Pipeline
    :members:


In your new pipeline class, specify a default `model_name` to choose a model and instantiate a :ref:`DataUtils <DataUtils>`.

.. code-block:: python

    from SciAssist.pipelines.pipeline import Pipeline
    from SciAssist.utils.pdf2text import process_pdf_file, get_reference

    class MyPipeline(Pipeline):

        def __init__(
                self, model_name: Optional[str] = "new-model", device: Optional[str] = "gpu",
                cache_dir = None,
                output_dir = None,
                temp_dir = None,
                tokenizer: PreTrainedTokenizer = None,
                checkpoint="allenai/scibert_scivocab_uncased",
                model_max_length=512,
        ):

            super().__init__(task_name="reference-string-parsing", model_name=model_name, device=device,
                            cache_dir=cache_dir, output_dir=output_dir, temp_dir=temp_dir)

            # Instantiate a datautils
            self.data_utils = self.data_utils(
                tokenizer=tokenizer,
                checkpoint=checkpoint,
                model_max_length=model_max_length
            )



Implement the ``predict()`` function
""""""""""""""""""""""""""""""""""""""

Next, you need to fill in the ``predict()`` function.
The pipeline works by this function, so its input should be directly 
path to a file, string or a list of string. And it return the results users expect.
You can use ``self.model`` to get your model for inference.

.. code-block:: python

    class MyPipeline(Pipeline):

        def predict(self, input, type="pdf", output_dir=None, temp_dir=None, save_results=True):
            # Get results from the input

            if output_dir is None:
                output_dir = self.output_dir
            if temp_dir is None:
                temp_dir = self.temp_dir

            
            if type in ["str", "string"]:
                results = self._predict_for_string(example=input)
            elif type in ["txt", "text"]:
                results = self._predict_for_text(filename=input)
            elif type == "pdf":
                results = self._predict_for_pdf(filename=input)

            # Save predicted results as a text file
            if save_results and type not in ["str", "string"]:
                os.makedirs(output_dir, exist_ok=True)
                output_file = os.path.basename(input)

                # The output filename is default to input filename with the abbr. of the task name appending to it.
                with open(os.path.join(output_dir, f"{output_file[:-4]}_rsp.txt"), "w") as output:
                    for res in results:
                        output.write(res["tagged_text"] + "\n")


            return results


An example of a task-specific pipeline and the ``predict()``:

.. autofunction:: SciAssist.ReferenceStringParsing.predict
    :noindex:
    
Make the new pipeline easy to import 
""""""""""""""""""""""""""""""""""""""""""""""

After you get a new pipeline, import it in ``SciAssist/__init__.py``.

.. code-block:: python

    from SciAssist.pipelines.new_task import MyPipeline

Finally, users can import it directly from ``SciAssist``.

.. code-block:: python

    from SciAssist import MyPipeline

    pipeline = MyPipeline()
    res = pipeline.predict(input)

You can change other configs in this way too. For example:

Train model with default configuration:

.. code-block:: bash

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
