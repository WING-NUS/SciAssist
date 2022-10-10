Usage
=====

.. _Usage:

API
---------

Pipeline usage
''''''''''''''

SciAssist provides pipelines to make it simple
to use any provided model for inference on various tasks.

Each task is associated with a pipeline, 
and the pipeline automatically loads a default model capable of inference for your task.

1.  Start by creating a task-specific pipeline. Taking ``reference string parsing`` as example:

.. code-block:: python

    >>> from SciAssist import ReferenceStringParsing

    >>> pipeline = ReferenceStringParsing()

2.  Pass your input string to the pipeline:
   
.. code-block:: python

    >>> pipeline.predict(
    ...     "Waleed Ammar, Matthew E. Peters, Chandra Bhagavat- ula, and Russell Power. 2017. The ai2 system at semeval-2017 task 10 (scienceie): semi-supervised end-to-end entity and relation extraction. In ACL workshop (SemEval).", 
    ...     type="str"
    ... )
    [{'tagged_text': '<author>Waleed</author> <author>Ammar,</author> <author>Matthew</author> <author>E.</author> <author>Peters,</author> <author>Chandra</author> <author>Bhagavat-</author> <author>ula,</author> <author>and</author> <author>Russell</author> <author>Power.</author> <date>2017.</date> <title>The</title> <title>ai2</title> <title>system</title> <title>at</title> <title>semeval-2017</title> <title>task</title> <title>10</title> <title>(scienceie):</title> <title>semi-supervised</title> <title>end-to-end</title> <title>entity</title> <title>and</title> <title>relation</title> <title>extraction.</title> <booktitle>In</booktitle> <booktitle>ACL</booktitle> <booktitle>workshop</booktitle> <booktitle>(SemEval).</booktitle>', 
    'tokens': ['Waleed', 'Ammar,', 'Matthew', 'E.', 'Peters,', 'Chandra', 'Bhagavat-', 'ula,', 'and', 'Russell', 'Power.', '2017.', 'The', 'ai2', 'system', 'at', 'semeval-2017', 'task', '10', '(scienceie):', 'semi-supervised', 'end-to-end', 'entity', 'and', 'relation', 'extraction.', 'In', 'ACL', 'workshop', '(SemEval).'], 
    'tags': ['author', 'author', 'author', 'author', 'author', 'author', 'author', 'author', 'author', 'author', 'author', 'date', 'title', 'title', 'title', 'title', 'title', 'title', 'title', 'title', 'title', 'title', 'title', 'title', 'title', 'title', 'booktitle', 'booktitle', 'booktitle', 'booktitle']}]

If you have more than one string, use ``predict()`` and pass your input as a list:

.. code-block:: python
    
    >>> pipeline.predict(
    ...     [
    ...         "Waleed Ammar, Matthew E. Peters, Chandra Bhagavat- ula, and Russell Power. 2017. The ai2 system at semeval-2017 task 10 (scienceie): semi-supervised end-to-end entity and relation extraction. In ACL workshop (SemEval).",
    ...         "Isabelle Augenstein, Mrinal Das, Sebastian Riedel, Lakshmi Vikraman, and Andrew D. McCallum. 2017. Semeval 2017 task 10 (scienceie): Extracting keyphrases and relations from scientific publications. In ACL workshop (SemEval)."
    ...     ], 
    ...     type="str"
    ... )
    [{'tagged_text': '<author>Waleed</author> <author>Ammar,</author> <author>Matthew</author> <author>E.</author> <author>Peters,</author> <author>Chandra</author> <author>Bhagavat-</author> <author>ula,</author> <author>and</author> <author>Russell</author> <author>Power.</author> <date>2017.</date> <title>The</title> <title>ai2</title> <title>system</title> <title>at</title> <title>semeval-2017</title> <title>task</title> <title>10</title> <title>(scienceie):</title> <title>semi-supervised</title> <title>end-to-end</title> <title>entity</title> <title>and</title> <title>relation</title> <title>extraction.</title> <booktitle>In</booktitle> <booktitle>ACL</booktitle> <booktitle>workshop</booktitle> <booktitle>(SemEval).</booktitle>', 'tokens': ['Waleed', 'Ammar,', 'Matthew', 'E.', 'Peters,', 'Chandra', 'Bhagavat-', 'ula,', 'and', 'Russell', 'Power.', '2017.', 'The', 'ai2', 'system', 'at', 'semeval-2017', 'task', '10', '(scienceie):', 'semi-supervised', 'end-to-end', 'entity', 'and', 'relation', 'extraction.', 'In', 'ACL', 'workshop', '(SemEval).'], 'tags': ['author', 'author', 'author', 'author', 'author', 'author', 'author', 'author', 'author', 'author', 'author', 'date', 'title', 'title', 'title', 'title', 'title', 'title', 'title', 'title', 'title', 'title', 'title', 'title', 'title', 'title', 'booktitle', 'booktitle', 'booktitle', 'booktitle']}, 
    {'tagged_text': '<author>Isabelle</author> <author>Augenstein,</author> <author>Mrinal</author> <author>Das,</author> <author>Sebastian</author> <author>Riedel,</author> <author>Lakshmi</author> <author>Vikraman,</author> <author>and</author> <author>Andrew</author> <author>D.</author> <author>McCallum.</author> <date>2017.</date> <title>Semeval</title> <title>2017</title> <title>task</title> <title>10</title> <title>(scienceie):</title> <title>Extracting</title> <title>keyphrases</title> <title>and</title> <title>relations</title> <title>from</title> <title>scientific</title> <title>publications.</title> <booktitle>In</booktitle> <booktitle>ACL</booktitle> <booktitle>workshop</booktitle> <booktitle>(SemEval).</booktitle>', 'tokens': ['Isabelle', 'Augenstein,', 'Mrinal', 'Das,', 'Sebastian', 'Riedel,', 'Lakshmi', 'Vikraman,', 'and', 'Andrew', 'D.', 'McCallum.', '2017.', 'Semeval', '2017', 'task', '10', '(scienceie):', 'Extracting', 'keyphrases', 'and', 'relations', 'from', 'scientific', 'publications.', 'In', 'ACL', 'workshop', '(SemEval).'], 'tags': ['author', 'author', 'author', 'author', 'author', 'author', 'author', 'author', 'author', 'author', 'author', 'author', 'date', 'title', 'title', 'title', 'title', 'title', 'title', 'title', 'title', 'title', 'title', 'title', 'title', 'booktitle', 'booktitle', 'booktitle', 'booktitle', 'date']}]

Any additional parameters for the specific task can also be included in ``predict(...)``.
For example, the ``reference string parsing`` task has a ``dehyphen`` parameter. 
If you want to remove hyphens in the raw text, set the ``dehyphen``:

.. code-block:: python
    
    >>> pipeline.predict(
    ...     "Waleed Ammar, Matthew E. Peters, Chandra Bhagavat- ula, and Russell Power. 2017. The ai2 system at semeval-2017 task 10 (scienceie): semi-supervised end-to-end entity and relation extraction. In ACL workshop (SemEval).", 
    ...     type="str",
    ...     dehyphen=True
    ... )


Choose a model and tokenizer
""""""""""""""""""""""""""""
You can choose a model you'd like to use for your task.
All provided models are shown in :doc:`Models`. 

For example, create a pipeline to summarize a document and specify a model and tokenizer: 

.. code-block:: python

    >>> from SciAssist import Summarization
    >>> from transformers import AutoTokenizer

    >>> tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
    >>> pipeline = ReferenceStringParsing(model_name="bart-cnn-on-mup", tokenizer=tokenizer)


The task-specific pipelines
'''''''''''''''''''''''''''



Reference string parsing
""""""""""""""""""""""""

.. _ReferenceStringParsing:

.. autoclass:: SciAssist.ReferenceStringParsing
    :members: predict

    


Single document summarization 
"""""""""""""""""""""""""""""

.. _Summarization:

.. autoclass:: SciAssist.Summarization
    :members: predict

