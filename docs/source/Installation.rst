Installation
============

.. _Installation:


Install with pip
""""""""""""""""""""""""""""""""
You should install SciAssist in a virtual environment. 
If you’re unfamiliar with Python virtual environments, take a look at this
`guide <https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/>`_:

First, creating a virtual environment in your project directory: 

.. code-block:: bash

    python -m venv .env

Activate the virtual environment. On Linux and MacOs:

.. code-block:: bash
    
    source .env/bin/activate

On Windows:

.. code-block:: bash

    .env/Scripts/activate


Option:
    You may need to install pytorch according to `instructions <https://pytorch.org/get-started/>`_.

Now you're ready to install SciAssist with pip:

.. code-block:: bash

    pip install SciAssist

Setup Grobid for pdf processing (Only for Linux)
"""""""""""""""""""""""""""""""""""""""""""""""""

If you want to process pdf files with Grobid server, then you need to install it first.

.. note:: 

    **Grobid is not available for Windows.**
    So we use **pyminer.six** to process pdfs on Windows, and Windows users can skip the following part.
    For Linux and MacOs, we use **Grobid** by default. 

You will need to have Java installed on your machine. Then, you can install 
your own version of Grobid and get it running.
After you install the package, you can simply setup grobid with the CLI:

.. code-block:: bash

    setup_grobid

This will setup Grobid. And after installation, starts the Grobid server with:

.. code-block:: bash

    run_grobid


Now you are ready for inference on your task.
