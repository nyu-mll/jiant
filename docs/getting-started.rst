.. role:: mono

================
Getting Started
================

--------------------
1. Get :mono:`jiant`
--------------------

You can get :mono:`jiant` by cloning from GitHub:

.. code-block:: shell

   git clone https://github.com/nyu-mll/jiant.git

------------------------------------
2. Install dependencies
------------------------------------

We recommend installing :mono:`jiant`'s dependencies in a virtual environment:

.. code-block:: shell

   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt

.. note:: If you plan to contribute to :mono:`jiant`, install additional dependencies with ``pip install -r requirements-dev.txt``.     

--------------------
3. Use :mono:`jiant`
--------------------

.. code-block:: shell

   python jiant

.. note:: Running :mono:`jiant` without providing an experiment config argument (e.g., ``python jiant``) will run a minimal example experiment. See :ref:`examples` for examples that specify custom experiemental configuration via command line arguments or via configuration files.