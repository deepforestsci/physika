Installation
============

Requirements
------------

- Python >= 3.9
- PyTorch
- PLY
- NumPy

Install from source
-------------------

.. code-block:: bash

   git clone https://github.com/deepforestsci/physika.git
   cd physika
   pip install -e ".[dev]"

Running a program
-----------------

.. code-block:: bash

   python -m physika.execute examples/example_arrays.phyk

Flags
-----

``--print-ast``
    Print the parsed Abstract Syntax Tree and exit.

``--print-code``
    Print the generated PyTorch source code and exit.
