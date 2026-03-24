Examples
========

All examples files are under ``examples/`` directory. Run any of them with:

.. code-block:: bash

   physika examples/<name>.phyk

Add ``--print-ast`` or ``--print-code`` to inspect the generated AST or
PyTorch source.

----

Data Types
----------

Arrays
~~~~~~

.. literalinclude:: ../examples/example_arrays.phyk
   :language: text

Matrices
~~~~~~~~

.. literalinclude:: ../examples/example_matrices.phyk
   :language: text

Tensors (contravariant / covariant)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../examples/example_tensors.phyk
   :language: text

----

Control Flow Operators & Differentiation
----------------------------------------

Conditionals
~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../examples/if_else_contexts.phyk
   :language: text

Differentiable conditional
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../examples/diff_ifelse.phyk
   :language: text

Differentiable sin / cos
~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../examples/diff_sincos.phyk
   :language: text

Threshold optimisation (gradient descent)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../examples/diff_threshold.phyk
   :language: text

For loops and for-expressions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../examples/for.phyk
   :language: text

Factorial (recursive)
~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../examples/factorial.phyk
   :language: text

----

Classes
-------


Neural network classes
~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../examples/networks.phyk
   :language: text

Training a fully connected network
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../examples/train_fully_connected.phyk
   :language: text

----

Physics Simulations
-------------------

.. note::
   The following three examples open a GUI window (matplotlib / PyVista).

Harmonic oscillator
~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../examples/harmonic_oscillator.phyk
   :language: text

Simple pendulum (RK4)
~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../examples/pendulum_rk4.phyk
   :language: text

Spring pendulum (RK4)
~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../examples/spring_pendulum_rk4.phyk
   :language: text

Hamiltonian Neural Network (HNN)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../examples/hnn.phyk
   :language: text