Easy Language Feature (ELF)
===========================

Every Physika program is processed through a pipeline of parser and lexer rules that produce an Abstract Syntax Tree (AST).
Physika's type system then verifies each AST node for type correctness by comparing inferred types with declared types.
If no type errors are found, the AST is passed to the code generator, which emits PyTorch/Python code as strings for both forward and backward passes.
ELFs allow new rules to be easily added at each stage of this pipeline, parsing, type checking, and code generation, without modifying the core Physika codebase, which facilitate implementation and maintenance.

Each ``ELF`` subclass declares a unique ``name`` and overrides four methods that control its behavior in Physika:

* **Parser rules**: PLY grammar functions that introduce new syntax rules.
* **Lexer rules**: New reserved keywords and token names for the lexer.
* **Type rules**: Physika's type checker handlers that receive AST nodes and update the type environment following Hindley-Milner type inference.
* **Forward rules**: Code generation handlers that emit PyTorch code as strings from AST nodes.
* **Backward rules**: Differentiation handlers for custom gradient computation.

Feature Registry
----------------

Once an ELF is defined, its rules must be registered so that each Physika module can use them. This is handled by ``FeatureRegistry``, which stores incoming rules and dispatches them to the appropriate pipeline stage.
``FeatureRegistry`` class have seven methods:

* ``register``: Accepts an ELF instance and stores its rules in dispatch tables keyed by ``ELF.name``.
* ``add_lexer_rules``: Adds new PLY tokens and reserved keywords to ``physika.lexer``. If any function-based token rules are added, the lexer is rebuilt and the updated instance is swapped into the ``IndentLexer`` wrapper class at ``physika.lexer``.
* ``add_parser_rules``: Injects PLY grammar functions into ``physika.parser`` so that ``yacc.yacc()`` register them.
* ``has_type_rule``: Returns ``True`` if a type inference handler is registered for a given AST node tag.
* ``dispatch_type``: Calls the registered type inference handler for an AST node tag and returns its inferred type.
* ``dispatch_forward``: Calls the registered code generation handler for an AST node tag and returns the emitted Python source string.
* ``dispatch_backward``: Calls the registered differentiation handler for an AST node tag and returns the emitted Pytorch gradient code.

Example: While Loop Feature
----------------------------

The following example implements a ``while_loop`` statement as a complete ELF,
exercising all five rule types: lexer, parser, type, forward, and backward.

.. code-block:: python

    from physika.elf import ELF
    from physika.utils.ast_utils import generate_statement, condition_to_expr

    class WhileLoopFeature(ELF):
        name = "while_loop"

        def lexer_rules(self):
            # Adds "while" as a reserved keyword mapped to the WHILE token.
            return {"reserved": {"while": "WHILE"}, "tokens": ["WHILE"]}

        def parser_rules(self):
            def p_while(p):
                """statement : WHILE condition COLON NEWLINE INDENT statements DEDENT"""
                p[0] = ("while_loop", p[2], p[6])
            return [p_while]

        def type_rules(self):
            def check(node, env, s, func_env, class_env, add_error, infer_expr):
                _, cond, _ = node
                cond_t, s = infer_expr(cond, env, s, func_env, class_env, add_error)
                if cond_t != ("scalar",):
                    add_error("while condition must be scalar")
                return None, s
            return {"while_loop": check}

        def forward_rules(self):
            def emit(node):
                _, cond, body = node
                body_lines = [f"    {generate_statement(s, set())}" for s in body]
                body_code = "\n".join(body_lines) if body_lines else "    pass"
                return f"while {condition_to_expr(cond)}:\n{body_code}"
            return {"while_loop": emit}

        def backward_rules(self):
            def grad(grad_output):
                # Adjoint method: reverse through the recorded tape,
                # applying the body VJP at each step.
                return (
                    f"_adj = {grad_output}\n"
                    f"for _state in reversed(_while_tape):\n"
                    f"    _adj = _body_vjp(_state, _adj)"
                )
            return {"while_loop": grad}

Once the ELF is defined, register it with ``FeatureRegistry`` and use the dispatch
methods to type check and generate code for ``while_loop`` nodes (Each registry will occur in the appropiate file path):

.. code-block:: python

    # At __init__.py of ELFs dir
    from physika.elf import FeatureRegistry

    reg = FeatureRegistry()
    reg.register(WhileLoopFeature())

.. code-block:: python

    # At physika/utils/types.py
    # Check that type and forward rules were registered
    reg.has_type_rule("while_loop")   # True

.. code-block:: python

    # At physika/utils/ast_utils.py
    # Forward dispatch: emit Python code for a while_loop AST node
    # Parser rules will add a node like this to AST:
    # node = (
    #    "while_loop",
    #    ("cond_lt", ("var", "n"), ("num", 10.0)),
    #    [("assign", "n", ("add", ("var", "n"), ("num", 1.0)), 1)],
    #)
    reg.dispatch_forward("while_loop", node)
    # 'while n < 10.0:\n    n = (n + 1.0)'

    # Backward dispatch: emit adjoint gradient code
    reg.dispatch_backward("while_loop", "dL_dn")
    # '_adj = dL_dn\nfor _state in reversed(_while_tape):\n    _adj = _body_vjp(_state, _adj)'


Features
--------
New ELF subclasses are defined at ``physika.features`` directory as python files, where lexer, parser, and code generation rules are added.
Tests for new language features should be added to ``physika/tests/test_features/`` folder.

Classes
~~~~~~~
The classes ELF subclass adds support for defining classes in Physika. Two new lexer tokens were added. ``DOT`` (for field and method access) and the ``CLASS`` reserved keyword.

A class can be defined with or without explicit constructor parameters:

.. code-block:: text

    # No constructor parameters
    class Particle:
        mass : ℝ
        def ke() : ℝ:
            return 0.5 * this.mass

    # Explicit constructor parameters
    class Linear(w: ℝ, b: ℝ):
        def λ(x: ℝ) → ℝ:
            return w * x + b

**AST node types** produced by the new parser rules:

* ``("class_def", name)``: The class definition is stored in the parser's symbol table under class ``name`` and is retrieved in code generation step.
* ``("field_decl", name, type)``: A field declaration inside the class body. For example, ``mass : ℝ`` produces ``("field_decl", "mass", "ℝ")``.
* ``("method_def", method_dict)``: Method definition, where ``method_dict`` holds the method name, parameter list, return type, body statements.
* ``("struct_type", name)``: Type annotation that refers to an instance of current or another class. For example, ``pos : Particle`` produces ``("struct_type", "Particle")``.
* ``("field_access", obj, field)``: Reads a field from an instance.
* ``("method_call", obj, method, args)``: Calls a method on an instance.

``make_parser_rules`` produce sixteen PLY grammar functions that handles class declarations with and without constructor parameters, field declarations, methods with and without parameters, methods with intermediate statements, single-value returns, and two-value tuple returns.

**this vs self**

Inside a Physika method body, the current instance is referred as ``this``. The parser produces ``("field_access", ("var", "this"), "mass")`` (AST) for ``this.mass`` (Physika code). During code generation, ``emit_method`` rewrites every occurrence of ``this`` to ``self`` so the emitted Python is runnable. The alias ``this = self`` is also inserted at the top of each method to avoid issues at runtime.

**Code generation**

``generate_class`` transforms a parsed class definition into a ``torch.nn.Module`` subclass:

* The class inherits from ``nn.Module``.
* An ``__init__`` method is generated from the constructor parameters. Scalar (``ℝ``) and tensor parameters are converted with ``torch.as_tensor`` objects. If the class defines learnable parameters (e.g. inside a forward method), these are wrapped in ``nn.Parameter``.
* Each Physika method is emitted by ``emit_method``, walking down the AST, which handles ``this`` to ``self`` rewriting and fields substitution via ``replace_class_params``
* A ``params`` property and an ``update`` method are appended to every generated class to support manual gradient-descent updates.

**Example**

The following physika program shows some example on Physika clases:

.. literalinclude:: ../examples/physika_class.phyk
   :language: text


Random sampling
~~~~~~~~~~~~~~~
``RandomnessFeature`` ELF adds support for random sampling from probability distributions.
Physika random sampling syntax allows users to declare a random variable with a distribution (and its arguments) and shape.

.. code-block:: text

    # Sample a scalar from a normal distribution with mean 0 and std 1
    x : ℝ ~ Normal(0.0, 1.0)

    # Sample a 3x2 matrix from a normal distribution with mean 0 and std 1
    y : ℝ[3, 2] ~ for i : ℕ(3) → ε : ℝ[2] ~ Normal(μ, σ, 2)

    # Sample a 10x5x2 tensor from a normal distribution with mean 0 and std 1
    z : ℝ[10, 5, 2] ~ for i : ℕ(10) → for j : ℕ(5) → ε : ℝ[2] ~ Normal(μ, σ, 2)

Physika supports differentiable sampling following Stochastich Computation Graphs (SCG) framework [1]_, where sampling statements
are represented as stochastic nodes in the computation graph and gradients are computed by backpropagating through these nodes
with the reparameterization trick (for continous distributions) or score function estimators (for non-continous distributions).
``RandomnessFeature`` default code generation emits reparameterized sampling for continous distributions (``Normal/Gaussian``, 
``Beta``, ``Uniform``, ``Gamma``) and score function estimators
for ``Bernoulli``.

Estimators can be defined per distribution by given "reparam", "socre", or "none" argument, for example::

    # Sampling using pathwise derivative estimator (reparameterization trick)
    x : ℝ ~ Normal(0.0, 1.0, "reparam")

    # Sampling using score function estimator
    y : ℝ ~ Normal(0.0, 1.0, "score")


Supported Distributions
^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :widths: 14 20 22 14 12 18
   :header-rows: 1

   * - Name
     - Physika syntax
     - Parameters
     - Default estimator
     - Unicode alias
     - PyTorch backend
   * - **Normal**
     - ``x ~ Normal(μ, σ)``
     - μ: mean, σ: std dev
     - ``reparam``
     - ``𝒩``
     - ``torch.distributions.Normal``
   * - **Uniform**
     - ``x ~ Uniform(a, b)``
     - a: lower bound, b: upper bound
     - ``reparam``
     - ``𝒰``
     - ``torch.distributions.Uniform``
   * - **Beta**
     - ``x ~ Beta(α, β)``
     - α: concentration1, β: concentration0
     - ``reparam``
     - ``ℬ``
     - ``torch.distributions.Beta``
   * - **Gamma**
     - ``x ~ Gamma(k, θ)``
     - k: concentration, θ: rate
     - ``reparam``
     - ``Γ``
     - ``torch.distributions.Gamma``
   * - **Bernoulli**
     - ``x ~ Bernoulli(p)``
     - p: probability of 1
     - ``score`` (fixed)
     - —
     - ``torch.distributions.Bernoulli``

Aliases for probability distributions are also supported, for ``Normal``, ``Uniform``, ``Beta`` distributions. These are as follows::
    
* ``𝒩`` → ``Normal``
* ``𝒰`` → ``Uniform``
* ``ℬ`` → ``Beta``

For adding new distributions, new lexer and code generation rules must be added to ``RandomnessFeature``. First, add a function handler to emit Pytorch code for the new distribution at ``features/randomness.py``.
Including an alias for a distribution is optional and must be done at ``lexer_rules()`` method. Finally, add the newly defined distribution emit code handler at ``forward_rules()`` dispatch table as ``"call:NewDist": make_call_emit(new_dist),``

**Type checking**

``RandomnessFeature`` checks that sampling statements are well-typed by verifying that the distribution call is consistent with the declared type of the variable being sampled.
If type annotations are not included, type system infers an registers in type enviroment to continue checking Physika programs.

The number of size arguments in the distribution call must match the rank of the declared type.
A scalar declaration (``ℝ``) requires no size args, and a 1D array declaration (``ℝ[n]``) requires one and so on.
A mismatch is recorded as a type error:

  .. code-block:: text

      # Error: declared ℝ but Normal(...) produces a ℝ[n] sample
      x : ℝ ~ Normal(0.0, 1.0, 100)

      # Error: declared ℝ[100] but Normal(...) produces a ℝ sample
      x : ℝ[100] ~ Normal(0.0, 1.0)

When ranks match, each declared dimension is compared against the corresponding size argument.


References
----------

.. [1] John Schulman, Nicolas Heess, Theophane Weber, and Pieter Abbeel.
       Gradient estimation using stochastic computation graphs. Advances
       in neural information processing systems, 28, 2015.