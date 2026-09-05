A Undirected Graph class
========================

This tutorial implements an undirected graph, the ``UndirectedGraph`` class
familiar from a dict-of-adjacency-lists implementation, in Physika.

Design
------

Physika has no ``dict`` and no growable container, and its values are
immutable, so the design differs from the Python in three ways:

- The adjacency structure is a fixed-size matrix, ``adjacency: ℝ[n, n]``,
  with a ``1.0`` at ``[u, v]`` wherever an edge connects ``u`` and ``v``;
- A method that would mutate the graph instead returns a **new** ``Graph``;
- Adding a vertex changes the matrix shape, which a method cannot do to its
  own ``this`` in place, so the new vertex count is passed in explicitly as
  a ``ℕ`` parameter.

The UndirectedGraph class
----------------

.. code-block:: text

   class UndirectedGraph():
       adjacency: ℝ[n, n]
       def num_vertices() → ℝ:
           return len(this.adjacency) * 1.0
       def has_edge(u: ℝ, v: ℝ) → ℝ:
           m: ℝ[n, n] = this.adjacency
           r: ℝ[n] = m[u]
           return r[v]
       def degree(u: ℝ) → ℝ:
           m: ℝ[n, n] = this.adjacency
           r: ℝ[n] = m[u]
           return sum(r)
       def neighbors(u: ℝ) → ℝ[n]:
           m: ℝ[n, n] = this.adjacency
           return m[u]
       def add_edge(u: ℝ, v: ℝ):
           m: ℝ[n, n] = this.adjacency
           k: ℕ = len(m)
           new_adj: ℝ[n, n] = for a : ℕ(k) → for b : ℕ(k) → m[a, b]
           new_adj[u, v] = 1.0
           new_adj[v, u] = 1.0
           this.adjacency = new_adj
       def grow_adjacency(new_n: ℕ) → ℝ[new_n, new_n]:
           old: ℝ[n, n] = this.adjacency
           result: ℝ[new_n, new_n] = for a : ℕ(new_n) → for b : ℕ(new_n) → (a + b) * 0.0
           m: ℕ = len(old)
           for a : ℕ(m):
               for b : ℕ(m):
                   result[a, b] = old[a, b]
           return result
       def add_vertex(new_n: ℕ):
           this.adjacency = this.grow_adjacency(new_n)

``add_edge`` copies the matrix and flips two entries, so its return type
``ℝ[n, n]`` is unchanged. ``add_vertex`` cannot do the same: it needs a
bigger matrix, so it calls ``grow_adjacency``, whose ``new_n`` is bound as
its own ``ℕ`` parameter and reused in the return type ``ℝ[new_n, new_n]`` --
the only way Physika lets a method's output shape differ from ``this``'s.
The old matrix is copied into the top-left block; the new row and column
are left zero, i.e. the new vertex starts isolated.

A ``Graph`` is built through a function rather than a literal matrix:

.. code-block:: text

   def empty_graph(n_vertices: ℕ): Graph:
       z: ℝ[n_vertices, n_vertices] = for a : ℕ(n_vertices) → for b : ℕ(n_vertices) → (a + b) * 0.0
       g: UndirectedGraph = UndirectedGraph()
       g.adjacency = z
       return g

Example
-------

.. code-block:: text

   n0: ℕ = 3
   g = empty_graph(n0)
   g.num_vertices()

   g.add_edge(0.0, 1.0)
   g.add_edge(1.0, 2.0)

   g.neighbors(1.0)
   g.degree(1.0)
   g.has_edge(0.0, 2.0)

   n3: ℕ = 4
   g.add_vertex(n3)
   g.add_edge(2.0, 3.0)
   g.degree(3.0)

Output::

   3.0 ∈ ℝ
   [1.0, 0.0, 1.0] ∈ ℝ[3]
   2.0 ∈ ℝ
   0.0 ∈ ℝ
   1.0 ∈ ℝ

Vertex ``1`` connects to both ``0`` and ``2``, so its degree is ``2``;
``0`` and ``2`` are not directly connected. After ``add_vertex`` the graph
has 4 vertices, and connecting the new vertex ``3`` to vertex ``2`` gives it
degree ``1``.
