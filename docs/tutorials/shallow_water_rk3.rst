Two-dimensional shallow water with Runge–Kutta time stepping
============================================================

What happens when one part of a water surface is higher than the rest?
Water flows away from the raised area, and waves spread across the surface.
In this tutorial, we simulate that motion starting from a small, smooth
bump in otherwise still water.

We use the shallow-water equations, which describe flow when the water is
much shallower than the horizontal distance over which it changes. Instead
of tracking motion at every depth, we track the water depth and the average
horizontal flow in two directions. This makes the model simpler than a full
three-dimensional simulation.

We divide a square into small cells, calculate how water and momentum move
between neighbouring cells, and advance the solution in time using a
Runge–Kutta method. The square has periodic boundaries: a wave leaving one
edge returns at the opposite edge. We build the Physika implementation step
by step below, with the complete program at the end. For installation and
language basics, see :doc:`intro_to_physika` and :doc:`/install`.

What “shallow water” means
--------------------------

Each grid cell represents a water column, with velocities averaged over its
depth. The grid resolves two horizontal directions, not vertical structure.
Shallow-water theory assumes depth is small relative to horizontal scales and
pressure is hydrostatic. We use a flat, stationary bed without friction,
wind, rotation, or external sources.

Let :math:`x` and :math:`y` be horizontal coordinates and :math:`t` be time.
The total water depth is :math:`h(x,y,t)`, the constant reference depth is
:math:`H`, and :math:`\eta(x,y,t)` is the surface displacement from that
reference level:

.. math::

   h = H + \eta, \qquad \eta = h-H.

Thus ``h`` is not just the bump height. The bump is the smaller displacement
:math:`\eta`. The horizontal depth-averaged velocities are :math:`u(x,y,t)`
in the positive :math:`x` direction and :math:`v(x,y,t)` in the positive
:math:`y` direction. The positive parameter :math:`g` is gravity in the units
used by the simulation.

Units and scales
~~~~~~~~~~~~~~~~

All numbers in the program are dimensionless. One consistent conversion to
physical units uses a horizontal length scale :math:`L_*`, a depth scale
:math:`D_*`, and physical gravitational acceleration :math:`g_*`. The star
on these constants marks a dimensional scale. Define the velocity and time
scales :math:`U_*` and :math:`T_*` by

.. math::

   U_* = \sqrt{g_*D_*}, \qquad T_* = \frac{L_*}{U_*}.

With the subscript “phys” denoting dimensional quantities, the conversion is

.. math::

   \begin{aligned}
   x_{\mathrm{phys}} &= L_*x, & y_{\mathrm{phys}} &= L_*y,\\
   t_{\mathrm{phys}} &= T_*t, & h_{\mathrm{phys}} &= D_*h,\\
   u_{\mathrm{phys}} &= U_*u, & v_{\mathrm{phys}} &= U_*v,\\
   g &= \frac{g_*D_*}{U_*^2}=1.
   \end{aligned}

Depth, reference depth, and surface displacement share the scale :math:`D_*`.
Setting :math:`H=1` does not imply a deep ocean: horizontal and vertical scales
differ. Choose :math:`D_*/L_*` small, with the bump also much wider than the depth.

From moving water to conservation laws
--------------------------------------

Water transports mass and momentum between cells; pressure also transfers
momentum. We evolve depth and both momentum components together.

The *primitive* fields are depth and velocities. The *conserved* state
:math:`\mathbf q` and the recovery of its primitive fields are

.. math::

   \mathbf q = \begin{pmatrix}h\\hu\\hv\end{pmatrix},
   \qquad u=\frac{hu}{h}, \qquad v=\frac{hv}{h}, \qquad h>0.

Here :math:`hu` and :math:`hv` mean products, not extra velocity variables.
For constant density they are horizontal momenta per unit horizontal area
with density divided out. Their integrals, and the integral of depth, are
conserved on this periodic domain.

Let :math:`\mathbf F` be the vector of physical fluxes in the positive
:math:`x` direction and :math:`\mathbf G` the corresponding vector in the
positive :math:`y` direction. A flux measures transport across a face per unit
face length and per unit time. The equations are

.. math::

   \frac{\partial\mathbf q}{\partial t}
   +\frac{\partial\mathbf F(\mathbf q)}{\partial x}
   +\frac{\partial\mathbf G(\mathbf q)}{\partial y}=\mathbf 0,

   \mathbf F(\mathbf q)=
   \begin{pmatrix}hu\\hu^2+\tfrac12gh^2\\huv\end{pmatrix},
   \qquad
   \mathbf G(\mathbf q)=
   \begin{pmatrix}hv\\huv\\hv^2+\tfrac12gh^2\end{pmatrix}.

Here :math:`\partial` denotes a partial derivative and :math:`\mathbf 0` a
zero vector. The rows conserve water, x momentum, and y momentum respectively.
The hydrostatic contribution is

.. math::

   p=\frac12gh^2.

Here :math:`p` is depth-integrated pressure divided by density, named
``pressure`` in the program. It is not the pressure at a single vertical
position. The remaining momentum-flux terms describe transport by the flow.
This nonlinear system is the PDE (partial differential equation) we discretize.

From the PDE to cell averages
-----------------------------

Divide the unit square into :math:`N_x` cells along :math:`x` and :math:`N_y`
cells along :math:`y`. Their widths and centre coordinates are

.. math::

   \begin{aligned}
   \Delta x &= \frac1{N_x}, & \Delta y &= \frac1{N_y},\\
   x_i &= (i+\tfrac12)\Delta x, & y_j &= (j+\tfrac12)\Delta y,\\
   i &= 0,\ldots,N_x-1, & j &= 0,\ldots,N_y-1.
   \end{aligned}

Indices :math:`i,j` label cells. The cell :math:`C_{ij}`, centred at
:math:`(x_i,y_j)`, has area :math:`\Delta x\Delta y` and average state

.. math::

   \overline{\mathbf q}_{ij}(t)
   =\frac1{\Delta x\Delta y}\int_{C_{ij}}\mathbf q(x,y,t)\,dx\,dy.

The bar denotes an average over the cell; the integral sums the field over
its area. Integrating the PDE over the cell turns spatial derivatives into
boundary fluxes:

.. math::

   \frac{d\overline{\mathbf q}_{ij}}{dt}
   =-\frac{\overline{\mathbf F}_{i+1/2,j}
                  -\overline{\mathbf F}_{i-1/2,j}}{\Delta x}
    -\frac{\overline{\mathbf G}_{i,j+1/2}
                  -\overline{\mathbf G}_{i,j-1/2}}{\Delta y}.

Barred fluxes are averages along cell faces; :math:`d/dt` is a time derivative.
Faces :math:`i+1/2,i-1/2` are right/left, and :math:`j+1/2,j-1/2` are top/bottom.
The minus signs give incoming minus outgoing transport.

Our numerical state :math:`\mathbf q_{ij}` approximates this cell average;
we drop its bar below. We approximate face fluxes using constant states in
each cell, giving first-order spatial accuracy.

Array axes and periodic neighbours
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``q[k, i, j]`` has type ``ℝ[3, m, n]``. Component ``k`` is 0 for depth,
1 for x momentum, and 2 for y momentum; ``m,n`` are the grid sizes ``Nx,Ny``.
``ℝ`` denotes real values and ``ℕ`` natural-number counts and indices.
The shape parameter ``n`` is unrelated to the time index below.

Each extracted component is a ``ℝ[m, n]`` array: **axis 0 is x and axis 1
is y**. These axis numbers apply after selecting ``q[k]``. Transpose a
component when plotting it with image software that puts rows vertically.

``roll(w, -1, 0)`` places the next x-cell value at each current cell;
``roll(w, 1, 0)`` supplies the previous x-cell value. Axis 1 does the same in
y. Here ``w`` is any one component field. Algebraically,

.. math::

   [\operatorname{roll}(w,-1,0)]_{ij}=w_{i+1,j}, \qquad
   [\operatorname{roll}(w,-1,1)]_{ij}=w_{i,j+1}.

The operator :math:`\operatorname{roll}` wraps indices at each edge. Waves
leaving one edge re-enter at the opposite edge: these are periodic boundaries,
not walls or open boundaries.

Rusanov fluxes at cell faces
----------------------------

A shared face flux makes the amount leaving one cell enter its neighbour.
Rusanov averages adjacent physical fluxes and damps their state jump. This
adds numerical diffusion, not physical viscosity.

Gravity waves move relative to the fluid with speed :math:`c`. The nonnegative
cell speed bounds :math:`s^x` and :math:`s^y` combine advection and waves:

.. math::

   c_{ij}=\sqrt{g h_{ij}}, \qquad
   s^x_{ij}=|u_{ij}|+c_{ij}, \qquad
   s^y_{ij}=|v_{ij}|+c_{ij}.

The absolute-value bars give speed regardless of the sign of velocity.
Superscripts x and y label directions, not powers. The wave-speed bounds at
the right and top faces are :math:`a^x` and :math:`a^y`:

.. math::

   a^x_{i+1/2,j}=\max(s^x_{ij},s^x_{i+1,j}), \qquad
   a^y_{i,j+1/2}=\max(s^y_{ij},s^y_{i,j+1}).

The maximum selects the larger speed from the two cells sharing a face.
The ``face_speed`` function uses the elementwise identity

.. math::

   \max(a,b)=\tfrac12(a+b+|a-b|),

where :math:`a` and :math:`b` are any two real values. In the function, ``a``
is the cell-speed array, ``neighbor`` is its forward neighbour array, and
``axis`` selects the spatial direction.

.. code-block:: text

   def face_speed(a: ℝ[m, n], axis: ℕ): ℝ[m, n]:
       neighbor: ℝ[m, n] = roll(a, -1, axis)
       return 0.5 * (a + neighbor + abs(a - neighbor))

Use a hat to distinguish a numerical face flux from a physical flux evaluated
inside a cell. The vector Rusanov fluxes are

.. math::

   \widehat{\mathbf F}_{i+1/2,j}
   =\tfrac12\left[\mathbf F(\mathbf q_{ij})+
                          \mathbf F(\mathbf q_{i+1,j})\right]
    -\tfrac12a^x_{i+1/2,j}(\mathbf q_{i+1,j}-\mathbf q_{ij}),

   \widehat{\mathbf G}_{i,j+1/2}
   =\tfrac12\left[\mathbf G(\mathbf q_{ij})+
                          \mathbf G(\mathbf q_{i,j+1})\right]
    -\tfrac12a^y_{i,j+1/2}(\mathbf q_{i,j+1}-\mathbf q_{ij}).

Each face uses its two adjacent cell states and the same speed bound for
all three conserved components.

From fluxes to the right-hand side
----------------------------------

Replacing the exact face averages in the cell balance defines the spatial
operator :math:`\mathcal L`, which maps the entire grid state to its rate of
change:

.. math::

   \frac{d\mathbf q_{ij}}{dt}=\mathcal L(\mathbf q)_{ij},
   \qquad
   \mathcal L(\mathbf q)_{ij}
   =-\frac{\widehat{\mathbf F}_{i+1/2,j}
                   -\widehat{\mathbf F}_{i-1/2,j}}{\Delta x}
    -\frac{\widehat{\mathbf G}_{i,j+1/2}
                   -\widehat{\mathbf G}_{i,j-1/2}}{\Delta y}.

This is an ODE (ordinary differential equation) for the array values: space
has been discretized, but time is still continuous. The function
``divergence`` computes one component of this negative flux divergence.
Its ``w`` argument is the conserved component; ``f`` and ``flux_y`` are that
component's physical x- and y-fluxes. ``ax`` and ``ay`` are the face-speed
arrays, and ``dx`` and ``dy`` are the cell widths. The arrays ``F`` and
``G`` store the numerical fluxes on each cell's right and top faces.
Shifting those fluxes backwards supplies the left and bottom faces.

.. code-block:: text

   def divergence(w: ℝ[m, n], f: ℝ[m, n], flux_y: ℝ[m, n],
                  ax: ℝ[m, n], ay: ℝ[m, n], dx: ℝ, dy: ℝ): ℝ[m, n]:
       F: ℝ[m, n] = 0.5 * (f + roll(f, -1, 0))
       G: ℝ[m, n] = 0.5 * (flux_y + roll(flux_y, -1, 1))
       F: ℝ[m, n] = F - 0.5 * ax * (roll(w, -1, 0) - w)
       G: ℝ[m, n] = G - 0.5 * ay * (roll(w, -1, 1) - w)
       return -(F - roll(F, 1, 0)) / dx - (G - roll(G, 1, 1)) / dy

The function ``rhs`` means “right-hand side” and implements
:math:`\mathcal L`. It recovers velocities from the input state, computes wave
speeds and pressure, then calls ``divergence`` for depth and both momenta.
The returned ``dh``, ``dhu``, and ``dhv`` are time derivatives, not yet changes
over a time step.



.. code-block:: text

   def rhs(q: ℝ[3, m, n], g: ℝ, dx: ℝ, dy: ℝ): ℝ[3, m, n]:
       h, hu, hv: ℝ[m, n] = q[0], q[1], q[2]
       u, v: ℝ[m, n] = hu / h, hv / h
       c: ℝ[m, n] = sqrt(g * h)
       ax: ℝ[m, n] = face_speed(abs(u) + c, 0)
       ay: ℝ[m, n] = face_speed(abs(v) + c, 1)
       pressure: ℝ[m, n] = 0.5 * g * h * h
       dh, dhu, dhv: ℝ[m, n] = divergence(h, hu, hv, ax, ay, dx, dy), divergence(
           hu, hu * u + pressure, hv * u, ax, ay, dx, dy), divergence(
           hv, hu * v, hv * v + pressure, ax, ay, dx, dy)
       return [dh, dhu, dhv]

On a periodic grid, summing a flux difference over all cells cancels every
face contribution with its neighbour's. This explains discrete conservation
of total depth and momentum in exact arithmetic. Floating-point roundoff can
leave a small residual.

Advancing time with SSPRK3
--------------------------

A forward Euler step adds the current rate of change multiplied by the time
step. The `SSPRK3 method of Gottlieb and Shu <https://doi.org/10.1090/S0025-5718-98-00913-2>`_
combines three such evaluations, allowing each new state to
change the velocities, pressure, and wave speeds used by the next evaluation.
Keeping all three conserved components together avoids using stale velocities.

Let :math:`n` be the completed time-step index, :math:`\Delta t` the fixed time
step, and :math:`t^n=n\Delta t` the time after that many steps, starting at
zero. The full grid state there is :math:`\mathbf q^n`. Define the Euler map
:math:`E` and the internal RK states by

.. math::

   E(\mathbf q)=\mathbf q+\Delta t\,\mathcal L(\mathbf q),

   \begin{aligned}
   \mathbf q^{(1)} &= E(\mathbf q^n),\\
   \mathbf q^{(2)} &= \tfrac34\mathbf q^n+\tfrac14E(\mathbf q^{(1)}),\\
   \mathbf q^{n+1} &= \tfrac13\mathbf q^n+\tfrac23E(\mathbf q^{(2)}).
   \end{aligned}

Superscripts :math:`(1),(2)` label internal stages, while :math:`n+1` labels
the next completed step. These formulas act on the whole grid. Each weighted
pair is a convex average: its coefficients are nonnegative and sum to one.

Strong stability preservation carries suitable forward-Euler stability
properties through these averages under a timestep restriction. For smooth
solutions, the method is third order in time; spatial accuracy remains
first order.


``q1,q2`` are internal states; ``q_next`` is the completed update. Each
``rhs`` call recomputes both directional fluxes from its own stage state.


.. code-block:: text

   def rk3_step(q: ℝ[3, m, n], g: ℝ, dx: ℝ, dy: ℝ, Δt: ℝ): ℝ[3, m, n]:
       q1: ℝ[3, m, n] = q + Δt * rhs(q, g, dx, dy)
       q2: ℝ[3, m, n] = 0.75 * q + 0.25 * (q1 + Δt * rhs(q1, g, dx, dy))
       q_next: ℝ[3, m, n] = q / 3.0 + (2.0 / 3.0) * (q2 + Δt * rhs(q2, g, dx, dy))
       return q_next

Choosing a time step
~~~~~~~~~~~~~~~~~~~~

The Courant–Friedrichs–Lewy (CFL) condition compares how far a disturbance
can travel in one time step with the size of a cell. For shallow water,
gravity waves travel relative to the fluid at speed

.. math::

   \boxed{c=\sqrt{gh}},

where :math:`g` is gravitational acceleration and :math:`h` is the local
total water depth. The flow also carries the disturbance along. For the
three-component shallow-water system used here, the characteristic speeds
in the x and y directions are, respectively,

.. math::

   u-c,\quad u,\quad u+c,
   \qquad
   v-c,\quad v,\quad v+c.

Here :math:`u` and :math:`v` are the depth-averaged flow velocities. The
middle speed describes transport of transverse momentum; the other two
include gravity-wave motion. Taking the largest absolute speed gives

.. math::

   \boxed{s^x=|u|+\sqrt{gh}}, \qquad
   \boxed{s^y=|v|+\sqrt{gh}}.

The `shallow-water reference <https://www.clawpack.org/riemann_book/html/Shallow_water.html>`_
explains the gravity-wave speeds. In ``rhs``, the following lines compute
these speed bounds and select the larger bound on each side of a cell face:

.. code-block:: text

   c: ℝ[m, n] = sqrt(g * h)
   ax: ℝ[m, n] = face_speed(abs(u) + c, 0)
   ay: ℝ[m, n] = face_speed(abs(v) + c, 1)

Starting from a smooth periodic bump
------------------------------------

A Gaussian written using ordinary distance from the centre need not join
smoothly across a periodic boundary. The example instead uses cosines inside
an exponential, so the height and its derivatives match at opposite edges.
The initial condition is

.. math::

   B(x,y)=\exp\!\left(
     \frac{\cos(2\pi(x-x_c))+\cos(2\pi(y-y_c))-2}{\beta}
   \right).

.. math::

   \begin{aligned}
   h(x,y,0) &= H+A B(x,y),\\
   u(x,y,0) &= v(x,y,0)=0.
   \end{aligned}

.. math::

   \begin{aligned}
   x_c=y_c &= 0.5, & \beta &= 0.12,\\
   A &= 0.05, & H &= 1.
   \end{aligned}

The shape :math:`B` peaks at one at its centre :math:`(x_c,y_c)`.
:math:`A` (``amplitude``) sets the peak displacement; smaller positive
:math:`\beta` gives a narrower bump. Here :math:`\exp` is the natural
exponential, :math:`\cos` uses radians, and ``pi`` stores :math:`\pi`.

``initial_state`` samples the profile at cell centres to approximate cell
averages. On this even-sized grid, no cell centre hits the peak exactly.
The nested for-expressions build ``bump``; ``zero`` initializes both momenta
at rest.



.. code-block:: text

   def initial_state(Nx: ℕ, Ny: ℕ, H: ℝ, amplitude: ℝ): ℝ[3, Nx, Ny]:
       x: ℝ[Nx] = for i:ℕ(Nx) -> (i + 0.5) / Nx
       y: ℝ[Ny] = for j:ℕ(Ny) -> (j + 0.5) / Ny
       pi: ℝ = 3.141592653589793
       bump: ℝ[Nx, Ny] = for i:ℕ(Nx) -> for j:ℕ(Ny) -> exp(
           (cos(2 * pi * (x[i] - 0.5)) + cos(2 * pi * (y[j] - 0.5)) - 2.0) / 0.12)
       h: ℝ[Nx, Ny] = H + amplitude * bump
       zero: ℝ[Nx, Ny] = 0.0 * h
       return [h, zero, zero]

The positive bump adds water relative to the uniform reference depth. Its
spatial mean is not zero, so the mean initial depth is greater than
:math:`H`. Conservation preserves this initial mean; it does not force the
mean to return to the reference depth.

Interpreting the example
------------------------

The globals ``Nx`` and ``Ny`` set the cell counts, ``H`` the reference depth,
``g`` gravity, ``dx`` and ``dy`` the cell widths, and ``Δt`` the time step.
``steps`` is the number of updates. ``initial`` stores the initial conserved
state; ``final`` starts from that state and is replaced on each iteration.
The loop index ``step`` counts the updates, starting at zero. The final
simulation time :math:`t_{\mathrm{end}}` is

.. math::

   t_{\mathrm{end}}=40\Delta t=0.12.

.. code-block:: text

   Nx: ℕ = 24
   Ny: ℕ = 24
   H, g: ℝ = 1.0, 1.0
   dx, dy: ℝ = 1.0 / Nx, 1.0 / Ny
   # Fixed step for this small bump. The 2D Courant number is
   # Δt * (max(|u| + c) / dx + max(|v| + c) / dy), with maxima over cells.
   # Check it and h > 0 at every stage when changing the setup; neither is enforced here.
   Δt: ℝ = 0.003
   steps: ℕ = 40
   initial: ℝ[3, Nx, Ny] = initial_state(Nx, Ny, H, 0.05)
   final: ℝ[3, Nx, Ny] = initial
   for step:ℕ(steps):
       final = rk3_step(final, g, dx, dy, Δt)

The only scalar diagnostic printed by this program is
``mean_depth_change``. In mathematical notation, call it
:math:`\delta\overline h`:

.. math::

   \delta\overline h=\frac1{N_xN_y}
   \sum_{i=0}^{N_x-1}\sum_{j=0}^{N_y-1}
   \left(h^{\mathrm{final}}_{ij}-h^{\mathrm{initial}}_{ij}\right).

The superscripts “initial” and “final” identify the depth components of those
two arrays; the sums include every cell. Since all cells have equal area,
this arithmetic mean measures the change in mean depth. It should be near
zero up to floating-point error because of the shared periodic fluxes. A
CPU run with the default single-precision tensors gave approximately
``4.24e-7``; the exact roundoff depends on the execution environment. This
small drift checks conservation, not the accuracy of wave propagation.
The program itself does not produce a plot or a stage-by-stage stability
report; intermediate states were checked separately during validation.

To interpret the state in your own diagnostics, ``final[0]`` is depth,
``final[0] - H`` is surface displacement, and the two momentum components
divided by ``final[0]`` recover velocities. A smooth surface disturbance
should drive outward flow and gravity waves. Numerical diffusion smooths
those waves, and periodic wrapping matters once waves reach an edge. These
are qualitative expectations, not measured snapshots supplied by this page.

Limits and useful checks
~~~~~~~~~~~~~~~~~~~~~~~~

* The spatial method is deliberately diffusive. Third-order time stepping
  does not make the spatial approximation third order. Resolve the bump on
  progressively finer grids and reduce the time step consistently before
  trusting fine details.
* Division by depth and its square root require positive depth. There is no
  dry-cell treatment, positivity repair, or wetting-and-drying model here.
  Check depth and finite values at internal stages as well as completed steps.
* Monitor the two-dimensional CFL budget when changing the setup. A fixed
  time step that suits one grid or amplitude may fail for another.
* Conservation of mean depth alone cannot establish accuracy. Uniform water
  at rest should remain unchanged; periodic flux cancellation should also
  conserve both total momentum components up to roundoff. Grid and time-step
  comparisons test different errors from these conservation checks.
* This is a flat-bed, depth-averaged gravity-wave demonstration, not a resolved
  three-dimensional simulation, a model of breaking waves, or a reproduction
  of a particular paper's numerical scheme.

Checking the implementation
---------------------------

The implementation was checked using separate numerical tests during
development; those test files are not included in this tutorial-only change.
The checks compiled the Physika source and exercised its generated functions.
They checked uniform water at rest and in motion, conservation of water and
both momentum components, symmetry when swapping x and y, and pressure-driven
acceleration. They also inspect every Runge–Kutta stage of the default example
for positive finite depth and the CFL budget. In the checked run, the maximum
budget was approximately ``0.1471``, below the chosen ``0.2`` target.

At a fixed grid, timestep refinement showed approximately third-order
convergence against an exact semidiscrete shear solution and between
nonlinear runs. Keeping the grid fixed isolates temporal error; spatial
accuracy remains first order.

On this checkout, the CLI reports that some constructs fall back from CIC
verification to the standard type checker and code generator, then reports
``No type errors found``. The example runs through that supported path; these
numerical tests are not a claim of full formal verification.

Full code
---------

The complete implementation is shown below.

.. code-block:: text

   # 2D shallow water: q_t + F(q)_x + G(q)_y = 0 on a periodic unit square.
   # q = [h, hu, hv]: depth and depth-integrated momenta; u, v are velocities.
   # This depth-averaged, flat-bed model omits viscosity and external forcing.
   # It is a reduced model, not a full 3D flow or a reproduction of Silvestri's scheme.
   # All quantities are dimensionless; the reference depth H and gravity g are 1.
   # q[k, i, j] stores component k in cell (i, j). For each component field,
   # axis 0 is x and axis 1 is y; roll(w, -1, axis) selects the next periodic cell.

   # At each positive face, use max(a_left, a_right) = (a + b + |a - b|) / 2.
   def face_speed(a: ℝ[m, n], axis: ℕ): ℝ[m, n]:
       neighbor: ℝ[m, n] = roll(a, -1, axis)
       return 0.5 * (a + neighbor + abs(a - neighbor))

   # For one conserved component w, f and flux_y are its x and y physical fluxes.
   # Rusanov flux: F_face = (f_left + f_right) / 2 - a_face * (w_right - w_left) / 2.
   # Piecewise-constant states give first-order spatial accuracy and numerical diffusion.
   # Shared face fluxes cancel between adjacent cells in the negative divergence.
   def divergence(w: ℝ[m, n], f: ℝ[m, n], flux_y: ℝ[m, n],
                  ax: ℝ[m, n], ay: ℝ[m, n], dx: ℝ, dy: ℝ): ℝ[m, n]:
       F: ℝ[m, n] = 0.5 * (f + roll(f, -1, 0))
       G: ℝ[m, n] = 0.5 * (flux_y + roll(flux_y, -1, 1))
       F: ℝ[m, n] = F - 0.5 * ax * (roll(w, -1, 0) - w)
       G: ℝ[m, n] = G - 0.5 * ay * (roll(w, -1, 1) - w)
       return -(F - roll(F, 1, 0)) / dx - (G - roll(G, 1, 1)) / dy

   # q_t = L(q), with physical fluxes
   #   F = [hu, hu * u + g * h² / 2, hu * v],
   #   G = [hv, hv * u, hv * v + g * h² / 2].
   # Recompute velocities and gravity-wave speed c from this stage's entire state.
   # The divisions and square root require h > 0; there is no dry-bed treatment.
   def rhs(q: ℝ[3, m, n], g: ℝ, dx: ℝ, dy: ℝ): ℝ[3, m, n]:
       h, hu, hv: ℝ[m, n] = q[0], q[1], q[2]
       u, v: ℝ[m, n] = hu / h, hv / h
       c: ℝ[m, n] = sqrt(g * h)
       ax: ℝ[m, n] = face_speed(abs(u) + c, 0)
       ay: ℝ[m, n] = face_speed(abs(v) + c, 1)
       pressure: ℝ[m, n] = 0.5 * g * h * h
       dh, dhu, dhv: ℝ[m, n] = divergence(h, hu, hv, ax, ay, dx, dy), divergence(
           hu, hu * u + pressure, hv * u, ax, ay, dx, dy), divergence(
           hv, hu * v, hv * v + pressure, ax, ay, dx, dy)
       return [dh, dhu, dhv]

   # SSPRK(3,3): three coupled stages, third-order in time for the semidiscrete ODE.
   def rk3_step(q: ℝ[3, m, n], g: ℝ, dx: ℝ, dy: ℝ, Δt: ℝ): ℝ[3, m, n]:
       q1: ℝ[3, m, n] = q + Δt * rhs(q, g, dx, dy)
       q2: ℝ[3, m, n] = 0.75 * q + 0.25 * (q1 + Δt * rhs(q1, g, dx, dy))
       q_next: ℝ[3, m, n] = q / 3.0 + (2.0 / 3.0) * (q2 + Δt * rhs(q2, g, dx, dy))
       return q_next

   # Start at rest with a smooth periodic bump, sampled at cell centers.
   def initial_state(Nx: ℕ, Ny: ℕ, H: ℝ, amplitude: ℝ): ℝ[3, Nx, Ny]:
       x: ℝ[Nx] = for i:ℕ(Nx) -> (i + 0.5) / Nx
       y: ℝ[Ny] = for j:ℕ(Ny) -> (j + 0.5) / Ny
       pi: ℝ = 3.141592653589793
       bump: ℝ[Nx, Ny] = for i:ℕ(Nx) -> for j:ℕ(Ny) -> exp(
           (cos(2 * pi * (x[i] - 0.5)) + cos(2 * pi * (y[j] - 0.5)) - 2.0) / 0.12)
       h: ℝ[Nx, Ny] = H + amplitude * bump
       zero: ℝ[Nx, Ny] = 0.0 * h
       return [h, zero, zero]

   Nx: ℕ = 24
   Ny: ℕ = 24
   H, g: ℝ = 1.0, 1.0
   dx, dy: ℝ = 1.0 / Nx, 1.0 / Ny
   # Fixed step for this small bump. The 2D Courant number is
   # Δt * (max(|u| + c) / dx + max(|v| + c) / dy), with maxima over cells.
   # Check it and h > 0 at every stage when changing the setup; neither is enforced here.
   Δt: ℝ = 0.003
   steps: ℕ = 40
   initial: ℝ[3, Nx, Ny] = initial_state(Nx, Ny, H, 0.05)
   final: ℝ[3, Nx, Ny] = initial
   for step:ℕ(steps):
       final = rk3_step(final, g, dx, dy, Δt)

   # Periodic flux cancellation conserves mean depth, up to floating-point roundoff.
   mean_depth_change: ℝ = mean(final[0] - initial[0])
   mean_depth_change

References
----------

- `Shallow-water equations and wave speeds — Clawpack <https://www.clawpack.org/riemann_book/html/Shallow_water.html>`_
- `Total variation diminishing Runge–Kutta schemes — Gottlieb and Shu (1998) <https://doi.org/10.1090/S0025-5718-98-00913-2>`_
