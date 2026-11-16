Differentiable 1D shallow-water bathymetry reconstruction
===========================================================

A wave moving across an uneven bottom is delayed, accelerated, and partly
reflected by changes in water depth.  *Bathymetry* is the bottom elevation,
and inferring it from surface measurements is an inverse problem: instead of
predicting a wave from a known bed, we adjust a bed model until the simulated
wave gauges agree with observations.

This tutorial is a compact solver-in-the-loop example.  It creates synthetic
measurements from a known smooth bed, runs a one-dimensional shallow-water
solver for a candidate bed, and differentiates the gauge mismatch with respect
to the three bed parameters.  It is a hydrostatic, entirely wet teaching model,
not an implementation or validation of a coastal or non-hydrostatic solver.

Model and experiment
--------------------

On a periodic unit interval, the conserved state is
:math:`\mathbf q=(h,hu)^\mathsf T`.  Here :math:`h=\eta-b>0`, :math:`\eta`
is free-surface elevation, :math:`b` is upward-positive bed elevation, and
:math:`u` is depth-averaged velocity.  The balance law is

.. math::

   h_t+(hu)_x=0,
   \qquad
   (hu)_t+\left(hu^2+\tfrac12gh^2\right)_x=-ghb_x.

The first equation conserves water.  The second balances momentum transport,
hydrostatic pressure, and the slope of the fixed bed.  A positive bed bump
reduces the water column depth at a fixed surface elevation, which is why the
sign convention :math:`h=\eta-b` matters.

The bed is not an independent parameter in every cell.  We infer the three
parameters :math:`\theta=(A,x_0,\sigma)` of a Gaussian bump,

.. math::

   b_\theta(x)=A\exp\!\left[-\frac{(x-x_0)^2}{2\sigma^2}\right].

This low-dimensional parameterisation keeps the inverse problem interpretable:
:math:`A` controls bump height, :math:`x_0` its location, and :math:`\sigma`
its width.  The source maps precisely to this equation:

.. code-block:: text

   def bathymetry(θ: ℝ[3]): ℝ[Nx]:
       A, x0, σ: ℝ = θ[0], θ[1], θ[2]
       return for i:ℕ(Nx) -> A * exp(-(x[i] - x0)**2 / (2.0 * σ**2))

The experiment uses :math:`N_x=32` cells, :math:`\Delta t=0.002`, thirty fixed
steps, and three final-time surface gauges.  It is entirely wet: there is no
positivity limiter, wetting/drying, friction, wind, rotation, or adaptive time
stepping.

CFL condition
~~~~~~~~~~~~~

For this first-order Rusanov finite-volume discretisation, a useful one-
dimensional Courant number is

.. math::

   \nu=\frac{\Delta t}{\Delta x}
   \max_i\left(|u_i|+\sqrt{g h_i}\right).

The gravity-wave speed relative to the flow is :math:`\sqrt{gh}`; adding
:math:`|u|` bounds either propagation direction.  The maximum is taken over
cells (and, conservatively, should be checked over each RK stage).  For the
configured small pulse and entirely wet Gaussian bed, :math:`\nu` is well below
one.  If the grid, time step, bed, or initial pulse changes, verify both
:math:`\nu\lesssim1` and :math:`h>0`; neither constraint is enforced by the
example.

Hydrostatic reconstruction
---------------------------

A flat-bed Rusanov divergence plus an independently centred bed-slope source
does not preserve a still lake over variable topography.  This tutorial uses
hydrostatic reconstruction [BathymetryAudusse2004]_.  At the right face of
cell :math:`i`, define

.. math::

   z_{i+1/2}=\max(b_i,b_{i+1}),\qquad
   h^*_{L,R}=\max(0,\eta_{L,R}-z_{i+1/2}).

The Rusanov flux is evaluated using these reconstructed depths.  The
cell-sided pressure correction is then

.. math::

   F^{hu}_{i+1/2,L}=
   \widehat F^{hu}_{i+1/2}
   +\tfrac12g\left(h_i^2-(h^*_{L})^2\right),

with the corresponding :math:`h_R^*` correction at the left face.  At lake at
rest, :math:`\eta` is constant and :math:`hu=0`; both corrected momentum
fluxes are :math:`gh_i^2/2`, so their difference is zero.  The tutorial test
checks this complete semidiscrete residual.

The following excerpt makes the reconstruction and face speed explicit.  The
absolute-value forms implement ``max`` without an opaque array primitive:

.. code-block:: text

   z_face: ℝ[n] = for i:ℕ(n) -> 0.5 * (
       b[i] + b_right[i] + abs(b[i] - b_right[i]))
   h_left_face: ℝ[n] = for i:ℕ(n) -> 0.5 * (
       η[i] - z_face[i] + abs(η[i] - z_face[i]))
   h_right_face: ℝ[n] = for i:ℕ(n) -> 0.5 * (
       η_right[i] - z_face[i] + abs(η_right[i] - z_face[i]))

   s_left: ℝ[n] = for i:ℕ(n) -> abs(u[i]) + sqrt(g * h_left_face[i])
   s_right: ℝ[n] = for i:ℕ(n) -> abs(u_right[i]) + sqrt(g * h_right_face[i])
   a: ℝ[n] = for i:ℕ(n) -> 0.5 * (
       s_left[i] + s_right[i] + abs(s_left[i] - s_right[i]))

Advancing the finite-volume state
---------------------------------

After spatial discretisation, ``rhs`` represents :math:`d\mathbf q/dt`.
The tutorial advances it with SSPRK(3,3), retaining the three-stage method
used by the companion shallow-water tutorial.  SSPRK3 is sufficient here and
keeps the source focused on the inverse problem; substituting RK4 would be a
different tutorial design rather than a correctness fix.

.. code-block:: text

   def rk3_step(n: ℕ, q: ℝ[2, n], b: ℝ[n]): ℝ[2, n]:
       q1: ℝ[2, n] = q + Δt * rhs(n, q, b)
       q2: ℝ[2, n] = 0.75 * q + 0.25 * (
           q1 + Δt * rhs(n, q1, b))
       return q / 3.0 + (2.0 / 3.0) * (
           q2 + Δt * rhs(n, q2, b))

``solve`` returns the **final state**, not a trajectory.  It starts from one
small free-surface pulse, overwrites ``q`` once for each completed step, then
returns :math:`\mathbf q^{\mathrm{final}}`.  Final-time gauges are enough for
this small demonstration, but a realistic inverse problem may use multiple
times, more gauges, regularisation, and uncertainty quantification.

.. code-block:: text

   def solve(θ: ℝ[3]): ℝ[2, Nx]:
       b: ℝ[Nx] = bathymetry(θ)
       q: ℝ[2, Nx] = initial_state(b)
       for step:ℕ(steps):
           q = rk3_step(Nx, q, b)
       return q

Differentiating the solver
--------------------------

The synthetic truth is deliberately chosen as the smooth, entirely wet bump
:math:`\theta_{\mathrm{true}}=(0.14,0.58,0.11)`.  It is not estimated from an
external dataset.  The tutorial simulates this known configuration once,
records the three resulting free-surface gauges, and treats those values as
fixed observations:

.. code-block:: text

   θ_true: ℝ[3] = [0.14, 0.58, 0.11]
   b_true: ℝ[Nx] = bathymetry(θ_true)
   gauge_obs: ℝ[3] = gauges(solve(θ_true), b_true)

For a candidate :math:`\theta`, the objective is the mean squared gauge error,

.. math::

   \mathcal L(\theta)=\frac13\sum_{r=1}^3
   \left[\eta_{j_r}^{\mathrm{pred}}(\theta)-
   \eta_{j_r}^{\mathrm{obs}}\right]^2.

``∇(loss, θ)`` differentiates this scalar loss with respect to all three
components of :math:`\theta` through the reconstruction, fluxes, SSPRK stages,
and fixed time loop.  The observations are formed outside ``loss`` and remain
fixed while the candidate simulation is differentiated.

.. code-block:: text

   loss_gradient: ℝ[3] = ∇(loss, θ)

The inversion is deliberately local: three noiseless gauges from one short
experiment do not establish parameter uniqueness.  It demonstrates a
computable solver-in-the-loop gradient, not a validated bathymetric survey
method.

Full source
-----------

The complete executable Physika program is rendered below.  The source in
``tutorials/`` remains compact because tutorial discovery executes it in CI;
the listing is included from that exact source to prevent documentation drift.

.. literalinclude:: ../../tutorials/differentiable_bathymetry.phyk
   :language: text

References
----------

.. [BathymetryAudusse2004] E. Audusse, F. Bouchut, M.-O. Bristeau, R. Klein, and
   B. Perthame. “A fast and stable well-balanced scheme with hydrostatic
   reconstruction for shallow water flows.” *SIAM Journal on Scientific
   Computing*, 25(6), 2050--2065, 2004.
   DOI: https://doi.org/10.1137/S1064827503431090
