Learning a Neural Flux Limiter for the 1D Euler Equations
==========================================================

In this tutorial we train a small neural network to act as the *flux
limiter* inside a finite-volume solver for the one-dimensional Euler
equations, by differentiating through the entire simulation. Shock-capturing
schemes blend a robust low-order flux with an accurate high-order flux, and
the limiter :math:`\phi(r)` decides the blend at every cell interface from
the local smoothness ratio :math:`r`. Classical limiters — minmod, van Leer,
superbee — are hand-designed compromises. Here the limiter is a learnable
function, trained so that the simulated density profile of a Sod shock tube
matches the exact solution.

The Equations
-------------

The 1D Euler equations describe an inviscid compressible gas as a system of
conservation laws for the density :math:`\rho`, momentum :math:`\rho u`, and
total energy :math:`E`:

.. math::

   \frac{\partial}{\partial t}
   \begin{pmatrix} \rho \\ \rho u \\ E \end{pmatrix}
   +
   \frac{\partial}{\partial x}
   \begin{pmatrix} \rho u \\ \rho u^2 + p \\ u (E + p) \end{pmatrix}
   = 0,
   \qquad
   p = (\gamma - 1)\left(E - \tfrac{1}{2} \rho u^2\right),

with :math:`\gamma = 1.4` for an ideal diatomic gas. The training problem is
the canonical **Sod shock tube** [Sod1978]_: a diaphragm at :math:`x = 0.5`
separates a hot, dense gas :math:`(\rho, u, p) = (1, 0, 1)` from a cold,
light one :math:`(0.125, 0, 0.1)`. When the diaphragm is removed the
solution splits into the three characteristic waves of the Euler system: a
rarefaction fan moving left, and a contact discontinuity and a shock moving
right. The problem has an exact solution, which we embed in the tutorial
(``true_rho``) as the training target at :math:`t = 0.1`.

The Numerical Scheme
--------------------

The solver is a flux-limited finite-volume scheme on 100 cells, applied to
each conserved component :math:`q` with physical flux :math:`f`. The
interface flux between cells :math:`i` and :math:`i+1` is

.. math::

   F_{i+1/2}
   = \underbrace{\tfrac{1}{2}\left(f_i + f_{i+1}\right)
   - \tfrac{1}{2} a \, \Delta q}_{\text{Rusanov (first order, robust)}}
   \;+\;
   \underbrace{\tfrac{1}{2} a \left(1 - a \tfrac{\Delta t}{\Delta x}\right)
   \phi(r) \, \Delta q}_{\text{limited anti-diffusion}},

where :math:`\Delta q = q_{i+1} - q_i` and :math:`a = |\bar u| + \bar c` is
the local maximum wave speed at the interface. With :math:`\phi = 0` this is
the very diffusive but unconditionally robust Rusanov scheme; with
:math:`\phi = 1` it becomes a second-order Lax–Wendroff-type scheme that
oscillates at discontinuities. The limiter reads the smoothness ratio

.. math::

   r_{i+1/2} = \frac{\Delta q_{\mathrm{upwind}}}{\Delta q_{i+1/2}},

the ratio of the neighbouring jump on the upwind side to the local jump
(:math:`r \approx 1` in smooth regions, :math:`r \leq 0` at extrema), and
turns the accurate flux off exactly where it would oscillate. For scalar
advection this construction reduces to the classical TVD framework of
[Sweby1984]_; for the Euler system, limiting component-by-component is a
standard practical simplification of characteristic-wise limiting
[Toro2009]_. The upwind side is selected smoothly from the sign of the
interface velocity, and everything is vectorized with ``roll`` — the whole
time step is a handful of array expressions:

.. code-block:: text

   Δρ = roll(ρ, -1) - ρ
   r_ρ = w_up * roll(Δρ, 1) / (Δρ + ε) + (1.0 - w_up) * roll(Δρ, -1) / (Δρ + ε)
   F_ρ = 0.5 * (mom + roll(mom, -1)) - 0.5 * a_face * Δρ + lw * this(r_ρ) * Δρ

Because no wave reaches the domain boundary by :math:`t = 0.1`, the first
and last cells simply keep their initial values (a mask zeroes their
updates), which enforces the Dirichlet conditions exactly.

The classical limiters use the identities 
:math:`\min(a,b) = \tfrac{1}{2}(a + b - |a - b|)` and
:math:`\max(a,b) = \tfrac{1}{2}(a + b + |a - b|)`, which vectorize with
``abs``:

.. code-block:: text

   def minmod(r: ℝ[m]): ℝ[m]:
       # max(0, min(1, r))
       m1 = 0.5 * (1.0 + r - abs(1.0 - r))
       phi = 0.5 * (m1 + abs(m1))
       return phi

The Neural Flux Limiter
-----------------------

A limiter is just a scalar function :math:`\phi(r)`, so a tiny MLP
(1 → 8 → 1, tanh) is enough. Rather than predicting :math:`\phi` directly,
the network outputs a sigmoid gate :math:`s(r) \in (0, 1)` that blends two
classical limiters, following [Huang2026]_:

.. math::

   \phi(r) = \big(1 - s(r)\big)\, \phi_{\mathrm{minmod}}(r)
           + s(r)\, \phi_{\mathrm{superbee}}(r).

minmod is the most diffusive classical limiter and superbee the most
compressive, so for *any* network weights the learned limiter stays inside
the envelope spanned by the classical schemes — in particular
:math:`\phi(r \le 0) = 0`, so the scheme falls back to pure Rusanov at
extrema. Training can explore freely without losing shock-capturing
robustness. The blend also gives two exact baselines for free: with zero
weights the gate is constant, so ``b2 = -20`` pins the limiter to minmod and
``b2 = +20`` to superbee:

.. code-block:: text

   minmod_limiter = NeuralFluxLimiter(θ_zero, -20.0, θ_zero, θ_zero, 0.0, 0.0, 0.0)
   superbee_limiter = NeuralFluxLimiter(θ_zero, 20.0, θ_zero, θ_zero, 0.0, 0.0, 0.0)

The class owns both the network and the differentiable solver, so the loss
for an initial condition is one method call, and ``grad`` differentiates the
final density profile with respect to the weights through all 61 solver
steps. Gradients that survive 61 time steps are small (:math:`\sim 10^{-5}`),
so the tutorial hand-rolls **Adam** — which rescales each parameter's step
by its gradient history — in ``update_params``, carrying the optimizer state
(``m_θ``, ``v_θ``, ...) explicitly as class fields:

.. code-block:: text

   def train(X: ℝ[1,6], Y: ℝ[1,100], epochs: ℕ, lr: ℝ) → ℝ:
       last: ℝ = 0
       for e:ℕ(epochs):
           for j:ℕ(0, 1):
               current_loss = this.loss_ic(X[j], Y[j])
               learnable_grads = grad(current_loss, this.learnable_params)
               this.update_params(lr, learnable_grads)
               last = current_loss
       return last

Running the Tutorial
--------------------

.. code-block:: text

   physika tutorials/learn_flux_limiter_euler_equations.phyk

The program prints the density MSE of the two classical baselines, the
untrained network, and the trained network, followed by the limiter values
on a few sample ratios:

.. code-block:: text

   0.0004899767227470875 ∈ ℝ    # minmod
   0.00039836575160734355 ∈ ℝ   # superbee
   0.00047969529987312853 ∈ ℝ   # neural, before training
   0.0003983769565820694 ∈ ℝ    # neural, after training
   [0.0, 0.0, 0.0, 0.0, 0.4999, 0.9999, 0.9999, 1.0, 1.4999, 1.9999, 1.9999, 1.9999] ∈ ℝ[12]
   [0.0, 0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0, 1.0] ∈ ℝ[12]
   [0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.5, 2.0, 2.0, 2.0] ∈ ℝ[12]

The gate bias is initialized to ``b2_0 = -3``, so the untrained network
starts close to the minmod end of the envelope; over sixteen Adam epochs it
traverses the whole envelope and converges to the superbee loss:

.. figure:: /_static/tutorial_files/flux_limiter_loss_curve.png
   :width: 85%
   :align: center

   Training loss per epoch. The dashed and dotted lines are the losses of
   the two classical limiters that bound the learnable envelope; the
   network starts near the minmod end and converges onto the superbee
   floor in about ten epochs.

The last three printed lines tell the same story pointwise: the learned
:math:`\phi` (first line) has moved onto superbee (third line) at every
sample ratio. For this problem and scheme, the most compressive limiter in
the envelope is optimal — and the network discovers that end-to-end, purely
from the mismatch between simulated and exact density:

.. figure:: /_static/tutorial_files/learned_flux_limiter_euler.png
   :width: 85%
   :align: center

   The learned limiter (solid) settles on superbee (dotted) across the
   whole Sweby diagram, having started as a soft blend between minmod and
   superbee.

.. figure:: /_static/tutorial_files/flux_limiter_density_profiles.png
   :width: 85%
   :align: center

   Density at :math:`t = 0.1`: both solvers capture the
   rarefaction–contact–shock structure, and the trained limiter resolves
   the contact discontinuity and shock slightly more sharply than minmod.

Full Code
----------

.. code-block:: text

   γ: ℝ = 1.4
   Nx: ℕ = 100
   dx: ℝ = 0.01
   dt: ℝ = 0.00163934
   Nt: ℕ = 61
   ε: ℝ = 1e-8

   initial_states: ℝ[1,6] = [
      [1, 0, 1, 0.125, 0, 0.1]]

   true_rho: ℝ[1,100] = [
      [1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 0.976824, 0.909632,
         0.84619, 0.786338, 0.729922, 0.676791, 0.6268, 0.579808, 0.535677, 0.494276,
         0.455475, 0.426319, 0.426319, 0.426319, 0.426319, 0.426319, 0.426319, 0.426319,
         0.426319, 0.426319, 0.426319, 0.265574, 0.265574, 0.265574, 0.265574, 0.265574,
         0.265574, 0.265574, 0.265574, 0.265574, 0.125, 0.125, 0.125, 0.125,
         0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125,
         0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125,
         0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125,
         0.125, 0.125, 0.125, 0.125]]

   def zero_1d_array(len: ℝ): ℝ[m]:
      results: ℝ[len] = for i: ℕ(len) -> i*0
      return results

   def minmod(r: ℝ[m]): ℝ[m]:
      # max(0, min(1, r))
      m1 = 0.5 * (1.0 + r - abs(1.0 - r))
      phi = 0.5 * (m1 + abs(m1))
      return phi

   def superbee(r: ℝ[m]): ℝ[m]:
      # max(0, min(2r, 1), min(r, 2))
      s1 = 0.5 * (2.0 * r + 1.0 - abs(2.0 * r - 1.0))
      s2 = 0.5 * (r + 2.0 - abs(r - 2.0))
      s3 = 0.5 * (s1 + s2 + abs(s1 - s2))
      phi = 0.5 * (s3 + abs(s3))
      return phi

   def tanh_act(x: ℝ[m]): ℝ[m]:
      # tanh written so that large |x| saturates instead of overflowing
      y = 1.0 - 2.0 / (exp(2.0 * x) + 1.0)
      return y

   # Boundary mask: interior cells evolve, the first and last cell keep their
   # initial (exact) values (Dirichlet treatment). No wave reaches the boundary 
   # by t = 0.1.
   mask: ℝ[100] = zero_1d_array(Nx)
   for i:ℕ(1, 99):
      mask[i] = 1.0

   class NeuralFluxLimiter(θ: ℝ[3,8], b2: ℝ, m_θ: ℝ[3,8], v_θ: ℝ[3,8], m_b: ℝ, v_b: ℝ, t_adam: ℝ):
      def λ(r: ℝ[m]) → ℝ[m]:
         # clamp r to [-2, 10]: both classical limiters are constant outside
         rc = 0.5 * (r + 10.0 - abs(r - 10.0))
         rc = 0.5 * (rc - 2.0 + abs(rc + 2.0))
         w_in = this.θ[0]
         b_h = this.θ[1]
         w_out = this.θ[2]
         z = 0.0 * rc + this.b2
         for k:ℕ(0, 8):
               z = z + w_out[k] * tanh_act(w_in[k] * rc + b_h[k])
         s = 1.0 / (1.0 + exp(0.0 - z))
         phi = (1.0 - s) * minmod(rc) + s * superbee(rc)
         return phi
      def solve(ic: ℝ[6]) → ℝ[100]:
         # piecewise-constant initial state: left half / right half
         ρ = zero_1d_array(Nx)
         u0 = zero_1d_array(Nx)
         p0 = zero_1d_array(Nx)
         for i:ℕ(0, 50):
               ρ[i] = ic[0]
               u0[i] = ic[1]
               p0[i] = ic[2]
         for i:ℕ(50, 100):
               ρ[i] = ic[3]
               u0[i] = ic[4]
               p0[i] = ic[5]
         mom = ρ * u0
         E = p0 / (γ - 1.0) + 0.5 * ρ * u0 * u0
         for n:ℕ(0, Nt):
               u = mom / ρ
               p = (γ - 1.0) * (E - 0.5 * mom * u)
               c = sqrt(γ * p / ρ)
               # local max wave speed at the interface and LW correction factor
               a = abs(u) + c
               a_face = 0.5 * (a + roll(a, -1) + abs(a - roll(a, -1)))
               lw = 0.5 * a_face * (1.0 - a_face * dt / dx)
               # smooth upwind selector from the interface velocity sign
               u_face = 0.5 * (u + roll(u, -1))
               w_up = 0.5 * (1.0 + u_face / (abs(u_face) + ε))
               # density flux
               Δρ = roll(ρ, -1) - ρ
               r_ρ = w_up * roll(Δρ, 1) / (Δρ + ε) + (1.0 - w_up) * roll(Δρ, -1) / (Δρ + ε)
               F_ρ = 0.5 * (mom + roll(mom, -1)) - 0.5 * a_face * Δρ + lw * this(r_ρ) * Δρ
               # momentum flux
               f_m = mom * u + p
               Δm = roll(mom, -1) - mom
               r_m = w_up * roll(Δm, 1) / (Δm + ε) + (1.0 - w_up) * roll(Δm, -1) / (Δm + ε)
               F_m = 0.5 * (f_m + roll(f_m, -1)) - 0.5 * a_face * Δm + lw * this(r_m) * Δm
               # energy flux
               f_E = u * (E + p)
               ΔE = roll(E, -1) - E
               r_E = w_up * roll(ΔE, 1) / (ΔE + ε) + (1.0 - w_up) * roll(ΔE, -1) / (ΔE + ε)
               F_E = 0.5 * (f_E + roll(f_E, -1)) - 0.5 * a_face * ΔE + lw * this(r_E) * ΔE
               # conservative update on interior cells
               ρ = ρ - dt / dx * mask * (F_ρ - roll(F_ρ, 1))
               mom = mom - dt / dx * mask * (F_m - roll(F_m, 1))
               E = E - dt / dx * mask * (F_E - roll(F_E, 1))
         return ρ
      def loss_ic(ic: ℝ[6], target: ℝ[100]) → ℝ:
         pred = this.solve(ic)
         diff = pred - target
         l = mean(diff * diff)
         return l
      def evaluate(X: ℝ[1,6], Y: ℝ[1,100]) → ℝ:
         l = this.loss_ic(X[0], Y[0])
         return l
      def train(X: ℝ[1,6], Y: ℝ[1,100], epochs: ℕ, lr: ℝ) → ℝ:
         last: ℝ = 0
         for e:ℕ(epochs):
               # loop over the training problems (one here; extend by adding rows)
               for j:ℕ(0, 1):
                  current_loss = this.loss_ic(X[j], Y[j])
                  learnable_grads = grad(current_loss, this.learnable_params)
                  this.update_params(lr, learnable_grads)
                  last = current_loss
         return last
      def update_params(lr: ℝ, learnable_grads: ℝ[m]):
         # one Adam step; gradients through 61 solver steps are ~1e-5, so
         # plain SGD would crawl — Adam rescales per-parameter (as in the
         # full-scale experiment)
         β1: ℝ = 0.9
         β2: ℝ = 0.999
         this.t_adam = this.t_adam + 1.0
         this.m_θ = β1 * this.m_θ + (1.0 - β1) * learnable_grads[0]
         this.v_θ = β2 * this.v_θ + (1.0 - β2) * learnable_grads[0] * learnable_grads[0]
         this.θ = this.θ - lr * (this.m_θ / (1.0 - β1 ** this.t_adam)) / (sqrt(this.v_θ / (1.0 - β2 ** this.t_adam)) + 1e-8)
         this.m_b = β1 * this.m_b + (1.0 - β1) * learnable_grads[1]
         this.v_b = β2 * this.v_b + (1.0 - β2) * learnable_grads[1] * learnable_grads[1]
         this.b2 = this.b2 - lr * (this.m_b / (1.0 - β1 ** this.t_adam)) / (sqrt(this.v_b / (1.0 - β2 ** this.t_adam)) + 1e-8)

   θ_zero: ℝ[3,8] = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
   minmod_limiter = NeuralFluxLimiter(θ_zero, -20.0, θ_zero, θ_zero, 0.0, 0.0, 0.0)
   superbee_limiter = NeuralFluxLimiter(θ_zero, 20.0, θ_zero, θ_zero, 0.0, 0.0, 0.0)

   # density MSE at t = 0.1: minmod, then superbee
   minmod_loss = minmod_limiter.evaluate(initial_states, true_rho)
   minmod_loss
   superbee_loss = superbee_limiter.evaluate(initial_states, true_rho)
   superbee_loss

   θ_0: ℝ[3,8] = [[0.5, -0.5, 0.3, -0.3, 0.8, -0.8, 0.2, -0.2],
                  [0.1, -0.1, 0.4, -0.4, 0.2, -0.2, -0.3, 0.3],
                  [0.3, -0.2, 0.25, -0.3, 0.2, 0.15, -0.25, 0.2]]
   b2_0: ℝ = -3.0   # gate starts near minmod, so training traverses the envelope
   adam_m0: ℝ[3,8] = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
   adam_v0: ℝ[3,8] = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]

   net = NeuralFluxLimiter(θ_0, b2_0, adam_m0, adam_v0, 0.0, 0.0, 0.0)

   # loss before training
   loss_before = net.evaluate(initial_states, true_rho)
   loss_before

   epochs: ℕ = 16
   lr: ℝ = 0.1
   final_loss = net.train(initial_states, true_rho, epochs, lr)

   # loss after training: converges to the superbee loss — for this problem
   # and scheme the most compressive limiter in the envelope is optimal, and
   # the network discovers that end-to-end through the solver
   loss_after = net.evaluate(initial_states, true_rho)
   loss_after

   r_sample: ℝ[12] = [-2.0, -1.0, -0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0, 8.0]
   learned_phi = net(r_sample)
   learned_phi
   minmod_phi = minmod(r_sample)
   minmod_phi
   superbee_phi = superbee(r_sample)
   superbee_phi

References
----------

.. [Sod1978] G. A. Sod, *A survey of several finite difference methods for
   systems of nonlinear hyperbolic conservation laws*, Journal of
   Computational Physics 27 (1978).

.. [Sweby1984] P. K. Sweby, *High resolution schemes using flux limiters for
   hyperbolic conservation laws*, SIAM Journal on Numerical Analysis 21
   (1984).

.. [Toro2009] E. F. Toro, *Riemann Solvers and Numerical Methods for Fluid
   Dynamics*, 3rd ed., Springer (2009).

.. [Huang2026] C. Huang, A. S. Sebastian, and V. Viswanathan, *Learning
   second-order total variation diminishing flux limiters using
   differentiable solvers*, Physics of Fluids 38, 036102 (2026).
   `Article <https://pubs.aip.org/aip/pof/article/38/3/036102/3381829>`__.

- `Flux limiter (Wikipedia) <https://en.wikipedia.org/wiki/Flux_limiter>`_


