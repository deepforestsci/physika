Modeling lithium ion batteries - Single particle model
======================================================

Batteries are a complex multiphysics system. To accurately model them one
needs to couple the thermodynamics of the electrodes, the kinetic properties
of the chemical reactions, and the transport of species in the electrodes and electrolyte.

In this tutorial, we will implement the **Single Particle Model** (SPM) -- a low fidelity physics based battery model --
in Physika. The goal is produce the discharge voltage curve of the `LG-M50 <https://www.batterydesign.net/lg-21700-m50/>`_ cell under
a constant applied current. We shall also use Physika to find the appropriate value of
active material fraction in the positive electrode by fitting the model to match high fidelity simulation data.


The Single Particle Model
-------------------------

A lithium-ion cell has a porous negative electrode, a separator, and a porous
positive electrode. During discharge:

* lithium leaves the negative particles,
* lithium ions move through the electrolyte,
* and lithium enters the positive particles.

Each electrode has multiple particles, however we replaces all particles in one electrode with a single representative
sphere. It also assumes that the current and electrolyte conditions are uniform
through each electrode. The resulting model has only two spatial problems: one
radial diffusion equation for the negative particle and one for the positive
particle.

.. figure:: /_static/tutorial_files/spm_single_particle_schematic.png
   :alt: Porous anode, separator, and cathode reduced to representative spherical particles
   :align: center
   :width: 850px

   The SPM replaces each porous electrode with one representative spherical
   particle.

In this model we neglect the effects associated with the electrolyte, thus this approximation
is only accurate at low and moderate currents, where concentration gradients in the electrolyte are not dominant.

We shall let :math:`k \in \{p, n\}`, where :math:`p, n` denote the positive and negative electrode
respectively.


From C-Rate to Surface Flux
---------------------------

A C-rate expresses current relative to cell capacity. The LG-M50 cell has a
nominal capacity of ``5 Ah``, so

.. math::

   I = (\text{C-rate})\,Q_{\mathrm{nominal}}.

The current is therefore ``0.5 A`` at ``0.1C``, ``0.25 A`` at ``0.05C``, and
``1.0 A`` at ``0.2C``.

For spherical particles, the active surface area per unit electrode volume is

.. math::

   a_k = \frac{3\varepsilon_{s,k}}{R_k},

where :math:`R_k` is particle radius and :math:`\varepsilon_{s,k}` is the
active-material volume fraction. Dividing the cell current by electrode area,
thickness, active surface area, and Faraday's constant gives molar flux at the
particle surface:

.. math::

   j_n = \frac{I}{F A L_n a_n},
   \qquad
   j_p = -\frac{I}{F A L_p a_p}.

We define flux as positive **out of** a particle. During discharge,
:math:`j_n>0` because the negative particle loses lithium, while
:math:`j_p<0` because the positive particle gains lithium.

.. code-block:: text

   def negative_surface_flux(current: ℝ): ℝ:
       area_density: ℝ = 3.0 * ε_s_n / radius_n
       return current / (
           faraday * electrode_area * thickness_n * area_density
       )

   def positive_surface_flux(current: ℝ, ε_s_p: ℝ): ℝ:
       area_density: ℝ = 3.0 * ε_s_p / radius_p
       return -current / (
           faraday * electrode_area * thickness_p * area_density
       )

Diffusion Inside a Particle
---------------------------

Let :math:`c_k(r,t)` be lithium concentration inside particle :math:`k`. For a
sphere with constant diffusivity, Fick's law gives

.. math::

   \frac{\partial c_k}{\partial t}
   = \frac{1}{r^2}\frac{\partial}{\partial r}
     \left(D_k r^2 \frac{\partial c_k}{\partial r}\right).

Symmetry requires zero flux at the particle center. At the surface, the outward
diffusive flux is ``j_k`` by definition:

.. math::

   \left.\frac{\partial c_k}{\partial r}\right|_{r=0}=0,
   \qquad
   \left.-D_k\frac{\partial c_k}{\partial r}\right|_{r=R_k}=j_k.

Finite Volumes in Spherical Coordinates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We divide each particle into five concentric shells. For shell :math:`i`,
conservation of lithium is

.. math::

   \frac{d c_i}{dt}
   = -\frac{A_{i+1}N_{i+1}-A_iN_i}{V_i},
   \qquad
   N=-D\frac{\partial c}{\partial r},

with face area :math:`A_i=4\pi r_i^2` and shell volume
:math:`V_i=\tfrac{4\pi}{3}(r_{i+1}^3-r_i^3)`.

That gives the compact spherical finite-volume update used in ``particle_rhs``:

.. code-block:: text

   def particle_rhs(c: ℝ[m], radius: ℝ,
                    diffusivity: ℝ, j_out: ℝ): ℝ[m]:
       dc: ℝ[nr] = zero_1d_array(nr)
       dr: ℝ = radius / nr
       for i:ℕ(nr):
           shell_volume_factor = (i + 1)**3 - i**3
           if i == first_cell:
               dc[i] = 3.0 * diffusivity * (c[1] - c[0]) / dr**2
           else:
               if i < last_cell:
                   outward_gradient = (i + 1)**2 * (c[i+1] - c[i])
                   inward_gradient = i**2 * (c[i] - c[i-1])
                   dc[i] = 3.0 * diffusivity * (
                       outward_gradient - inward_gradient
                   ) / (dr**2 * shell_volume_factor)
               else:
                   diffusion_in = (
                       i**2 * diffusivity * (c[i] - c[i-1]) / dr
                   )
                   dc[i] = 3.0 * (
                       -(i + 1)**2 * j_out - diffusion_in
                   ) / (dr * shell_volume_factor)
       return dc

The special first-cell expression enforces symmetry at the center. The
last-cell expression inserts the applied surface flux.

Advancing in Time
~~~~~~~~~~~~~~~~~

We integrate the shell concentrations with fourth-order Runge--Kutta (RK4):

.. code-block:: text

   def rk4_particle(c: ℝ[m], radius: ℝ, diffusivity: ℝ,
                    j_out: ℝ, Δt: ℝ): ℝ[m]:
       k1: ℝ[nr] = particle_rhs(c, radius, diffusivity, j_out)
       k2: ℝ[nr] = particle_rhs(
           c + 0.5 * Δt * k1, radius, diffusivity, j_out
       )
       k3: ℝ[nr] = particle_rhs(
           c + 0.5 * Δt * k2, radius, diffusivity, j_out
       )
       k4: ℝ[nr] = particle_rhs(
           c + Δt * k3, radius, diffusivity, j_out
       )
       return c + Δt * (k1 + 2.0*k2 + 2.0*k3 + k4) / 6.0

The tutorial uses a :math:`\Delta t = 20 \text{s}` timestep.

From Surface Concentration to Voltage
-------------------------------------

Voltage depends on the particle **surface** concentrations because reactions
occur at the particle-electrolyte interface. We first convert surface
concentration to stoichiometry:

.. math::

   \theta_n=\frac{c_{n,\mathrm{surf}}}{c_{n,\max}},
   \qquad
   \theta_p=\frac{c_{p,\mathrm{surf}}}{c_{p,\max}}.

We make use of the functions provided in Chen et al. [Chen2020]_, for the open circuit potentials :math:`U_n(\theta_n)`,
:math:`U_p(\theta_p)` of each electrode.

We define the  exchange-current density as

.. math::

   i_{0,k}=k_{0,k}
  \sqrt{c_e c_{k,\mathrm{surf}}
  (c_{k,\max}-c_{k,\mathrm{surf}})},

where :math:`c_e` is the electrolyte concentration. To model the rate of the reaction at the interface we
use the symmetric Butler--Volmer given by

.. math::

   \eta_k=\frac{2RT}{F}
   \operatorname{asinh}\!\left(\frac{i_k}{2i_{0,k}}\right).

With outward molar flux positive, the signed interfacial current density is
:math:`i_k=Fj_k`. Thus :math:`i_n>0` and :math:`i_p<0` during discharge, giving
a positive negative-electrode overpotential and a negative positive-electrode
overpotential.

The terminal voltage is then defined as

.. math::

   V = U_p(\theta_p)-U_n(\theta_n)
       +\eta_p-\eta_n.

.. code-block:: text

   def terminal_voltage(c_n: ℝ[m], c_p: ℝ[m],
                        ε_s_p: ℝ, current: ℝ): ℝ:
       c_ns: ℝ = clip_concentration(c_n[nr-1], c_n_max)
       c_ps: ℝ = clip_concentration(c_p[nr-1], c_p_max)
       θ_n: ℝ = c_ns / c_n_max
       θ_p: ℝ = c_ps / c_p_max
       j_n: ℝ = negative_surface_flux(current)
       j_p: ℝ = positive_surface_flux(current, ε_s_p)
       i0_n: ℝ = exchange_current(c_ns, c_n_max, k0_n)
       i0_p: ℝ = exchange_current(c_ps, c_p_max, k0_p)
       η_n: ℝ = bv_overpotential(faraday * j_n, i0_n)
       η_p: ℝ = bv_overpotential(faraday * j_p, i0_p)
       return ocp_positive(θ_p) - ocp_negative(θ_n) \
           + η_p - η_n

Cell Parameters
---------------

To model the LG-M50 cell we use the parameter set given in Chen et al. [Chen2020]_.

.. list-table::
   :header-rows: 1
   :widths: 42 25 25

   * - Parameter
     - Negative electrode
     - Positive electrode
   * - Thickness [m]
     - ``8.52e-5``
     - ``7.56e-5``
   * - Particle radius [m]
     - ``5.86e-6``
     - ``5.22e-6``
   * - Solid diffusivity [m2/s]
     - ``3.3e-14``
     - ``4.0e-15``
   * - Active-material volume fraction
     - ``0.75``
     - ``0.665``
   * - Initial concentration [mol/m3]
     - ``29866``
     - ``17038``
   * - Maximum concentration [mol/m3]
     - ``33133``
     - ``63104``
   * - Exchange-current prefactor
     - ``6.48e-7``
     - ``3.42e-6``

The electrode area is ``0.1027 m2``, electrolyte concentration is
``1000 mol/m3``, temperature is ``298.15 K``, and nominal capacity is ``5 Ah``.

Assembling the Forward Solver
-----------------------------

The solver starts both particles at their initial concentrations, calculates
their constant-current surface fluxes, advances both diffusion problems, and
records voltage after each group of RK4 steps:

.. code-block:: text

   def spm_solver(ε_s_p: ℝ, c_rate: ℝ,
                  substeps_per_sample: ℕ, Δt: ℝ): ℝ[m]:
       c_n: ℝ[nr] = [c_n_initial, c_n_initial, c_n_initial,
                         c_n_initial, c_n_initial]
       c_p: ℝ[nr] = [c_p_initial, c_p_initial, c_p_initial,
                         c_p_initial, c_p_initial]
       voltage: ℝ[num_samples] = zero_1d_array(num_samples)
       current: ℝ = c_rate * nominal_capacity_ah
       j_n: ℝ = negative_surface_flux(current)
       j_p: ℝ = positive_surface_flux(current, ε_s_p)
       voltage[0] = terminal_voltage(c_n, c_p, ε_s_p, current)

       for sample:ℕ(0, num_samples - 1):
           for substep:ℕ(0, substeps_per_sample):
               c_n = rk4_particle(
                   c_n, radius_n, diffusivity_n, j_n, Δt
               )
               c_p = rk4_particle(
                   c_p, radius_p, diffusivity_p, j_p, Δt
               )
           voltage[sample+1] = terminal_voltage(
               c_n, c_p, ε_s_p, current
           )
       return voltage


Learning the Active-Material Fraction
-------------------------------------

We now use the differentiable forward model for an inverse problem. We use
voltage data from a ``0.1C`` discharge simulated with PyBaMM's [PyBaMM2021]_
implementation of the high-fidelity Doyle--Fuller--Newman model as ground
truth.

We fit the positive active-material volume fraction
:math:`\varepsilon_{s,p}`. This parameter controls positive-electrode surface
area and therefore the molar flux required at each particle surface. Starting
from the deliberately poor guess ``0.3`` makes its effect on usable capacity
easy to see.

The loss is mean squared voltage error:

.. math::

   \mathcal{L}(\varepsilon_{s,p})=
   \frac{1}{N}\sum_{q=1}^{N}
   \left[V_{\mathrm{SPM}}(t_q;\varepsilon_{s,p})
   -V_{\mathrm{data}}(t_q)\right]^2.

.. code-block:: text

   def calculate_loss(ε_fit: ℝ): ℝ:
       predicted: ℝ[m] = spm_solver(
           ε_fit, train_c_rate,
           train_substeps_per_sample, Δt
       )
       return mean((predicted - dfn_train_voltage)**2)

   learning_rate: ℝ = 0.86
   learning_rate_decay: ℝ = 0.25
   epochs: ℕ = 5
   loss_history: ℝ[6] = zero_1d_array(epochs + 1)
   loss_history[0] = initial_train_rmse**2

   for epoch:ℕ(0, epochs):
       g = grad(calculate_loss, ε_s_p)
       ε_s_p = ε_s_p - learning_rate * g
       loss_history[epoch+1] = calculate_loss(ε_s_p)
       learning_rate = learning_rate * learning_rate_decay

``grad`` differentiates through both particle solvers, every RK4 stage, the
surface kinetics, and the voltage equation. The first gradient step makes the
large correction from the deliberately poor initial guess. Multiplying the
learning rate by ``0.25`` after each epoch allows the remaining steps to refine
the parameter without overshooting the narrow minimum.

We see that the optimization successfully moves :math:`\varepsilon_{s,p}` from ``0.3`` to ``0.6471``,as the voltage RMSE falls from ``287.0 mV`` to ``3.66 mV``.

.. figure:: /_static/tutorial_files/spm_training_fit.png
   :alt: Training voltage curves and mean-squared-error loss by epoch
   :align: center
   :width: 900px

   Left: the initial and fitted Physika SPM voltage curves against the 0.1C DFN
   training data. Right: the MSE loss recorded after each optimization epoch.


Validation on a Dynamic Current Profile
---------------------------------------

A fitted model should also respond correctly when the applied current changes.
We therefore test the the model on the following six-hour protocol:

.. math::

   I(t)=
   \begin{cases}
   1.0\ \mathrm{A} & 0 \leq t < 2\ \mathrm{h},\\
   0 & 2 \leq t < 4\ \mathrm{h},\\
   0.25\ \mathrm{A} & 4 \leq t \leq 6\ \mathrm{h}.
   \end{cases}


.. code-block:: text

   def validation_protocol_solver(ε_s_p: ℝ,
                                  Δt: ℝ): ℝ[m]:
       # Initial particle concentrations are defined as in spm_solver.
       voltage: ℝ[validation_num_samples] = zero_1d_array(
           validation_num_samples
       )
       current: ℝ = validation_current_a[0]
       first_transition: ℕ = 8
       second_transition: ℕ = 17
       voltage[0] = terminal_voltage(c_n, c_p, ε_s_p, current)

       for sample:ℕ(0, validation_num_samples - 1):
           current = validation_current_a[sample+1]
           j_n = negative_surface_flux(current)
           j_p = positive_surface_flux(current, ε_s_p)
           if sample != first_transition:
               if sample != second_transition:
                   for substep:ℕ(0, validation_substeps_per_sample):
                       c_n = rk4_particle(
                           c_n, radius_n, diffusivity_n, j_n, Δt
                       )
                       c_p = rk4_particle(
                           c_p, radius_p, diffusivity_p, j_p, Δt
                       )
           voltage[sample+1] = terminal_voltage(
               c_n, c_p, ε_s_p, current
           )
       return voltage

   validation_voltage: ℝ[m] = validation_protocol_solver(
       ε_s_p, Δt
   )
   validation_rmse: ℝ = rmse(
       validation_voltage, pybamm_validation_voltage
   )

The reference curve is generated with the using the SPM in PyBaMM.
The slight differences we see are a result of differences in numerical discretization
and solver choice.

.. figure:: /_static/tutorial_files/spm_validation.png
   :alt: Current protocol and Physika versus PyBaMM SPM validation voltage
   :align: center
   :width: 700px

   Top: The applied current. Bottom: Voltage trace of the fitted
   Physika SPM against the PyBaMM SPM.

Generating the Figures
----------------------

The plotting functions are not built into Physika. As in the other tutorials,
add the following helpers to ``physika/runtime.py`` to plot the tensors
computed by the Physika program:

.. code-block:: python

   def plot_spm_training(dfn_voltage, initial_voltage, fitted_voltage,
                         loss_history, fitted_ε_s_p, initial_rmse,
                         fitted_rmse, sample_interval_s):
       import matplotlib.pyplot as plt
       import numpy as np

       def as_numpy(value):
           return value.detach().cpu().numpy()

       def as_float(value):
           if hasattr(value, "detach"):
               return float(value.detach().cpu())
           return float(value)

       dfn = as_numpy(dfn_voltage)
       initial = as_numpy(initial_voltage)
       fitted = as_numpy(fitted_voltage)
       losses = as_numpy(loss_history)
       time_h = (
           np.arange(dfn.size) * as_float(sample_interval_s) / 3600.0
       )

       fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
       axes[0].plot(time_h, dfn, color="black", lw=2.0,
                    label="Chen2020 DFN data")
       axes[0].plot(time_h, initial, color="#8c8c8c", ls=":", lw=2.0,
                    label=r"Physika SPM, initial $\varepsilon_{s,p}=0.3$")
       axes[0].plot(
           time_h, fitted, color="#0072B2", ls="--", lw=2.0,
           label=rf"Physika SPM, fitted $\varepsilon_{{s,p}}="
                 rf"{as_float(fitted_ε_s_p):.3f}$",
       )
       axes[0].set_title("Fit at 0.1C")
       axes[0].set_xlabel("Time [h]")
       axes[0].set_ylabel("Terminal voltage [V]")
       axes[0].set_ylim(2.45, 4.22)
       axes[0].grid(True, alpha=0.25)
       axes[0].legend(fontsize=8, loc="lower left")
       axes[0].text(
           0.98, 0.96,
           f"RMSE: {1e3 * as_float(initial_rmse):.1f} to "
           f"{1e3 * as_float(fitted_rmse):.2f} mV",
           transform=axes[0].transAxes, ha="right", va="top", fontsize=8,
       )

       epoch = np.arange(losses.size)
       axes[1].semilogy(epoch, losses, color="#D55E00",
                        marker="o", ms=5, lw=2.0)
       axes[1].set_title("Optimization history")
       axes[1].set_xlabel("Epoch")
       axes[1].set_ylabel(r"MSE loss [$\mathrm{V}^2$]")
       axes[1].set_xticks(epoch)
       axes[1].grid(True, which="both", alpha=0.25)
       fig.tight_layout()
       plt.show()

   def plot_spm_validation(time_h, current_a, pybamm_voltage,
                           validation_voltage, validation_rmse):
       import matplotlib.pyplot as plt

       def as_numpy(value):
           return value.detach().cpu().numpy()

       def as_float(value):
           if hasattr(value, "detach"):
               return float(value.detach().cpu())
           return float(value)

       time = as_numpy(time_h)
       current = as_numpy(current_a)
       reference = as_numpy(pybamm_voltage)
       validation = as_numpy(validation_voltage)

       fig, (ax_current, ax_voltage) = plt.subplots(
           2, 1, figsize=(7.2, 6.0), sharex=True,
           gridspec_kw={"height_ratios": [1, 2]},
       )

       ax_current.plot(time, current, color="#333333", lw=2.0)
       ax_current.set_ylabel("Current [A]")
       ax_current.set_ylim(-0.08, 1.08)
       ax_current.set_yticks([0.0, 0.25, 1.0])
       ax_current.grid(True, alpha=0.25)
       ax_current.set_title("Discharge-rest-discharge validation")

       ax_voltage.plot(time, reference, color="black", lw=2.0,
                       marker="o", ms=3.5, label="PyBaMM SPM")
       ax_voltage.plot(time, validation, color="#0072B2", ls="--",
                       lw=2.0, label="Physika SPM")
       ax_voltage.set_xlabel("Time [h]")
       ax_voltage.set_ylabel("Terminal voltage [V]")
       ax_voltage.set_xlim(0.0, 6.0)
       ax_voltage.set_ylim(3.65, 4.20)
       ax_voltage.grid(True, alpha=0.25)
       ax_voltage.legend(fontsize=8, loc="lower left")
       ax_voltage.text(
           0.98, 0.05,
           f"RMSE: {1e3 * as_float(validation_rmse):.2f} mV",
           transform=ax_voltage.transAxes,
           ha="right", va="bottom", fontsize=8,
       )
       fig.tight_layout()
       plt.show()

After adding the helpers, uncomment these final lines in
``single_particle_model.phyk``:

.. code-block:: text

   plot_spm_training(dfn_train_voltage, initial_train_voltage,
       fitted_train_voltage, loss_history, ε_s_p, initial_train_rmse,
       fitted_train_rmse, train_substeps_per_sample * Δt)
   plot_spm_validation(validation_time_h, validation_current_a,
       pybamm_validation_voltage, validation_voltage, validation_rmse)


Full Source
-----------

.. code-block:: text

    def zero_1d_array(len: ℝ): ℝ[m]:
        results: ℝ[len] = for i: ℕ(len) -> i*0
        return results

    def get_1d_array_length(x: ℝ[m]): ℝ:
        total: ℝ = 0
        temp: ℝ = 0
        for i:
            temp = x[i]
            total += 1
        return total

    def clip_stoichiometry(θ: ℝ): ℝ:
        x: ℝ = θ
        if x < 0.002:
            x = 0.002
        if x > 0.995:
            x = 0.995
        return x

    def clip_concentration(c_s: ℝ, c_s_max: ℝ): ℝ:
        c: ℝ = c_s
        if c < 1.0:
            c = 1.0
        if c > c_s_max - 1.0:
            c = c_s_max - 1.0
        return c

    def tanh_scalar(z: ℝ): ℝ:
        return (exp(z) - exp(-z)) / (exp(z) + exp(-z))

    # Chen2020 graphite and NMC open-circuit potentials.
    def ocp_negative(θ: ℝ): ℝ:
        x: ℝ = clip_stoichiometry(θ)
        return 1.9793 * exp(-39.3631 * x) + 0.2482 - 0.0909 * tanh_scalar(29.8538 * (x - 0.1234)) - 0.04478 * tanh_scalar(14.9159 * (x - 0.2769)) - 0.0205 * tanh_scalar(30.4444 * (x - 0.6103))

    def ocp_positive(θ: ℝ): ℝ:
        x: ℝ = clip_stoichiometry(θ)
        return -0.8090 * x + 4.4875 - 0.0428 * tanh_scalar(18.5138 * (x - 0.5542)) - 17.7326 * tanh_scalar(15.7890 * (x - 0.3117)) + 17.5842 * tanh_scalar(15.9308 * (x - 0.3120))

    def exchange_current(c_s_surface: ℝ, c_s_max: ℝ, k0: ℝ): ℝ:
        c_s: ℝ = clip_concentration(c_s_surface, c_s_max)
        return k0 * sqrt(c_e * c_s * (c_s_max - c_s))

    def asinh_scalar(z: ℝ): ℝ:
        return log(z + sqrt(z**2 + 1.0))

    def bv_overpotential(reaction_current: ℝ, i0: ℝ): ℝ:
        z: ℝ = reaction_current / (2.0 * i0)
        return 2.0 * gas_constant * temperature / faraday * asinh_scalar(z)

    # A sphere has active surface area per electrode volume a = 3*ε_s/R.
    def negative_surface_flux(current: ℝ): ℝ:
        area_density: ℝ = 3.0 * ε_s_n / radius_n
        return current / (faraday * electrode_area * thickness_n * area_density)

    def positive_surface_flux(current: ℝ, ε_s_p: ℝ): ℝ:
        area_density: ℝ = 3.0 * ε_s_p / radius_p
        return -current / (faraday * electrode_area * thickness_p * area_density)

    # Finite-volume discretization of spherical solid diffusion. The r^2 factors
    # are the face areas; shell_volume_factor is proportional to shell volume.
    def particle_rhs(c: ℝ[m], radius: ℝ, diffusivity: ℝ, j_out: ℝ): ℝ[m]:
        dc: ℝ[nr] = zero_1d_array(nr)
        dr: ℝ = radius / nr
        shell_volume_factor: ℝ = 0.0
        outward_gradient: ℝ = 0.0
        inward_gradient: ℝ = 0.0
        diffusion_in: ℝ = 0.0
        first_cell: ℕ = 0
        last_cell: ℕ = nr - 1
        for i:ℕ(nr):
            shell_volume_factor = (i + 1)**3 - i**3
            if i == first_cell:
                dc[i] = 3.0 * diffusivity * (c[1] - c[0]) / dr**2
            else:
                if i < last_cell:
                    outward_gradient = (i + 1)**2 * (c[i+1] - c[i])
                    inward_gradient = i**2 * (c[i] - c[i-1])
                    dc[i] = 3.0 * diffusivity * (outward_gradient - inward_gradient) / (dr**2 * shell_volume_factor)
                else:
                    diffusion_in = i**2 * diffusivity * (c[i] - c[i-1]) / dr
                    dc[i] = 3.0 * (-(i + 1)**2 * j_out - diffusion_in) / (dr * shell_volume_factor)
        return dc

    def rk4_particle(c: ℝ[m], radius: ℝ, diffusivity: ℝ, j_out: ℝ, Δt: ℝ): ℝ[m]:
        k1: ℝ[nr] = particle_rhs(c, radius, diffusivity, j_out)
        k2: ℝ[nr] = particle_rhs(c + 0.5 * Δt * k1, radius, diffusivity, j_out)
        k3: ℝ[nr] = particle_rhs(c + 0.5 * Δt * k2, radius, diffusivity, j_out)
        k4: ℝ[nr] = particle_rhs(c + Δt * k3, radius, diffusivity, j_out)
        return c + Δt * (k1 + 2.0*k2 + 2.0*k3 + k4) / 6.0

    def terminal_voltage(c_n: ℝ[m], c_p: ℝ[m], ε_s_p: ℝ, current: ℝ): ℝ:
        c_ns: ℝ = clip_concentration(c_n[nr-1], c_n_max)
        c_ps: ℝ = clip_concentration(c_p[nr-1], c_p_max)
        θ_n: ℝ = c_ns / c_n_max
        θ_p: ℝ = c_ps / c_p_max
        j_n: ℝ = negative_surface_flux(current)
        j_p: ℝ = positive_surface_flux(current, ε_s_p)
        i0_n: ℝ = exchange_current(c_ns, c_n_max, k0_n)
        i0_p: ℝ = exchange_current(c_ps, c_p_max, k0_p)
        η_n: ℝ = bv_overpotential(faraday * j_n, i0_n)
        η_p: ℝ = bv_overpotential(faraday * j_p, i0_p)
        return ocp_positive(θ_p) - ocp_negative(θ_n) + η_p - η_n - current * contact_resistance

    # Advance both particles and record voltage after each group of RK4 substeps.
    def spm_solver(ε_s_p: ℝ, c_rate: ℝ, substeps_per_sample: ℕ, Δt: ℝ): ℝ[m]:
        c_n: ℝ[nr] = [c_n_initial, c_n_initial, c_n_initial, c_n_initial, c_n_initial]
        c_p: ℝ[nr] = [c_p_initial, c_p_initial, c_p_initial, c_p_initial, c_p_initial]
        voltage: ℝ[num_samples] = zero_1d_array(num_samples)
        current: ℝ = c_rate * nominal_capacity_ah
        j_n: ℝ = negative_surface_flux(current)
        j_p: ℝ = positive_surface_flux(current, ε_s_p)
        voltage[0] = terminal_voltage(c_n, c_p, ε_s_p, current)
        for sample:ℕ(0, num_samples - 1):
            for substep:ℕ(0, substeps_per_sample):
                c_n = rk4_particle(c_n, radius_n, diffusivity_n, j_n, Δt)
                c_p = rk4_particle(c_p, radius_p, diffusivity_p, j_p, Δt)
            voltage[sample+1] = terminal_voltage(c_n, c_p, ε_s_p, current)
        return voltage

    # Run three consecutive validation segments without resetting either particle.
    # Duplicate samples at 2 h and 4 h retain both sides of each current jump.
    def validation_protocol_solver(ε_s_p: ℝ, Δt: ℝ): ℝ[m]:
        c_n: ℝ[nr] = [c_n_initial, c_n_initial, c_n_initial, c_n_initial, c_n_initial]
        c_p: ℝ[nr] = [c_p_initial, c_p_initial, c_p_initial, c_p_initial, c_p_initial]
        voltage: ℝ[validation_num_samples] = zero_1d_array(validation_num_samples)
        current: ℝ = validation_current_a[0]
        j_n: ℝ = negative_surface_flux(current)
        j_p: ℝ = positive_surface_flux(current, ε_s_p)
        first_transition: ℕ = 8
        second_transition: ℕ = 17
        voltage[0] = terminal_voltage(c_n, c_p, ε_s_p, current)
        for sample:ℕ(0, validation_num_samples - 1):
            current = validation_current_a[sample+1]
            j_n = negative_surface_flux(current)
            j_p = positive_surface_flux(current, ε_s_p)
            if sample != first_transition:
                if sample != second_transition:
                    for substep:ℕ(0, validation_substeps_per_sample):
                        c_n = rk4_particle(c_n, radius_n, diffusivity_n, j_n, Δt)
                        c_p = rk4_particle(c_p, radius_p, diffusivity_p, j_p, Δt)
            voltage[sample+1] = terminal_voltage(c_n, c_p, ε_s_p, current)
        return voltage

    def rmse(a: ℝ[m], b: ℝ[n]): ℝ:
        loss: ℝ = 0.0
        n: ℝ = get_1d_array_length(a)
        for i:ℕ(n):
            loss += (a[i] - b[i])**2
        return sqrt(loss / n)

    # Chen2020 parameters.
    faraday: ℝ = 96485.33212
    gas_constant: ℝ = 8.31446261815324
    temperature: ℝ = 298.15
    c_e: ℝ = 1000.0
    electrode_area: ℝ = 0.1027
    nominal_capacity_ah: ℝ = 5.0
    contact_resistance: ℝ = 0.0

    thickness_n: ℝ = 8.52e-5
    thickness_p: ℝ = 7.56e-5
    radius_n: ℝ = 5.86e-6
    radius_p: ℝ = 5.22e-6
    diffusivity_n: ℝ = 3.3e-14
    diffusivity_p: ℝ = 4.0e-15
    ε_s_n: ℝ = 0.75
    c_n_initial: ℝ = 29866.0
    c_p_initial: ℝ = 17038.0
    c_n_max: ℝ = 33133.0
    c_p_max: ℝ = 63104.0
    k0_n: ℝ = 6.48e-7
    k0_p: ℝ = 3.42e-6

    nr: ℕ = 5
    num_samples: ℕ = 21
    train_c_rate: ℝ = 0.1
    train_substeps_per_sample: ℕ = 90
    Δt: ℝ = 20.0

    # Validation.
    validation_num_samples: ℕ = 27
    validation_substeps_per_sample: ℕ = 45
    validation_time_h: ℝ[27] = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.0, 4.25, 4.5, 4.75, 5.0, 5.25, 5.5, 5.75, 6.0]
    validation_c_rate_profile: ℝ[27] = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
    validation_current_a: ℝ[27] = validation_c_rate_profile * nominal_capacity_ah

    # Observations used for the inverse problem
    dfn_train_voltage: ℝ[21] = [4.158395248, 4.084192602, 4.071392813, 4.049195473, 4.006860924, 3.961241395, 3.916527071, 3.861774530, 3.814408034, 3.772015162, 3.725484224, 3.682248062, 3.645685812, 3.609974847, 3.565901908, 3.513874271, 3.468832454, 3.424676253, 3.317194985, 3.146746296, 2.789451932]

    initial_ε_s_p: ℝ = 0.3
    chen2020_ε_s_p: ℝ = 0.665
    ε_s_p: ℝ = initial_ε_s_p

    def calculate_loss(ε_fit: ℝ): ℝ:
        predicted: ℝ[m] = spm_solver(ε_fit, train_c_rate, train_substeps_per_sample, Δt)
        return mean((predicted - dfn_train_voltage)**2)

    # Start with a large step, then reduce it to refine the fit without overshooting.
    learning_rate: ℝ = 0.86
    learning_rate_decay: ℝ = 0.25
    epochs: ℕ = 5

    initial_train_voltage: ℝ[m] = spm_solver(initial_ε_s_p, train_c_rate, train_substeps_per_sample, Δt)
    initial_train_rmse: ℝ = rmse(initial_train_voltage, dfn_train_voltage)
    loss_history: ℝ[6] = zero_1d_array(epochs + 1)
    loss_history[0] = initial_train_rmse**2

    for epoch:ℕ(0, epochs):
        g = grad(calculate_loss, ε_s_p)
        ε_s_p = ε_s_p - learning_rate * g
        loss_history[epoch+1] = calculate_loss(ε_s_p)
        learning_rate = learning_rate * learning_rate_decay

    fitted_train_voltage: ℝ[m] = spm_solver(ε_s_p, train_c_rate, train_substeps_per_sample, Δt)
    fitted_train_rmse: ℝ = rmse(fitted_train_voltage, dfn_train_voltage)

    # Reference SPM solution
    pybamm_validation_ε_s_p: ℝ = 0.6471111178398132
    pybamm_validation_voltage: ℝ[27] = [4.142996394, 4.067772943, 4.057471567, 4.027229404, 3.981657244, 3.937880704, 3.895608468, 3.838549294, 3.788632364, 3.813228110, 3.833002643, 3.833640604, 3.833695039, 3.833696392, 3.833696305, 3.833696268, 3.833696266, 3.833696264, 3.827385441, 3.811877071, 3.800818412, 3.789583321, 3.778002657, 3.766121504, 3.754073902, 3.742039259, 3.730207631]
    validation_voltage: ℝ[m] = validation_protocol_solver(ε_s_p, Δt)
    validation_rmse: ℝ = rmse(validation_voltage, pybamm_validation_voltage)

    physika_print(ε_s_p)
    physika_print(initial_train_rmse)
    physika_print(fitted_train_rmse)
    physika_print(validation_rmse)
    physika_print(fitted_train_voltage)
    physika_print(validation_voltage)
    physika_print(loss_history)


References
----------

.. [Chen2020] Chen, C.-H., Brosa Planella, F., O’Regan, K., Gastol, D., Widanage, W. D., & Kendrick, E. (2020). Development of Experimental Techniques for Parameterization of Multi-scale Lithium-ion Battery Models. Journal of The Electrochemical Society, 167(8), 080534. https://doi.org/10.1149/1945-7111/ab9050
.. [PyBaMM2021] Sulzer, V., Marquis, S. G., Timms, R., Robinson, M., & Chapman, S. J. (2021). Python Battery Mathematical Modelling (PyBaMM). Journal of Open Research Software, 9(1), 14. https://doi.org/10.5334/jors.309
