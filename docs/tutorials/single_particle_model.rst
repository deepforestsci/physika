Modeling lithium ion batteries - Single particle model
======================================================

Batteries are a complex multiphysics system. To accurately model them one
needs to couple the thermodynamics of the electrodes, the kinetic properties
of the chemical reactions, and the transport of species in the electrodes and electrolyte.

In this tutorial, we will implement the **Single Particle Model** (SPM) -- a low fidelity physics based battery model --
in Physika. The goal is produce the discharge voltage curve of the `LG-M50 <https://www.batterydesign.net/lg-21700-m50/>`_ cell under
a constant applied current. We shall also use Physika to find the appropriate value of
active material fraction in the positive electrode by fitting the model to match high fidelity simulation data.

.. We will:
..
.. * represent each porous electrode by one spherical active-material particle,
.. * solve lithium diffusion inside both particles,
.. * calculate reaction overpotentials at their surfaces,
.. * combine those terms with open-circuit potentials to obtain cell voltage,
.. * fit one material parameter by differentiating through the Physika solver,
.. * and test the fitted model at a different C-rate.
..
.. The implementation uses the Chen2020 cell parameters. Everything needed to run
.. the example is stored in ``single_particle_model.phyk``; no battery-modeling
.. package is imported by the tutorial.

The Single Particle Model
-------------------------

A lithium-ion cell has a porous negative electrode, a separator, and a porous
positive electrode. Each electrode contains many active-material particles. During discharge:

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
is only accurate at low and moderate currents, where electrolyte gradients are not dominant.

We shall let :math:`k \in \{p, n\}`, where :math:`p, n` denote the positive and negative electrode
respectively.


From C-Rate to Surface Flux
---------------------------

A C-rate expresses current relative to cell capacity. The LG-M50 cell has a
nominal capacity of ``5 Ah``, so

.. math::

   I = (\text{C-rate})\,Q_{\mathrm{nominal}}.

The current is therefore ``0.5 A`` at ``0.1C`` and ``0.25 A`` at ``0.05C``.

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
       area_density: ℝ = 3.0 * eps_s_n / radius_n
       return current / (
           faraday * electrode_area * thickness_n * area_density
       )

   def positive_surface_flux(current: ℝ, eps_s_p: ℝ): ℝ:
       area_density: ℝ = 3.0 * eps_s_p / radius_p
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

The factors of :math:`4\pi` cancel. That gives the compact spherical
finite-volume update used in ``particle_rhs``:

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
                    j_out: ℝ, step_dt: ℝ): ℝ[m]:
       k1: ℝ[nr] = particle_rhs(c, radius, diffusivity, j_out)
       k2: ℝ[nr] = particle_rhs(
           c + 0.5 * step_dt * k1, radius, diffusivity, j_out
       )
       k3: ℝ[nr] = particle_rhs(
           c + 0.5 * step_dt * k2, radius, diffusivity, j_out
       )
       k4: ℝ[nr] = particle_rhs(
           c + step_dt * k3, radius, diffusivity, j_out
       )
       return c + step_dt * (k1 + 2.0*k2 + 2.0*k3 + k4) / 6.0

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

and use the symmetric Butler--Volmer kinetics given by

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
                        eps_s_p: ℝ, current: ℝ): ℝ:
       c_ns: ℝ = clip_concentration(c_n[nr-1], c_n_max)
       c_ps: ℝ = clip_concentration(c_p[nr-1], c_p_max)
       theta_n: ℝ = c_ns / c_n_max
       theta_p: ℝ = c_ps / c_p_max
       j_n: ℝ = negative_surface_flux(current)
       j_p: ℝ = positive_surface_flux(current, eps_s_p)
       i0_n: ℝ = exchange_current(c_ns, c_n_max, k0_n)
       i0_p: ℝ = exchange_current(c_ps, c_p_max, k0_p)
       eta_n: ℝ = bv_overpotential(faraday * j_n, i0_n)
       eta_p: ℝ = bv_overpotential(faraday * j_p, i0_p)
       return ocp_positive(theta_p) - ocp_negative(theta_n) \
           + eta_p - eta_n

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

   def spm_solver(eps_s_p: ℝ, c_rate: ℝ,
                  substeps_per_sample: ℕ, step_dt: ℝ): ℝ[m]:
       c_n: ℝ[nr] = [c_n_initial, c_n_initial, c_n_initial,
                         c_n_initial, c_n_initial]
       c_p: ℝ[nr] = [c_p_initial, c_p_initial, c_p_initial,
                         c_p_initial, c_p_initial]
       voltage: ℝ[num_samples] = zero_1d_array(num_samples)
       current: ℝ = c_rate * nominal_capacity_ah
       j_n: ℝ = negative_surface_flux(current)
       j_p: ℝ = positive_surface_flux(current, eps_s_p)
       voltage[0] = terminal_voltage(c_n, c_p, eps_s_p, current)

       for sample:ℕ(0, num_samples - 1):
           for substep:ℕ(0, substeps_per_sample):
               c_n = rk4_particle(
                   c_n, radius_n, diffusivity_n, j_n, step_dt
               )
               c_p = rk4_particle(
                   c_p, radius_p, diffusivity_p, j_p, step_dt
               )
           voltage[sample+1] = terminal_voltage(
               c_n, c_p, eps_s_p, current
           )
       return voltage

At ``0.1C``, the solver records one voltage every ``0.5 h`` for ``10 h``. At
``0.05C``, it records one voltage every hour for ``20 h``. Both runs therefore
cover the same nominal discharged capacity.

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

   def calculate_loss(eps_fit: ℝ): ℝ:
       predicted: ℝ[m] = spm_solver(
           eps_fit, train_c_rate,
           train_substeps_per_sample, rk4_step_s
       )
       return mean((predicted - dfn_train_voltage)**2)

   learning_rate: ℝ = 0.86
   learning_rate_decay: ℝ = 0.25
   epochs: ℕ = 5
   loss_history: ℝ[6] = zero_1d_array(epochs + 1)
   loss_history[0] = initial_train_rmse**2

   for epoch:ℕ(0, epochs):
       g = grad(calculate_loss, eps_s_p)
       eps_s_p = eps_s_p - learning_rate * g
       loss_history[epoch+1] = calculate_loss(eps_s_p)
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

The initial curve falls too quickly because ``eps_s_p=0.3`` gives too little
positive active material. After fitting, the Physika curve follows both the
gradual mid-discharge decline and the sharp end-of-discharge knee.


Validation at a New C-Rate
--------------------------

A fitted model should work outside the data used to fit it. We therefore reuse
the learned value without further training and simulate at ``0.05C``:

.. code-block:: text

   validation_voltage: ℝ[m] = spm_solver(
       eps_s_p, validation_c_rate,
       validation_substeps_per_sample, rk4_step_s
   )
   validation_rmse: ℝ = rmse(
       validation_voltage, dfn_validation_voltage
   )

The validation RMSE is ``4.66 mV``.

.. figure:: /_static/tutorial_files/spm_validation.png
   :alt: Physika SPM validation voltage curve at 0.05C
   :align: center
   :width: 600px

   The fitted Physika SPM evaluated at 0.05C without further parameter updates.
   The dashed SPM curve closely follows the independent DFN validation data.

Generating the Figures
----------------------

The plotting functions are not built into Physika. As in the other tutorials,
add the following helpers to ``physika/runtime.py`` to plot the tensors
computed by the Physika program:

.. code-block:: python

   def plot_spm_training(dfn_voltage, initial_voltage, fitted_voltage,
                         loss_history, fitted_eps, initial_rmse,
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
                 rf"{as_float(fitted_eps):.3f}$",
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

   def plot_spm_validation(dfn_voltage, validation_voltage,
                           validation_rmse, sample_interval_s):
       import matplotlib.pyplot as plt
       import numpy as np

       def as_numpy(value):
           return value.detach().cpu().numpy()

       def as_float(value):
           if hasattr(value, "detach"):
               return float(value.detach().cpu())
           return float(value)

       dfn = as_numpy(dfn_voltage)
       validation = as_numpy(validation_voltage)
       time_h = (
           np.arange(dfn.size) * as_float(sample_interval_s) / 3600.0
       )

       fig, ax = plt.subplots(figsize=(6.2, 4.4))
       ax.plot(time_h, dfn, color="black", lw=2.0,
               label="Chen2020 DFN data")
       ax.plot(time_h, validation, color="#0072B2", ls="--", lw=2.0,
               label="Physika SPM with fitted parameter")
       ax.set_title("Validation at 0.05C")
       ax.set_xlabel("Time [h]")
       ax.set_ylabel("Terminal voltage [V]")
       ax.set_ylim(2.45, 4.22)
       ax.grid(True, alpha=0.25)
       ax.legend(fontsize=8, loc="upper right")
       ax.text(
           0.98, 0.05,
           f"RMSE: {1e3 * as_float(validation_rmse):.2f} mV",
           transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
       )
       fig.tight_layout()
       plt.show()

After adding the helpers, uncomment these final lines in
``single_particle_model.phyk``:

.. code-block:: text

   plot_spm_training(dfn_train_voltage, initial_train_voltage,
       fitted_train_voltage, loss_history, eps_s_p, initial_train_rmse,
       fitted_train_rmse, train_substeps_per_sample * rk4_step_s)
   plot_spm_validation(dfn_validation_voltage, validation_voltage,
       validation_rmse, validation_substeps_per_sample * rk4_step_s)

The code generator detects these calls and imports the helpers from
``physika.runtime``. The checked-in documentation images are generated by a
non-interactive wrapper around the same Physika outputs. It does not duplicate
the battery equations.

Regenerate the documentation images with:

.. code-block:: console

   source .venv/bin/activate
   python scripts/gen_single_particle_model_figure.py

Running the Tutorial
--------------------

Run the complete model with:

.. code-block:: console

   source .venv/bin/activate
   physika physika/tutorials/single_particle_model.phyk

The program prints the fitted active-material fraction, the initial, fitted,
and validation RMSE values, followed by the training and validation voltage
arrays and the loss history used by the plotting helper.

What the SPM Leaves Out
-----------------------

The SPM is intentionally compact. This implementation does not resolve:

* electrolyte concentration or potential gradients,
* variation through electrode thickness,
* particle-size distributions,
* thermal behavior,
* degradation reactions,
* or voltage-cutoff events.

Those effects become increasingly important at high currents, low temperature,
or near operating limits. The SPM remains valuable because it connects solid
diffusion, interfacial kinetics, and terminal voltage with a small,
differentiable state model.

Full Source
-----------

The complete runnable implementation is in
``physika/tutorials/single_particle_model.phyk``.


References
----------

.. [Chen2020] Chen, C.-H., Brosa Planella, F., O’Regan, K., Gastol, D., Widanage, W. D., & Kendrick, E. (2020). Development of Experimental Techniques for Parameterization of Multi-scale Lithium-ion Battery Models. Journal of The Electrochemical Society, 167(8), 080534. https://doi.org/10.1149/1945-7111/ab9050
.. [PyBaMM2021] Sulzer, V., Marquis, S. G., Timms, R., Robinson, M., & Chapman, S. J. (2021). Python Battery Mathematical Modelling (PyBaMM). Journal of Open Research Software, 9(1), 14. https://doi.org/10.5334/jors.309
