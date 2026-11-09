Differentiable 1D shallow-water bathymetry reconstruction
===========================================================

A wave gauge measures the free surface at a few locations.  Here, synthetic
measurements are generated over a smooth bed and used to differentiate a
shallow-water solver with respect to the three bed parameters.  This is a
small hydrostatic, one-dimensional teaching problem, inspired by
solver-in-the-loop inversion, not an implementation of a coastal or
non-hydrostatic model.

Model and experiment
--------------------

On the periodic unit interval, the conserved state is
:math:`\mathbf q=(h,hu)^\mathsf T`, where :math:`h=\eta-b>0`, :math:`\eta` is
free-surface elevation, and upward-positive :math:`b` is bed elevation.  The
balance law is

.. math::

   h_t+(hu)_x=0,
   \qquad
   (hu)_t+\left(hu^2+\tfrac12gh^2\right)_x=-ghb_x.

The fixed bed is parameterised rather than learned independently in every
cell:

.. math::

   b_\theta(x)=A\exp\!\left[-\frac{(x-x_0)^2}{2\sigma^2}\right],
   \qquad \theta=(A,x_0,\sigma).

The program uses :math:`N_x=32`, fixed :math:`\Delta t=0.002`, thirty SSPRK3
steps, and three final-time gauges.  It is entirely wet; no positivity limiter,
wetting/drying, friction, wind, rotation, or adaptive time step is supplied.
When changing parameters, verify :math:`h>0` and a suitable CFL number.

Hydrostatic reconstruction
---------------------------

A flat-bed Rusanov divergence plus an independent centred source does not
preserve a lake at rest.  At face :math:`i+\tfrac12`, this tutorial reconstructs
[BathymetryAudusse2004]_

.. math::

   z_{i+1/2}=\max(b_i,b_{i+1}),\qquad
   h^*_{L,R}=\max(0,\eta_{L,R}-z_{i+1/2}),

then evaluates the Rusanov flux from :math:`(h_L^*,hu_i)` and
:math:`(h_R^*,hu_{i+1})`.  The momentum flux entering the update from the
right side of cell :math:`i` is corrected by

.. math::

   F^{hu}_{i+1/2,L}=
   \widehat F^{hu}_{i+1/2}
   +\tfrac12g\left(h_i^2-(h^*_{L})^2\right).

The left-face correction uses :math:`h^*_R` instead.  For a constant free
surface and zero momentum, both corrected face fluxes equal
:math:`gh_i^2/2`; their difference is zero.  The numerical test therefore
checks the full semidiscrete residual, rather than merely asserting the
continuous equilibrium.

Differentiating the solver
--------------------------

``gauge_obs`` is evaluated once from ``θ_true`` outside ``loss``.  In the loss,
the candidate bed is passed through all RK3 stages and the gauge mismatch is

.. math::

   \mathcal L(\theta)=\frac13\sum_{r=1}^3
   \left[\eta_{j_r}^{\mathrm{pred}}(\theta)-
   \eta_{j_r}^{\mathrm{obs}}\right]^2.

``grad(loss, θ)`` lowers to PyTorch reverse-mode automatic differentiation.
It therefore differentiates through the periodic reconstruction, each Rusanov
flux, every RK3 stage, and the fixed time loop.  The observations are fixed
program globals during this differentiated candidate evaluation, so they do
not receive a gradient path.  ``θ`` is supported by the Physika lexer; the
vector gradient is named ``loss_gradient`` because the lexer does not accept
``∇`` as an identifier.

The inversion is deliberately local: three noiseless gauges and one short
experiment do not establish parameter uniqueness.  It demonstrates a
computable gradient, not a validated bathymetric survey method.

Full source
-----------

.. literalinclude:: ../../tutorials/differentiable_bathymetry.phyk
   :language: text

References
----------

.. [BathymetryAudusse2004] E. Audusse, F. Bouchut, M.-O. Bristeau, R. Klein, and
   B. Perthame. “A fast and stable well-balanced scheme with hydrostatic
   reconstruction for shallow water flows.” *SIAM Journal on Scientific
   Computing*, 25(6), 2050--2065, 2004.
   DOI: https://doi.org/10.1137/S1064827503431090
