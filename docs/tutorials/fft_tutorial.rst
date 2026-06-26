Fast Fourier Transform
======================

In this tutorial we will go through what the Fast Fourier Transform (FFT) is, how
to use it in Physika, and how gradients flow through it with ``grad``.

The FFT is a fast algorithm for computing the Discrete Fourier Transform (DFT),
which represents a signal as a combination of pure sinusoids and reports the
amplitude and phase of each frequency it contains.

The Discrete Fourier Transform
------------------------------

We start from what the FFT actually computes. The signal arrives as :math:`N`
samples :math:`x_0, \dots, x_{N-1}`, its values measured at equally spaced points
in time or space. The DFT maps these to :math:`N` spectrum coefficients
:math:`X_0, \dots, X_{N-1}`, one per frequency:

.. math::

    \begin{align*}
    X_k = \sum_{n=0}^{N-1} x_n \, e^{-2\pi i \, kn / N}, \qquad
    x_n = \frac{1}{N} \sum_{k=0}^{N-1} X_k \, e^{+2\pi i \, kn / N}
    \end{align*}

with :math:`k, n = 0, \dots, N-1`. The factor :math:`e^{-2\pi i \, kn/N}` is a
pure wave that completes :math:`k` cycles across the :math:`N` samples. The
forward sum multiplies the signal by that wave and adds the result up, so
:math:`X_k` measures how much of frequency :math:`k` the signal contains:
:math:`|X_k|` is its amplitude and :math:`\arg(X_k)` its phase. The inverse sum
runs the other way, rebuilding the signal as those waves added back up, each
weighted by its :math:`X_k`, which returns the original samples exactly.


For a real signal the spectrum is conjugate-symmetric, :math:`X_{N-k} =\overline{X_k}`, so the upper half mirrors the lower half, and a frequency
present in the signal appears at both :math:`k` and :math:`N-k` with equal
magnitude.


Why it is fast
---------------

The FFT is not a different transform but an algorithm that computes the same DFT
more cheaply. Evaluating the sum directly costs :math:`O(N^2)`, since each of the
:math:`N` coefficients is a sum over :math:`N` terms.

The Cooley-Tukey algorithm brings this down by divide and conquer. We split the
forward sum into its even-indexed samples (:math:`n = 2m`) and its odd-indexed
ones (:math:`n = 2m + 1`):

.. math::

    X_k = \sum_{m=0}^{N/2-1} x_{2m} \, e^{-2\pi i \, k(2m)/N}
        + \sum_{m=0}^{N/2-1} x_{2m+1} \, e^{-2\pi i \, k(2m+1)/N}.

In the first sum the :math:`2` cancels into the :math:`N`, leaving a half-length
DFT over the even samples; the second is the same over the odd samples once we
pull an extra :math:`e^{-2\pi i \, k/N}` out of its shifted exponent. Writing
those two half transforms as

.. math::

    E_k = \sum_{m=0}^{N/2-1} x_{2m} \, e^{-2\pi i \, km/(N/2)}, \qquad
    O_k = \sum_{m=0}^{N/2-1} x_{2m+1} \, e^{-2\pi i \, km/(N/2)},

the length-:math:`N` DFT becomes

.. math::

    X_k = E_k + e^{-2\pi i \, k/N} \, O_k.

The factor :math:`e^{-2\pi i \, k/N}`, the algorithm's twiddle factor, restores the
one-sample offset dropped when the odd samples were reindexed by :math:`m`, since
shifting a sample by one position multiplies frequency :math:`k` by exactly
:math:`e^{-2\pi i \, k/N}`.

Each half transform repeats with period :math:`N/2`, so :math:`E_{k+N/2} = E_k`
and likewise for :math:`O_k`, and the phase factor flips sign over that half
period, :math:`e^{-2\pi i (k+N/2)/N} = -e^{-2\pi i \, k/N}`. One computed pair
:math:`(E_k, O_k)` therefore fills two outputs at once,

.. math::

    X_k = E_k + e^{-2\pi i \, k/N} O_k, \qquad
    X_{k+N/2} = E_k - e^{-2\pi i \, k/N} O_k,

so only :math:`k = 0, \dots, N/2 - 1` need be formed. Applying the split
recursively gives :math:`\log_2 N` stages of :math:`O(N)` work, and the whole
transform costs :math:`O(N \log N)` for the same result.


The FFT in Physika
------------------

Now we turn to using the transform in Physika, which exposes it as six built-ins
that map directly onto ``torch.fft``:

==========  ====================  ===================================
Physika     PyTorch               Transform
==========  ====================  ===================================
``fft``     ``torch.fft.fft``     1D forward
``ifft``    ``torch.fft.ifft``    1D inverse
``fft2``    ``torch.fft.fft2``    2D forward
``ifft2``   ``torch.fft.ifft2``   2D inverse
``fftn``    ``torch.fft.fftn``    N-dimensional forward
``ifftn``   ``torch.fft.ifftn``   N-dimensional inverse
==========  ====================  ===================================

A call such as ``fft(x)`` compiles straight to ``torch.fft.fft(x)``. The input
may be real or complex, and the transform always returns a complex spectrum, so
its result is typed ``ℂ[N]``. Physika follows PyTorch's default normalization:
the forward transform is unnormalized and the inverse carries the :math:`1/N`
factor, so ``ifft(fft(x))`` returns the original signal.

A one-dimensional transform
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In one dimension, ``fft`` turns an array of :math:`N` samples into its :math:`N`
complex spectrum coefficients, and ``ifft`` turns them back. To see this, take a
signal built from two cosines, a strong one at frequency 1 and a weaker one at
frequency 3:

.. code-block:: text

    Ns: ℕ = 9
    two_pi: ℝ = 6.283185307179586
    center: ℝ = 4.0

    f: ℝ[Ns] = for n: ℕ(Ns) -> cos(two_pi * (n - center) / Ns) + 0.5 * cos(two_pi * 3 * (n - center) / Ns)
    F: ℂ[Ns] = fft(f)
    abs(F)
    ifft(F)



Output::

    [0.0, 4.5, 0.0, 2.25, 0.0, 0.0, 2.25, 0.0, 4.5] ∈ ℝ[9]
    [(-1.19+0j), 0j, (-0.076+0j), (0.516+0j), (1.5+0j), (0.516+0j), (-0.076+0j), 0j, (-1.19+0j)] ∈ ℂ[9]


``fft`` recovers exactly the two frequencies the signal was built from: a peak of
height 4.5 at :math:`k = 1` and a peak of height 2.25 at :math:`k = 3`, each
mirrored at :math:`N - k` (so at :math:`k = 8` and :math:`k = 6`) by the
conjugate symmetry. Each height is :math:`N/2` times the cosine's amplitude, so
the weaker cosine gives the shorter peak. ``ifft`` then returns the original
samples.

.. figure:: /_static/tutorial_files/fft_signal_spectrum.png
   :alt: a two-cosine signal and its magnitude spectrum
   :align: center
   :width: 700px

   A signal made of two cosines and its spectrum. Each cosine appears as a
   mirrored pair of peaks, the taller pair for the larger-amplitude cosine.



A two-dimensional transform
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In two dimensions, ``fft2`` transforms a grid over both of its axes into a complex
spectrum, and ``ifft2`` turns it back. Take a small matrix:

.. code-block:: text

    M: ℝ[3, 3] = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    G: ℂ[3, 3] = fft2(M)
    G
    ifft2(G)

Output::

    [[(45+0j), (-4.5+2.6j), (-4.5-2.6j)],
     [(-13.5+7.79j), 0j, 0j],
     [(-13.5-7.79j), 0j, 0j]] ∈ ℂ[3,3]
    [[(1+0j), (2+0j), (3+0j)],
     [(4+0j), (5+0j), (6+0j)],
     [(7+0j), (8+0j), (9+0j)]] ∈ ℂ[3,3]

The top-left coefficient holds the sum of all entries (45) and the rest capture
how the grid varies across each axis. ``ifft2`` returns ``M``, its imaginary parts
zero because the grid was real.


An N-dimensional transform
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The same extends to any rank: ``fftn`` and ``ifftn`` transform over every axis of
an array, the form used on the three-dimensional grids of a physics simulation.
Take a small 2x2x2 grid:

.. code-block:: text

    V: ℝ[2, 2, 2] = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
    S: ℂ[2, 2, 2] = fftn(V)
    S
    ifftn(S)

Output::

    [[[(36+0j), (-4+0j)], [(-8+0j), 0j]],
     [[(-16+0j), 0j], [0j, 0j]]] ∈ ℂ[2,2,2]
    [[[(1+0j), (2+0j)], [(3+0j), (4+0j)]],
     [[(5+0j), (6+0j)], [(7+0j), (8+0j)]]] ∈ ℂ[2,2,2]

As in two dimensions, the corner coefficient is the sum of the grid (36), and
``ifftn`` recovers ``V``.


These six built-ins cover the transforms in everyday use, but two related
features of ``torch.fft`` are not yet exposed in Physika.

.. note::

   The ``norm`` argument, which selects a different normalization, cannot be
   passed: Physika has no string type, so every transform uses the default
   convention described above. The real-input variants ``rfft`` and ``irfft``
   are also left out, because they return a half-length spectrum and need the
   original signal length to invert, which does not round-trip cleanly without
   that extra argument.

Differentiating through the FFT
-------------------------------

We now differentiate through the transform. Because the DFT is linear, it is a
matrix multiply :math:`X = F x` with the constant matrix
:math:`F_{kn} = e^{-2\pi i \, kn/N}`. The derivative of a linear map is the map
itself, so the Jacobian :math:`\partial X / \partial x = F` is constant: it does
not depend on :math:`x`.

Reverse-mode autodiff propagates a gradient backwards through this map. Given the
gradient of a scalar loss :math:`L` with respect to the output, the cotangent
:math:`\bar{X} = \partial L / \partial X`, the gradient with respect to the input
is the adjoint of the Jacobian, its conjugate transpose, applied to that
cotangent:

.. math::

    \bar{x} = \frac{\partial L}{\partial x} = F^{H} \, \bar{X}.

For the DFT, the adjoint :math:`F^{H}` is the inverse transform up to the
:math:`1/N` factor. So the backward pass of an FFT is itself an inverse FFT: the
gradient is obtained by running an inverse transform on the cotangent. This makes
the gradient exact and as cheap as the forward transform, :math:`O(N \log N)`.

In Physika this needs no special handling. ``grad(expr, x)`` compiles to
``compute_grad``, which calls ``torch.autograd.grad``, and because each transform
is a plain ``torch.fft`` call on the graph, PyTorch applies the adjoint above for
us.

.. note::

   ``grad`` differentiates with respect to real variables. A complex variable is
   differentiated through its real and imaginary parts separately, so to optimize
   a complex input you carry it as a pair of real variables, its real and
   imaginary parts, and take the gradient with respect to each.

As an example, we differentiate the spectral energy of a signal, the total power
in its spectrum, with respect to the signal itself. This is the building block of
a spectral loss, the kind used to shape a signal in the frequency domain:

.. code-block:: text

    def spectral_energy(x: ℝ[N]): ℝ:
        return sum(abs(fft(x))**2)

    x: ℝ[N] = [1, 2, 3, 4, 5, 6, 7, 8]
    grad(spectral_energy(x), x)

Output::

    [16.0, 32.0, 48.0, 64.0, 80.0, 96.0, 112.0, 128.0] ∈ ℝ[8]

The gradient is the signal scaled by :math:`2N`. This is the adjoint at work: the
loss runs a forward transform, its gradient runs the inverse, and the two compose
back to :math:`N` times the identity, so the gradient points straight back along
the signal (the factor 2 comes from squaring the magnitudes). Differentiation
flowed through both the ``fft`` and the magnitude, with nothing written by hand.



References
----------

- `Fast Fourier Transform (Wikipedia) <https://en.wikipedia.org/wiki/Fast_Fourier_transform>`_
- `Discrete Fourier Transform (Wikipedia) <https://en.wikipedia.org/wiki/Discrete_Fourier_transform>`_
- `Cooley-Tukey FFT Algorithm (Wikipedia) <https://en.wikipedia.org/wiki/Cooley%E2%80%93Tukey_FFT_algorithm>`_
- `torch.fft (PyTorch Documentation) <https://pytorch.org/docs/stable/fft.html>`_
- `Autograd for Complex Numbers (PyTorch Documentation) <https://pytorch.org/docs/stable/notes/autograd.html#autograd-for-complex-numbers>`_
