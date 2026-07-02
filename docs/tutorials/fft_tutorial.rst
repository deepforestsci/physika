Fast Fourier Transform
======================

In this tutorial we will cover what the Discrete Fourier Transform (DFT) is, the
Fast Fourier Transform (FFT) that computes it efficiently, how to use them in
Physika, and how gradients flow through them with ``grad``.

Many of the signals we work with, such as a pressure trace, a time series, or a
row of a simulation grid, are more naturally described by the oscillations they
contain than by their raw sample values. The DFT [DFTWiki]_ resolves a signal
into exactly those frequency components, reporting the amplitude and phase of
each one it contains. The FFT [FFTWiki]_ computes this same decomposition, using
an algorithm that reduces its cost from :math:`O(N^2)` to :math:`O(N \log N)`.

Discrete Fourier Transform
------------------------------

The DFT takes :math:`N` samples :math:`x_0, \dots, x_{N-1}`, measured at equally
spaced points in time or space, and returns :math:`N` coefficients
:math:`X_0, \dots, X_{N-1}`, one per frequency :math:`k`. The samples are the
signal's time-domain representation and the coefficients are its frequency-domain
representation, or spectrum. Each :math:`X_k` is a complex number, carrying both
the strength of frequency :math:`k` and its phase:

.. math::

    X_k = \sum_{n=0}^{N-1} x_n \, e^{-2\pi i \, kn / N}, \qquad
    x_n = \frac{1}{N} \sum_{k=0}^{N-1} X_k \, e^{+2\pi i \, kn / N}

with :math:`k, n = 0, \dots, N-1`. The term :math:`e^{-2\pi i \, kn/N}` is a
complex sinusoid, a wave oscillating at frequency :math:`k`. Writing it with
Euler's formula, :math:`e^{-2\pi i \, kn/N} = \cos(2\pi kn/N) - i\,\sin(2\pi kn/N)`,
shows it as a cosine and sine that together complete :math:`k` full cycles across
the :math:`N` samples.

The forward sum multiplies the input by this sinusoid term by term and adds the
products. When the signal contains frequency :math:`k`, the two line up and the
sum is large, so :math:`X_k` measures how strongly frequency :math:`k` is present.
Its magnitude :math:`|X_k|` is proportional to the amplitude of that frequency and
its angle :math:`\arg(X_k)` records the phase.

The inverse sum reverses this. It scales each sinusoid by its coefficient
:math:`X_k` and adds them all back together, recovering the original samples
:math:`x_n` exactly.

As a small example, take a cosine that completes one full cycle over four samples,
:math:`x_n = \cos(2\pi n / 4)`, which gives :math:`x = [1, 0, -1, 0]`. Only
:math:`x_0` and :math:`x_2` are nonzero, so the sum reduces to

.. math::

    X_k = x_0 + x_2 \, e^{-2\pi i \, k \cdot 2 / 4}
        = 1 - e^{-i\pi k} = 1 - (-1)^k,

which evaluates to :math:`X = [0, 2, 0, 2]`. The nonzero coefficients fall at
:math:`k = 1` and :math:`k = 3`, the single frequency the
cosine carries.


Cooley-Tukey Algorithm
--------------------------

Evaluating the DFT sum directly costs :math:`O(N^2)`, since each of the :math:`N`
coefficients :math:`X_k` is itself a sum over :math:`N` terms. For the small
signal this is fine, but a simulation grid can easily reach :math:`N` in
the millions, where an :math:`O(N^2)` cost becomes impractical. The Fast Fourier
Transform is not a different transform. It computes exactly the same :math:`X_k`
values, but does so in :math:`O(N \log N)` time by avoiding redundant work in the
direct sum.

The most widely used FFT is the Cooley-Tukey algorithm [CooleyTukey1965]_, which
brings this cost down by divide and conquer. We split the
forward sum into its even-indexed samples (:math:`n = 2m`) and its odd-indexed
ones (:math:`n = 2m + 1`):

.. math::

    X_k = \sum_{m=0}^{N/2-1} x_{2m} \, e^{-2\pi i \, k(2m)/N}
        + \sum_{m=0}^{N/2-1} x_{2m+1} \, e^{-2\pi i \, k(2m+1)/N}.

In the even sum the exponent simplifies because :math:`k(2m)/N = km/(N/2)`, so
:math:`e^{-2\pi i \, k(2m)/N} = e^{-2\pi i \, km/(N/2)}`. This is a DFT that runs
over just the :math:`N/2` even-indexed samples, a transform of half the original
size. The odd sum carries the same exponent times one extra factor, since
:math:`e^{-2\pi i \, k(2m+1)/N} = e^{-2\pi i \, km/(N/2)} \, e^{-2\pi i \, k/N}`,
so pulling that factor out front leaves a second half-size DFT, this one over the
:math:`N/2` odd-indexed samples. Writing those two half transforms as

.. math::

    E_k = \sum_{m=0}^{N/2-1} x_{2m} \, e^{-2\pi i \, km/(N/2)}, \qquad
    O_k = \sum_{m=0}^{N/2-1} x_{2m+1} \, e^{-2\pi i \, km/(N/2)},

The DFT over all :math:`N` samples becomes


.. math::

    X_k = E_k + e^{-2\pi i \, k/N} \, O_k.

The factor :math:`e^{-2\pi i \, k/N}` multiplying :math:`O_k` is the twiddle
factor. It accounts for the odd samples sitting one position after the even ones,
since delaying a signal by one sample multiplies its frequency-:math:`k` component
by exactly :math:`e^{-2\pi i \, k/N}`.

Each half transform has period :math:`N/2`, so its values repeat, with
:math:`E_{k+N/2} = E_k` and :math:`O_{k+N/2} = O_k`. Over that same half period
the twiddle factor only flips sign,
:math:`e^{-2\pi i \, (k+N/2)/N} = -e^{-2\pi i \, k/N}`. A single pair
:math:`(E_k, O_k)` therefore produces two output coefficients at once,

.. math::

    X_k = E_k + e^{-2\pi i \, k/N} O_k, \qquad
    X_{k+N/2} = E_k - e^{-2\pi i \, k/N} O_k,

so the lower output :math:`X_k` and the upper output :math:`X_{k+N/2}` come from
the very same pair :math:`(E_k, O_k)`, differing only in the sign of the twiddle
factor. We therefore compute :math:`E_k` and :math:`O_k` once, for
:math:`k = 0, \dots, N/2 - 1`, and read off all :math:`N` outputs from them. This
two-input, two-output combine is the FFT's butterfly.

Each of those two half-length DFTs is computed the same way, splitting again into
even and odd parts, then each quarter-length DFT after that, down to transforms of
length one. This gives :math:`\log_2 N` levels of splitting, each costing
:math:`O(N)` to recombine, so the whole transform runs in :math:`O(N \log N)`,
against :math:`O(N^2)` for the direct sum.


FFT in Physika
------------------

We have seen what the transform computes and why the FFT computes it faster.
Physika provides these as built-in functions, which compile to PyTorch's FFT
functions [TorchFFT]_:

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
its result is typed ``ℂ[N]``. Physika follows PyTorch's default normalization, where
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
    π: ℝ = 3.141592653589793
    center: ℝ = 4.0

    f: ℝ[Ns] = for n: ℕ(Ns) -> cos(2*π * (n - center) / Ns) + 0.5 * cos(2*π * 3 * (n - center) / Ns)
    F: ℂ[Ns] = fft(f)
    abs(F)
    ifft(F)



Output::

    [0.0, 4.5, 0.0, 2.25, 0.0, 0.0, 2.25, 0.0, 4.5] ∈ ℝ[9]
    [(-1.19+0j), 0j, (-0.076+0j), (0.516+0j), (1.5+0j), (0.516+0j), (-0.076+0j), 0j, (-1.19+0j)] ∈ ℂ[9]


``fft`` recovers exactly the two frequencies the signal was built from. A peak of
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

The same extends to any rank. ``fftn`` and ``ifftn`` transform over every axis of
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


.. note::

   The ``norm`` argument, which selects a different normalization, cannot be
   passed because Physika has no string type, so every transform uses the default
   convention described above.

Differentiating through the FFT
-------------------------------

We now differentiate through the transform, then see it on two examples in
Physika.

The DFT is a linear map, :math:`X = Fx`, with the constant matrix
:math:`F_{kn} = e^{-2\pi i \, kn/N}`. Backpropagating through a linear map applies
its adjoint, the conjugate transpose :math:`F^{H}`. A gradient :math:`\bar{X}`
arriving on the output becomes the gradient :math:`\bar{x} = F^{H} \, \bar{X}` on
the input. For the DFT, this adjoint is the inverse transform without its
:math:`1/N` normalization, so the backward pass of an FFT is itself an inverse
FFT, exact and just as cheap, :math:`O(N \log N)`.

In Physika this needs no special handling. ``grad(expr, x)`` compiles to
``compute_grad``, which calls ``torch.autograd.grad``.

.. note::

   ``grad`` differentiates with respect to real variables. A complex variable is
   differentiated through its real and imaginary parts separately, so to optimize
   a complex input you carry it as a pair of real variables, its real and
   imaginary parts, and take the gradient with respect to each. [TorchComplexAutograd]_

We now see this on two examples.

Gradient of a single coefficient
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We start with the most direct case, differentiating one spectrum coefficient with
respect to the input. Take a short real signal and transform it:

.. code-block:: text

    x: ℝ[4] = [1.0, 2.0, 3.0, 4.0]
    X: ℂ[4] = fft(x)

A coefficient :math:`X_k` is complex, and ``grad`` needs a real-valued output, so
we differentiate its real part rather than the coefficient itself:

.. code-block:: text

    y: ℝ = real(X[1])
    grad(y, x)

Output::

    [1.0, 0.0, -1.0, 0.0] ∈ ℝ[4]

From the DFT sum, the real part of the :math:`k = 1` coefficient is
:math:`\mathrm{Re}(X_1) = x_0 - x_2` when :math:`N = 4`. Its gradient is therefore
:math:`1` at :math:`x_0`, :math:`-1` at :math:`x_2`, and zero elsewhere, matching
the output above.

The inverse transform differentiates the same way. Running ``ifft`` on the
spectrum reconstructs the signal, and we differentiate the first reconstructed
sample:

.. code-block:: text

    invX: ℂ[4] = ifft(X)
    w: ℝ = real(invX[0])
    grad(w, x)

Output::

    [1.0, 0.0, 0.0, 0.0] ∈ ℝ[4]

Since ``ifft(fft(x))`` returns ``x``, the first reconstructed sample is just
:math:`x_0`, so its gradient is :math:`1` at :math:`x_0` and zero elsewhere.

Gradient of the spectral energy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The second example is the one used in practice. The *energy* of a signal is the
sum of its squared samples, proportional to the physical energy it carries. By
Parseval's theorem [ParsevalWiki]_ the same value can be read from the spectrum:

.. math::

    E = \sum_{n=0}^{N-1} x_n^2 = \frac{1}{N} \sum_{k=0}^{N-1} |X_k|^2 .

Each :math:`|X_k|^2` is the energy at frequency :math:`k`, so the sum shows how the
signal's energy is spread across frequencies. We compute the energy both ways and
differentiate each with respect to the signal:

.. code-block:: text

    x2: ℝ[8] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    X2: ℂ[8] = fft(x2)

    energy_time: ℝ = sum(x2**2)
    energy_freq: ℝ = sum(abs(X2)**2) / len(x2)

    energy_time
    energy_freq
    grad(energy_time, x2)
    grad(energy_freq, x2)

Output::

    204.0 ∈ ℝ
    204.0 ∈ ℝ
    [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0] ∈ ℝ[8]
    [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0] ∈ ℝ[8]

Both forms give the same energy, :math:`204`, a numerical check of Parseval's
theorem, and the same gradient :math:`\partial E / \partial x_n = 2 x_n`. That the
frequency-domain version matches confirms ``grad`` flows correctly back through
``fft``.


References
----------

.. [FFTWiki] Fast Fourier transform. *Wikipedia*.
   `<https://en.wikipedia.org/wiki/Fast_Fourier_transform>`_.

.. [DFTWiki] Discrete Fourier transform. *Wikipedia*.
   `<https://en.wikipedia.org/wiki/Discrete_Fourier_transform>`_.

.. [CooleyTukey1965] Cooley, J. W. and Tukey, J. W. An algorithm for the
   machine calculation of complex Fourier series. *Mathematics of Computation*,
   19(90):297–301, 1965. doi: `10.1090/S0025-5718-1965-0178586-1
   <https://doi.org/10.1090/S0025-5718-1965-0178586-1>`_.

.. [TorchFFT] torch.fft. *PyTorch Documentation*.
   `<https://pytorch.org/docs/stable/fft.html>`_.

.. [TorchComplexAutograd] Autograd for complex numbers. *PyTorch Documentation*.
   `<https://pytorch.org/docs/stable/notes/autograd.html#autograd-for-complex-numbers>`_.

.. [ParsevalWiki] Parseval's theorem. *Wikipedia*.
   `<https://en.wikipedia.org/wiki/Parseval%27s_theorem>`_.