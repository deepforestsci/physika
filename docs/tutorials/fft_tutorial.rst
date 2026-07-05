Fast Fourier Transform
======================

In this tutorial we will cover what the Discrete Fourier Transform (DFT) is, the
Fast Fourier Transform (FFT) that computes it efficiently, how to use them in
Physika, and how gradients flow through them with ``grad``.

Many of the signals we work with, such as a pressure trace, a time series, or a
row of a simulation grid, are more naturally described by the oscillations they
contain than by their raw sample values. The DFT [DFTWiki]_ resolves a signal
into its constituent frequency components, reporting the amplitude and phase of each frequency.
The FFT [FFTWiki]_ computes this same decomposition, using
an algorithm that reduces its cost from :math:`O(N^2)` to :math:`O(N \log N)`.

Discrete Fourier Transform
------------------------------

The DFT takes :math:`N` samples :math:`x_0, \dots, x_{N-1}`, measured at equally
spaced points in time or space, and returns :math:`N` coefficients
:math:`X_0, \dots, X_{N-1}`, one per frequency :math:`k`. The samples are the
signal's time-domain representation and the coefficients are its frequency-domain
representation, or spectrum. Each :math:`X_k` is a complex number, carrying both
the strength of frequency :math:`k` and its phase:

**Forward Discrete Fourier Transform (DFT):**

.. math::

    X_k = \sum_{n=0}^{N-1} x_n \, e^{-2\pi i \, kn / N}

**Inverse Discrete Fourier Transform (IDFT):**

.. math::

    x_n = \frac{1}{N} \sum_{k=0}^{N-1} X_k \, e^{+2\pi i \, kn / N}

with :math:`k, n = 0, \dots, N-1`. The term :math:`e^{-2\pi i \, kn/N}` is a
complex sinusoid, a wave oscillating at frequency :math:`k`.

The forward transform computes the inner product of
the signal with the complex sinusoid :math:`e^{-2\pi i kn/N}` for each frequency :math:`k`. 
Since complex sinusoids at different frequencies are orthogonal over the
:math:`N` samples, contributions from other frequencies cancel out, leaving
only the component at frequency :math:`k` in :math:`X_k`.
Its magnitude :math:`|X_k|` is proportional to the amplitude of that frequency and
its angle :math:`\arg(X_k)` records the phase.

The inverse transform reverses this. It scales each sinusoid by its coefficient
:math:`X_k` and adds them all back together, recovering the original samples
:math:`x_n` exactly.

Computing DFT in Physika
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The forward and inverse Discrete Fourier Transforms (DFT) can be implemented in Physika as follows:

.. code-block:: text

    π: ℝ = 3.141592653589793
    # Discrete Fourier Transform (Forward and Inverse):

    def dft(x: ℂ[m]): ℂ[m]:
        s = len(x)                        # number of samples
        coeffs = for k: ℕ(s) → sum(for n: ℕ(s) → x[n] * (cos(2*π*k*n / s) - 1j * sin(2*π*k*n / s)))  # one coefficient per frequency k
        return coeffs

    def idft(X: ℂ[m]): ℂ[m]:
        s = len(X)                        # number of samples
        samples = for n: ℕ(s) → sum(for k: ℕ(s) → X[k] * (cos(2*π*k*n / s) + 1j * sin(2*π*k*n / s))) / s  # one reconstructed sample per n
        return samples

For example, consider a cosine signal that completes one full cycle over four samples, 
:math:`x_n = \cos(2\pi n / 4)`:

.. code-block:: text

    signal: ℂ[4] = for n: ℕ(4) → cos(2*π*n / 4)

    forward_transform: ℂ[4] = dft(signal)
    abs(forward_transform)
    idft(forward_transform)

Output::

    [0.0, 2.0, 0.0, 2.0] ∈ ℝ[4]
    [(1+0j), 0j, (-1+0j), 0j] ∈ ℂ[4]

The nonzero coefficients fall at :math:`k = 1` and :math:`k = 3`, the single
frequency the cosine carries, and ``idft`` recovers the original samples
exactly.

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

    X_k &= \sum_{n=0}^{N-1} x_n \, e^{-2\pi i \, kn/N} \\
        &= \sum_{m=0}^{N/2-1} x_{2m} \, e^{-2\pi i \, k(2m)/N}
         + \sum_{m=0}^{N/2-1} x_{2m+1} \, e^{-2\pi i \, k(2m+1)/N} \\
        &= \sum_{m=0}^{N/2-1} x_{2m} \, e^{-2\pi i \, km/(N/2)}
         + e^{-2\pi i \, k/N} \sum_{m=0}^{N/2-1} x_{2m+1} \, e^{-2\pi i \, km/(N/2)}

We've split the forward sum into two half-size DFTs, one over the even-indexed
samples and one over the odd-indexed samples, tied together by the factor
:math:`e^{-2\pi i \, k/N}`. Writing those two half transforms as

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

Each half transform has period :math:`N/2`, and the twiddle factor flips sign
over that same half period:

.. math::

    E_{k+N/2} &= E_k \\
    O_{k+N/2} &= O_k \\
    e^{-2\pi i \, (k+N/2)/N} &= -e^{-2\pi i \, k/N}

So a single pair :math:`(E_k, O_k)` produces two output coefficients at once:

.. math::

    X_k &= E_k + e^{-2\pi i \, k/N} \, O_k \\
    X_{k+N/2} &= E_k - e^{-2\pi i \, k/N} \, O_k

We compute :math:`E_k` and :math:`O_k` once, for :math:`k = 0, \dots, N/2 - 1`,
and read off all :math:`N` outputs from them. This two-input, two-output
combine is the FFT's butterfly [ButterflyWiki]_.

Each of those two half-length DFTs is computed the same way, splitting again into
even and odd parts, then each quarter-length DFT after that, down to transforms of
length one. This gives :math:`\log_2 N` levels of splitting, each costing
:math:`O(N)` to recombine, so the whole transform runs in :math:`O(N \log N)`,
against :math:`O(N^2)` for the direct sum.

Computing FFT (Cooley-Tukey) in Physika
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The recursive Cooley-Tukey FFT algorithm can be implemented in Physika as follows:

.. code-block:: text

    π: ℝ = 3.141592653589793
    # Fast Fourier Transform, recursive radix-2 Cooley-Tukey (O(N log N)):

    def ct_fft(x: ℂ[m], s: ℝ): ℂ[m]:
        if s == 1.0:
            return x                                   # base case: a length-1 transform is the sample itself
        even = for a: ℕ(s / 2) → x[2 * a]               # even-indexed samples
        odd  = for a: ℕ(s / 2) → x[2 * a + 1]           # odd-indexed samples
        E = ct_fft(even, s / 2)                         # transform of the even half
        O = ct_fft(odd, s / 2)                          # transform of the odd half
        twiddle = for k: ℕ(s / 2) → cos(2*π*k / s) - 1j * sin(2*π*k / s)  # twiddle factor
        lower = for k: ℕ(s / 2) → E[k] + twiddle[k] * O[k]                # first half of the output
        upper = for k: ℕ(s / 2) → E[k] - twiddle[k] * O[k]                # second half of the output
        return concat(lower, upper)


    def ict_fft(X: ℂ[m], s: ℝ): ℂ[m]:
        if s == 1.0:
            return X                                   # base case: a length-1 inverse transform is the sample itself
        even = for a: ℕ(s / 2) → X[2 * a]               # even-indexed coefficients
        odd  = for a: ℕ(s / 2) → X[2 * a + 1]           # odd-indexed coefficients
        E = ict_fft(even, s / 2)                        # inverse transform of the even half
        O = ict_fft(odd, s / 2)                         # inverse transform of the odd half
        twiddle = for k: ℕ(s / 2) → cos(2*π*k / s) + 1j * sin(2*π*k / s)  # twiddle factor, conjugated for the inverse
        lower = for k: ℕ(s / 2) → E[k] + twiddle[k] * O[k]                # first half of the output
        upper = for k: ℕ(s / 2) → E[k] - twiddle[k] * O[k]                # second half of the output
        return concat(lower, upper)

Using the same cosine signal as before:

.. code-block:: text

    signal: ℂ[4] = for n: ℕ(4) → cos(2*π*n / 4)
    fft_sum: ℂ[4] = ct_fft(signal, 4)
    abs(fft_sum)
    ict_fft(fft_sum, 4) / 4

Output::

    [0.0, 2.0, 0.0, 2.0] ∈ ℝ[4]
    [(1+0j), 0j, (-1+0j), 0j] ∈ ℂ[4]

The FFT produces the same result as ``dft``/``idft``, computed in
:math:`O(N \log N)` instead of :math:`O(N^2)`.

FFT Builtins
------------------

The above implementations aren't fast in practice, and that's expected. They are
written to illustrate how the DFT and FFT work, not to maximize performance.
Each executes as a Python-level loop that rebuilds tensors on every iteration,
whereas PyTorch's fft module uses highly optimized compiled C/C++ backends. Physika
exposes these directly as builtins, which compile to PyTorch's FFT functions
[TorchFFT]_. Let's take a look at them.

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

Gradient of the signal energy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A common practical application is computing the energy of a signal. The *energy* of a 
discrete signal is defined as the sum of its squared samples. By
Parseval's theorem [ParsevalWiki]_ the total energy can be computed either from the signal samples or from its Fourier spectrum:

.. math::

    E = \sum_{n=0}^{N-1} x_n^2 = \frac{1}{N} \sum_{k=0}^{N-1} |X_k|^2 .


Each frequency component contributes :math:`|X_k|^2 / N` to the total energy, so the sum shows how the
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

Both forms give the same energy, :math:`204`, providing a numerical
verification of Parseval's theorem. They also produce the same gradient,
:math:`\partial E/\partial x_n = 2x_n`, confirming that ``grad`` correctly
propagates derivatives through fft.


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

.. [ButterflyWiki] Butterfly diagram. *Wikipedia*.
   `<https://en.wikipedia.org/wiki/Butterfly_diagram>`_.