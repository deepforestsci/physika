Fast Fourier Transform
======================

In this tutorial we will cover what the Discrete Fourier Transform (DFT) is, the
Fast Fourier Transform (FFT) that computes it efficiently, how to use them in
Physika, and how gradients flow through them with ``grad``.

Many of the signals we work with, such as a pressure trace, a time series, or a
row of a simulation grid, are more naturally described by the oscillations they
contain than by their raw sample values. The Discrete Fourier Transform (DFT) [DFTWiki]_ resolves a signal
into its constituent frequency components, reporting the amplitude and phase of each frequency.
The Fast Fourier Transform (FFT) [FFTWiki]_ computes this same decomposition, using
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
        s: ℝ = len(x)                        # number of samples
        coeffs: ℂ[s] = for k: ℕ(s) → sum(for n: ℕ(s) → x[n] * (cos(2*π*k*n / s) - 1j * sin(2*π*k*n / s)))  # one coefficient per frequency k
        return coeffs

    def idft(X: ℂ[m]): ℂ[m]:
        s: ℝ = len(X)                        # number of samples
        samples: ℂ[s] = for n: ℕ(s) → sum(for k: ℕ(s) → X[k] * (cos(2*π*k*n / s) + 1j * sin(2*π*k*n / s))) / s  # one reconstructed sample per n
        return samples

For example, consider a cosine signal that completes one full cycle over four samples,
:math:`x_n = \cos(2\pi n / 4)` for :math:`n = 0, 1, 2, 3`, giving the samples
:math:`x = (1, 0, -1, 0)`:

.. code-block:: text

    signal: ℂ[4] = for n: ℕ(4) → cos(2*π*n / 4)

    dft_spectrum: ℂ[4] = dft(signal)        # forward transform of the input signal
    dft_spectrum
    abs(dft_spectrum)                       # magnitude of the spectrum
    idft(dft_spectrum)                      # inverse transform

Output::

    [0j, (2+0j), 0j, (2+0j)] ∈ ℂ[4]
    [0.0, 2.0, 0.0, 2.0] ∈ ℝ[4]
    [(1+0j), 0j, (-1+0j), 0j] ∈ ℂ[4]

The Fourier coefficients are complex numbers, although for this signal they happen
to be purely real. The nonzero coefficients fall at :math:`k = 1` and
:math:`k = 3`, the single frequency the cosine carries. Taking abs gives
the corresponding magnitude spectrum, and idft recovers the original
samples exactly.

Cooley-Tukey FFT Algorithm
---------------------------

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

The two summations are now recognizable as DFTs of length :math:`N/2`: one over the 
even-indexed samples and one over the odd-indexed samples. The odd transform carries 
the additional factor :math:`e^{-2\pi i\,k/N}`, obtained by rewriting
:math:`e^{-2\pi i\,k(2m+1)/N}` as
:math:`e^{-2\pi i\,km/(N/2)}e^{-2\pi i\,k/N}`. Writing the two half transforms
as

.. math::

    E_k = \sum_{m=0}^{N/2-1} x_{2m} \, e^{-2\pi i \, km/(N/2)}, \qquad
    O_k = \sum_{m=0}^{N/2-1} x_{2m+1} \, e^{-2\pi i \, km/(N/2)},

The DFT over all :math:`N` samples therefore becomes

.. math::

    X_k = E_k + e^{-2\pi i \, k/N} \, O_k.

The factor :math:`e^{-2\pi i\,k/N}` multiplying :math:`O_k` is the twiddle
factor [TwiddleWiki]_. Since

.. math::

    e^{-2\pi i\,k/N} = \cos(2\pi k/N) - i\sin(2\pi k/N), \qquad
    \left|e^{-2\pi i\,k/N}\right| = 1,

multiplying :math:`O_k` by it changes only its phase, rotating it by
:math:`-2\pi k/N` radians while leaving its magnitude unchanged. That rotation
is necessary because :math:`O_k` is a length-:math:`N/2` DFT of the odd
samples taken on their own, as if they started at time zero, when in the
original signal they actually sit one position after the even samples. The
twiddle factor reinstates that one-sample offset, aligning :math:`O_k` with
:math:`E_k` before the two are added.

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

    def ct_fft(x: ℂ[m], s: ℝ): ℂ[m]:                  # x: input signal array; s: size (length) of the array
        if s == 1.0:
            return x                                        # base case: a length-1 transform is the sample itself
        half_size: ℝ = s / 2                                 # size of each half-transform
        even: ℂ[half_size] = for a: ℕ(half_size) → x[2 * a]      # even-indexed samples
        odd:  ℂ[half_size] = for a: ℕ(half_size) → x[2 * a + 1]  # odd-indexed samples
        E: ℂ[half_size] = ct_fft(even, half_size)                # transform of the even half
        O: ℂ[half_size] = ct_fft(odd, half_size)                 # transform of the odd half
        twiddle: ℂ[half_size] = for k: ℕ(half_size) → cos(2*π*k / s) - 1j * sin(2*π*k / s)  # twiddle factor
        lower: ℂ[half_size] = for k: ℕ(half_size) → E[k] + twiddle[k] * O[k]                # first half of the output
        upper: ℂ[half_size] = for k: ℕ(half_size) → E[k] - twiddle[k] * O[k]                # second half of the output
        return concat(lower, upper)

    def ict_fft(X: ℂ[m], s: ℝ): ℂ[m]:                 # X: input spectrum array; s: size (length) of the array
        if s == 1.0:
            return X                                        # base case: a length-1 inverse transform is the sample itself
        half_size: ℝ = s / 2                                 # size of each half-transform
        even: ℂ[half_size] = for a: ℕ(half_size) → X[2 * a]      # even-indexed coefficients
        odd:  ℂ[half_size] = for a: ℕ(half_size) → X[2 * a + 1]  # odd-indexed coefficients
        E: ℂ[half_size] = ict_fft(even, half_size)               # inverse transform of the even half
        O: ℂ[half_size] = ict_fft(odd, half_size)                # inverse transform of the odd half
        twiddle: ℂ[half_size] = for k: ℕ(half_size) → cos(2*π*k / s) + 1j * sin(2*π*k / s)  # twiddle factor, conjugated for the inverse
        lower: ℂ[half_size] = for k: ℕ(half_size) → E[k] + twiddle[k] * O[k]                # first half of the output
        upper: ℂ[half_size] = for k: ℕ(half_size) → E[k] - twiddle[k] * O[k]                # second half of the output
        return concat(lower, upper)

Using the same cosine signal as before:

.. code-block:: text

    signal: ℂ[4] = for n: ℕ(4) → cos(2*π*n / 4)
    fft_spectrum: ℂ[4] = ct_fft(signal, 4)            # forward transform via Cooley-Tukey
    fft_spectrum
    abs(fft_spectrum)                                 # magnitude of the spectrum
    ict_fft(fft_spectrum, 4) / 4                      # inverse transform

Output::

    [0j, (2+0j), 0j, (2+0j)] ∈ ℂ[4]
    [0.0, 2.0, 0.0, 2.0] ∈ ℝ[4]
    [(1+0j), 0j, (-1+0j), 0j] ∈ ℂ[4]

The FFT produces the same result as ``dft``/ ``idft``, computed in
:math:`O(N \log N)` instead of :math:`O(N^2)`.

Built-in FFTs
------------------

The ``dft`` and ``ct_fft`` implementations above illustrate the underlying algorithms, but in
practice Physika provides FFTs as builtins that compile directly to PyTorch's highly 
optimized FFT functions [TorchFFT]_. We first introduce these builtins, then compare 
them with the implementations above, verifying that they produce the same results while 
achieving substantially better performance.

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

Comparing implementations
~~~~~~~~~~~~~~~~~~~~~~~~~~

Before using the builtin FFT, we can verify that all three implementations
compute the same transform. Using the cosine signal from earlier:

.. code-block:: text

    dft(signal)
    ct_fft(signal, 4)
    fft(signal)

Output::

    [0j, (2+0j), 0j, (2+0j)] ∈ ℂ[4]
    [0j, (2+0j), 0j, (2+0j)] ∈ ℂ[4]
    [0j, (2+0j), 0j, (2+0j)] ∈ ℂ[4]

All three produce the same Fourier transform, up to floating-point roundoff.

Having verified that all three implementations compute the same transform, 
we can now compare how efficiently they do so. We benchmark the implementations on randomly 
generated complex signals of different lengths :math:`N`, where :math:`N` is the number of samples in the signal. 
Since Physika's ``fft`` builtin lowers directly to ``torch.fft.fft``, we compare
``torch.fft.fft`` alongside our ``dft`` and ``ct_fft`` implementations,
using Python's ``timeit`` module:

.. note::
   This benchmark is plain Python, run separately from the ``.phyk`` file.
   Here, ``dft`` and ``ct_fft`` refer to the generated PyTorch functions
   Physika compiles them to (see ``--print-code`` in :doc:`intro_to_physika`).

.. code-block:: python

    import timeit
    import torch

    def time_it(fn, *args, number=20, repeat=5):
        timer = timeit.Timer(lambda: fn(*args))
        return min(timer.repeat(repeat=repeat, number=number)) / number

    for N in [64, 512]:
        x = torch.randn(N, dtype=torch.complex64)
        t_dft = time_it(dft, x, number=5, repeat=3)
        t_ct  = time_it(ct_fft, x, float(N), number=20, repeat=5)
        t_fft = time_it(torch.fft.fft, x, number=1000, repeat=5)
        print(f"N={N}  dft={t_dft*1e3:.1f} ms  ct_fft={t_ct*1e3:.1f} ms  fft={t_fft*1e6:.1f} us")

gives:

==========  ============  ============  ================
N           dft           ct_fft        fft (builtin)
==========  ============  ============  ================
64          895 ms        36.4 ms       6.2 μs
512         55.8 s        503.8 ms      22.1 μs
==========  ============  ============  ================

The benchmark clearly demonstrates the performance advantage of the FFT algorithm.
At :math:`N = 64`, the builtin ``fft`` is already about :math:`10^5` times faster than ``dft``, 
while at :math:`N = 512` the gap grows to more than :math:`10^6`. This reflects the difference 
between the :math:`O(N^2)` cost of the direct DFT and the :math:`O(N \log N)` complexity of the FFT. 
Although ``ct_fft`` uses the same :math:`O(N \log N)` algorithm, the builtin ``fft`` is still substantially 
faster because it is implemented by PyTorch's highly optimized compiled backend [TorchFFTBackend]_.

Having seen that the builtins produce the same results much more efficiently, we now look at 
how to use them, starting with the one-dimensional transform.

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

    signal_1d: ℝ[Ns] = for n: ℕ(Ns) -> cos(2*π * (n - center) / Ns) + 0.5 * cos(2*π * 3 * (n - center) / Ns)  # two cosines: strong at k=1, weaker at k=3
    spectrum_1d: ℂ[Ns] = fft(signal_1d)   # 1D Fourier transform of signal_1d
    abs(spectrum_1d)                       # magnitude at each frequency, peaks mark k=1 and k=3
    ifft(spectrum_1d)                      # inverse transform, recovers signal_1d



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

In two dimensions, ``fft2`` treats a matrix as a two-dimensional signal and computes its 
complex frequency spectrum by applying the Fourier transform along both rows and columns. 
``ifft2`` reconstructs the original matrix from that spectrum.

.. code-block:: text

    matrix: ℝ[3, 3] = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]         # a small 3x3 grid
    spectrum_2d: ℂ[3, 3] = fft2(matrix)                          # 2D Fourier transform of matrix
    spectrum_2d
    ifft2(spectrum_2d)                                           # inverse transform, recovers matrix

Output::

    [[(45+0j), (-4.5+2.6j), (-4.5-2.6j)],
     [(-13.5+7.79j), 0j, 0j],
     [(-13.5-7.79j), 0j, 0j]] ∈ ℂ[3,3]
    [[(1+0j), (2+0j), (3+0j)],
     [(4+0j), (5+0j), (6+0j)],
     [(7+0j), (8+0j), (9+0j)]] ∈ ℂ[3,3]

The top-left coefficient is the zero-frequency (DC) component and equals the sum of all entries (45). 
The remaining coefficients describe the matrix's spatial frequency content. ``ifft2`` reconstructs the 
original matrix.


An N-dimensional transform
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The same idea extends to any number of dimensions. ``fftn`` computes the complex
frequency spectrum of a N-dimensional tensor over all its axes, and ``ifftn`` reconstructs the
original tensor. Take a small 2×2×2 tensor as an example:

.. code-block:: text

    tensor: ℝ[2, 2, 2] = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]    # a small 2x2x2 tensor
    spectrum_Nd: ℂ[2, 2, 2] = fftn(tensor)                        # N-D Fourier transform of tensor
    spectrum_Nd
    ifftn(spectrum_Nd)                                            # inverse transform, recovers tensor  

Output::

    [[[(36+0j), (-4+0j)], [(-8+0j), 0j]],
     [[(-16+0j), 0j], [0j, 0j]]] ∈ ℂ[2,2,2]
    [[[(1+0j), (2+0j)], [(3+0j), (4+0j)]],
     [[(5+0j), (6+0j)], [(7+0j), (8+0j)]]] ∈ ℂ[2,2,2]

As in two dimensions, the corner coefficient is the zero-frequency (DC) component
and equals the sum of all entries (36). ``ifftn`` reconstructs the original
tensor.


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
    X

Output::

    [(10+0j), (-2+2j), (-2+0j), (-2-2j)] ∈ ℂ[4]

A coefficient :math:`X_k` is complex, and ``grad`` needs a real-valued output, so
we differentiate its real part rather than the coefficient itself. Writing
:math:`X_1` out from its defining sum, with :math:`N = 4`,

.. math::

    X_1 = \sum_{n=0}^{3} x_n \, e^{-2\pi i \, n/4}
        = x_0 + x_1 e^{-i\pi/2} + x_2 e^{-i\pi} + x_3 e^{-i3\pi/2}

Since :math:`e^{-i\pi/2} = -i`, :math:`e^{-i\pi} = -1`, and :math:`e^{-i3\pi/2} = i`,

.. math::

    X_1 = x_0 - i x_1 - x_2 + i x_3 = (x_0 - x_2) + i(x_3 - x_1)

so the real part is :math:`\mathrm{Re}(X_1) = x_0 - x_2`, a linear function of only
:math:`x_0` and :math:`x_2`. Its gradient is therefore :math:`1` at :math:`x_0`,
:math:`-1` at :math:`x_2`, and zero elsewhere. With :math:`x_0 = 1.0` and
:math:`x_2 = 3.0`, this predicts :math:`\mathrm{Re}(X_1) = -2.0`:

.. code-block:: text

    y: ℝ = real(X[1])
    y
    grad(y, x)

Output::

    -2.0 ∈ ℝ
    [1.0, 0.0, -1.0, 0.0] ∈ ℝ[4]

The same gradient is obtained if we replace the builtin ``fft`` with ``dft`` or ``ct_fft``. 
This is because ``grad`` differentiates the computation that produced the output, 
regardless of which implementation produced it.

.. code-block:: text

    dft_spectrum: ℂ[4] = dft(x)
    dft_coeff: ℝ = real(dft_spectrum[1])
    grad(dft_coeff, x)

    ct_fft_spectrum: ℂ[4] = ct_fft(x, 4)
    ct_fft_coeff: ℝ = real(ct_fft_spectrum[1])
    grad(ct_fft_coeff, x)

Output::

    [1.0, 0.0, -1.0, 0.0] ∈ ℝ[4]
    [1.0, 0.0, -1.0, 0.0] ∈ ℝ[4]

All three implementations produce the same gradient of ``real(X[1])`` with
respect to the input :math:`x`, up to floating-point roundoff, confirming that
``grad`` differentiates correctly regardless of which FFT implementation is
used.

So far we have differentiated through the forward transform. The inverse transform 
is equally differentiable, allowing gradients to propagate back through signal 
reconstruction as well. Running ``ifft`` on the spectrum reconstructs the signal,
and we differentiate the same :math:`k = 1` sample as before, this time from the
reconstructed signal:

.. code-block:: text

    inv_signal: ℂ[4] = ifft(X)
    inv_sample: ℝ = real(inv_signal[1])
    inv_sample
    grad(inv_sample, x)

Output::

    2.0 ∈ ℝ
    [0.0, 1.0, 0.0, 0.0] ∈ ℝ[4]

Since ``ifft(fft(x))`` returns ``x``, the reconstructed sample at index 1 is just
:math:`x_1`, so its gradient is :math:`1` at :math:`x_1` and zero elsewhere.

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

    energy_time: ℝ = sum(x2**2)                  # energy of x2 - sum of its squared samples
    energy_freq: ℝ = sum(abs(X2)**2) / len(x2)   # same energy - sum of squared spectrum magnitudes, normalized by N

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
verification of Parseval's theorem. ``energy_time`` never touches ``fft``, so
its gradient is just the plain derivative of a sum of squares,
:math:`\partial E/\partial x_n = 2x_n`. ``energy_freq`` is built from
``fft(x2)``, so computing its gradient exercises the FFT's backward pass
instead. The two come out identical, confirming that ``grad`` correctly
propagates derivatives through ``fft``.


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

.. [TwiddleWiki] Twiddle factor. *Wikipedia*.
    `<https://en.wikipedia.org/wiki/Twiddle_factor>`_.

.. [TorchFFTBackend] The torch.fft module: Accelerated Fast Fourier
   Transforms with Autograd in PyTorch. *PyTorch Blog*.
   `<https://pytorch.org/blog/the-torch.fft-module-accelerated-fast-fourier-transforms-with-autograd-in-pyTorch/>`_.