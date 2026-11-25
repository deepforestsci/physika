Inverse Kinematics with a Two-Bone Arm
======================================

In this tutorial we solve the **inverse kinematics** of a planar two-bone
arm: given a target point, find the two joint angles that put the hand on
it. The opposite direction, **forward kinematics**, is straightforward: add
up the two bones and find where the hand ends up. The problem is useful in
robotics and animation [Aristidou2018]_.

We write the forward map from joint angles to hand position and let
Physika's ``∇`` compute the gradient of the squared miss. Gradient descent then
finds the angles, without needing an inverse formula. For two bones, a
closed-form solution exists, so we use it as a check. We also fit the bone
lengths from measured hand positions.


The Arm
-------

We consider a planar arm with two bones of lengths :math:`l_1` and
:math:`l_2`, connected at the elbow. The shoulder is fixed at the origin,
and two joint angles :math:`\theta_0` and :math:`\theta_1` set where the
hand ends up. The first angle is measured from the x-axis, the second
relative to the first bone.

.. figure:: /_static/tutorial_files/ik_two_bone_arm.png
   :alt: Planar two-bone arm seen from above, with the shoulder at the origin, bone 1 of length l1 at angle theta 0 from the x-axis, the elbow, bone 2 of length l2 bent by theta 1, and the hand on the target at (5, 5).
   :align: center
   :width: 60%

   A two-bone arm in the pose that reaches the target at (5, 5).

A bone of length :math:`l` pointing at angle :math:`\phi` reaches
:math:`(l\cos\phi,\; l\sin\phi)` from where it starts. The first bone points at
:math:`\phi_1 = \theta_0` and the second at :math:`\phi_2 = \theta_0 + \theta_1`,
so the hand is at

.. math::

    \begin{align*}
    x &= l_1 \cos\phi_1 + l_2 \cos\phi_2, \\
    y &= l_1 \sin\phi_1 + l_2 \sin\phi_2.
    \end{align*}

Here :math:`(x, y)` is the hand position. We use :math:`l_1 = l_2 = 4`. The angles are stored as
:math:`\theta = [\theta_0, \theta_1]`, which is ``θ[0]`` and ``θ[1]`` in the
code.

.. note::

   Both angles are in radians; :math:`0.2987` rad is about
   :math:`17^\circ`. When :math:`\theta_1 = 0` the two bones line up and the
   arm is straight.


Part 1: Finding the Joint Angles
--------------------------------

1.1 The Forward Map
~~~~~~~~~~~~~~~~~~~

``hand`` is the formula from The Arm written as a loop, so it works for any
number of bones.

.. math::

    \phi_i = \theta_0 + \dots + \theta_i, \qquad
    (x, y) = \sum_i l_i \,(\cos\phi_i,\; \sin\phi_i).

.. code-block:: text

    def hand(θ: ℝ[m], l: ℝ[m], n: ℕ): ℝ[2]:
        x, y, φ: ℝ = 0.0, 0.0, 0.0
        for i: ℕ(n):
            φ = φ + θ[i]
            x = x + l[i] * cos(φ)
            y = y + l[i] * sin(φ)
        return [x, y]

Each pass of the loop adds one bone. ``φ`` is the running sum of the joint
angles, so it is the direction :math:`\phi_i` of the current bone:
:math:`\theta_0` after the first pass and :math:`\theta_0 + \theta_1` after the
second. ``x``, ``y`` and ``φ`` all start at zero.

For example, with the angles :math:`\theta = [0.1, 0.5]` the function returns
about :math:`(7.28, 2.66)`.


1.2 The Arm and the Target
~~~~~~~~~~~~~~~~~~~~~~~~~~

We use two bones, each of length 4, so the maximum reach is 8. We picked the
target, :math:`(5, 5)`, by hand. It is about 7.07 units from the
shoulder and is therefore within reach.

.. code-block:: text

    l: ℝ[2] = [4.0, 4.0]
    n_bones: ℕ = len(l)
    reach: ℝ = sum(l)
    target: ℝ[2] = [5.0, 5.0]


1.3 The Loss
~~~~~~~~~~~~

The miss is the hand position minus the target, and the loss is its squared
length, so it is zero exactly when the hand is on the target. Squaring skips the
square root and leaves a smooth function with a simple gradient.

.. math::

    \mathbf{d}(\theta) = \mathrm{hand}(\theta) - \text{target}, \qquad
    L(\theta) = \lVert \mathbf{d}(\theta) \rVert^2 = d_x^2 + d_y^2.

.. code-block:: text

    def L(θ: ℝ[m], l: ℝ[m], n: ℕ, target: ℝ[2]): ℝ:
        d: ℝ[2] = hand(θ, l, n) - target
        return sum(d * d)

In the code, ``d`` is the miss vector, and ``sum(d * d)`` is
:math:`d_x^2 + d_y^2`.

For the starting angles :math:`\theta = [0.1, 0.5]`, the hand is at about
(7.28, 2.66) and the target is at (5, 5), so the miss is
(2.2814, −2.3421). The positive x means the hand is too far
to the right, and the negative y means it is too low. Squaring and adding the
two entries gives a loss of about 10.69.


1.4 The Gradient
~~~~~~~~~~~~~~~~

The Jacobian ``J`` tells us how the hand moves when a joint angle changes. It
has one row for each hand coordinate (x, then y) and one column for each joint
angle. So the entry in row x, column :math:`\theta_0` says how much the hand's
x changes when we turn the first joint a little.

Physika's ``∇`` computes this Jacobian directly from the ``hand`` function. We
don't need to derive its entries by hand.

.. math::

    L(\theta) = d_x^2 + d_y^2,

    \frac{\partial L}{\partial \theta_c}
    = 2\left(d_x J_{0c} + d_y J_{1c}\right),

    \nabla L = 2J^\mathsf{T}\mathbf{d}.

This is the chain rule. The target doesn't move, so the miss changes with the
angles exactly the way the hand does, and each entry of the gradient is 2 times
the miss dotted with that joint's column of :math:`J`.

We start from :math:`\theta = [0.1, 0.5]` and get the gradient two ways:
directly with ``∇`` on the loss, and from the Jacobian with the formula
above.

.. code-block:: text

    θ: ℝ[2] = [0.1, 0.5]

    J = ∇(hand(θ, l, n_bones), θ)
    print(J)

    d: ℝ[2] = hand(θ, l, n_bones) - target
    print(∇(L(θ, l, n_bones, target), θ))
    print(2.0 * (d @ J))

The first print gives the Jacobian at the starting angles:

.. math::

    J =
    \begin{bmatrix}
    -2.6579 & -2.2586\\
    7.2814 & 3.3013
    \end{bmatrix}.

The next two prints give the gradient from ``∇`` and from the Jacobian
expression. ``d @ J`` is a row vector times :math:`J`, which is the same as
:math:`J^\mathsf{T}` times the miss.

For the first angle, the gradient is

.. math::

    2\left(2.2814(-2.6579) + (-2.3421)(7.2814)\right)
    \approx -46.2346.

For the second angle,

.. math::

    2\left(2.2814(-2.2586) + (-2.3421)(3.3013)\right)
    \approx -25.7693.

Both ways give the same gradient, about [-46.2346, -25.7693].

.. note::

    ``J`` has no type annotation. The type checker reports ``∇`` of a vector
    function with the shape of ``θ``, but at run time ``J`` is the 2 × 2 matrix
    printed above.


1.5 Gradient Descent
~~~~~~~~~~~~~~~~~~~~

The gradient points in the direction where the loss increases fastest, so we
update the angles in the opposite direction. The learning rate controls the
size of each step. We use a learning rate of 0.005 and run 1000 steps.

.. math::

    \theta_{k+1} = \theta_k - \eta\,\nabla L(\theta_k),

where :math:`\eta` is the learning rate.

We also save the angles and the gradient at each step in ``history`` and
``gradients`` so we can plot the search afterwards.

.. code-block:: text

    η: ℝ = 0.005
    steps: ℕ = 1000

    history: ℝ[steps, n_bones] = zeros(steps, n_bones)
    gradients: ℝ[steps, n_bones] = zeros(steps, n_bones)

    for i: ℕ(steps):
        history[i] = θ
        g: ℝ[2] = ∇(L(θ, l, n_bones, target), θ)
        gradients[i] = g
        θ = θ - η * g

    print(θ)
    print(hand(θ, l, n_bones))
    print(target)
    print(L(θ, l, n_bones, target))

In the first step both gradient entries are negative, so the update increases
both angles.

The first gradient is [-46.2346, -25.7693]. Multiplying by 0.005 gives
[-0.2312, -0.1288], and subtracting that takes the angles from [0.1, 0.5] to
about [0.3312, 0.6288].

.. figure:: /_static/tutorial_files/ik_two_bone_descent.gif
   :alt: Animation of the two-bone arm during gradient descent: it starts low and to the right, then swings up and settles with its hand on the target at (5, 5). A light blue line traces the path of the hand.
   :align: center
   :width: 60%

   The arm during gradient descent, from step 0 to step 260.

After 1000 steps, the angles converge to about [0.2987, 0.9734],
placing the hand at (5, 5). The final loss is about :math:`1.8 \times 10^{-12}`, and it falls below
:math:`10^{-6}` after 243 steps.

The plot below shows the loss on a log scale.

.. figure:: /_static/tutorial_files/ik_loss.png
   :alt: Loss versus gradient descent step on a logarithmic scale, falling from about 10.7 at step 0 to about 1e-12 by step 1000 and passing 1e-6 at step 243.
   :align: center
   :width: 70%

   Loss during gradient descent.


1.6 Checking with the Closed Form
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For two bones, we can also find the joint angles directly using the geometry of a triangle, which lets us check the result from gradient descent. A target inside the reach has two poses, mirror images of each other, with the elbow bent in opposite directions. At the farthest reach, 8, the arm is straight and there is only one.

Let the target be :math:`(x, y)`, and let :math:`r` be its distance from the shoulder:

.. math::

    r^2 = x^2 + y^2.

The shoulder, elbow, and target form a triangle with sides :math:`l_1`, :math:`l_2`, and :math:`r`. Let :math:`a` be the angle at the shoulder, between the first bone and the line to the target, and let :math:`b` be the angle inside the elbow, between the two bones. Applying the cosine rule gives these two triangle angles:

.. math::

    \cos a = \frac{l_1^2 + r^2 - l_2^2}{2 l_1 r}, \qquad
    \cos b = \frac{l_1^2 + l_2^2 - r^2}{2 l_1 l_2}.

These are triangle angles, not the joint angles used by ``hand``. Let :math:`\psi = \operatorname{atan2}(y, x)` be the direction from the shoulder to the target. The first bone points at an angle :math:`\psi - a`, and the bend at the elbow is :math:`\pi - b`. This gives the joint angles:

.. math::

    \theta_0 = \psi - a, \qquad \theta_1 = \pi - b.

The conversion from the triangle angles to these joint angles is our own step. The cosine-rule formulas come from the analytic two-link solution in [Aristidou2018]_, and also appear in the INF585 lecture slides [INF585]_. Aristidou et al. give only one of the two poses; the mirror pose is the other.

The triangle formulas give a cosine, and Physika has no ``acos`` to turn it into an angle. So we use ``atan2``, which needs a sine as well. The sine is :math:`\sqrt{1 - c^2}`, with ``abs`` inside so rounding cannot make it negative.

The function ``clip_reach`` moves a target beyond the maximum reach back along the same direction, to the reachable boundary. The formula above only works when the target is within reach and is not at the shoulder itself, where :math:`r = 0`. In this example, the target is already within reach, so ``clip_reach`` leaves it unchanged.

.. code-block:: text

    π: ℝ = 3.141592653589793

    def clip_reach(t: ℝ[2], reach: ℝ): ℝ[2]:
        dist: ℝ = sqrt(sum(t * t))
        if dist > reach:
            return t * (reach / dist)
        else:
            return t

    def closed_form(t: ℝ[2], l1: ℝ, l2: ℝ): ℝ[2]:
        r2: ℝ = t[0] * t[0] + t[1] * t[1]
        ca: ℝ = (l1 * l1 + r2 - l2 * l2) / (2.0 * l1 * sqrt(r2))
        cb: ℝ = (l1 * l1 + l2 * l2 - r2) / (2.0 * l1 * l2)
        a: ℝ = atan2(sqrt(abs(1.0 - ca * ca)), ca)
        b: ℝ = atan2(sqrt(abs(1.0 - cb * cb)), cb)
        ψ: ℝ = atan2(t[1], t[0])
        return [ψ - a, π - b]

    goal: ℝ[2] = clip_reach(target, reach)
    exact: ℝ[2] = closed_form(goal, l[0], l[1])
    print(exact)
    print(hand(exact, l, n_bones))

In the code, ``r2`` is :math:`r^2`, ``ca`` and ``cb`` are :math:`\cos a` and :math:`\cos b`, and ``ψ`` is the same :math:`\psi` as above. The elbow bends by :math:`\pi - b` because a straight arm has :math:`b = \pi`.

For our target :math:`(5, 5)` and bone lengths :math:`(4, 4)`, :math:`r = \sqrt{50} \approx 7.0711`. The cosine-rule values are :math:`\cos a \approx 0.8839` and :math:`\cos b \approx -0.5625`, giving :math:`a \approx 0.4867` and :math:`b \approx 2.1682`. Since :math:`\psi = \pi/4 \approx 0.7854`, the resulting joint angles are about :math:`[0.2987, 0.9734]`.

This matches the result from gradient descent in 1.5. The same target also has a mirror pose, with the elbow on the other side of the shoulder-to-target line.

.. figure:: /_static/tutorial_files/ik_two_bone_poses.png
   :alt: Three poses of a two-bone arm for the target (5, 5): the grey dashed arm shows the starting guess, while the two blue arms show the two solutions with the elbow on opposite sides of the shoulder-to-target line.
   :align: center

   The starting guess and the two possible poses that reach the target.


Part 2: Learning the Bone Lengths
---------------------------------

In Part 1 we knew the bone lengths and found the joint angles. Here it is the other way round. We know the joint angles and want to find the bone lengths.

We move the arm to four known joint settings, picked by hand, and note where the hand ends up. We make these measurements ourselves, using bone lengths :math:`(4, 4)`, so we know the right answer to check against. The fit only gets the measurements, not the lengths.

We start from a wrong guess, :math:`[3, 5]`. For this guess, we work out where the hand would be at each of the four settings and square the miss from the measured position. The loss is the sum of the four:

.. math::

    L(\mathbf{l}) =
    \sum_k \left\lVert
    \mathrm{hand}(\theta_k;\mathbf{l}) - \mathbf{p}_k
    \right\rVert^2.

Here :math:`\mathbf{p}_k` is the measured hand position for the :math:`k`-th setting, and :math:`\theta_k` is its pair of joint angles.

``hand`` takes the bone lengths as an input, so ``∇`` can give us the gradient with respect to the lengths, the same way it did for the angles. Each step moves the lengths a little downhill:

.. math::

    \mathbf{l} \leftarrow
    \mathbf{l} - \eta\,\nabla L(\mathbf{l}).

We use a learning rate of 0.02 and 500 steps. That is enough here. With a different guess or learning rate, you may need a different number of steps.

.. code-block:: text

    def make_obs(angs: ℝ[p, 2], l_true: ℝ[2], count: ℕ): ℝ[p, 2]:
        obs: ℝ[count, 2] = zeros(count, 2)
        for k: ℕ(count):
            obs[k] = hand(angs[k], l_true, len(l_true))
        return obs

    def fit_loss(guess: ℝ[m], angs: ℝ[p, 2], obs: ℝ[p, 2], count: ℕ): ℝ:
        total: ℝ = 0.0
        for k: ℕ(count):
            d: ℝ[2] = hand(angs[k], guess, len(guess)) - obs[k]
            total = total + sum(d * d)
        return total

    angs: ℝ[4, 2] = [[0.2, 0.6], [0.9, -0.4], [1.4, 0.8], [2.0, -0.7]]
    n_obs: ℕ = len(angs)
    l_true: ℝ[2] = l
    obs: ℝ[n_obs, 2] = make_obs(angs, l_true, n_obs)

    l_fit: ℝ[2] = [3.0, 5.0]
    η_l: ℝ = 0.02
    length_steps: ℕ = 500

    for s: ℕ(length_steps):
        g_fit: ℝ[2] = ∇(fit_loss(l_fit, angs, obs, n_obs), l_fit)
        l_fit = l_fit - η_l * g_fit

    print(l_fit)

``make_obs`` creates the four measurements by running ``hand`` with the true bone lengths. ``fit_loss`` adds up the squared misses for a guess. The loop is the same as in 1.5, except that it updates ``l_fit`` instead of the angles.

After 500 steps, ``l_fit`` is about [4.0, 4.0], the same as the true lengths.


What to Know
------------

Gradient descent does the job here, but it can go wrong in a few ways. The numbers below come from running this tutorial with one thing changed each time.

The learning rate has to be small. The loss starts at 10.69. With a learning rate of 0.005 it drops below 0.000001 in 243 steps, and with 0.01 in 120. With 0.02 the first step goes too far and the loss jumps to 26.3, and with 0.03 it jumps to 70.4. Neither one settles down. After 1000 steps the loss is still 11.9 and 28.0.

.. list-table::
   :header-rows: 1

   * - Learning rate
     - Steps until loss < 1e-6
     - Loss after 1000 steps
   * - 0.005
     - 243
     - 1.8e-12
   * - 0.01
     - 120
     - 4.5e-13
   * - 0.02
     - not reached
     - 11.9
   * - 0.03
     - not reached
     - 28.0

We keep 0.005. If this simple update isn't enough, [Aristidou2018]_ describe other methods built on the Jacobian, like damped least squares (Section 5.1.3).

Targets close to the edge of the reach are slow. For (7, −3) the loss needs 656 steps instead of 243, and for (0, 7.9), which is just 0.1 short of the maximum reach, it is still not below 0.000001 after 1000 steps. The arm is almost straight there, so bending the elbow barely moves the hand and the gradient is tiny. A target beyond the reach is not an error either. The arm just stretches toward it as far as it can. For (6, −6), which is 8.49 from the shoulder, it points straight at the target and stays a little short, and for (9, 0) the hand stops at (8, 0).

Where you start matters too.


Full Code
---------

.. code-block:: text

    # Hand position from the joint angles
    def hand(θ: ℝ[m], l: ℝ[m], n: ℕ): ℝ[2]:
        x, y, φ: ℝ = 0.0, 0.0, 0.0
        for i: ℕ(n):
            φ = φ + θ[i]
            x = x + l[i] * cos(φ)
            y = y + l[i] * sin(φ)
        return [x, y]

    # Squared distance from hand to target
    def L(θ: ℝ[m], l: ℝ[m], n: ℕ, target: ℝ[2]): ℝ:
        d: ℝ[2] = hand(θ, l, n) - target
        return sum(d * d)

    l: ℝ[2] = [4.0, 4.0]
    n_bones: ℕ = len(l)
    reach: ℝ = sum(l)

    target: ℝ[2] = [5.0, 5.0]

    θ: ℝ[2] = [0.1, 0.5]
    η: ℝ = 0.005
    steps: ℕ = 1000

    J = ∇(hand(θ, l, n_bones), θ)
    print(J)

    d: ℝ[2] = hand(θ, l, n_bones) - target
    print(∇(L(θ, l, n_bones, target), θ))
    print(2.0 * (d @ J))

    history: ℝ[steps, n_bones] = zeros(steps, n_bones)
    gradients: ℝ[steps, n_bones] = zeros(steps, n_bones)

    for i: ℕ(steps):
        history[i] = θ
        g: ℝ[2] = ∇(L(θ, l, n_bones, target), θ)
        gradients[i] = g
        θ = θ - η * g

    print(θ)
    print(hand(θ, l, n_bones))
    print(target)
    print(L(θ, l, n_bones, target))

    π: ℝ = 3.141592653589793

    def clip_reach(t: ℝ[2], reach: ℝ): ℝ[2]:
        dist: ℝ = sqrt(sum(t * t))
        if dist > reach:
            return t * (reach / dist)
        else:
            return t

    # Closed form, used to check gradient descent
    def closed_form(t: ℝ[2], l1: ℝ, l2: ℝ): ℝ[2]:
        r2: ℝ = t[0] * t[0] + t[1] * t[1]
        ca: ℝ = (l1 * l1 + r2 - l2 * l2) / (2.0 * l1 * sqrt(r2))
        cb: ℝ = (l1 * l1 + l2 * l2 - r2) / (2.0 * l1 * l2)
        a: ℝ = atan2(sqrt(abs(1.0 - ca * ca)), ca)
        b: ℝ = atan2(sqrt(abs(1.0 - cb * cb)), cb)
        ψ: ℝ = atan2(t[1], t[0])
        return [ψ - a, π - b]

    goal: ℝ[2] = clip_reach(target, reach)
    exact: ℝ[2] = closed_form(goal, l[0], l[1])
    print(exact)
    print(hand(exact, l, n_bones))

    def make_obs(angs: ℝ[p, 2], l_true: ℝ[2], count: ℕ): ℝ[p, 2]:
        obs: ℝ[count, 2] = zeros(count, 2)
        for k: ℕ(count):
            obs[k] = hand(angs[k], l_true, len(l_true))
        return obs

    def fit_loss(guess: ℝ[m], angs: ℝ[p, 2], obs: ℝ[p, 2], count: ℕ): ℝ:
        total: ℝ = 0.0
        for k: ℕ(count):
            d: ℝ[2] = hand(angs[k], guess, len(guess)) - obs[k]
            total = total + sum(d * d)
        return total

    angs: ℝ[4, 2] = [[0.2, 0.6], [0.9, -0.4], [1.4, 0.8], [2.0, -0.7]]
    n_obs: ℕ = len(angs)
    l_true: ℝ[2] = l
    obs: ℝ[n_obs, 2] = make_obs(angs, l_true, n_obs)

    l_fit: ℝ[2] = [3.0, 5.0]
    η_l: ℝ = 0.02
    length_steps: ℕ = 500

    for s: ℕ(length_steps):
        g_fit: ℝ[2] = ∇(fit_loss(l_fit, angs, obs, n_obs), l_fit)
        l_fit = l_fit - η_l * g_fit

    print(l_fit)


References
----------

.. [Aristidou2018] A\. Aristidou, J. Lasenby, Y. Chrysanthou, and A. Shamir, "Inverse Kinematics Techniques in Computer Graphics: A Survey," *Computer Graphics Forum*, 37(6), 35–58, 2018. DOI: https://doi.org/10.1111/cgf.13310. Free author copy (the section numbers on this page refer to it): https://www.andreasaristidou.com/publications/papers/IK_survey.pdf

.. [INF585] "Animating Virtual Characters," INF585 lecture slides, slide 35, "IK Example with two bones." https://imagecomputing.net/old_storage/old_website2/teaching/2019_2020/semester_2/INF585_computer_animation/class/06_character_animation/presentation.pdf
