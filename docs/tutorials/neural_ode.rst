Neural ODE 
===========

A **Neural ODE** replaces the right-hand side of an ordinary differential
equation with a neural network.  Instead of hand-crafting the vector field
that governs a physical system, we *learn* it from observed trajectories.


.. math::

   \frac{d\mathbf{z}(t)}{dt} = f_\theta\!\bigl(\mathbf{z}(t),\, t\bigr)

where :math:`f_\theta` is a neural network with parameters :math:`\theta`.
Given an initial state :math:`\mathbf{z}(t_0)`, the state at any later time
is obtained by integrating:

.. math::

   \mathbf{z}(t_1) = \mathbf{z}(t_0)
                    + \int_{t_0}^{t_1} f_\theta\!\bigl(\mathbf{z}(t),t\bigr)\,dt

This document shows how to implement and train a Neural ODE in **Physika**
to learn the dynamics of a simple pendulum.


Loss Function
-------------

Given a predicted trajectory
:math:`\hat{\mathbf{y}}_{1:T} = \{\hat{\mathbf{y}}_1, \ldots, \hat{\mathbf{y}}_T\}`
and the true trajectory
:math:`\mathbf{y}_{1:T}^*`, we minimise the mean-squared error:

.. math::

   \mathcal{L}(\theta)
   = \frac{1}{T} \sum_{i=1}^{T}
     \bigl\|\hat{\mathbf{y}}_i(\theta) - \mathbf{y}_i^*\bigr\|^2

Gradients :math:`\nabla_\theta \mathcal{L}` are computed by
back-propagating through the ODE solver with PyTorch autograd.


.. code-block::

   def mse_loss(pred_traj: R[m, 2, 1], true_traj: R[m, 2, 1]): R:
       diff = pred_traj - true_traj
       return mean(diff**2)


Numerical Solver — RK4
-----------------------

We integrate the ODE with the classical fourth-order Runge–Kutta scheme.
Given state :math:`\mathbf{y}_n` at time :math:`t_n`, the state at
:math:`t_{n+1} = t_n + \Delta t` is:

.. math::

   k_1 &= f_\theta(\mathbf{y}_n) \\
   k_2 &= f_\theta\!\left(\mathbf{y}_n + \tfrac{\Delta t}{2}\,k_1\right) \\
   k_3 &= f_\theta\!\left(\mathbf{y}_n + \tfrac{\Delta t}{2}\,k_2\right) \\
   k_4 &= f_\theta\!\left(\mathbf{y}_n + \Delta t\,k_3\right) \\[4pt]
   \mathbf{y}_{n+1} &= \mathbf{y}_n
        + \frac{\Delta t}{6}\bigl(k_1 + 2k_2 + 2k_3 + k_4\bigr)

.. code-block::

   def rk4_step(state: ℝ[2, 1], t: ℝ, dt: ℝ, ode_func: ODEFunc): ℝ[2, 1]:
       k1: ℝ[2, 1] = ode_func(state)
       k2_state: ℝ[2, 1] = state + 0.5 * dt * k1
       k2: ℝ[2, 1] = ode_func(k2_state)
       k3_state: ℝ[2, 1] = state + 0.5 * dt * k2
       k3: ℝ[2, 1] = ode_func(k3_state)
       k4_state: ℝ[2, 1] = state + dt * k3
       k4: ℝ[2, 1] = ode_func(k4_state)
       return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

The ODE solver rolls out the full trajectory by calling ``rk4_step``
at every time point:

.. code-block::

   def odesolver(ode_func: ODEFunc, y0: R[2, 1], dt: R, timesteps: R[n]): R[m, 2, 1]:
       n_times: N = len(timesteps)
       trajectory: ℝ[n_times, 2, 1] = zeros(n_times, 2, 1)
       state: R[2, 1] = y0
       trajectory[0] = state
       for i: N(1, n_times):
           current_t = timesteps[i-1]
           state = rk4_step(state, current_t, dt, ode_func)
           trajectory[i] = state
       return trajectory


Create Dataset
--------------


The simple pendulum (unit mass, unit length, no damping) obeys:

.. math::

   \dot{q} &= p \\
   \dot{p} &= -\sin(q)

where :math:`q` is the angle and :math:`p` is the angular momentum.
The state vector is :math:`\mathbf{y} = [q,\, p]^\top \in \mathbb{R}^{2 \times 1}`.

Our goal: observe a trajectory of this pendulum and *recover* its dynamics
using a neural network, without ever telling the network the equations above.


We generate the ground-truth trajectory by integrating the *known*
pendulum equations with the same RK4 solver.

The true vector field is:

.. math::

   f^*(\mathbf{y}) =
   \begin{bmatrix} p \\ -\sin(q) \end{bmatrix}

.. code-block::

   def simple_pendulum(state: ℝ[2, 1]): ℝ[2, 1]:
       q: ℝ = state[0, 0]
       p: ℝ = state[1, 0]
       dq: ℝ = p
       dp: ℝ = 0.0 - sin(q)
       return [[dq], [dp]]

   def generate_dataset(y0: ℝ[2, 1], dt: ℝ, timesteps: R[n]): ℝ[m, 2, 1]:
       n_times = len(timesteps)
       trajectory: ℝ[n_times, 2, 1] = zeros(n_times, 2, 1)
       state: ℝ[2, 1] = y0
       trajectory[0] = state
       for i: ℕ(1, n_times):
           current_t = timesteps[i-1]
           state = rk4_step(state, current_t, dt, simple_pendulum)
           trajectory[i] = state
       return trajectory

   # Time grid and initial condition
   timesteps: ℝ[100] = linspace(0.0, 1.0, 100)
   y0: ℝ[2, 1] = [[1.0], [0.0]]    # q=1 rad, p=0
   dt: R = 0.1

   true_trajectory = generate_dataset(y0, 0.1, timesteps)


Define the Neural Network (ODEFunc)
------------------------------------



The SGD weight update at each epoch is:

.. math::

   \theta \leftarrow \theta - \eta\,\nabla_\theta \mathcal{L}

where :math:`\eta` is the learning rate and gradients are obtained via
``grad(loss, θ)``.

.. code-block::

   class ODEFunc:
       W1: ℝ[Z, 2]
       B1: ℝ[Z, 1]
       W2: ℝ[2, Z]
       B2: ℝ[2, 1]

       def λ(x: ℝ[2, 1]) → ℝ[2, 1]:
           h1: R[Z, 1] = tanh(W1 @ x + B1)
           out: R[2, 1] = W2 @ h1 + B2
           return out

       def update(lr: R, dW1: R[Z,2], dB1: R[Z,1], dW2: R[2,Z], dB2: R[2,1]) → R:
           this.W1 = this.W1 - lr * dW1
           this.B1 = this.B1 - lr * dB1
           this.W2 = this.W2 - lr * dW2
           this.B2 = this.B2 - lr * dB2
           return 0.0

       def train(y0: R[2, 1], lr: R, epochs: N) → R:
           loss: R = 0
           for i: N(epochs):
               pred_traj = odesolver(model, y0, dt, timesteps)
               loss = mse_loss(pred_traj, true_trajectory)
               dW1 = grad(loss, model.W1)
               dB1 = grad(loss, model.B1)
               dW2 = grad(loss, model.W2)
               dB2 = grad(loss, model.B2)
               this.update(lr, dW1, dB1, dW2, dB2)
           return loss


Initialise Weights
------------------

Weights are sampled from a Normal distribution
:math:`\mathcal{N}(\mu, \sigma^2)` with small variance to keep the
initial vector field close to zero:

.. math::

   W_1 \sim \mathcal{N}(0,\, 0.1^2), \quad
   W_2 \sim \mathcal{N}(0,\, 0.1^2), \quad
   b_1 = b_2 = 0.01

.. code-block::

   n_neurons: ℕ = 256

   μ: ℝ = 0.0
   σ: ℝ = 0.1

   W1: ℝ[n_neurons, 2] = for i: ℕ(n_neurons) -> ε: ℝ[2] ~ Normal(μ, σ, 2)
   B1: ℝ[n_neurons, 1] = for i: ℕ(n_neurons) -> [0.01]
   W2: ℝ[2, n_neurons] = for i: ℕ(2) -> ε: ℝ[n_neurons] ~ Normal(μ, σ, n_neurons)
   B2: ℝ[2, 1] = [[0.01], [0.01]]


Training
--------

We instantiate the model and run SGD for 2000 epochs:

.. math::

   \theta^{(i+1)} = \theta^{(i)}
                  - \eta\,\nabla_\theta \mathcal{L}\!\left(\theta^{(i)}\right)

.. code-block::

   model = ODEFunc(W1, B1, W2, B2)

   epochs: N = 2000
   lr: R = 0.01

   model.train(y0, lr, epochs)


Results
-------

After training we compare the predicted and true trajectories in phase
space :math:`(q, p)`:

.. code-block::

   predicted_trajectory: ℝ[m, 2, 1] = odesolver(model, y0, dt, timesteps)
   plot_phase_space(true_trajectory)
   plot_phase_space(predicted_trajectory)

.. note::

   ``plot_trajectories`` is not a built-in Physika function. To use it,
   add the following helper to ``physika/runtime.py``:

   .. code-block:: python

        import matplotlib.pyplot as plt
        def plot_phase_space(true_trajectory, pred_trajectory):
            true_trajectory = true_trajectory.detach().numpy()
            pred_trajectory = pred_trajectory.detach().numpy()

            true_x = true_trajectory[:, 0, 0]
            true_y = true_trajectory[:, 1, 0]

            pred_x = pred_trajectory[:, 0, 0]
            pred_y = pred_trajectory[:, 1, 0]

            plt.figure(figsize=(6, 6))
            plt.plot(true_x, true_y)
            plt.plot(pred_x, pred_y, linestyle="--")

            plt.xlabel("x")
            plt.ylabel("y")
            plt.title("Phase Space Trajectory (True vs Predicted)")

            plt.axis("equal")
            plt.grid(True)

            plt.show()


.. figure:: /_static/tutorial_files/neural_ode_plot.png
   :alt: Learned ODE trajectory vs ground truth
   :align: center
   :width: 700px

   Comparison between ground truth and learned trajectory after training.