Neural Ordinary differential equation (Neural ODE)
=======================================================

In previous tutorials, we have used Ordinary differential equations (ODEs) to describe how a system evolves over time.
We have also seen how to learn parameters of an ODE system using differentiation. **But what if we do not know the equation
itself that describes the system?**

For example, the dynamics of system are described by:

.. math::

    \frac{dy}{dt} = f(y, t)

If the function :math:`f` is known, we can provide an initial condition :math:`y0` to an ODE solver and compute the 
evolution of the system (trajectory).

In many problems the dynamics are not known explicitly, we may only have observations of how the system evolves over time.
So instead of manually define function :math:`f`, we can use neural networks to represent it:

.. math::

    \frac{dy}{dt} = f_\theta(y, t)

where :math:`f_\theta(y, t)` is a neural network with learnable parameters :math:`\theta`.

An ODE solver can be used to generate trajectories using this neural networks, and can be corrected by optimizing the parameters.
This is the overall idea of Neural Ordinary Differential equations. (NeuralODE).

In this tutorial we will implement a simple Neural ODE model and train it to predict dynamics of damped oscillator. It is also recommended to 
read the NeuralODE paper which has explained the concepts in more detail. [Chen_NeuralODE]_


Define the Neural network (ODEFunc)
-------------------------------------

So lets quickly implement the ODEFunc which will be a Feed forward neural network:

.. code-block:: text

    class ODEFunc(W1: ℝ[Z, 2], B1: ℝ[Z, 1], W2: ℝ[Z, Z], B2: ℝ[2, 1]):
        def λ(x: ℝ[2, 1]) → ℝ[2, 1]:
            h1: R[Z, 1] = tanh(this.W1 @ x + this.B1)
            out: R[2, 1] = this.W2 @ h1 + this.B2
            return out


This is a simple Feed forward neural network with one hidden layer, we can initialize the ``ODEFunc`` class by:

.. code-block:: text

    n_neurons: ℕ = 128

    μ, σ: ℝ = 0.0, 0.01

    W1: ℝ[n_neurons, 2] = for i: ℕ(n_neurons) -> ε: ℝ[2] ~ Normal(μ, σ, 2)
    B1: ℝ[n_neurons, 1] = for i: ℕ(n_neurons) -> [0.01]
    W2: ℝ[2, n_neurons] = for i: ℕ(2) -> ε: ℝ[n_neurons] ~ Normal(μ, σ, n_neurons)
    B2: ℝ[2, 1] = [[0.01], [0.01]]

    model: ODEFunc = ODEFunc(W1, B1, W2, B2)

Input layer has 2 neurons, which corresponds to two components :math:`[q, p]` which is explained in dataset section, Hidden layer has
total 128 neurons and output layer has again 2 neurons which represents :math:`[\dot{q}, \dot{p}]`.



Create damped oscillator dataset
-----------------------------------

For this tutorial, we will train our Neural ODE to learn the dynamics of a damped oscillator. The non-dimensional form of
the damped oscillator can be written as: [JKU_Hamiltonian]_

.. math::

    \dot{q} = p, \qquad
    \dot{p} = -q - \alpha p

where :math:`q` represents the position, :math:`p` represents the velocity, and :math:`\alpha` is the damping coefficient.

We can represent this equation in physika as:

.. code-block:: text

    def damped_oscillator(state: ℝ[2,1]): ℝ[2,1]:
        q, p: ℝ = state[0,0], state[1,0]
        α: ℝ = 0.2
        dq: ℝ = p
        dp: ℝ = 0.0 - q - α * p
        return [[dq], [dp]] 

To generate the dataset, we need to numerically integrate the damped oscillator equation. In this tutorial, we will use the classical
fourth-order Runge-Kutta method (RK4).


.. math::

    k_1 &= f(y_n, t_n) \\
    k_2 &= f\left(y_n + \frac{\Delta t}{2}k_1,
                t_n + \frac{\Delta t}{2}\right) \\
    k_3 &= f\left(y_n + \frac{\Delta t}{2}k_2,
                t_n + \frac{\Delta t}{2}\right) \\
    k_4 &= f\left(y_n + \Delta t\,k_3,
                t_n + \Delta t\right) \\
    y_{n+1} &= y_n + \frac{\Delta t}{6} \left(k_1 + 2k_2 + 2k_3 + k_4\right)

We can implement one RK4 step in Physika as:

.. code-block:: text

    def rk4_step(state: ℝ[2, 1], t: ℝ, Δt: ℝ, ode_func: ODEFunc): ℝ[2, 1]:
        k1: ℝ[2, 1] = ode_func(state)
        k2_state: ℝ[2, 1] = state + 0.5 * Δt * k1
        k2: ℝ[2, 1] = ode_func(k2_state)
        k3_state: ℝ[2, 1] = state + 0.5 * Δt * k2
        k3: ℝ[2, 1] = ode_func(k3_state)
        k4_state: ℝ[2, 1] = state + Δt * k3
        k4: ℝ[2, 1] = ode_func(k4_state)
        return state + (Δt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

Now lets create a function to generate dataset:

.. code-block:: text

    def generate_dataset(y0: ℝ[2,1], Δt: ℝ, timesteps: ℝ[n]): ℝ[m,2,1]:
        n_times: ℝ = len(timesteps)
        trajectory: ℝ[n_times, 2, 1] = zeros(n_times, 2, 1)
        state: ℝ[2,1] = y0
        trajectory[0] = state
        for i: ℕ(1, n_times):
            current_t = timesteps[i-1]
            state = rk4_step(state, current_t, Δt, damped_oscillator)
            trajectory[i] = state
        return trajectory

``generate_dataset`` takes three arguments, :math:`y_0` is inital condition, :math:`\Delta t` is step size and
:math:`timesteps` contains total number of points in time we want to store trajectory.

We can create a trajectory by:

.. code-block:: text

    t_start, t_end, Δt: ℝ = 0.0, 15.0, 0.1

    n_steps: ℕ = int((t_end - t_start) / Δt) + 1
    timesteps: ℝ[n_steps] = linspace(t_start, t_end, n_steps)
    y0: ℝ[2,1] = [[1.0], [0.0]]

    true_trajectory: ℝ[n_steps, 2, 1] = generate_dataset(y0, Δt, timesteps)


``true_trajectory`` is what we will compare with our NeuralODE predictions and then will use optimizer to adjust the weights
which eventually helps in predicting better results.


.. figure:: /_static/tutorial_files/neural_ode/true_trajectory.png
   :alt: neural_ode_true_trajectory
   :align: center
   :width: 500px
   :name: neural_ode_true_trajectory

   Figure 1: Phase space trajectory (true-trajectory)



Neural ODE algorithm
----------------------

In this section we will implement the main algorithm of NeuralODE, in paper it is given as:

.. figure:: /_static/tutorial_files/neural_ode/algorithm_1.png
   :alt: algorithm_1
   :align: center
   :width: 700px
   :name: algorithm_1

   Figure 2: Reverse-mode derivative of an ODE initial value problem


Before implementing algorithm, lets try to understand the easiest part, which is **Input** and **return** values, They are
part of a outer function which we are going to define later:

Input
^^^^^^

- dynamics parameters :math:`\theta` : are the parameters of our Neural networks ``ODEFunc``.
- start time :math:`t_0` : represents starting time which is ``t_start``.
- stop time :math:`t_1` : represents stop time which is ``t_end``.
- final state :math:`z(t_1)` : is only the final value of full trajectory
- loss gradient :math:`{\partial L}/{\partial \mathbf{z}(t_1)}` : is Loss w.r.t final state.


return
^^^^^^

- :math:`\frac{\partial L}{\partial \mathbf{z}(t_0)}` : loss gradients w.r.t initial state.
- :math:`\frac{\partial L}{\partial \theta}` : loss gradients w.r.t model parameters.

aug_dynamics
^^^^^^^^^^^^^

After this, lets implement ``aug_dynamics`` function, where the main differentiation happens:



.. figure:: /_static/tutorial_files/neural_ode/aug_dynamics.png
   :alt: aug_dynamics
   :align: center
   :width: 700px
   :name: aug_dynamics

   Figure 3: aug_dynamics function


We can implement this function as:

.. code-block:: text

    def aug_dynamics(z: ℝ[2,1], a: ℝ[2,1], model: ODEFunc): list:
        z: ℝ[2, 1] = detach_grad(z)
        a: ℝ[2, 1] = detach(a)
        f_val: ℝ[2, 1] = model(z)
        scalar: ℝ = sum(a * f_val)
        parameter_list: list = [z, model.W1, model.B1, model.W2, model.B2]
        result: list = grad(scalar, parameter_list)
        dadt: ℝ[2, 1] = -result[0]
        dW1: ℝ[n_neurons, 2] = -result[1]
        dB1: ℝ[n_neurons, 1] = -result[2]
        dW2: ℝ[2, n_neurons] = -result[3]
        dB2: ℝ[2, 1] = -result[4]
        f_val: ℝ[2, 1] = detach(f_val)
        vjps_results: list = [f_val, dadt, dW1, dB1, dW2, dB2]
        return vjps_results

since :math:`\theta` represents model parameters, for our ``ODEFunc`` this are :math:`[W1, B1, W2, B2]`

lets map our Physika code with the actual function, 


.. list-table:: 
   :header-rows: 1
   :widths: 50 50

   * - aug_dynamics (input)
     - Physika code
   * - :math:`\mathbf{z}(t)`
     - ``z: ℝ[2,1]``
   * - :math:`\mathbf{a}(t)`
     - ``a: ℝ[2,1]``
  

.. list-table:: 
   :header-rows: 1
   :widths: 50 50

   * - aug_dynamics (returns)
     - Physika code
   * - :math:`f(\mathbf{z}(t), t, \theta)`
     - ``f_val``
   * - :math:`-\mathbf{a}(t)^\top \frac{\partial f}{\partial \mathbf{z}}`
     - ``dadt = -result[0]``
   * - :math:`-\mathbf{a}(t)^\top \frac{\partial f}{\partial \theta}`
     - ``dW1, dB1, dW2, dB2``



Just to recall from our algorithm, we just learned what is **Input** to our outer function, and what it **returns**, and we also learned and implemented **aug_dynamics** function:
The only remaining parts are, defining augmented state and solving reverse time ODE, which we will implement it in outer function.

.. figure:: /_static/tutorial_files/neural_ode/algorithm_1.png
   :alt: algorithm_1
   :align: center
   :width: 700px
   :name: 

   



The reason this algorithm called as Reverse mode derivative is because, once we do forward solve from :math:`t_0` to :math:`t_1`, we again solve
the same ODE backwards but this time we keep a adjoint value which is considered as final state of forward trajectory, and then we use
``aug_dynamics`` to calculate how its changing, but as we used ``rk4_step`` while going forward, we need another to go backwards which accepts model parameters, this is the main
logic of the outer function.

In physika, we are defining the outer function named as ``adjoint_solver`` and the reverse rk4-step as ``rk4_step_adjoint``:

Here is the implementation of ``rk4_step_adjoint``:

.. code-block:: text

    def rk4_step_adjoint(z: ℝ[2, 1], a: ℝ[2, 1], W1g: ℝ[n_neurons, 2], B1g: ℝ[n_neurons, 1], W2g: ℝ[n_neurons, 2], B2g: ℝ[n_neurons, 1], dt: ℝ, model: ODEFunc): list:
        # k1
        dz1, da1, dW1_1, dB1_1, dW2_1, dB2_1 = aug_dynamics(z, a, model)
        # k2
        z2: ℝ[2, 1] = z + 0.5 * dt * dz1
        a2: ℝ[2, 1] = a + 0.5 * dt * da1
        W1g2: ℝ[n_neurons, 2] = W1g + 0.5 * dt * dW1_1
        B1g2: ℝ[n_neurons, 1] = B1g + 0.5 * dt * dB1_1
        W2g2: ℝ[n_neurons, 2] = W2g + 0.5 * dt * dW2_1
        B2g2: ℝ[n_neurons, 1] = B2g + 0.5 * dt * dB2_1
        dz2, da2, dW1_2, dB1_2, dW2_2, dB2_2 = aug_dynamics(z2, a2, model)
        # k3
        z3: ℝ[2, 1] = z + 0.5 * dt * dz2
        a3: ℝ[2, 1] = a + 0.5 * dt * da2
        W1g3: ℝ[n_neurons, 2] = W1g + 0.5 * dt * dW1_2
        B1g3: ℝ[n_neurons, 1] = B1g + 0.5 * dt * dB1_2
        W2g3: ℝ[n_neurons, 2] = W2g + 0.5 * dt * dW2_2
        B2g3: ℝ[n_neurons, 1] = B2g + 0.5 * dt * dB2_2
        dz3, da3, dW1_3, dB1_3, dW2_3, dB2_3 = aug_dynamics(z3, a3, model)
        # k4
        z4: ℝ[2, 1] = z + dt * dz3
        a4: ℝ[2, 1] = a + dt * da3
        W1g4: ℝ[n_neurons, 2] = W1g + dt * dW1_3
        B1g4: ℝ[n_neurons, 1] = B1g + dt * dB1_3
        W2g4: ℝ[n_neurons, 2] = W2g + dt * dW2_3
        B2g4: ℝ[n_neurons, 1] = B2g + dt * dB2_3
        dz4, da4, dW1_4, dB1_4, dW2_4, dB2_4 = aug_dynamics(z4, a4, model)
        # update
        new_z: ℝ[2, 1] = z + (dt / 6.0) * (dz1 + 2*dz2 + 2*dz3 + dz4)
        new_a: ℝ[2, 1] = a + (dt / 6.0) * (da1 + 2*da2 + 2*da3 + da4)
        new_W1g: ℝ[n_neurons, 2] = W1g + (dt / 6.0) * (dW1_1 + 2*dW1_2 + 2*dW1_3 + dW1_4)
        new_B1g: ℝ[n_neurons, 1] = B1g + (dt / 6.0) * (dB1_1 + 2*dB1_2 + 2*dB1_3 + dB1_4)
        new_W2g: ℝ[n_neurons, 2] = W2g + (dt / 6.0) * (dW2_1 + 2*dW2_2 + 2*dW2_3 + dW2_4)
        new_B2g: ℝ[n_neurons, 1] = B2g + (dt / 6.0) * (dB2_1 + 2*dB2_2 + 2*dB2_3 + dB2_4)
        rk4_results: list = [new_z, new_a, new_W1g, new_B1g, new_W2g, new_B2g]
        return rk4_results


and here is the main outer loop which we are defining as ``adjoint_solver`` : 

.. code-block:: text

    def adjoint_solver(pred_traj: ℝ[m, 2, 1], true_trajectory: ℝ[m, 2, 1], Δt: ℝ, n_steps: ℕ, model: ODEFunc): list:
        z: ℝ[2, 1] = pred_traj[n_steps - 1]
        a: ℝ[2, 1] = 2 * (pred_traj[n_steps - 1] - true_trajectory[n_steps - 1])
        W1g: ℝ[n_neurons, 2] = zeros(n_neurons, 2)
        B1g: ℝ[n_neurons, 1] = zeros(n_neurons, 1)
        W2g: ℝ[2, n_neurons] = zeros(2, n_neurons)
        B2g: ℝ[2, 1] = zeros(2, 1)
        for i:ℕ(n_steps - 1):
            # RK4 backward one step
            aug_state = rk4_step_adjoint(z, a, W1g, B1g, W2g, B2g, -Δt, model)
            z, a, W1g, B1g, W2g, B2g = aug_state
            idx = n_steps - 2 - i
            a = a + 2 * (pred_traj[idx] - true_trajectory[idx])
        a_t0, W1g, B1g, W2g, B2g = aug_state[1], aug_state[2], aug_state[3], aug_state[4], aug_state[5]
        results: list = [a_t0, W1g, B1g, W2g, B2g]
        return results

now lets map this Physika implmentation with algorithm side by side:

.. math:: 

    s_0 = \left[z(t_1),\; \frac{\partial \mathcal{L}}{\partial z(t_1)},\; 0_{|\theta|}\right]

where,


.. list-table:: 
   :header-rows: 1
   :widths: 50 50

   * - Define initial augmented state
     - Physika code
   * - :math:`z(t_1)`
     - ``z: ℝ[2, 1] = pred_traj[n_steps - 1]``
   * - :math:`\frac{\partial \mathcal{L}}{\partial z(t_1)}`
     - ``a: ℝ[2, 1] = 2 * (pred_traj[n_steps - 1] - true_trajectory[n_steps - 1])``
   * - :math:`0_{|\theta|}`
     - ``W1g, B1g, W2g, B2g``

.. math::

    \left[z(t_0), \frac{\partial L}{\partial z(t_0)}, \frac{\partial L}{\partial \theta}\right]
    = \operatorname{ODESolve}\left(s_0, \operatorname{aug\_dynamics}, t_1, t_0, \theta\right)


.. list-table:: 
   :header-rows: 1
   :widths: 50 50

   * - solver reverse time ode
     - Physika code
   * - :math:`\operatorname{ODESolve}\left(s_0, \operatorname{aug\_dynamics}, t_1, t_0, \theta\right)`
     - ``rk4_step_adjoint(z, a, W1g, B1g, W2g, B2g, -Δt, model)``
   * - :math:`[z(t_0), \frac{\partial L}{\partial z(t_0)}, \frac{\partial L}{\partial \theta}]`
     - ``aug_state``




Training the model
----------------------

We are using AdamOptimizer to train this model, In Physika we can define it by:

.. code-block:: text

    class AdamOptimizer:
        lr, beta1, beta2, eps, t: ℝ
        m, v: ℝ[m, n]
        def step(param: ℝ[m,n], grad: ℝ[m,n]) → ℝ[m,n]:
            this.t = this.t + 1.0
            this.m = this.beta1 * this.m + (1.0 - this.beta1) * grad
            this.v = this.beta2 * this.v + (1.0 - this.beta2) * grad**2
            m_hat: ℝ[m, n] = this.m / (1.0 - this.beta1**this.t)
            v_hat: ℝ[m, n] = this.v / (1.0 - this.beta2**this.t)
            param_new: ℝ[m, n] = param - this.lr * m_hat / (sqrt(v_hat) + this.eps)
            return param_new


    lr: ℝ = 0.01
    adam_W1: AdamOptimizer = AdamOptimizer(lr, 0.9, 0.999, 1e-8, 0.0, zeros(n_neurons,2), zeros(n_neurons,2))
    adam_B1: AdamOptimizer = AdamOptimizer(lr, 0.9, 0.999, 1e-8, 0.0, zeros(n_neurons,1), zeros(n_neurons,1))
    adam_W2: AdamOptimizer = AdamOptimizer(lr, 0.9, 0.999, 1e-8, 0.0, zeros(2,n_neurons), zeros(2,n_neurons))
    adam_B2: AdamOptimizer = AdamOptimizer(lr, 0.9, 0.999, 1e-8, 0.0, zeros(2,1), zeros(2,1))


and here is the training loop to train the model


.. code-block:: text

    epochs: ℕ = 200

    for i:ℕ(epochs):
        print(i)
        # -------------------
        # Forward pass 
        # -------------------
        pred_traj = odesolver(model, y0, Δt, timesteps)
        diff = pred_traj - true_trajectory
        loss = mean(diff**2)
        # -------------------
        # adjoint method
        # -------------------
        at0, dW1, dB1, dW2, dB2 = adjoint_solver(pred_traj, true_trajectory, Δt, n_steps, model)
        model.W1 = adam_W1.step(model.W1, dW1)
        model.B1 = adam_B1.step(model.B1, dB1)
        model.W2 = adam_W2.step(model.W2, dW2)
        model.B2 = adam_B2.step(model.B2, dB2)
        print(loss)

To visualize results, we can predict the trajectory with trained model as:

.. code-block:: text

    predicted_trajectory: ℝ[m, 2, 1] = odesolver(model, y0, Δt, timesteps)
    plot_phase_space(true_trajectory, predicted_trajectory)




.. note::
   ``plot_phase_space`` is not a built-in Physika function. To use it,
   add the following helper function to ``physika/runtime.py``.

   .. code-block:: python

        def plot_phase_space(true_trajectory, pred_trajectory):
            import matplotlib.pyplot as plt

            true_trajectory = true_trajectory.detach().numpy()
            pred_trajectory = pred_trajectory.detach().numpy()

            true_x = true_trajectory[:, 0, 0]
            true_y = true_trajectory[:, 1, 0]

            pred_x = pred_trajectory[:, 0, 0]
            pred_y = pred_trajectory[:, 1, 0]

            plt.figure(figsize=(6, 6))

            plt.plot(true_x, true_y, label="True", linewidth=2)
            plt.plot(
                pred_x,
                pred_y,
                "--",
                color="orange",
                label="Predicted",
                linewidth=2,
            )

            plt.xlabel("x")
            plt.ylabel("y")
            plt.title("Phase Space Trajectory")

            plt.axis("equal")
            plt.grid(True)
            plt.legend()

            plt.show()



.. figure:: /_static/tutorial_files/neural_ode/final_results.png
   :alt: neural_ode_results
   :align: center
   :width: 600px
   :name: neural_ode_results

   Predicted vs True phase space trajectory



Full code
------------


.. code-block:: text


  # ---------------------------------
  # Helper functions
  # --------------------------------


  def linspace(start: ℝ, end: ℝ, n: ℕ): ℝ[n]:
      x: ℝ[n] = zeros(n)
      dx: ℝ = (end - start) / (n - 1)
      for i:ℕ(0, n):
          x[i] = start + i * dx
      return x

  def tanh(x: ℝ): ℝ:
      return (exp(x) - exp(0.0 - x)) / (exp(x) + exp(0.0 - x))


  # ------------------------------------------------------------
  # ODE solver function with rk4
  # ------------------------------------------------------------


  def rk4_step(state: ℝ[2, 1], t: ℝ, Δt: ℝ, ode_func: ODEFunc): ℝ[2, 1]:
      k1: ℝ[2, 1] = ode_func(state)
      k2_state: ℝ[2, 1] = state + 0.5 * Δt * k1
      k2: ℝ[2, 1] = ode_func(k2_state)
      k3_state: ℝ[2, 1] = state + 0.5 * Δt * k2
      k3: ℝ[2, 1] = ode_func(k3_state)
      k4_state: ℝ[2, 1] = state + Δt * k3
      k4: ℝ[2, 1] = ode_func(k4_state)
      return state + (Δt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)



  def odesolver(ode_func: ODEFunc, y0: ℝ[2, 1], Δt: ℝ, timesteps: ℝ[n]): ℝ[m, 2, 1]:
      n_times: ℕ = len(timesteps)
      trajectory: ℝ[n_times,2,1] = zeros(n_times, 2, 1)
      state: ℝ[2, 1] = y0
      trajectory[0] = state
      for i:ℕ(1, n_times):
          current_t = timesteps[i-1]
          state = rk4_step(state, current_t, Δt, ode_func)
          trajectory[i] = state
      return trajectory


  # ------------------------------------------------------------
  # Dataset script
  # ------------------------------------------------------------


  def damped_oscillator(state: ℝ[2,1]): ℝ[2,1]:
      q, p: ℝ = state[0,0], state[1,0]
      gamma: ℝ = 0.2
      dq: ℝ = p
      dp: ℝ = 0.0 - q - gamma * p
      return [[dq], [dp]] 


  def generate_dataset(y0: ℝ[2,1], Δt: ℝ, timesteps: ℝ[n]): ℝ[m,2,1]:
      n_times: ℝ = len(timesteps)
      trajectory: ℝ[n_times, 2, 1] = zeros(n_times, 2, 1)
      state: ℝ[2,1] = y0
      trajectory[0] = state
      for i: ℕ(1, n_times):
          current_t = timesteps[i-1]
          state = rk4_step(state, current_t, Δt, damped_oscillator)
          trajectory[i] = state
      return trajectory


  t_start, t_end, Δt: ℝ = 0.0, 15.0, 0.1

  n_steps: ℕ = int((t_end - t_start) / Δt) + 1
  timesteps: ℝ[n_steps] = linspace(t_start, t_end, n_steps)
  y0: ℝ[2,1] = [[1.0], [0.0]]

  true_trajectory: ℝ[n_steps, 2, 1] = generate_dataset(y0, Δt, timesteps)


  # ------------------------------------------------------------
  # ODE Func model
  # ------------------------------------------------------------




  class ODEFunc(W1: ℝ[Z, 2], B1: ℝ[Z, 1], W2: ℝ[Z, Z], B2: ℝ[2, 1]):
      def λ(x: ℝ[2, 1]) → ℝ[2, 1]:
          h1: R[Z, 1] = tanh(this.W1 @ x + this.B1)
          out: R[2, 1] = this.W2 @ h1 + this.B2
          return out



  n_neurons: ℕ = 128

  μ, σ: ℝ = 0.0, 0.01

  W1: ℝ[n_neurons, 2] = for i: ℕ(n_neurons) -> ε: ℝ[2] ~ Normal(μ, σ, 2)
  B1: ℝ[n_neurons, 1] = for i: ℕ(n_neurons) -> [0.01]
  W2: ℝ[2, n_neurons] = for i: ℕ(2) -> ε: ℝ[n_neurons] ~ Normal(μ, σ, n_neurons)
  B2: ℝ[2, 1] = [[0.01], [0.01]]

  model: ODEFunc = ODEFunc(W1, B1, W2, B2)


  # -----------------------------------------------------------
  # Neural ODE adjoint method
  # ----------------------------------------------------------


  def compute_vjps(z: ℝ[2,1], a: ℝ[2,1], model: ODEFunc): list:
      z: ℝ[2, 1] = detach_grad(z)
      a: ℝ[2, 1] = detach(a)
      f_val: ℝ[2, 1] = model(z)
      scalar: ℝ = sum(a * f_val)
      parameter_list: list = [z, model.W1, model.B1, model.W2, model.B2]
      result: list = grad(scalar, parameter_list)
      dadt: ℝ[2, 1] = -result[0]
      dW1: ℝ[n_neurons, 2] = -result[1]
      dB1: ℝ[n_neurons, 1] = -result[2]
      dW2: ℝ[2, n_neurons] = -result[3]
      dB2: ℝ[2, 1] = -result[4]
      f_val: ℝ[2, 1] = detach(f_val)
      vjps_results: list = [f_val, dadt, dW1, dB1, dW2, dB2]
      return vjps_results


  def rk4_step_adjoint(z: ℝ[2, 1], a: ℝ[2, 1], W1g: ℝ[n_neurons, 2], B1g: ℝ[n_neurons, 1], W2g: ℝ[2, n_neurons], B2g: ℝ[2, 1], dt: ℝ, model: ODEFunc): list:
      # k1
      dz1, da1, dW1_1, dB1_1, dW2_1, dB2_1 = compute_vjps(z, a, model)
      # k2
      z2 = z + 0.5 * dt * dz1
      a2 = a + 0.5 * dt * da1
      W1g2 = W1g + 0.5 * dt * dW1_1
      B1g2 = B1g + 0.5 * dt * dB1_1
      W2g2 = W2g + 0.5 * dt * dW2_1
      B2g2 = B2g + 0.5 * dt * dB2_1
      dz2, da2, dW1_2, dB1_2, dW2_2, dB2_2 = compute_vjps(z2, a2, model)
      # k3
      z3 = z + 0.5 * dt * dz2
      a3 = a + 0.5 * dt * da2
      W1g3 = W1g + 0.5 * dt * dW1_2
      B1g3 = B1g + 0.5 * dt * dB1_2
      W2g3 = W2g + 0.5 * dt * dW2_2
      B2g3 = B2g + 0.5 * dt * dB2_2
      dz3, da3, dW1_3, dB1_3, dW2_3, dB2_3 = compute_vjps(z3, a3, model)
      # k4
      z4 = z + dt * dz3
      a4 = a + dt * da3
      W1g4 = W1g + dt * dW1_3
      B1g4 = B1g + dt * dB1_3
      W2g4 = W2g + dt * dW2_3
      B2g4 = B2g + dt * dB2_3
      dz4, da4, dW1_4, dB1_4, dW2_4, dB2_4 = compute_vjps(z4, a4, model)
      # update
      new_z = z + (dt / 6.0) * (dz1 + 2*dz2 + 2*dz3 + dz4)
      new_a = a + (dt / 6.0) * (da1 + 2*da2 + 2*da3 + da4)
      new_W1g = W1g + (dt / 6.0) * (dW1_1 + 2*dW1_2 + 2*dW1_3 + dW1_4)
      new_B1g = B1g + (dt / 6.0) * (dB1_1 + 2*dB1_2 + 2*dB1_3 + dB1_4)
      new_W2g = W2g + (dt / 6.0) * (dW2_1 + 2*dW2_2 + 2*dW2_3 + dW2_4)
      new_B2g = B2g + (dt / 6.0) * (dB2_1 + 2*dB2_2 + 2*dB2_3 + dB2_4)
      rk4_results: list = [new_z, new_a, new_W1g, new_B1g, new_W2g, new_B2g]
      return rk4_results


  def adjoint_solver(pred_traj: ℝ[m, 2, 1], true_trajectory: ℝ[m, 2, 1], Δt: ℝ, n_steps: ℕ, model: ODEFunc): list:
      z: ℝ[2, 1] = pred_traj[n_steps - 1]
      a: ℝ[2, 1] = 2 * (pred_traj[n_steps - 1] - true_trajectory[n_steps - 1])
      W1g: ℝ[n_neurons, 2] = zeros(n_neurons, 2)
      B1g: ℝ[n_neurons, 1] = zeros(n_neurons, 1)
      W2g: ℝ[2, n_neurons] = zeros(2, n_neurons)
      B2g: ℝ[2, 1] = zeros(2, 1)
      for i:N(n_steps - 1):
          # RK4 backward one step
          aug_state = rk4_step_adjoint(z, a, W1g, B1g, W2g, B2g, -Δt, model)
          z, a, W1g, B1g, W2g, B2g = aug_state
          idx = n_steps - 2 - i
          a = a + 2 * (pred_traj[idx] - true_trajectory[idx])
      a_t0, W1g, B1g, W2g, B2g = aug_state[1], aug_state[2], aug_state[3], aug_state[4], aug_state[5]
      results: list = [a_t0, W1g, B1g, W2g, B2g]
      return results


  # -----------------------------------------------
  # AdamOptimizer
  # -----------------------------------------------


  class AdamOptimizer:
      lr, beta1, beta2, eps, t: ℝ
      m, v: R[m, n]
      def step(param: ℝ[m,n], grad: ℝ[m,n]) → ℝ[m,n]:
          this.t = this.t + 1.0
          this.m = this.beta1 * this.m + (1.0 - this.beta1) * grad
          this.v = this.beta2 * this.v + (1.0 - this.beta2) * grad**2
          m_hat: ℝ[m, n] = this.m / (1.0 - this.beta1**this.t)
          v_hat: ℝ[m, n] = this.v / (1.0 - this.beta2**this.t)
          param_new: ℝ[m, n] = param - this.lr * m_hat / (sqrt(v_hat) + this.eps)
          return param_new


  lr: ℝ = 0.01
  adam_W1: AdamOptimizer = AdamOptimizer(lr, 0.9, 0.999, 1e-8, 0.0, zeros(n_neurons,2), zeros(n_neurons,2))
  adam_B1: AdamOptimizer = AdamOptimizer(lr, 0.9, 0.999, 1e-8, 0.0, zeros(n_neurons,1), zeros(n_neurons,1))
  adam_W2: AdamOptimizer = AdamOptimizer(lr, 0.9, 0.999, 1e-8, 0.0, zeros(2,n_neurons), zeros(2,n_neurons))
  adam_B2: AdamOptimizer = AdamOptimizer(lr, 0.9, 0.999, 1e-8, 0.0, zeros(2,1), zeros(2,1))


  # -----------------------------------------------
  # Training loop
  # -----------------------------------------------


  epochs: ℕ = 200

  for i:ℕ(epochs):
      print(i)
      # -------------------
      # Forward pass 
      # -------------------
      pred_traj = odesolver(model, y0, Δt, timesteps)
      diff = pred_traj - true_trajectory
      loss = mean(diff**2)
      # -------------------
      # adjoint method
      # -------------------
      a_t0, dW1, dB1, dW2, dB2 = adjoint_solver(pred_traj, true_trajectory, Δt, n_steps, model)
      model.W1 = adam_W1.step(model.W1, dW1)
      model.B1 = adam_B1.step(model.B1, dB1)
      model.W2 = adam_W2.step(model.W2, dW2)
      model.B2 = adam_B2.step(model.B2, dB2)
      print(loss)



  predicted_trajectory: ℝ[m, 2, 1] = odesolver(model, y0, Δt, timesteps)
  plot_phase_space(true_trajectory, predicted_trajectory)





References
------------

.. [Chen_NeuralODE] Ricky T. Q. Chen, Yulia Rubanova, Jesse Bettencourt, and David Duvenaud,
   *\*Neural Ordinary Differential Equations\**, NeurIPS 2018.

   [https://arxiv.org/pdf/1806.07366](https://arxiv.org/pdf/1806.07366)


.. [JKU_Hamiltonian] Johannes Kepler University Linz,
   *Hamiltonian Mechanics*, Chapter 3.2, p. 11.

   https://www3.risc.jku.at/publications/download/risc_3966/Hamiltonian_2010_02_26.pdf#page=11
