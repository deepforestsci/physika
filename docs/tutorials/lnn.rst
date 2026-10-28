Lagrangian Neural networks
==========================

In this tutorial we will implement Lagrangian Neural network in Physika and also integrate
the model to predict complete trajectory [LagrangianNNs]_ .
Before implementing the model, lets understand the basics of lagrangian mechanics


Lagrangian mechanics
--------------------

The euler lagrangian equation is defined as: [EulerLagrangeWiki]_


.. math::

    \frac{d}{dt}\left(\frac{\partial \mathcal{L}}{\partial \dot{q}}\right)
    =
    \frac{\partial \mathcal{L}}{\partial q}
    \label{euler_lagrangian_equation}


    

Here :math:`L` is the scalar value obtained from given Lagrangian function, :math:`q` represents
position term and :math:`\dot{q}` represents velocity term, both :math:`q` and :math:`\dot{q}` are
passed as input state to :math:`L` function.

We can apply chain rule to expand the equation using time derivative as:

.. math::

    \frac{\partial^2 \mathcal{L}}{\partial \dot{q}\,\partial q}\dot{q}
    +
    \frac{\partial^2 \mathcal{L}}{\partial \dot{q}\,\partial \dot{q}}\ddot{q}
    =
    \frac{\partial \mathcal{L}}{\partial q}


rearranging it gives us:

.. math::

    \frac{\partial^2 \mathcal{L}}{\partial \dot{q}\,\partial \dot{q}}\ddot{q}
    =
    \frac{\partial \mathcal{L}}{\partial q}
    -
    \frac{\partial^2 \mathcal{L}}{\partial \dot{q}\,\partial q}\dot{q}
    \label{lagrangian_final_form}

now this equation is in form of :math:`Ax = b` so we can use linear solve to find the acceleartion term
:math:`\ddot{q}`. In general there is also other way, where we can rearrange this equation and use matrix inverse [LagrangianNNsCode]_ .
Since Physika already have gaussian solve we are choosing the first method to find acceleartion term :math:`\ddot{q}`.


Dataset
-------

We are using simple mass-spring system to generate dataset 

.. math::

    \mathcal{L}(q,\dot{q}) = \frac{1}{2}m\dot{q}^2 - \frac{1}{2}kq^2

where :math:`m` is the mass and :math:`k` is the spring constant.
For our example, we set :math:`m=2` and :math:`k=4`.

we can calculate acceleartion as: [ClassicalMechanicsNotes]_


.. math::

    m\ddot{q} = -kq

add this below python function in ``physika/runtime.py`` file:

.. code-block:: python

    def create_dataset(n_samples):
        m = 2.0
        k = 4.0

        dataset = []

        for _ in range(n_samples):
            state = torch.distributions.Normal(0.0, 0.2).sample((2,))

            q = state[0].item()
            qdot = state[1].item()
            true_qddot = -(k / m) * q

            x = torch.tensor([[q], [qdot]], dtype=torch.float32)
            # Each sample is a list: [input_tensor, scalar_target]
            sample = [x, true_qddot]
            dataset.append(sample)

        indices = torch.randperm(n_samples).tolist()
        train_size = int(0.8 * n_samples)

        train_data = [dataset[i] for i in indices[:train_size]]
        test_data = [dataset[i] for i in indices[train_size:]]
        
        return train_data, test_data


LNN class
---------

Equation :math:`\eqref{lagrangian_final_form}` is the final form which we got from euler lagrangian equation :math:`\eqref{euler_lagrangian_equation}`,
to make it simple we can use a Hessian term :math:`H` as:

.. math::

    H_{\dot{q}\dot{q}}\ddot{q}
    =
    \nabla_q \mathcal{L}
    -
    H_{q\dot{q}}^{T}\dot{q}


where,

.. math::

    H_{\dot{q}\dot{q}}
    =
    \frac{\partial^2 \mathcal{L}}
    {\partial \dot{q}\,\partial \dot{q}}

.. math::

    H_{q\dot{q}}
    =
    \frac{\partial^2 \mathcal{L}}
    {\partial q\,\partial \dot{q}}

To build a LNN class, we can model the Lagrangian :math:`L` by a neural network:

.. math::

    L = f_\theta

We can define it in physika as:



.. code-block:: text
    
    class LNN(W1: ℝ[H, I], b1: ℝ[H, 1], W2: ℝ[1, H], b2: ℝ):
        def λ(x: ℝ[I, 1]) → ℝ[d]:
            d: ℕ = I / 2
            L: ℝ[1, 1] = W2 @ softplus(W1 @ x + b1) + b2
            dL: ℝ[I, 1] = grad(L, x)
            hessian: ℝ[I, I] = zeros(I, I)
            for i:ℕ(I):
                row: ℝ[1, I] = transpose(grad(dL[i], x), 0, 1)
                hessian[i, :] = row
            H_dq_dq: ℝ[d, d] = hessian[d:I, d:I]
            H_q_dq: ℝ[d, d] = hessian[0:d, d:I]
            dL_dq = squeeze(dL[0:d], 1)
            qdot = squeeze(x[d:I], 1)
            b: ℝ[d] = dL_dq - transpose(H_q_dq, 0, 1) @ qdot
            q_ddot: ℝ[d] = gaussian_solve(H_dq_dq, b)
            return q_ddot

Here, :math:`L` the scalar value predicted by neural network, :math:`dL` calculates gradients w.r.t input state :math:`x`
after that we build hessian matrix where, 
``H_dq_dq`` is represents :math:`H_{\dot{q}\dot{q}}` term and ``H_q_dq`` represents :math:`H_{q\dot{q}}`.
Finally we calculate :math:`q_ddot` acceleartion value using gaussian solve function.



We can write a training loop inside ``LNN`` class as:

.. code-block:: text

    def train(epochs: N, lr: ℝ): ℝ:
        train_dataset_len = len(train_dataset)
        epoch_loss: ℝ = 0
        for idx:ℕ(epochs):
            print(idx)
            epoch_loss: R = 0
            for j:ℕ(train_dataset_len):    
                sample = train_dataset[j]
                x = sample[0]
                true_qddot = sample[1]
                pred = this(x)
                loss = mse_loss(pred, true_qddot)
                epoch_loss += loss
                learnable_grads = grad(loss, this.learnable_params)
                this.update_params(lr, learnable_grads)
            epoch_loss = epoch_loss / train_dataset_len
            print(epoch_loss)
        return epoch_loss


This passes each state :math:`x` (:math:`q` and :math:`\dot{q}`) to model's forward method. The loss is calculated
using MSE (Mean Square error) loss function defined as:

.. code-block:: text

    def mse_loss(pred: ℝ, true: ℝ): R:
        return (pred - true)**2


Integrate the model
-------------------

Once the model gets trained, we can integrate the model using an euler solver to get predicted trajectory

We are using RK-4 euler method defined as:

.. code-block:: text

    def rk4_step(state: ℝ[2, 1], Δt: ℝ,lnn: LNN): ℝ[2, 1]:
        k1: ℝ[2, 1] = ode_derivative(state, lnn)
        k2_state: ℝ[2, 1] = state + 0.5 * Δt * k1
        k2: ℝ[2, 1] = ode_derivative(k2_state, lnn)
        k3_state: ℝ[2, 1] = state + 0.5 * Δt * k2
        k3: ℝ[2, 1] = ode_derivative(k3_state, lnn)
        k4_state: ℝ[2, 1] = state + Δt * k3
        k4: ℝ[2, 1] = ode_derivative(k4_state, lnn)
        return state + (Δt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


where, ``ode_derivate`` is a helper function which returns which wraps :math:`\dot{q}` and :math:`\ddot{q}` in a state:

.. code-block:: text

    def ode_derivative(state: ℝ[2, 1], lnn: LNN): ℝ[2, 1]:
        derivative: ℝ[2, 1] = zeros(2, 1)
        derivative[0, 0] = state[1, 0]
        derivative[1, 0] = lnn(state)[0]
        return derivative


We can define a ode solver function which takes LNN model, initial state, :math:`\Delta t`  and total number of steps and 
it returns the predicted trajectory

.. code-block:: text

    def ode_solver(lnn: LNN, initial_state: ℝ[2, 1],Δt: ℝ,
                num_steps: ℕ): ℝ[num_steps + 1, 2, 1]:
        trajectory: ℝ[num_steps + 1, 2, 1] = zeros(num_steps + 1, 2, 1)
        state: ℝ[2, 1] = initial_state
        trajectory[0, :, :] = state
        for i: ℕ(num_steps):
            state = rk4_step(state, Δt, lnn)
            trajectory[i + 1, :, :] = state
        return trajectory

Here is how we can use test dataset to generated predicted trajectory:

.. code-block:: text

    test_sample = test_dataset[0]
    initial_state: ℝ[2, 1] = test_sample[0]
    Δt: ℝ = 0.01
    num_steps: ℕ = 1000

    trajectory = ode_solver(model, initial_state, Δt, num_steps)


To visualize trajectory, we can add below python function in ``physika/runtime.py`` file:

.. code-block:: python

    def visualize_trajectory(trajectory, dt):
        import math
        import matplotlib.pyplot as plt
        trajectory = trajectory.detach().cpu().numpy()

        q = trajectory[:, 0, 0]
        qdot = trajectory[:, 1, 0]
        t = [i * dt for i in range(len(q))]

        q0 = q[0]
        qdot0 = qdot[0]
        omega = math.sqrt(2)

        true_q = [
            q0 * math.cos(omega * ti)
            + (qdot0 / omega) * math.sin(omega * ti)
            for ti in t
        ]
        true_qdot = [
            -q0 * omega * math.sin(omega * ti)
            + qdot0 * math.cos(omega * ti)
            for ti in t
        ]

        plt.plot(true_q, true_qdot, color="lightblue", label="True trajectory")
        plt.plot(q, qdot, color="orange", linestyle="dashed",
                label="Predicted trajectory")
        plt.xlabel("Position")
        plt.ylabel("Velocity")
        plt.legend()
        plt.axis("equal")
        plt.show()


.. figure:: /_static/tutorial_files/lnn_results.png
   :alt: Learned trajectory vs ground truth
   :align: center
   :width: 600px

   Figure 1: Comparison between predicted trajectory and true trajectory.



Full code
---------

.. code-block:: text

    from tutorials.linear_solve_gaussian_elimination import gaussian_solve, get_2d_array_num_cols, get_2d_array_num_rows, get_1d_array_length, min, max


    def rand_2d_array(n:N, m:N, μ:ℝ): ℝ[n, m]:
        return for i:ℕ(n) → for j:ℕ(m) → μ * eps ~ 𝒩(0.5, 1.0)

    def softplus(x: ℝ[n,1]): ℝ[n,1]:
        return log(1 + exp(x))

    def mse_loss(pred: ℝ, true: ℝ): R:
        return (pred - true)**2

    dataset = create_dataset(50)
    train_dataset = dataset[0]
    test_dataset = dataset[1]

    class LNN(W1: ℝ[H, I], b1: ℝ[H, 1], W2: ℝ[1, H], b2: ℝ):
        def λ(x: ℝ[I, 1]) → ℝ[d]:
            d: ℕ = I / 2
            L: ℝ[1, 1] = W2 @ softplus(W1 @ x + b1) + b2
            dL: ℝ[I, 1] = grad(L, x)
            hessian: ℝ[I, I] = zeros(I, I)
            for i:ℕ(I):
                row: ℝ[1, I] = transpose(grad(dL[i], x), 0, 1)
                hessian[i, :] = row
            H_dq_dq: ℝ[d, d] = hessian[d:I, d:I]
            H_q_dq: ℝ[d, d] = hessian[0:d, d:I]
            dL_dq = squeeze(dL[0:d], 1)
            qdot = squeeze(x[d:I], 1)
            b: ℝ[d] = dL_dq - transpose(H_q_dq, 0, 1) @ qdot
            q_ddot: ℝ[d] = gaussian_solve(H_dq_dq, b)
            return q_ddot
        def train(epochs: N, lr: ℝ): ℝ:
            train_dataset_len = len(train_dataset)
            epoch_loss: ℝ = 0
            for idx:ℕ(epochs):
                print(idx)
                epoch_loss: R = 0
                for j:ℕ(train_dataset_len):    
                    sample = train_dataset[j]
                    x = sample[0]
                    true_qddot = sample[1]
                    pred = this(x)
                    loss = mse_loss(pred, true_qddot)
                    epoch_loss += loss
                    learnable_grads = grad(loss, this.learnable_params)
                    this.update_params(lr, learnable_grads)
                epoch_loss = epoch_loss / train_dataset_len
                print(epoch_loss)
            return epoch_loss
        def update_params(lr: ℝ, learnable_grads: ℝ[m]):
            this.W1 = this.W1 - lr * learnable_grads[0]
            this.b1 = this.b1 - lr * learnable_grads[1]
            this.W2 = this.W2 - lr * learnable_grads[2]
            this.b2 = this.b2 - lr * learnable_grads[3]  


    H, I: ℕ = 128, 2
    W1: ℝ[H, I] = rand_2d_array(H, I, 1.0)
    b1: ℝ[H, 1] = rand_2d_array(H, 1, 0.5)
    W2: ℝ[1, H] = rand_2d_array(1, H, 0.1)
    b2: ℝ = 0.0

    model: LNN = LNN(W1, b1, W2, b2)

    epochs: ℕ = 1000
    train_loss: ℝ = model.train(epochs, 0.001)
    print(train_loss)


    def ode_derivative(state: ℝ[2, 1], lnn: LNN): ℝ[2, 1]:
        derivative: ℝ[2, 1] = zeros(2, 1)
        derivative[0, 0] = state[1, 0]
        derivative[1, 0] = lnn(state)[0]
        return derivative


    def rk4_step(state: ℝ[2, 1], Δt: ℝ,lnn: LNN): ℝ[2, 1]:
        k1: ℝ[2, 1] = ode_derivative(state, lnn)
        k2_state: ℝ[2, 1] = state + 0.5 * Δt * k1
        k2: ℝ[2, 1] = ode_derivative(k2_state, lnn)
        k3_state: ℝ[2, 1] = state + 0.5 * Δt * k2
        k3: ℝ[2, 1] = ode_derivative(k3_state, lnn)
        k4_state: ℝ[2, 1] = state + Δt * k3
        k4: ℝ[2, 1] = ode_derivative(k4_state, lnn)
        return state + (Δt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


    def ode_solver(lnn: LNN, initial_state: ℝ[2, 1],Δt: ℝ,
                    num_steps: ℕ): ℝ[num_steps + 1, 2, 1]:
        trajectory: ℝ[num_steps + 1, 2, 1] = zeros(num_steps + 1, 2, 1)
        state: ℝ[2, 1] = initial_state
        trajectory[0, :, :] = state
        for i: ℕ(num_steps):
            state = rk4_step(state, Δt, lnn)
            trajectory[i + 1, :, :] = state
        return trajectory

    test_sample = test_dataset[0]
    initial_state: ℝ[2, 1] = test_sample[0]
    Δt: ℝ = 0.01
    num_steps: ℕ = 1000

    trajectory = ode_solver(model, initial_state, Δt, num_steps)

    visualize_trajectory(trajectory, Δt)






References
----------

.. [EulerLagrangeWiki] *Euler–Lagrange equation*. Wikipedia.
   https://en.wikipedia.org/wiki/Euler%E2%80%93Lagrange_equation

.. [LagrangianNNs] Cranmer, M., et al. *Lagrangian Neural Networks*.
   arXiv:2003.04630, 2020.
   https://arxiv.org/pdf/2003.04630

.. [LagrangianNNsCode] Cranmer, M. *Lagrangian Neural Networks*.
   GitHub repository.
   https://github.com/MilesCranmer/lagrangian_nns

.. [ClassicalMechanicsNotes] *Chapter 6: Lagrangian Mechanics*.
   IPCMS.
   https://www.ipcms.fr/uploads/2021/06/chap6.pdf