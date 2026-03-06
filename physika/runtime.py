import re
import copy
import inspect
from typing import Any, Sequence, Union

import torch
import torch.nn as nn
import torch.optim as optim

from utils.print_utils import _from_torch, _infer_type


def physika_print(value: Any) -> None:
    """Pretty-print a Physika value with its inferred type annotation.

    Converts PyTorch tensors, complex numbers, and Python scalars into a
    readable display form, infers the Physika type (e.g. ``ℝ``,
    ``ℝ[3]``, ``ℂ``), and prints ``<value> ∈ <type>``.

    Parameters
    ----------
    value : Any
        The value to print.  Supported types include ``torch.Tensor``,
        ``int``, ``float``, ``complex``, ``list`` (nested), and
        ``nn.Module`` subclasses.

    Examples
    --------
    >>> physika_print(3.0)
    3.0 ∈ ℝ
    >>> physika_print(torch.tensor([1.0, 2.0, 3.0]))
    [1.0, 2.0, 3.0] ∈ ℝ[3]
    >>> physika_print(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    [[1.0, 2.0], [3.0, 4.0]] ∈ ℝ[2,2]
    """
    display = _from_torch(value)
    type_str = _infer_type(value)
    print(f"{display} ∈ {type_str}")


def solve(*equations: str, **known_vars: float) -> tuple[torch.Tensor, ...]:
    """Solve a system of linear equations for unknown variables.

    Parses string equations of the form ``"var = expr"`` where ``expr``
    is a linear combination of unknowns and known variables.  Builds the
    coefficient matrix **A** and constant vector **b**, then solves
    ``Ax = b`` using ``torch.linalg.solve``.

    Unknowns are automatically detected: any variable on the right-hand
    side that is not in ``known_vars`` and is not a built-in (``exp``,
    ``sin``, ``cos``, ``sqrt``, ``i``) is treated as an unknown.

    Parameters
    ----------
    *equations : str
        One or more equation strings, each containing exactly one ``=``.
        Example: ``"F1 = m * a1 + m * a2"``.
    **known_vars : float
        Named values for known variables.  Example: ``m=2.0, F1=10.0``.

    Returns
    -------
    tuple[torch.Tensor, ...]
        A tuple of solved values, one per unknown, in sorted alphabetical
        order of the unknown variable names.

    Examples
    --------
    >>> from runtime import solve
    >>> x, = solve("y = 2 * x", y=6.0)
    >>> float(x)
    3.0
    """
    parsed = []
    for eq in equations:
        lhs, rhs = eq.split('=')
        parsed.append((lhs.strip(), rhs.strip()))

    all_rhs_vars = set()
    for lhs, rhs in parsed:
        tokens = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', rhs)
        all_rhs_vars.update(tokens)

    special = {'i', 'exp', 'sin', 'cos', 'sqrt'}
    unknowns = sorted([v for v in all_rhs_vars if v not in special and v not in known_vars])

    n = len(unknowns)
    use_complex = any('i' in rhs for _, rhs in parsed)

    dtype = torch.complex64 if use_complex else torch.float32
    A = torch.zeros((n, n), dtype=dtype)
    b = torch.zeros(n, dtype=dtype)

    for i, (lhs, rhs) in enumerate(parsed):
        b[i] = known_vars[lhs]

        for j, u in enumerate(unknowns):
            coeff = 0
            pattern = rf'([+-]?\s*(?:[\d.]*\s*\*\s*)?(?:i\s*\*\s*)?(?:[a-zA-Z_][a-zA-Z0-9_]*\s*\*\s*)*)\b{u}\b'
            matches = re.finditer(pattern, rhs)
            for m in matches:
                coeff_str = m.group(1).strip()
                if not coeff_str or coeff_str == '+':
                    coeff += 1
                elif coeff_str == '-':
                    coeff += -1
                else:
                    coeff_str = coeff_str.rstrip('* ')
                    coeff_str = coeff_str.replace('i', '1j')
                    for var, val in known_vars.items():
                        coeff_str = re.sub(rf'\b{var}\b', str(complex(val) if use_complex else float(val)), coeff_str)
                    try:
                        coeff += eval(coeff_str)
                    except:
                        coeff += 1
            A[i, j] = coeff

    solution = torch.linalg.solve(A, b)
    return tuple(solution[i] for i in range(n))


def train(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    epochs: Union[int , float],
    lr: float,
) -> nn.Module:
    """Train a Physika model using SGD on per-sample loss.

    Creates a deep copy of ``model`` so the original is not mutated,
    enables gradients on all parameters, then runs ``epochs`` of SGD.
    For each epoch every sample ``(X[i], y[i])`` is passed through the
    model; if the model defines a ``loss(pred, target[, input])`` method
    it is used, otherwise MSE ``(pred - y_i)**2`` is the default.

    Training progress is printed every ``epochs // 10`` epochs and on
    the final epoch.

    Parameters
    ----------
    model : nn.Module
        The Physika ``nn.Module`` to train (will be deep-copied).
    X : torch.Tensor
        Input data of shape ``(n_samples, ...)``.
    y : torch.Tensor
        Target data of shape ``(n_samples, ...)``.
    epochs : int or float
        Number of training epochs (cast to ``int`` internally).
    lr : float
        Learning rate for ``optim.SGD``.

    Returns
    -------
    nn.Module
        A new, trained copy of the model.

    Example
    --------
    >>> from runtime import train
    >>> trained = train(model, X, y, 100, 0.01)
    """
    trained_model = copy.deepcopy(model)

    for param in trained_model.parameters():
        param.requires_grad_(True)

    optimizer = optim.SGD(trained_model.parameters(), lr=lr)

    loss_takes_input = False
    if hasattr(trained_model, 'loss'):
        sig = inspect.signature(trained_model.loss)
        loss_takes_input = len(sig.parameters) == 3

    epochs = int(epochs)
    for epoch in range(epochs):
        optimizer.zero_grad()
        total_loss = torch.tensor(0.0, requires_grad=True)

        for i in range(X.shape[0]):
            x_i = X[i].clone().requires_grad_(True)
            y_i = y[i]
            pred = trained_model(x_i)

            if hasattr(trained_model, 'loss'):
                if loss_takes_input:
                    loss_i = trained_model.loss(pred, y_i, x_i)
                else:
                    loss_i = trained_model.loss(pred, y_i)
            else:
                loss_i = (pred - y_i) ** 2

            total_loss = total_loss + loss_i

        total_loss.backward()
        optimizer.step()

        if epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1:
            print(f"  Epoch {epoch}: Loss = {total_loss.item():.6f}")

    return trained_model


def evaluate(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> float:
    """Evaluate a trained model and return the mean per-sample loss.

    Iterates over every sample ``(X[i], y[i])``, computes the loss
    using the model's ``loss()`` method (if defined) or MSE, and
    returns the average.

    Parameters
    ----------
    model : nn.Module
        The Physika ``nn.Module`` to evaluate.
    X : torch.Tensor
        Input data of shape ``(n_samples, ...)``.
    y : torch.Tensor
        Target data of shape ``(n_samples, ...)``.

    Returns
    -------
    float
        The mean loss across all samples.

    Examples
    --------
    >>> from runtime import evaluate
    >>> avg_loss = evaluate(trained_model, X_test, y_test)
    """
    loss_takes_input = False
    if hasattr(model, 'loss'):
        sig = inspect.signature(model.loss)
        loss_takes_input = len(sig.parameters) == 3

    total_loss = 0.0
    n_samples = X.shape[0]

    for i in range(n_samples):
        x_i = X[i].clone().requires_grad_(True)
        y_i = y[i]
        pred = model(x_i)

        if hasattr(model, 'loss'):
            if loss_takes_input:
                loss_i = model.loss(pred, y_i, x_i)
            else:
                loss_i = model.loss(pred, y_i)
        else:
            loss_i = (pred - y_i) ** 2

        if isinstance(loss_i, torch.Tensor):
            loss_i = loss_i.item()
        total_loss += loss_i

    return total_loss / n_samples


def compute_grad(
    output: Union[torch.Tensor, float],
    input: Union[torch.Tensor, float]
) -> torch.Tensor:
    """Compute the gradient of ``output`` with respect to ``input``.

    Wraps ``torch.autograd.grad`` with automatic tensor conversion and
    ``requires_grad`` handling.  Both ``create_graph`` and
    ``retain_graph`` are set to ``True`` so that higher-order
    derivatives remain available.

    Parameters
    ----------
    output : torch.Tensor or float
        The scalar output whose gradient is computed.
    input : torch.Tensor or float
        The variable to differentiate with respect to.  If a plain
        ``float`` is given it is wrapped in a ``torch.tensor`` with
        ``requires_grad=True``.

    Returns
    -------
    torch.Tensor
        The gradient ``∂output/∂input``.

    Examples
    --------
    >>> from runtime import compute_grad
    >>> x = torch.tensor(2.0, requires_grad=True)
    >>> y = x ** 2
    >>> compute_grad(y, x)
    tensor(4., grad_fn=<MulBackward0>)
    """
    if not isinstance(input, torch.Tensor):
        input = torch.tensor(float(input), requires_grad=True)
    if not input.requires_grad:
        input = input.clone().requires_grad_(True)
    if not isinstance(output, torch.Tensor):
        output = torch.tensor(output, dtype=torch.float32)
    grads = torch.autograd.grad(output, input, create_graph=True, retain_graph=True)
    return grads[0]


def simulate(
    model: nn.Module,
    x0: Union[Sequence[float], torch.Tensor],
    nsteps: Union[int, float],
    dt: Union[float, torch.Tensor],
) -> None:
    """Simulate a dynamical system and visualise the trajectory.

    Rolls out ``model`` for ``nsteps`` discrete time-steps starting from
    initial state ``x0`` with step size ``dt``:

    .. code-block:: text

        x_{k+1} = model(x_k)

    The resulting trajectory is plotted with **matplotlib** (time
    evolution and, for multi-dimensional states, a phase-space plot).

    Parameters
    ----------
    model : nn.Module
        A Physika ``nn.Module`` whose ``forward`` maps a state vector to
        the next state vector.
    x0 : Sequence[float] or torch.Tensor
        The initial state, e.g. ``[theta_0, omega_0]``.
    nsteps : int or float
        Number of simulation steps (cast to ``int`` internally).
    dt : float or torch.Tensor
        Time-step size.

    Examples
    --------
    >>> from runtime import simulate
    >>> simulate(pendulum_model, [0.5, 0.0], 1000, 0.01)
    """
    import matplotlib.pyplot as plt
    x = torch.as_tensor(x0).float()
    nsteps = int(nsteps)
    dt_val = float(dt) if isinstance(dt, torch.Tensor) else float(dt)
    trajectory = [x.detach().clone()]
    with torch.no_grad():
        for i in range(nsteps):
            x = model(x)
            trajectory.append(x.detach().clone())
    states = torch.stack(trajectory)
    t = torch.arange(states.shape[0]).float() * dt_val
    if states.dim() == 1 or states.shape[-1] == 1:
        plt.figure(figsize=(10, 6))
        plt.plot(t.numpy(), states.squeeze().numpy())
        plt.ylabel("x")
        plt.xlabel("Time (s)")
        plt.title("Physika")
        plt.grid(True)
        plt.show()
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        labels = [f"x[{j}]" for j in range(states.shape[1])]
        for j in range(states.shape[1]):
            ax1.plot(t.numpy(), states[:, j].numpy(), label=labels[j])
        ax1.legend()
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("State")
        ax1.set_title("Time Evolution")
        ax1.grid(True)
        ax2.plot(states[:, 1].numpy(), states[:, 0].numpy(), linewidth=0.8)
        ax2.plot(states[0, 1].item(), states[0, 0].item(), 'ro', markersize=6, label='Start')
        ax2.set_xlabel("x[1]")
        ax2.set_ylabel("x[0]")
        ax2.set_title("Phase Space")
        ax2.legend()
        ax2.grid(True)
        ax2.set_aspect('equal', adjustable='datalim')
        fig.suptitle("Physika", fontsize=14)
        plt.tight_layout()
        plt.show()

    state_dim = states.shape[-1] if states.dim() > 1 else 1
    if state_dim in (2, 4):
        try:
            import pyvista as pv
            import numpy as np
            import time as time_module

            total_frames = states.shape[0]
            step_size = max(1, total_frames // 2000)
            idx = list(range(0, total_frames, step_size))
            sub_states = states[idx].numpy()
            sub_t = t[idx].numpy()

            if state_dim == 2:
                L = 1.0
                theta_vals = sub_states[:, 0]
                xs = L * np.sin(theta_vals)
                ys = -L * np.cos(theta_vals)
                title_str = "Physika \n\nSimple Pendulum Animation"
                rod_color = "black"
                bob_radius = 0.06
            else:
                r_vals = sub_states[:, 0]
                theta_vals = sub_states[:, 1]
                xs = r_vals * np.sin(theta_vals)
                ys = -r_vals * np.cos(theta_vals)
                title_str = "Physika \n\nSpring Pendulum Animation"
                rod_color = "black"
                bob_radius = 0.07

            pad = 0.3
            x_range = float(np.max(np.abs(xs))) + pad
            y_min = float(np.min(ys)) - pad
            y_max = max(float(np.max(ys)) + pad, pad)
            scene_range = max(x_range, abs(y_min), abs(y_max))

            plotter = pv.Plotter()
            plotter.add_title(title_str, font_size=20, font="times", shadow=True)

            axis_len = scene_range * 1.2
            for axis_pt in [((axis_len, 0, 0), "X"), ((0, axis_len, 0), "Y"), ((0, 0, axis_len), "Z")]:
                end, label = axis_pt
                neg = tuple(-c for c in end)
                line = pv.Line(neg, end, resolution=60)
                plotter.add_mesh(line, color="gray", style="wireframe", line_width=1, opacity=0.5)
                plotter.add_point_labels([end], [label], font_size=14, text_color="gray", shadow=False, shape=None)

            pivot = pv.Sphere(radius=0.025, center=(0, 0, 0))
            plotter.add_mesh(pivot, color="red")

            bob = pv.Sphere(radius=bob_radius, center=(xs[0], ys[0], 0))
            plotter.add_mesh(bob, color="blue")

            rod = pv.Line((0, 0, 0), (xs[0], ys[0], 0))
            rod_actor = plotter.add_mesh(rod, color=rod_color, line_width=3)

            trail_len = 80
            trail_pts = np.zeros((trail_len, 3))
            trail_pts[:, 0] = xs[0]
            trail_pts[:, 1] = ys[0]
            trail_line = pv.Spline(trail_pts, n_points=trail_len)
            trail_actor = plotter.add_mesh(trail_line, color="brown", line_width=2, opacity=0.6)

            plotter.camera_position = [(0, 0, 3 * scene_range),
                                       (0, (y_min + y_max) / 2, 0),
                                       (0, 1, 0)]

            anim_state = {"paused": False, "running": True, "recording": False, "request_record": False, "frames": []}
            gif_name = "physika_pendulum.gif" if state_dim == 2 else "physika_spring_pendulum.gif"

            def on_space():
                anim_state["paused"] = not anim_state["paused"]
            def on_quit():
                anim_state["running"] = False
            def on_save():
                if not anim_state["recording"]:
                    anim_state["request_record"] = True

            plotter.add_key_event("space", on_space)
            plotter.add_key_event("q", on_quit)
            plotter.add_key_event("s", on_save)

            if state_dim == 2:
                info = (f"t = {sub_t[0]:.3f}\n"
                        f"\u03b8 = {sub_states[0, 0]:.4f}\n"
                        f"\u03c9 = {sub_states[0, 1]:.4f}\n"
                        f"[SPACE: pause | S: save GIF | Q: quit]")
            else:
                info = (f"t = {sub_t[0]:.3f}\n"
                        f"r = {sub_states[0, 0]:.4f}  \u03b8 = {sub_states[0, 1]:.4f}\n"
                        f"dr = {sub_states[0, 2]:.4f}  d\u03b8 = {sub_states[0, 3]:.4f}\n"
                        f"[SPACE: pause | S: save GIF | Q: quit]")

            text_actor = plotter.add_text(info, position=(10, 10), font_size=13, font="times")

            plotter.show(auto_close=False, interactive_update=True)

            trail_history = []
            while anim_state["running"]:
                if anim_state["request_record"]:
                    anim_state["request_record"] = False
                    anim_state["recording"] = True
                    anim_state["frames"].clear()
                    print(f"[simulate] Recording GIF to {gif_name} ...")

                trail_history.clear()
                for i in range(len(xs)):
                    if not anim_state["running"]:
                        break
                    while anim_state["paused"] and anim_state["running"]:
                        plotter.update()
                        time_module.sleep(0.05)
                    if not anim_state["running"]:
                        break

                    bx, by = float(xs[i]), float(ys[i])
                    bob.points = pv.Sphere(radius=bob_radius, center=(bx, by, 0)).points

                    new_rod = pv.Line((0, 0, 0), (bx, by, 0))
                    plotter.remove_actor(rod_actor)
                    rod_actor = plotter.add_mesh(new_rod, color=rod_color, line_width=3)

                    trail_history.append([bx, by, 0.0])
                    if len(trail_history) > trail_len:
                        trail_history.pop(0)
                    if len(trail_history) >= 2:
                        tp = np.array(trail_history)
                        new_trail = pv.Spline(tp, n_points=len(tp))
                        plotter.remove_actor(trail_actor)
                        trail_actor = plotter.add_mesh(new_trail, color="brown", line_width=2, opacity=0.6)

                    if anim_state["recording"]:
                        pause_status = f"[RECORDING {gif_name} ...]"
                    elif anim_state["paused"]:
                        pause_status = "[PAUSED]"
                    else:
                        pause_status = "[SPACE: pause | S: save GIF | Q: quit]"
                    if state_dim == 2:
                        info = (f"t = {sub_t[i]:.3f}\n"
                                f"\u03b8 = {sub_states[i, 0]:.4f}\n"
                                f"\u03c9 = {sub_states[i, 1]:.4f}\n"
                                f"{pause_status}")
                    else:
                        info = (f"t = {sub_t[i]:.3f}\n"
                                f"r = {sub_states[i, 0]:.4f}  \u03b8 = {sub_states[i, 1]:.4f}\n"
                                f"dr = {sub_states[i, 2]:.4f}  d\u03b8 = {sub_states[i, 3]:.4f}\n"
                                f"{pause_status}")
                    text_actor.SetInput(info)

                    plotter.update()
                    if anim_state["recording"]:
                        anim_state["frames"].append(plotter.screenshot(return_img=True))
                    time_module.sleep(0.02)

                if anim_state["recording"]:
                    anim_state["recording"] = False
                    try:
                        import imageio
                        imageio.mimsave(gif_name, anim_state["frames"], fps=30, loop=0)
                        print(f"[simulate] GIF saved: {gif_name}")
                    except ImportError:
                        print("[simulate] Install imageio to save GIFs: pip install imageio")
                    anim_state["frames"].clear()

            plotter.close()
        except ImportError:
            pass
        except Exception as e:
            print(f"[simulate] PyVista animation error: {e}")


def animate(func: Any, *args: Any) -> None:
    """Animate a scalar function over a time range.

    Evaluates ``func`` at ``n_points`` evenly-spaced time values between
    ``time_min`` and ``time_max``, numerically differentiates to obtain
    velocity, and displays an interactive animation.

    If **PyVista** is installed an interactive 3-D scene is used
    (``SPACE`` to pause, ``Q`` to quit).  Otherwise falls back to a
    **matplotlib** ``FuncAnimation``.

    The last two (or three) positional ``args`` are interpreted as
    ``(time_min, time_max)`` or ``(time_min, time_max, n_points)``;
    everything before them is forwarded as extra arguments to ``func``.

    Parameters
    ----------
    func : callable
        A callable ``(*extra_args, t) -> scalar`` where ``t`` is the
        time parameter and ``extra_args`` are any fixed arguments.
    *args : Any
        Positional arguments laid out as
        ``[*extra_args, time_min, time_max]`` or
        ``[*extra_args, time_min, time_max, n_points]``.
        ``n_points`` defaults to ``200`` when omitted.

    Examples
    --------
    >>> from runtime import animate
    >>> animate(harmonic_oscillator, 1.0, 0.0, 0.0, 10.0)
    """
    import numpy as np
    import time as time_module

    if len(args) < 2:
        print("[animate] Need at least time_min and time_max")
        return

    last_arg = args[-1]
    if isinstance(last_arg, torch.Tensor):
        last_arg_val = last_arg.item()
    else:
        last_arg_val = last_arg

    def is_integer_like(val):
        if isinstance(val, int):
            return True
        if isinstance(val, float):
            return val == int(val)
        return False

    is_n_points_provided = is_integer_like(last_arg_val) and last_arg_val >= 10

    if is_n_points_provided:
        time_min = args[-3]
        time_max = args[-2]
        n_points = int(last_arg_val)
        fixed_args = list(args[:-3])
    else:
        time_min = args[-2]
        time_max = args[-1]
        n_points = 200
        fixed_args = list(args[:-2])

    if isinstance(time_min, torch.Tensor):
        time_min = time_min.item()
    if isinstance(time_max, torch.Tensor):
        time_max = time_max.item()

    time_vals = np.linspace(float(time_min), float(time_max), n_points)
    x_values = []

    for t in time_vals:
        call_args = []
        for a in fixed_args[:2]:
            if isinstance(a, torch.Tensor):
                call_args.append(a)
            else:
                call_args.append(torch.tensor(float(a), requires_grad=True))
        call_args.append(torch.tensor(float(t), requires_grad=True))
        for a in fixed_args[2:]:
            if isinstance(a, torch.Tensor):
                call_args.append(a)
            else:
                call_args.append(torch.tensor(float(a), requires_grad=True))

        result = func(*call_args)

        if isinstance(result, torch.Tensor):
            if result.is_complex():
                result = result.real
            x_values.append(result.item())
        elif isinstance(result, complex):
            x_values.append(result.real)
        else:
            x_values.append(float(result))

    x_values = np.array(x_values)

    dt = (float(time_max) - float(time_min)) / (n_points - 1)
    v_values = np.gradient(x_values, dt)

    try:
        import pyvista as pv
        HAS_PYVISTA = True
    except ImportError:
        HAS_PYVISTA = False

    try:
        import matplotlib.pyplot as plt
        HAS_MATPLOTLIB = True
    except ImportError:
        HAS_MATPLOTLIB = False

    if HAS_PYVISTA:
        plotter = pv.Plotter()
        plotter.add_title(
            "Physika \n\nHarmonic Oscillator Animation",
            font_size=24,
            font="times",
            shadow=True
        )

        sphere = pv.Sphere(radius=0.1, center=(x_values[0], 0, 0))
        plotter.add_mesh(sphere, color='blue')
        plotter.add_mesh(pv.Line((-2, 0, 0), (2, 0, 0)), color='black', line_width=3)

        x0_marker = pv.Sphere(radius=0.03, center=(x_values[0], 0, 0))
        plotter.add_mesh(x0_marker, color='red')

        plotter.camera_position = [(0, 5, 0), (0, 0, 0), (0, 0, 1)]

        anim_state = {"paused": False, "running": True}

        def on_key_press(key):
            if key == "space":
                anim_state["paused"] = not anim_state["paused"]
            elif key == "q" or key == "Escape":
                anim_state["running"] = False

        plotter.add_key_event("space", lambda: on_key_press("space"))
        plotter.add_key_event("q", lambda: on_key_press("q"))

        text_actor = plotter.add_text(
            f"t = {time_vals[0]:.3f}\nx = {x_values[0]:.4f}\nv = {v_values[0]:.4f}\n[SPACE: pause | Q: quit]",
            position=(10, 10), font_size=15, font="times"
        )

        plotter.show(auto_close=False, interactive_update=True)

        while anim_state["running"]:
            for i, x in enumerate(x_values):
                if not anim_state["running"]:
                    break
                while anim_state["paused"] and anim_state["running"]:
                    plotter.update()
                    time_module.sleep(0.05)
                if not anim_state["running"]:
                    break
                sphere.points = pv.Sphere(radius=0.1, center=(x, 0, 0)).points
                pause_status = "[PAUSED]" if anim_state["paused"] else "[SPACE: pause | Q: quit]"
                text_actor.SetInput(
                    f"t = {time_vals[i]:.3f}\nx = {x_values[i]:.4f}\nv = {v_values[i]:.4f}\n{pause_status}"
                )
                plotter.update()
                time_module.sleep(0.03)

        plotter.close()

    elif HAS_MATPLOTLIB:
        from matplotlib.animation import FuncAnimation

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.set_xlim(-2, 2)
        ax.set_ylim(-0.5, 0.5)
        ax.set_aspect('equal')
        ax.axhline(y=0, color='black', linewidth=2)
        ax.set_title("Harmonic Oscillator Animation [SPACE: pause/resume | R: reset]")

        mass, = ax.plot([], [], 'bo', markersize=20)
        spring, = ax.plot([], [], 'gray', linewidth=2)

        ax.plot([x_values[0]], [0], 'ro', markersize=8, label=f'x0 = {x_values[0]:.2f}')

        info_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                            verticalalignment='top', fontfamily='monospace',
                            fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        anim_state = {"paused": False, "frame": 0}
        ani_ref = [None]

        def init():
            mass.set_data([], [])
            spring.set_data([], [])
            info_text.set_text('')
            return mass, spring, info_text

        def anim(i):
            if anim_state["paused"]:
                i = anim_state["frame"]
            else:
                anim_state["frame"] = i
            mass.set_data([x_values[i]], [0])
            spring.set_data([0, x_values[i]], [0, 0])
            pause_str = " [PAUSED]" if anim_state["paused"] else ""
            info_text.set_text(f't = {time_vals[i]:.3f}{pause_str}\nx = {x_values[i]:.4f}\nv = {v_values[i]:.4f}')
            return mass, spring, info_text

        def on_key(event):
            if event.key == ' ':
                anim_state["paused"] = not anim_state["paused"]
                if not anim_state["paused"] and ani_ref[0] is not None:
                    ani_ref[0].frame_seq = ani_ref[0].new_frame_seq()
            elif event.key == 'r':
                anim_state["frame"] = 0
                anim_state["paused"] = False
                if ani_ref[0] is not None:
                    ani_ref[0].frame_seq = ani_ref[0].new_frame_seq()

        fig.canvas.mpl_connect('key_press_event', on_key)

        ani = FuncAnimation(fig, anim, init_func=init, frames=len(x_values),
                            interval=30, blit=True, repeat=True)
        ani_ref[0] = ani
        plt.show()
    else:
        print("[animate] No visualization backend available (install pyvista or matplotlib)")
