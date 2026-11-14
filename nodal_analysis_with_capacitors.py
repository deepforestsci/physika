import numpy as np


def _check_node(node, num_nodes):
    """Validate a node number; 0 is ground, 1..num_nodes are unknown nodes."""
    if not isinstance(node, (int, np.integer)) or isinstance(node, (bool, np.bool_)):
        raise ValueError(f"Invalid node number {node!r}; expected an integer.")
    if not 0 <= node <= num_nodes:
        raise ValueError(
            f"Node {node} is out of range; valid nodes are 0..{num_nodes}."
        )


def _stamp_conductance(G, n1, n2, conductance):
    """Add a branch conductance between nodes n1 and n2 to G."""
    if n1 != 0:
        G[n1 - 1, n1 - 1] += conductance
    if n2 != 0:
        G[n2 - 1, n2 - 1] += conductance
    if n1 != 0 and n2 != 0:
        G[n1 - 1, n2 - 1] -= conductance
        G[n2 - 1, n1 - 1] -= conductance


def _stamp_current_source(I, n1, n2, current):
    """Current flows from n1 to n2; positive current is drawn from n1."""
    if n1 != 0:
        I[n1 - 1] -= current
    if n2 != 0:
        I[n2 - 1] += current


def nodal_analysis(num_nodes, resistors, current_sources):
    """Solve a linear DC resistive circuit using nodal analysis.

    Parameters
    ----------
    num_nodes : int
        Number of non-ground nodes, numbered 1..num_nodes. Node 0 is ground.
    resistors : iterable of (node1, node2, resistance_ohms)
    current_sources : iterable of (node1, node2, current_amperes)
        Positive current flows from node1 to node2.

    Returns
    -------
    numpy.ndarray
        Node voltages [V1, V2, ...] in volts.

    Notes
    -----
    Capacitors are open circuits in DC steady state. Use
    nodal_analysis_transient() to model capacitor charging/discharging.
    """
    if not isinstance(num_nodes, (int, np.integer)) or isinstance(num_nodes, (bool, np.bool_)) or num_nodes < 1:
        raise ValueError("num_nodes must be a positive integer.")

    G = np.zeros((num_nodes, num_nodes), dtype=float)
    I = np.zeros(num_nodes, dtype=float)

    for n1, n2, resistance in resistors:
        _check_node(n1, num_nodes)
        _check_node(n2, num_nodes)
        if not np.isfinite(resistance) or resistance <= 0:
            raise ValueError("Resistance must be finite and greater than zero.")
        _stamp_conductance(G, n1, n2, 1.0 / resistance)

    for n1, n2, current in current_sources:
        _check_node(n1, num_nodes)
        _check_node(n2, num_nodes)
        if not np.isfinite(current):
            raise ValueError("Current-source values must be finite.")
        _stamp_current_source(I, n1, n2, current)

    try:
        return np.linalg.solve(G, I)
    except np.linalg.LinAlgError as exc:
        raise np.linalg.LinAlgError(
            "The DC conductance matrix is singular. Check for floating nodes, "
            "missing ground/reference connections, or an underdetermined circuit."
        ) from exc


def nodal_analysis_transient(
    num_nodes,
    resistors,
    capacitors,
    current_sources,
    dt,
    num_steps,
    initial_voltages=None,
):
    """Simulate a linear RC circuit over time using backward Euler integration.

    Parameters
    ----------
    num_nodes : int
        Number of non-ground nodes, numbered 1..num_nodes. Node 0 is ground.
    resistors : iterable of (node1, node2, resistance_ohms)
    capacitors : iterable of (node1, node2, capacitance_farads)
    current_sources : iterable of (node1, node2, current_amperes), or a callable
        accepting time t (seconds) and returning that iterable at time t.
        Positive current flows from node1 to node2.
    dt : float
        Positive time step in seconds.
    num_steps : int
        Number of time steps to simulate.
    initial_voltages : array-like, optional
        Initial node voltages [V1, ..., Vn] in volts. Defaults to all zeros.

    Returns
    -------
    times : numpy.ndarray
        Time points, from 0 to num_steps*dt.
    history : numpy.ndarray
        Shape (num_steps + 1, num_nodes); each row contains node voltages.

    Capacitor model
    ---------------
    Backward Euler approximates i_C = C*d(v_C)/dt with a conductance
    g_eq = C/dt plus a history-current injection g_eq*v_C(previous).
    This is the companion-model approach used by Circuit-Builder's transient
    solver, adapted to this simpler node-numbered implementation:
    https://github.com/robertscharlie/Circuit-Builder/blob/main/circuit_builder/core/simulation.py
    """
    if not isinstance(num_nodes, (int, np.integer)) or isinstance(num_nodes, (bool, np.bool_)) or num_nodes < 1:
        raise ValueError("num_nodes must be a positive integer.")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and greater than zero.")
    if not isinstance(num_steps, (int, np.integer)) or isinstance(num_steps, (bool, np.bool_)) or num_steps < 0:
        raise ValueError("num_steps must be a non-negative integer.")

    # Validate and materialize static component lists once, so generators work too.
    resistors = list(resistors)
    capacitors = list(capacitors)
    static_sources = None if callable(current_sources) else list(current_sources)

    for n1, n2, resistance in resistors:
        _check_node(n1, num_nodes)
        _check_node(n2, num_nodes)
        if not np.isfinite(resistance) or resistance <= 0:
            raise ValueError("Resistance must be finite and greater than zero.")

    for n1, n2, capacitance in capacitors:
        _check_node(n1, num_nodes)
        _check_node(n2, num_nodes)
        if not np.isfinite(capacitance) or capacitance < 0:
            raise ValueError("Capacitance must be finite and non-negative.")

    if initial_voltages is None:
        previous_voltages = np.zeros(num_nodes, dtype=float)
    else:
        previous_voltages = np.asarray(initial_voltages, dtype=float).copy()
        if previous_voltages.shape != (num_nodes,):
            raise ValueError("initial_voltages must have one value per non-ground node.")
        if not np.all(np.isfinite(previous_voltages)):
            raise ValueError("initial_voltages must contain only finite values.")

    times = np.arange(num_steps + 1, dtype=float) * dt
    history = np.zeros((num_steps + 1, num_nodes), dtype=float)
    history[0] = previous_voltages

    for step in range(1, num_steps + 1):
        t = times[step]
        G = np.zeros((num_nodes, num_nodes), dtype=float)
        I = np.zeros(num_nodes, dtype=float)

        # Resistors: ordinary conductance stamps.
        for n1, n2, resistance in resistors:
            _stamp_conductance(G, n1, n2, 1.0 / resistance)

        # Independent current sources: either fixed or evaluated at each time.
        sources_now = current_sources(t) if callable(current_sources) else static_sources
        for n1, n2, current in sources_now:
            _check_node(n1, num_nodes)
            _check_node(n2, num_nodes)
            if not np.isfinite(current):
                raise ValueError("Current-source values must be finite.")
            _stamp_current_source(I, n1, n2, current)

        # Capacitors: g_eq=C/dt in parallel with a history-current source.
        # The previous capacitor voltage is V_previous(n1)-V_previous(n2).
        for n1, n2, capacitance in capacitors:
            if capacitance == 0:
                continue
            g_eq = capacitance / dt
            _stamp_conductance(G, n1, n2, g_eq)

            v1_previous = previous_voltages[n1 - 1] if n1 != 0 else 0.0
            v2_previous = previous_voltages[n2 - 1] if n2 != 0 else 0.0
            i_history = g_eq * (v1_previous - v2_previous)

            # This history term is a current injection into n1 and out of n2.
            if n1 != 0:
                I[n1 - 1] += i_history
            if n2 != 0:
                I[n2 - 1] -= i_history

        try:
            voltages = np.linalg.solve(G, I)
        except np.linalg.LinAlgError as exc:
            raise np.linalg.LinAlgError(
                f"Singular conductance matrix at t={t:g} s. Check for floating "
                "nodes, missing ground/reference connections, or an underdetermined circuit."
            ) from exc

        history[step] = voltages
        previous_voltages = voltages

    return times, history


if __name__ == "__main__":
    # Test 1: the original DC example.
    resistors = [
        (1, 2, 1000),
        (1, 0, 1000),
        (2, 0, 1000),
    ]
    current_sources = [
        (0, 1, 0.001),  # 1 mA from ground to node 1
        (0, 2, 0.001),  # 1 mA from ground to node 2
    ]

    dc_voltages = nodal_analysis(2, resistors, current_sources)
    print("DC test:")
    for node, voltage in enumerate(dc_voltages, start=1):
        print(f"  V{node} = {voltage:.6f} V")

    # Test 2: 1 kOhm resistor in parallel with 1 uF capacitor, driven by 1 mA.
    rc_resistors = [(1, 0, 1000.0)]
    rc_capacitors = [(1, 0, 1e-6)]
    rc_sources = [(0, 1, 1e-3)]
    dt = 1e-5  # 10 us
    num_steps = 500  # simulate for 5 ms

    times, voltage_history = nodal_analysis_transient(
        num_nodes=1,
        resistors=rc_resistors,
        capacitors=rc_capacitors,
        current_sources=rc_sources,
        dt=dt,
        num_steps=num_steps,
        initial_voltages=[0.0],
    )

    simulated_final = voltage_history[-1, 0]
    exact_final = 1e-3 * 1000.0 * (1.0 - np.exp(-times[-1] / (1000.0 * 1e-6)))
    print("\nTransient RC test:")
    print(f"  simulated time = {times[-1] * 1e3:.3f} ms")
    print(f"  simulated V1   = {simulated_final:.6f} V")
    print(f"  exact V1       = {exact_final:.6f} V")
    print(f"  absolute error = {abs(simulated_final - exact_final):.6f} V")

    # Basic regression checks.
    assert np.allclose(dc_voltages, [1.0, 1.0], atol=1e-12)
    assert abs(simulated_final - exact_final) < 5e-4
    assert np.all(np.diff(voltage_history[:, 0]) >= 0)
    print("\nAll checks passed.")
