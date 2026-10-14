import numpy as np


def nodal_analysis(num_nodes, resistors, current_sources):
    """
    Solve a resistive circuit using nodal analysis.

    Parameters
    ----------
    num_nodes : int
        Number of non-ground nodes.
        Nodes are numbered 1..num_nodes.
        Node 0 is ground.

    resistors : list of tuples
        [(node1, node2, resistance), ...]

    current_sources : list of tuples
        [(node1, node2, current), ...]
        Current flows from node1 -> node2.

    Returns
    -------
    voltages : numpy.ndarray
        Node voltages [V1, V2, ...].
    """

    # Conductance matrix
    G = np.zeros((num_nodes, num_nodes))

    # Current vector
    I = np.zeros(num_nodes)

    # ------------------------------------------------
    # Resistors
    # ------------------------------------------------
    for n1, n2, R in resistors:
        g = 1.0 / R

        # Both nodes are non-ground
        if n1 != 0 and n2 != 0:
            G[n1 - 1, n1 - 1] += g
            G[n2 - 1, n2 - 1] += g
            G[n1 - 1, n2 - 1] -= g
            G[n2 - 1, n1 - 1] -= g

        # n2 is ground
        elif n1 != 0 and n2 == 0:
            G[n1 - 1, n1 - 1] += g

        # n1 is ground
        elif n1 == 0 and n2 != 0:
            G[n2 - 1, n2 - 1] += g

    # ------------------------------------------------
    # Current sources
    # ------------------------------------------------
    for n1, n2, current in current_sources:

        # Current leaves n1
        if n1 != 0:
            I[n1 - 1] -= current

        # Current enters n2
        if n2 != 0:
            I[n2 - 1] += current

    # ------------------------------------------------
    # Solve G V = I
    # ------------------------------------------------
    voltages = np.linalg.solve(G, I)

    return voltages


resistors = [
    (1, 2, 1000),
    (1, 0, 1000),
    (2, 0, 1000),
]

current_sources = [
    (0, 1, 0.001),   # 1 mA from ground -> node 1
    (0, 2, 0.001),   # 1 mA from ground -> node 1
]

voltages = nodal_analysis(
    num_nodes=2,
    resistors=resistors,
    current_sources=current_sources
)

print("Node voltages:")
for node, voltage in enumerate(voltages, start=1):
    print(f"V{node} = {voltage:.6f} V")
