import torch
import torch.nn as nn
import torch.optim as optim
from physika.runtime import DEVICE

from physika.runtime import print
from physika.runtime import compute_grad

# === Functions ===
def zeros_1d(n):
    return torch.stack([(i * 0.0) for _fi_i in range(int(n)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])

def zeros_2d(n):
    return torch.stack([torch.stack([(i * 0.0) for _fi_j in range(int(n)) for j in [torch.tensor(float(_fi_j), device=DEVICE)]]) for _fi_i in range(int(n)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])

def stamp_conductance(G, n1, n2, g):
    if n1 != 0.0:
        G[int((n1 - 1)), int((n1 - 1))] = (G[int((n1 - 1)), int((n1 - 1))] + g)
    if n2 != 0.0:
        G[int((n2 - 1)), int((n2 - 1))] = (G[int((n2 - 1)), int((n2 - 1))] + g)
    if n1 != 0.0:
        if n2 != 0.0:
            G[int((n1 - 1)), int((n2 - 1))] = (G[int((n1 - 1)), int((n2 - 1))] - g)
            G[int((n2 - 1)), int((n1 - 1))] = (G[int((n2 - 1)), int((n1 - 1))] - g)
    return G

def stamp_current_source(I, n1, n2, current):
    if n1 != 0.0:
        I[int((n1 - 1))] = (I[int((n1 - 1))] - current)
    if n2 != 0.0:
        I[int((n2 - 1))] = (I[int((n2 - 1))] + current)
    return I

def stamp_resistors(G, resistors, num_resistors):
    for r in range(int(0), int(num_resistors)):
        G = stamp_conductance(G, resistors[int(r), int(0)], resistors[int(r), int(1)], (1.0 / resistors[int(r), int(2)]))
    return G

def stamp_sources(I, sources, num_sources):
    for s in range(int(0), int(num_sources)):
        I = stamp_current_source(I, sources[int(s), int(0)], sources[int(s), int(1)], sources[int(s), int(2)])
    return I

def eye(n):
    E = zeros_2d(n)
    for i in range(int(0), int(n)):
        E[int(i), int(i)] = 1.0
    return E

def linear_solve(n, A, b):
    E = eye(n)
    for c in range(int(0), int(n)):
        b = torch.stack([(b[int(i)] - (((A[int(i), int(c)] - E[int(i), int(c)]) * b[int(c)]) / A[int(c), int(c)])) for _fi_i in range(int(n)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
        A = torch.stack([torch.stack([(A[int(i), int(j)] - (((A[int(i), int(c)] - E[int(i), int(c)]) * A[int(c), int(j)]) / A[int(c), int(c)])) for _fi_j in range(int(n)) for j in [torch.tensor(float(_fi_j), device=DEVICE)]]) for _fi_i in range(int(n)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
    return b

def nodal_analysis(num_nodes, resistors, num_resistors, sources, num_sources, k=None, m=None):
    if k is None:
        k = int(resistors.shape[0])
    if m is None:
        m = int(sources.shape[0])
    G = stamp_resistors(zeros_2d(num_nodes), resistors, num_resistors)
    I = stamp_sources(zeros_1d(num_nodes), sources, num_sources)
    return linear_solve(num_nodes, G, I)

def nodal_analysis_transient(num_nodes, resistors, num_resistors, capacitors, num_capacitors, sources, num_sources, dt, num_points, initial_voltages):
    history = torch.stack([torch.stack([(t * 0.0) for _fi_j in range(int(num_nodes)) for j in [torch.tensor(float(_fi_j), device=DEVICE)]]) for _fi_t in range(int(num_points)) for t in [torch.tensor(float(_fi_t), device=DEVICE)]])
    previous = initial_voltages
    for j in range(int(0), int(num_nodes)):
        history[int(0), int(j)] = previous[int(j)]
    G_static = stamp_resistors(zeros_2d(num_nodes), resistors, num_resistors)
    I_static = stamp_sources(zeros_1d(num_nodes), sources, num_sources)
    G = zeros_2d(num_nodes)
    I = zeros_1d(num_nodes)
    n1 = 0.0
    n2 = 0.0
    g_eq = 0.0
    v1_previous = 0.0
    v2_previous = 0.0
    i_history = 0.0
    for step in range(int(1), int(num_points)):
        G = (G_static * 1.0)
        I = (I_static * 1.0)
        for q in range(int(0), int(num_capacitors)):
            n1 = capacitors[int(q), int(0)]
            n2 = capacitors[int(q), int(1)]
            g_eq = (capacitors[int(q), int(2)] / dt)
            G = stamp_conductance(G, n1, n2, g_eq)
            v1_previous = 0.0
            v2_previous = 0.0
            if n1 != 0.0:
                v1_previous = previous[int((n1 - 1))]
            if n2 != 0.0:
                v2_previous = previous[int((n2 - 1))]
            i_history = (g_eq * (v1_previous - v2_previous))
            I = stamp_current_source(I, n2, n1, i_history)
        previous = linear_solve(num_nodes, G, I)
        for j in range(int(0), int(num_nodes)):
            history[int(step), int(j)] = previous[int(j)]
    return history

def v1_at_2ms(C_uF):
    caps = torch.stack([torch.stack([torch.as_tensor(1, dtype=torch.get_default_dtype(), device=DEVICE), torch.as_tensor(0, dtype=torch.get_default_dtype(), device=DEVICE), torch.as_tensor((C_uF * 1e-06), dtype=torch.get_default_dtype(), device=DEVICE)])])
    h = nodal_analysis_transient(1, rc_resistors, 1, caps, 1, rc_sources, 1, dt, 201, v0)
    return h[int(200), int(0)]

# === Program ===
resistors = torch.tensor([[1, 2, 1000], [1, 0, 1000], [2, 0, 1000]], device=DEVICE)
current_sources = torch.tensor([[0, 1, 0.001], [0, 2, 0.001]], device=DEVICE)
dc_voltages = nodal_analysis(2, resistors, 3, current_sources, 2, 3, 2)
print(dc_voltages)
rc_resistors = torch.tensor([[1, 0, 1000.0]], device=DEVICE)
rc_capacitors = torch.tensor([[1, 0, 1e-06]], device=DEVICE)
rc_sources = torch.tensor([[0, 1, 0.001]], device=DEVICE)
dt = 1e-05
num_steps = 500
v0 = torch.tensor([0.0], device=DEVICE)
voltage_history = nodal_analysis_transient(1, rc_resistors, 1, rc_capacitors, 1, rc_sources, 1, dt, (num_steps + 1), v0)
final_time = (num_steps * dt)
simulated_final = voltage_history[int(500), int(0)]
exact_final = ((0.001 * 1000.0) * (1.0 - torch.exp(((-final_time) / (1000.0 * 1e-06)) if isinstance(((-final_time) / (1000.0 * 1e-06)), torch.Tensor) else torch.tensor(float(((-final_time) / (1000.0 * 1e-06)))))))
abs_error = torch.abs((simulated_final - exact_final) if isinstance((simulated_final - exact_final), torch.Tensor) else torch.tensor(float((simulated_final - exact_final))))
print(final_time)
print(simulated_final)
print(exact_final)
print(abs_error)
target_time = 0.002
target_voltage = 0.9
C_uF = torch.tensor(1.0, requires_grad=True)
for it in range(int(0), int(5)):
    C_uF = (C_uF - ((v1_at_2ms(C_uF) - target_voltage) / compute_grad(lambda _dC_uF: v1_at_2ms(_dC_uF), C_uF)))
print(C_uF)
print(v1_at_2ms(C_uF))
print(((target_time / (1000.0 * torch.log(10.0 if isinstance(10.0, torch.Tensor) else torch.tensor(float(10.0))))) * 1000000.0))