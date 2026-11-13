import torch
import torch.nn as nn
import torch.optim as optim
from physika.runtime import DEVICE

from physika.runtime import print

# === Functions ===
def asinh_scalar(q):
    return torch.log((q + torch.sqrt(((q * q) + 1.0) if isinstance(((q * q) + 1.0), torch.Tensor) else torch.tensor(float(((q * q) + 1.0))))) if isinstance((q + torch.sqrt(((q * q) + 1.0) if isinstance(((q * q) + 1.0), torch.Tensor) else torch.tensor(float(((q * q) + 1.0))))), torch.Tensor) else torch.tensor(float((q + torch.sqrt(((q * q) + 1.0) if isinstance(((q * q) + 1.0), torch.Tensor) else torch.tensor(float(((q * q) + 1.0))))))))

def current_a(t):
    return current_nominal

def reaction_n(t, x, z):
    cs = (S_n @ x[int(0):int(ns_n)])
    ce = x[int((ns_n + ns_p)):int(((ns_n + ns_p) + nx_n))]
    phi_s = z[int(0):int(nx_n)]
    phi_e = z[int((nx_n + nx_p)):int(((nx_n + nx_p) + nx_n))]
    eta = ((((phi_s - phi_e) + u_n_initial) + eta_n_reference) - ocp_n((cs / cmax_n)))
    return ((2.0 * exchange_n(ce, cs)) * sinh(((faraday * eta) / ((2.0 * gas_constant) * temperature))))

def reaction_p(t, x, z):
    cs = (S_p @ x[int(ns_n):int((ns_n + ns_p))])
    ce = x[int((((ns_n + ns_p) + nx_n) + nx_s)):int(((ns_n + ns_p) + ne))]
    phi_s = z[int(nx_n):int((nx_n + nx_p))]
    phi_e = z[int((((nx_n + nx_p) + nx_n) + nx_s)):int(na)]
    eta = ((((phi_s - phi_e) + u_p_initial) + eta_p_reference) - ocp_p((cs / cmax_p)))
    return ((2.0 * exchange_p(ce, cs)) * sinh(((faraday * eta) / ((2.0 * gas_constant) * temperature))))

def electrolyte_current(t, x, z):
    ce = x[int((ns_n + ns_p)):int(((ns_n + ns_p) + ne))]
    phi_e = z[int((nx_n + nx_p)):int(na)]
    conductivity = (A_e @ (transport * electrolyte_conductivity(ce)))
    chi = ((((2.0 * (1.0 - t_plus)) * gas_constant) * temperature) / (faraday * maximum(ce, 0.01)))
    return (conductivity * ((harmonic_edges(chi, H_e, ne) * gradient_values(ce, width, ne)) - gradient_values(phi_e, width, ne)))

def dfn_rhs(t, x, z):
    jn = reaction_n(t, x, z)
    jp = reaction_p(t, x, z)
    ce = x[int((ns_n + ns_p)):int(((ns_n + ns_p) + ne))]
    aj = torch.cat([(a_n * jn), separator_zero, (a_p * jp)])
    flux = (((-harmonic_edges((transport * electrolyte_diffusivity(ce)), H_e, ne)) * gradient_values(ce, width, ne)) + ((t_plus * electrolyte_current(t, x, z)) / faraday))
    dcn = ((K_n @ (x[int(0):int(ns_n)] - initial_n)) + (B_n @ jn))
    dcp = ((K_p @ (x[int(ns_n):int((ns_n + ns_p))] - initial_p)) + (B_p @ jp))
    dce = (((-divergence_values(flux, width, ne)) + (aj / faraday)) / eps)
    return torch.cat([dcn, dcp, dce, torch.stack([torch.as_tensor((current_a(t) / 3600.0))])])

def dfn_algebraic(t, x, z):
    jn = reaction_n(t, x, z)
    jp = reaction_p(t, x, z)
    aj = torch.cat([(a_n * jn), separator_zero, (a_p * jp)])
    phin = z[int(0):int(nx_n)]
    phip = z[int(nx_n):int((nx_n + nx_p))]
    gradn = gradient_values(phin, width_n, nx_n)
    gradn[int(0)] = ((2.0 * phin[int(0)]) / width_n[int(0)])
    isn = ((-sigma_n) * gradn)
    boundary = positive_boundary
    isp = (((-sigma_p) * gradient_values(phip, width_p, nx_p)) + ((current_a(t) / area) * boundary))
    gn = ((length_total ** 2) * (divergence_values(isn, width_n, nx_n) + (a_n * jn)))
    gp = ((length_total ** 2) * (divergence_values(isp, width_p, nx_p) + (a_p * jp)))
    ge = ((length_total ** 2) * (divergence_values(electrolyte_current(t, x, z), width, ne) - aj))
    return torch.cat([gn, gp, ge])

def dfn_voltage(t, x, z):
    return ((ref_p + (1.5 * z[int(((nx_n + nx_p) - 1))])) - (0.5 * z[int(((nx_n + nx_p) - 2))]))

def physical_potentials(z):
    return torch.cat([z[int(0):int(nx_n)], (ref_p + z[int(nx_n):int((nx_n + nx_p))]), (ref_e + z[int((nx_n + nx_p)):int(na)])])

def surface_n(x):
    return (S_n @ x[int(0):int(ns_n)])

def surface_p(x):
    return (S_p @ x[int(ns_n):int((ns_n + ns_p))])

def dfn_cutoff(t, x, z, nd=None, na=None):
    if nd is None:
        nd = int(x.shape[0])
    if na is None:
        na = int(z.shape[0])
    return (dfn_voltage(t, x, z) - 2.5)

def trajectory_fields(states):
    fields = torch.zeros(int(82), int(nf), device=DEVICE)
    x = torch.zeros(int(nd), device=DEVICE)
    z = torch.zeros(int(na), device=DEVICE)
    t = 0.0
    for row in range(int(1), int((states[int(0), int(0)] + 1.0))):
        t = states[int(row), int(0)]
        x = states[int(row), int(1):int((nd + 1))]
        z = states[int(row), int((nd + 1)):int((nb + 1))]
        fields[int(row), :] = torch.cat([torch.stack([torch.as_tensor(dfn_voltage(t, x, z))]), surface_n(x), surface_p(x), physical_potentials(z)])
    return fields

def lithium_inventory_drift(states):
    initial = (states[int(1), int(1):int((nd + 1))] @ inventory_weights)
    drift = 0.0
    change = 0.0
    for row in range(int(1), int((states[int(0), int(0)] + 1.0))):
        change = (torch.abs(((states[int(row), int(1):int((nd + 1))] @ inventory_weights) - initial) if isinstance(((states[int(row), int(1):int((nd + 1))] @ inventory_weights) - initial), torch.Tensor) else torch.tensor(float(((states[int(row), int(1):int((nd + 1))] @ inventory_weights) - initial)))) / torch.abs(initial if isinstance(initial, torch.Tensor) else torch.tensor(float(initial))))
        if change > drift:
            drift = change
    return drift

def prediction_values(states, fields, row):
    return torch.stack([torch.as_tensor(states[int(row), int(0)]), torch.as_tensor(fields[int(row), int(0)]), torch.as_tensor(states[int(row), int(nd)])])

def minimum(q, b):
    result = (q + 0.0)
    for k in range(len(q)):
        if q[int(k)] > b:
            result[int(k)] = b
    return result

def maximum(q, b):
    result = (q + 0.0)
    for k in range(len(q)):
        if q[int(k)] < b:
            result[int(k)] = b
    return result

def softplus(q):
    return (maximum(q, 0.0) + torch.log((1.0 + torch.exp((-torch.abs(q if isinstance(q, torch.Tensor) else torch.tensor(float(q)))) if isinstance((-torch.abs(q if isinstance(q, torch.Tensor) else torch.tensor(float(q)))), torch.Tensor) else torch.tensor(float((-torch.abs(q if isinstance(q, torch.Tensor) else torch.tensor(float(q)))))))) if isinstance((1.0 + torch.exp((-torch.abs(q if isinstance(q, torch.Tensor) else torch.tensor(float(q)))) if isinstance((-torch.abs(q if isinstance(q, torch.Tensor) else torch.tensor(float(q)))), torch.Tensor) else torch.tensor(float((-torch.abs(q if isinstance(q, torch.Tensor) else torch.tensor(float(q)))))))), torch.Tensor) else torch.tensor(float((1.0 + torch.exp((-torch.abs(q if isinstance(q, torch.Tensor) else torch.tensor(float(q)))) if isinstance((-torch.abs(q if isinstance(q, torch.Tensor) else torch.tensor(float(q)))), torch.Tensor) else torch.tensor(float((-torch.abs(q if isinstance(q, torch.Tensor) else torch.tensor(float(q))))))))))))

def sinh(q):
    return ((torch.exp(q if isinstance(q, torch.Tensor) else torch.tensor(float(q))) - torch.exp((-q) if isinstance((-q), torch.Tensor) else torch.tensor(float((-q))))) / 2.0)

def tanh(q):
    result = (q + 0.0)
    for k in range(len(q)):
        if q[int(k)] >= 0.0:
            result[int(k)] = ((1.0 - torch.exp(((-2.0) * q[int(k)]) if isinstance(((-2.0) * q[int(k)]), torch.Tensor) else torch.tensor(float(((-2.0) * q[int(k)]))))) / (1.0 + torch.exp(((-2.0) * q[int(k)]) if isinstance(((-2.0) * q[int(k)]), torch.Tensor) else torch.tensor(float(((-2.0) * q[int(k)]))))))
        else:
            result[int(k)] = ((torch.exp((2.0 * q[int(k)]) if isinstance((2.0 * q[int(k)]), torch.Tensor) else torch.tensor(float((2.0 * q[int(k)])))) - 1.0) / (torch.exp((2.0 * q[int(k)]) if isinstance((2.0 * q[int(k)]), torch.Tensor) else torch.tensor(float((2.0 * q[int(k)])))) + 1.0))
    return result

def ocp_barrier(theta):
    return (205.0568621937484 * (softplus(((-6910.192179565431) * (theta + 0.00077))) - softplus(((-6910.192179565431) * ((1.0 - theta) + 0.00077)))))

def ocp_n(theta):
    q = maximum(minimum(theta, (1.0 - 1e-10)), 1e-10)
    return ((((((1.9793 * torch.exp(((-39.3631) * q) if isinstance(((-39.3631) * q), torch.Tensor) else torch.tensor(float(((-39.3631) * q))))) + 0.2482) - (0.0909 * tanh((29.8538 * (q - 0.1234))))) - (0.04478 * tanh((14.9159 * (q - 0.2769))))) - (0.0205 * tanh((30.4444 * (q - 0.6103))))) + ocp_barrier(theta))

def ocp_p(theta):
    q = maximum(minimum(theta, (1.0 - 1e-10)), 1e-10)
    return (((((((-0.809) * q) + 4.4875) - (0.0428 * tanh((18.5138 * (q - 0.5542))))) - (17.7326 * tanh((15.789 * (q - 0.3117))))) + (17.5842 * tanh((15.9308 * (q - 0.312))))) + ocp_barrier(theta))

def exchange_n(ce, cs):
    q = maximum(minimum(cs, cmax_n), 0.0)
    return ((((6.48e-07 * torch.exp(((35000.0 / gas_constant) * ((1.0 / 298.15) - (1.0 / temperature))) if isinstance(((35000.0 / gas_constant) * ((1.0 / 298.15) - (1.0 / temperature))), torch.Tensor) else torch.tensor(float(((35000.0 / gas_constant) * ((1.0 / 298.15) - (1.0 / temperature))))))) * torch.sqrt(maximum(ce, 0.0) if isinstance(maximum(ce, 0.0), torch.Tensor) else torch.tensor(float(maximum(ce, 0.0))))) * torch.sqrt(q if isinstance(q, torch.Tensor) else torch.tensor(float(q)))) * torch.sqrt((cmax_n - q) if isinstance((cmax_n - q), torch.Tensor) else torch.tensor(float((cmax_n - q)))))

def exchange_p(ce, cs):
    q = maximum(minimum(cs, cmax_p), 0.0)
    return ((((3.42e-06 * torch.exp(((17800.0 / gas_constant) * ((1.0 / 298.15) - (1.0 / temperature))) if isinstance(((17800.0 / gas_constant) * ((1.0 / 298.15) - (1.0 / temperature))), torch.Tensor) else torch.tensor(float(((17800.0 / gas_constant) * ((1.0 / 298.15) - (1.0 / temperature))))))) * torch.sqrt(maximum(ce, 0.0) if isinstance(maximum(ce, 0.0), torch.Tensor) else torch.tensor(float(maximum(ce, 0.0))))) * torch.sqrt(q if isinstance(q, torch.Tensor) else torch.tensor(float(q)))) * torch.sqrt((cmax_p - q) if isinstance((cmax_p - q), torch.Tensor) else torch.tensor(float((cmax_p - q)))))

def electrolyte_diffusivity(ce):
    q = (maximum(ce, 10.0) / 1000.0)
    return (((8.794e-11 * (q ** 2)) - (3.972e-10 * q)) + 4.862e-10)

def electrolyte_conductivity(ce):
    q = (maximum(ce, 10.0) / 1000.0)
    return (((0.1297 * (q ** 3)) - (2.51 * (q ** 1.5))) + (3.329 * q))

def arithmetic_matrix(width, n):
    result = torch.zeros(int((n + 1)), int(n), device=DEVICE)
    result[int(0), int(0)] = 1.5
    result[int(0), int(1)] = (-0.5)
    result[int(n), int((n - 1))] = 1.5
    result[int(n), int((n - 2))] = (-0.5)
    for k in range(int(1), int(n)):
        result[int(k), int((k - 1))] = (width[int(k)] / (width[int((k - 1))] + width[int(k)]))
        result[int(k), int(k)] = (width[int((k - 1))] / (width[int((k - 1))] + width[int(k)]))
    return result

def harmonic_matrix(width, n):
    result = torch.zeros(int((n + 1)), int(n), device=DEVICE)
    result[int(0), int(0)] = 1.0
    result[int(n), int((n - 1))] = 1.0
    for k in range(int(1), int(n)):
        result[int(k), int((k - 1))] = (width[int((k - 1))] / (width[int((k - 1))] + width[int(k)]))
        result[int(k), int(k)] = (width[int(k)] / (width[int((k - 1))] + width[int(k)]))
    return result

def matrix_vector(weights, c, n):
    return torch.stack([torch.sum((weights[int(k)] * c) if isinstance((weights[int(k)] * c), torch.Tensor) else torch.tensor(float((weights[int(k)] * c)))) for _fi_k in range(int(n)) for k in [torch.tensor(float(_fi_k), device=DEVICE)]])

def harmonic_edges(c, weights, n):
    result = (1.0 / matrix_vector(weights, (1.0 / c), (n + 1)))
    result[int(0)] = ((1.5 * c[int(0)]) - (0.5 * c[int(1)]))
    result[int(n)] = ((1.5 * c[int((n - 1))]) - (0.5 * c[int((n - 2))]))
    return result

def particle_matrix(nx, nr, radius, diffusivity):
    size = (nx * nr)
    result = torch.zeros(int(size), int(size), device=DEVICE)
    dr = (radius / nr)
    volume = 0.0
    coeff = 0.0
    for k in range(int(0), int(nx)):
        for q in range(int(0), int(nr)):
            volume = (((((q + 1) ** 3) - (q ** 3)) * (dr ** 3)) / 3.0)
            if q > 0:
                coeff = ((diffusivity * ((q * dr) ** 2)) / (dr * volume))
                result[int(((k * nr) + q)), int((((k * nr) + q) - 1))] = coeff
                result[int(((k * nr) + q)), int(((k * nr) + q))] = (result[int(((k * nr) + q)), int(((k * nr) + q))] - coeff)
            if q < (nr - 1):
                coeff = ((diffusivity * (((q + 1) * dr) ** 2)) / (dr * volume))
                result[int(((k * nr) + q)), int((((k * nr) + q) + 1))] = coeff
                result[int(((k * nr) + q)), int(((k * nr) + q))] = (result[int(((k * nr) + q)), int(((k * nr) + q))] - coeff)
    return result

def particle_source(nx, nr, radius, faraday):
    size = (nx * nr)
    result = torch.zeros(int(size), int(nx), device=DEVICE)
    volume = ((((nr ** 3) - ((nr - 1) ** 3)) * ((radius / nr) ** 3)) / 3.0)
    for k in range(int(0), int(nx)):
        result[int((((k * nr) + nr) - 1)), int(k)] = ((-(radius ** 2)) / (faraday * volume))
    return result

def surface_matrix(nx, nr):
    size = (nx * nr)
    result = torch.zeros(int(nx), int(size), device=DEVICE)
    for k in range(int(0), int(nx)):
        result[int(k), int((((k * nr) + nr) - 1))] = 1.5
        result[int(k), int((((k * nr) + nr) - 2))] = (-0.5)
    return result

def particle_inventory(nx, nr, radius, width, active):
    size = (nx * nr)
    result = torch.zeros(int(size), device=DEVICE)
    for k in range(int(0), int(nx)):
        for q in range(int(0), int(nr)):
            result[int(((k * nr) + q))] = (((active * width) * (((q + 1) ** 3) - (q ** 3))) / (nr ** 3))
    return result

def gradient_values(c, width, n):
    left = c[int(0):int((n - 1))]
    right = c[int(1):int(n)]
    wl = width[int(0):int((n - 1))]
    wr = width[int(1):int(n)]
    internal = ((2.0 * (right - left)) / (wl + wr))
    return torch.cat([torch.tensor([0.0], device=DEVICE), internal, torch.tensor([0.0], device=DEVICE)])

def divergence_values(flux, width, n):
    left = flux[int(0):int(n)]
    right = flux[int(1):int((n + 1))]
    return ((right - left) / width)

def gaussian_solve(A, b):
    a_row = get_2d_array_num_rows(A)
    a_col = get_2d_array_num_cols(A)
    new_col = (a_col + 1)
    aug = torch.zeros(int(a_row), int(new_col), device=DEVICE)
    aug[:, :int(a_col)] = A
    aug[:, int(a_col)] = b
    for i in range(int(0), int(a_row)):
        max_row = i
        magnitudes = torch.abs(aug[int(i):, int(i)] if isinstance(aug[int(i):, int(i)], torch.Tensor) else torch.tensor(float(aug[int(i):, int(i)])))
        largest = magnitudes[int(0)]
        for k in range(int((i + 1)), int(a_row)):
            if magnitudes[int((k - i))] > largest:
                max_row = k
                largest = magnitudes[int((k - i))]
        if max_row != i:
            lo = min(i, max_row)
            hi = max(i, max_row)
            rows_above = aug[:int(lo), :]
            pivot_row = aug[int(hi):int((hi + 1)), :]
            rows_between = aug[int((lo + 1)):int(hi), :]
            current_row = aug[int(lo):int((lo + 1)), :]
            rows_below = aug[int((hi + 1)):, :]
            aug = torch.cat([rows_above, pivot_row, rows_between, current_row, rows_below])
        pivot_row = aug[int(i):int((i + 1)), :]
        pivot_value = aug[int(i), int(i)]
        rows_below = aug[int((i + 1)):, :]
        elimination_factors = (rows_below[:, int(i):int((i + 1))] / pivot_value)
        eliminated_rows = (rows_below - (elimination_factors * pivot_row))
        aug = torch.cat([aug[:int((i + 1)), :], eliminated_rows])
    x = torch.zeros(int(0), device=DEVICE)
    for i in range(int(0), int(a_row)):
        idx = ((a_col - 1) - i)
        total = (aug[int(idx), int(a_col)] - torch.sum((aug[int(idx), int((idx + 1)):int(a_col)] * x) if isinstance((aug[int(idx), int((idx + 1)):int(a_col)] * x), torch.Tensor) else torch.tensor(float((aug[int(idx), int((idx + 1)):int(a_col)] * x)))))
        val = (total / aug[int(idx), int(idx)])
        val = torch.reshape(val, (1,) if isinstance(1, int) else 1)
        x = torch.cat([val, x])
    return x

def get_2d_array_num_rows(x):
    total = 0
    temp = 0
    for i in range(len(x)):
        temp = x[int(i)]
        total = total + 1
    return total

def get_2d_array_num_cols(x, m=None, n=None):
    if m is None:
        m = int(x.shape[0])
    if n is None:
        n = int(x.shape[1])
    return get_1d_array_length(x[int(0)])

def get_1d_array_length(x):
    total = 0
    temp = 0
    for i in range(len(x)):
        temp = x[int(i)]
        total = total + 1
    return total

def min(x, y):
    if x < y:
        return x
    else:
        return y

def max(x, y):
    if x > y:
        return x
    else:
        return y

def max_norm(v, n):
    result = 0.0
    for k in range(int(0), int(n)):
        if torch.abs(v[int(k)] if isinstance(v[int(k)], torch.Tensor) else torch.tensor(float(v[int(k)]))) > result:
            result = torch.abs(v[int(k)] if isinstance(v[int(k)], torch.Tensor) else torch.tensor(float(v[int(k)])))
    return result

def finite_values(v, n):
    valid = 1.0
    for k in range(int(0), int(n)):
        if torch.abs(v[int(k)] if isinstance(v[int(k)], torch.Tensor) else torch.tensor(float(v[int(k)]))) <= 3.402823466e+38:
            valid = valid
        else:
            valid = 0.0
    return valid

def initial_residual(u, x, t):
    return ((dfn_algebraic(t, x, (u * z_scale)) / g_scale) * (newton_tolerance / algebraic_tolerance))

def initial_component(u, x, t, k, na=None, nd=None):
    if na is None:
        na = int(u.shape[0])
    if nd is None:
        nd = int(x.shape[0])
    r = initial_residual(u, x, t)
    return r[int(k)]

def initial_jacobian(u, x, t):
    shape = [na, na]
    return torch.reshape(compute_grad(lambda _du: initial_residual(_du, x, t), u), (shape,) if isinstance(shape, int) else shape)

def consistent_potentials(x, guess, t):
    tolerance = newton_tolerance
    last_iteration = 20
    u = (guess / z_scale)
    active = 1.0
    success = 0.0
    iterations = 0.0
    alpha = 1.0
    found = 0.0
    r = torch.zeros(int(na), device=DEVICE)
    trial = torch.zeros(int(na), device=DEVICE)
    delta = torch.zeros(int(na), device=DEVICE)
    J = torch.zeros(int(na), int(na), device=DEVICE)
    rnorm = 0.0
    for iteration in range(int(0), int(21)):
        if active > 0.0:
            r = initial_residual(u, x, t)
            if finite_values(r, na) > 0.0:
                rnorm = max_norm(r, na)
                if rnorm <= tolerance:
                    success = 1.0
                    active = 0.0
                else:
                    if iteration < last_iteration:
                        J = initial_jacobian(u, x, t)
                        iterations = (iterations + 1.0)
                        delta = gaussian_solve(J, (-r))
                        if finite_values(delta, na) > 0.0:
                            alpha = 1.0
                            found = 0.0
                            for search in range(int(0), int(16)):
                                if found < 1.0:
                                    trial = initial_residual((u + (alpha * delta)), x, t)
                                    if finite_values(trial, na) > 0.0:
                                        if max_norm(trial, na) < rnorm:
                                            found = 1.0
                                    if found < 1.0:
                                        alpha = (alpha * 0.5)
                            if found > 0.0:
                                u = (u + (alpha * delta))
                            else:
                                active = 0.0
                        else:
                            active = 0.0
            else:
                active = 0.0
    return torch.cat([(u * z_scale), torch.stack([torch.as_tensor(success), torch.as_tensor(iterations)])])

def step_residual(u, a, history, t, h, alpha):
    y = (u * solver_scale)
    increment = y[int(0):int(nd)]
    z = y[int(nd):int(nb)]
    x = (a + increment)
    return torch.cat([((((alpha * increment) + history) - (h * dfn_rhs((t + h), x, z))) / x_scale), ((dfn_algebraic((t + h), x, z) / g_scale) * (newton_tolerance / algebraic_tolerance))])

def step_component(u, a, history, t, h, alpha, k):
    differential_size = nd
    y = (u * solver_scale)
    increment = y[int(0):int(nd)]
    z = y[int(nd):int(nb)]
    x = (a + increment)
    r = torch.zeros(int(nd), device=DEVICE)
    g = torch.zeros(int(na), device=DEVICE)
    if k < differential_size:
        r = ((((alpha * increment) + history) - (h * dfn_rhs((t + h), x, z))) / x_scale)
    if k < differential_size:
        return r[int(k)]
    g = ((dfn_algebraic((t + h), x, z) / g_scale) * (newton_tolerance / algebraic_tolerance))
    return g[int((k - nd))]

def step_jacobian(u, a, history, t, h, alpha):
    shape = [nb, nb]
    return torch.reshape(compute_grad(lambda _du: step_residual(_du, a, history, t, h, alpha), u), (shape,) if isinstance(shape, int) else shape)

def step_newton(a, b, history, t, h, alpha):
    tolerance = newton_tolerance
    last_iteration = 20
    u = (torch.cat([torch.zeros(int(nd), device=DEVICE), b]) / solver_scale)
    active = 1.0
    success = 0.0
    iterations = 0.0
    r = torch.zeros(int(nb), device=DEVICE)
    delta = torch.zeros(int(nb), device=DEVICE)
    defect = torch.zeros(int(nb), device=DEVICE)
    tangent = torch.zeros(int(nb), device=DEVICE)
    J = torch.zeros(int(nb), int(nb), device=DEVICE)
    rnorm = 0.0
    gnorm = 0.0
    correction = 0.0
    for iteration in range(int(0), int(21)):
        if active > 0.0:
            r = step_residual(u, a, history, t, h, alpha)
            if finite_values(r, nb) > 0.0:
                rnorm = max_norm(r, nb)
                if rnorm <= tolerance:
                    success = 1.0
                    active = 0.0
                else:
                    if iteration < last_iteration:
                        J = step_jacobian(u, a, history, t, h, alpha)
                        iterations = (iterations + 1.0)
                        delta = gaussian_solve(J, (-r))
                        if finite_values(delta, nb) > 0.0:
                            gnorm = max_norm(r[int(nd):int(nb)], na)
                            correction = max_norm(delta[int(0):int(nd)], nd)
                            if gnorm <= tolerance:
                                if correction > tolerance:
                                    defect = torch.cat([r[int(0):int(nd)], torch.zeros(int(na), device=DEVICE)])
                                    tangent = gaussian_solve(J, (-defect))
                                    if finite_values(tangent, nb) > 0.0:
                                        correction = max_norm(tangent[int(0):int(nd)], nd)
                                    else:
                                        active = 0.0
                                if correction <= tolerance:
                                    success = 1.0
                                    active = 0.0
                            if active > 0.0:
                                u = (u + delta)
                        else:
                            active = 0.0
            else:
                active = 0.0
    return torch.cat([(u * solver_scale), torch.stack([torch.as_tensor(success), torch.as_tensor(iterations)])])

def bdf_step(t, a, b, h, previous, previous_h, started):
    alpha = 1.0
    ratio = 0.0
    history = torch.zeros(int(nd), device=DEVICE)
    if started > 0.0:
        ratio = (h / previous_h)
        alpha = ((1.0 + (2.0 * ratio)) / (1.0 + ratio))
        history = (((ratio ** 2) / (1.0 + ratio)) * (previous - a))
    result = step_newton(a, b, history, t, h, alpha)
    return torch.cat([(a + result[int(0):int(nd)]), result[int(nd):int(nb)], result[int(nb):int((nb + 2))]])

def step_factor(error):
    factor = 2.0
    if error > 0.03:
        factor = 1.6
    if error > 0.2:
        factor = 1.2
    if error > 0.7:
        factor = 0.95
    return factor

def attempt_step(t, x, z, h, previous, previous_h, started):
    full = bdf_step(t, x, z, h, previous, previous_h, started)
    half = torch.zeros(int((nb + 2)), device=DEVICE)
    small = torch.zeros(int((nb + 2)), device=DEVICE)
    success = 0.0
    error = 2.0
    count = full[int((nb + 1))]
    a = full[int(0):int(nd)]
    b = full[int(nd):int(nb)]
    magnitude = torch.zeros(int(nd), device=DEVICE)
    weight = torch.zeros(int(nd), device=DEVICE)
    divisor = 1.0
    if full[int(nb)] > 0.0:
        half = bdf_step(t, x, z, (h / 2.0), previous, previous_h, started)
        count = (count + half[int((nb + 1))])
        if half[int(nb)] > 0.0:
            small = bdf_step((t + (h / 2.0)), half[int(0):int(nd)], half[int(nd):int(nb)], (h / 2.0), x, (h / 2.0), 1.0)
            count = (count + small[int((nb + 1))])
            if small[int(nb)] > 0.0:
                magnitude = (0.5 * ((torch.abs((a / x_scale) if isinstance((a / x_scale), torch.Tensor) else torch.tensor(float((a / x_scale)))) + torch.abs((x / x_scale) if isinstance((x / x_scale), torch.Tensor) else torch.tensor(float((x / x_scale))))) + torch.abs((torch.abs((a / x_scale) if isinstance((a / x_scale), torch.Tensor) else torch.tensor(float((a / x_scale)))) - torch.abs((x / x_scale) if isinstance((x / x_scale), torch.Tensor) else torch.tensor(float((x / x_scale))))) if isinstance((torch.abs((a / x_scale) if isinstance((a / x_scale), torch.Tensor) else torch.tensor(float((a / x_scale)))) - torch.abs((x / x_scale) if isinstance((x / x_scale), torch.Tensor) else torch.tensor(float((x / x_scale))))), torch.Tensor) else torch.tensor(float((torch.abs((a / x_scale) if isinstance((a / x_scale), torch.Tensor) else torch.tensor(float((a / x_scale)))) - torch.abs((x / x_scale) if isinstance((x / x_scale), torch.Tensor) else torch.tensor(float((x / x_scale))))))))))
                weight = (absolute_tolerance + (relative_tolerance * magnitude))
                if started > 0.0:
                    divisor = 3.0
                error = torch.sqrt((torch.sum(((((a - small[int(0):int(nd)]) / x_scale) / (divisor * weight)) ** 2) if isinstance(((((a - small[int(0):int(nd)]) / x_scale) / (divisor * weight)) ** 2), torch.Tensor) else torch.tensor(float(((((a - small[int(0):int(nd)]) / x_scale) / (divisor * weight)) ** 2)))) / nd) if isinstance((torch.sum(((((a - small[int(0):int(nd)]) / x_scale) / (divisor * weight)) ** 2) if isinstance(((((a - small[int(0):int(nd)]) / x_scale) / (divisor * weight)) ** 2), torch.Tensor) else torch.tensor(float(((((a - small[int(0):int(nd)]) / x_scale) / (divisor * weight)) ** 2)))) / nd), torch.Tensor) else torch.tensor(float((torch.sum(((((a - small[int(0):int(nd)]) / x_scale) / (divisor * weight)) ** 2) if isinstance(((((a - small[int(0):int(nd)]) / x_scale) / (divisor * weight)) ** 2), torch.Tensor) else torch.tensor(float(((((a - small[int(0):int(nd)]) / x_scale) / (divisor * weight)) ** 2)))) / nd))))
                if error <= 1.0:
                    success = 1.0
    return torch.cat([a, b, torch.stack([torch.as_tensor(success), torch.as_tensor(error), torch.as_tensor(count)])])

def integrate_dfn():
    output = torch.zeros(int(82), int((nb + 1)), device=DEVICE)
    x = x0
    z = z0
    previous = x0
    initial = consistent_potentials(x, z, 0.0)
    result = torch.zeros(int((nb + 3)), device=DEVICE)
    root = torch.zeros(int((nb + 2)), device=DEVICE)
    a = x0
    b = z0
    live = initial[int(na)]
    status = (-1.0)
    saved = 1
    t = 0.0
    h = 0.1
    hh = 0.1
    previous_h = 0.1
    started = 0.0
    stop = 0.0
    interval_done = 0.0
    retries = 0
    retry_limit = 15
    accepted = 0.0
    rejected = 0.0
    iterations = initial[int((na + 1))]
    max_algebraic = 0.0
    algebraic_norm = 0.0
    algebraic_limit = (algebraic_tolerance * 1.01)
    lo = 0.0
    hi = 0.0
    mid = 0.0
    new_t = 0.0
    root_ok = 1.0
    if live > 0.0:
        status = 0.0
        z = initial[int(0):int(na)]
        max_algebraic = max_norm((dfn_algebraic(0.0, x, z) / g_scale), na)
    output[int(1), int(0)] = 0.0
    output[int(1), int(1):int((nd + 1))] = x
    output[int(1), int((nd + 1)):int((nb + 1))] = z
    for point in range(int(1), int(81)):
        if live > 0.0:
            stop = (50.0 * point)
            interval_done = 0.0
            retries = 0
            for attempt in range(int(0), int(10000)):
                if interval_done < 1.0:
                    hh = h
                    if hh > (stop - t):
                        hh = (stop - t)
                    if hh < 1e-09:
                        status = (-2.0)
                        live = 0.0
                        interval_done = 1.0
                    if live > 0.0:
                        result = attempt_step(t, x, z, hh, previous, previous_h, started)
                        iterations = (iterations + result[int((nb + 2))])
                        if result[int(nb)] > 0.0:
                            a = result[int(0):int(nd)]
                            b = result[int(nd):int(nb)]
                            new_t = (t + hh)
                            root_ok = 1.0
                            if dfn_cutoff(new_t, a, b) <= 0.0:
                                lo = 0.0
                                hi = hh
                                for bracket in range(int(0), int(32)):
                                    if (hi - lo) > 1e-05:
                                        if root_ok > 0.0:
                                            mid = ((lo + hi) / 2.0)
                                            root = bdf_step(t, x, z, mid, previous, previous_h, started)
                                            iterations = (iterations + root[int((nb + 1))])
                                            if root[int(nb)] > 0.0:
                                                if dfn_cutoff((t + mid), root[int(0):int(nd)], root[int(nd):int(nb)]) <= 0.0:
                                                    hi = mid
                                                    a = root[int(0):int(nd)]
                                                    b = root[int(nd):int(nb)]
                                                else:
                                                    lo = mid
                                            else:
                                                root_ok = 0.0
                                new_t = (t + hi)
                                if root_ok > 0.0:
                                    status = 1.0
                                    live = 0.0
                                    interval_done = 1.0
                            if root_ok > 0.0:
                                previous = x
                                previous_h = hh
                                started = 1.0
                                t = new_t
                                x = a
                                z = b
                                accepted = (accepted + 1.0)
                                retries = 0
                                h = (hh * step_factor(result[int((nb + 1))]))
                                if h > 50.0:
                                    h = 50.0
                                if t >= (stop - 1e-10):
                                    interval_done = 1.0
                            else:
                                rejected = (rejected + 1.0)
                                retries = (retries + 1)
                                h = (hh / 2.0)
                        else:
                            rejected = (rejected + 1.0)
                            retries = (retries + 1)
                            h = (hh / 2.0)
                        if retries > retry_limit:
                            status = (-3.0)
                            live = 0.0
                            interval_done = 1.0
                        if accepted >= 10000.0:
                            status = (-4.0)
                            live = 0.0
                            interval_done = 1.0
            if interval_done < 1.0:
                status = (-5.0)
                live = 0.0
            if status >= 0.0:
                saved = (saved + 1)
                output[int(saved), int(0)] = t
                output[int(saved), int(1):int((nd + 1))] = x
                output[int(saved), int((nd + 1)):int((nb + 1))] = z
                algebraic_norm = max_norm((dfn_algebraic(t, x, z) / g_scale), na)
                if algebraic_norm > max_algebraic:
                    max_algebraic = algebraic_norm
                if algebraic_norm > algebraic_limit:
                    status = (-6.0)
                    live = 0.0
    output[int(0), int(0)] = saved
    output[int(0), int(1)] = status
    output[int(0), int(2)] = accepted
    output[int(0), int(3)] = rejected
    output[int(0), int(4)] = iterations
    output[int(0), int(5)] = max_algebraic
    output[int(0), int(6)] = initial[int((na + 1))]
    return output

# === Program ===
nx_n = 10
nx_s = 10
nx_p = 10
nr_n = 10
nr_p = 10
current_nominal = 5.0
faraday = 96485.33212
gas_constant = 8.31446261815324
temperature = 298.15
length_n = 8.52e-05
length_s = 1.2e-05
length_p = 7.56e-05
area = (0.065 * 1.58)
radius_n = 5.86e-06
radius_p = 5.22e-06
diffusivity_n = 3.3e-14
diffusivity_p = 4e-15
sigma_n = 215.0
sigma_p = 0.18
porosity_n = 0.25
porosity_s = 0.47
porosity_p = 0.335
active_n = 0.75
active_p = 0.665
cmax_n = 33133.0
cmax_p = 63104.0
initial_n = 29866.0
initial_p = 17038.0
initial_e = 1000.0
t_plus = 0.2594
ns_n = (nx_n * nr_n)
ns_p = (nx_p * nr_p)
ne = ((nx_n + nx_s) + nx_p)
nd = (((ns_n + ns_p) + ne) + 1)
na = ((nx_n + nx_p) + ne)
length_total = ((length_n + length_s) + length_p)
a_n = ((3.0 * active_n) / radius_n)
a_p = ((3.0 * active_p) / radius_p)
width_n = torch.stack([((length_n / nx_n) + (0.0 * k)) for _fi_k in range(int(nx_n)) for k in [torch.tensor(float(_fi_k), device=DEVICE)]])
width_s = torch.stack([((length_s / nx_s) + (0.0 * k)) for _fi_k in range(int(nx_s)) for k in [torch.tensor(float(_fi_k), device=DEVICE)]])
width_p = torch.stack([((length_p / nx_p) + (0.0 * k)) for _fi_k in range(int(nx_p)) for k in [torch.tensor(float(_fi_k), device=DEVICE)]])
width = torch.cat([width_n, width_s, width_p])
eps_n = torch.stack([(porosity_n + (0.0 * k)) for _fi_k in range(int(nx_n)) for k in [torch.tensor(float(_fi_k), device=DEVICE)]])
eps_s = torch.stack([(porosity_s + (0.0 * k)) for _fi_k in range(int(nx_s)) for k in [torch.tensor(float(_fi_k), device=DEVICE)]])
eps_p = torch.stack([(porosity_p + (0.0 * k)) for _fi_k in range(int(nx_p)) for k in [torch.tensor(float(_fi_k), device=DEVICE)]])
eps = torch.cat([eps_n, eps_s, eps_p])
transport = (eps ** 1.5)
separator_zero = torch.zeros(int(nx_s), device=DEVICE)
A_e = arithmetic_matrix(width, ne)
H_e = harmonic_matrix(width, ne)
positive_boundary = torch.zeros(int((nx_p + 1)), device=DEVICE)
positive_boundary[int(nx_p)] = 1.0
K_n = particle_matrix(nx_n, nr_n, radius_n, diffusivity_n)
K_p = particle_matrix(nx_p, nr_p, radius_p, diffusivity_p)
B_n = particle_source(nx_n, nr_n, radius_n, faraday)
B_p = particle_source(nx_p, nr_p, radius_p, faraday)
S_n = surface_matrix(nx_n, nr_n)
S_p = surface_matrix(nx_p, nr_p)
cn0 = torch.stack([(initial_n + (0.0 * k)) for _fi_k in range(int(ns_n)) for k in [torch.tensor(float(_fi_k), device=DEVICE)]])
cp0 = torch.stack([(initial_p + (0.0 * k)) for _fi_k in range(int(ns_p)) for k in [torch.tensor(float(_fi_k), device=DEVICE)]])
ce0 = torch.stack([(initial_e + (0.0 * k)) for _fi_k in range(int(ne)) for k in [torch.tensor(float(_fi_k), device=DEVICE)]])
x0 = torch.cat([cn0, cp0, ce0, torch.tensor([0.0], device=DEVICE)])
un0 = ocp_n(torch.stack([torch.as_tensor((initial_n / cmax_n))]))
u_n_initial = un0[int(0)]
up0 = ocp_p(torch.stack([torch.as_tensor((initial_p / cmax_p))]))
u_p_initial = up0[int(0)]
i0n_ref = exchange_n(torch.stack([torch.as_tensor(initial_e)]), torch.stack([torch.as_tensor(initial_n)]))
i0p_ref = exchange_p(torch.stack([torch.as_tensor(initial_e)]), torch.stack([torch.as_tensor(initial_p)]))
eta_n_reference = ((((2.0 * gas_constant) * temperature) / faraday) * asinh_scalar((current_nominal / ((((2.0 * area) * length_n) * a_n) * i0n_ref[int(0)]))))
eta_p_reference = ((((2.0 * gas_constant) * temperature) / faraday) * asinh_scalar(((-current_nominal) / ((((2.0 * area) * length_p) * a_p) * i0p_ref[int(0)]))))
ref_p = (((u_p_initial - u_n_initial) + eta_p_reference) - eta_n_reference)
ref_e = ((-u_n_initial) - eta_n_reference)
z0 = torch.zeros(int(na), device=DEVICE)
sxn = torch.stack([(cmax_n + (0.0 * k)) for _fi_k in range(int(ns_n)) for k in [torch.tensor(float(_fi_k), device=DEVICE)]])
sxp = torch.stack([(cmax_p + (0.0 * k)) for _fi_k in range(int(ns_p)) for k in [torch.tensor(float(_fi_k), device=DEVICE)]])
sxe = torch.stack([(initial_e + (0.0 * k)) for _fi_k in range(int(ne)) for k in [torch.tensor(float(_fi_k), device=DEVICE)]])
x_scale = torch.cat([sxn, sxp, sxe, torch.tensor([5.0], device=DEVICE)])
szn = torch.stack([(0.0001 + (0.0 * k)) for _fi_k in range(int(nx_n)) for k in [torch.tensor(float(_fi_k), device=DEVICE)]])
szp = torch.stack([(0.05 + (0.0 * k)) for _fi_k in range(int(nx_p)) for k in [torch.tensor(float(_fi_k), device=DEVICE)]])
sze = torch.stack([(0.05 + (0.0 * k)) for _fi_k in range(int(ne)) for k in [torch.tensor(float(_fi_k), device=DEVICE)]])
z_scale = torch.cat([szn, szp, sze])
gsn = torch.stack([((((length_total ** 2) * 5.0) / (area * width_n[int(k)])) + (0.0 * k)) for _fi_k in range(int(nx_n)) for k in [torch.tensor(float(_fi_k), device=DEVICE)]])
gsp = torch.stack([((((length_total ** 2) * 5.0) / (area * width_p[int(k)])) + (0.0 * k)) for _fi_k in range(int(nx_p)) for k in [torch.tensor(float(_fi_k), device=DEVICE)]])
gse = torch.stack([((((length_total ** 2) * 5.0) / (area * width[int(k)])) + (0.0 * k)) for _fi_k in range(int(ne)) for k in [torch.tensor(float(_fi_k), device=DEVICE)]])
g_scale = torch.cat([gsn, gsp, gse])
wn = particle_inventory(nx_n, nr_n, radius_n, (length_n / nx_n), active_n)
wp = particle_inventory(nx_p, nr_p, radius_p, (length_p / nx_p), active_p)
inventory_weights = torch.cat([wn, wp, (eps * width), torch.tensor([0.0], device=DEVICE)])
nb = (nd + na)
solver_scale = torch.cat([x_scale, z_scale])
newton_tolerance = 2e-06
algebraic_tolerance = 0.002
relative_tolerance = 0.0001
absolute_tolerance = 1e-06
nf = (((1 + nx_n) + nx_p) + na)
trajectory = integrate_dfn()
prediction_fields = trajectory_fields(trajectory)
inventory_drift = lithium_inventory_drift(trajectory)
if trajectory[int(0), int(1)] == 1.0:
    diagnostic_label = 'Predictions: [time (s), voltage (V), discharged capacity (Ah)]'
    print(diagnostic_label)
    print(prediction_values(trajectory, prediction_fields, 1.0))
    print(prediction_values(trajectory, prediction_fields, 3.0))
    print(prediction_values(trajectory, prediction_fields, 31.0))
    print(prediction_values(trajectory, prediction_fields, 61.0))
    diagnostic_label = 'Cutoff prediction: [time (s), voltage (V), discharged capacity (Ah)]'
    print(diagnostic_label)
    print(prediction_values(trajectory, prediction_fields, trajectory[int(0), int(0)]))
else:
    diagnostic_label = 'DFN solve did not reach cutoff; solver status:'
    print(diagnostic_label)
print(print(trajectory[int(0), int(1)]))