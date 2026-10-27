import torch
import torch.nn as nn
import torch.optim as optim
from physika.runtime import DEVICE

from physika.runtime import print

# === Functions ===
def asinh_scalar(q):
    return torch.log((q + torch.sqrt(((q * q) + 1.0) if isinstance(((q * q) + 1.0), torch.Tensor) else torch.tensor(float(((q * q) + 1.0))))) if isinstance((q + torch.sqrt(((q * q) + 1.0) if isinstance(((q * q) + 1.0), torch.Tensor) else torch.tensor(float(((q * q) + 1.0))))), torch.Tensor) else torch.tensor(float((q + torch.sqrt(((q * q) + 1.0) if isinstance(((q * q) + 1.0), torch.Tensor) else torch.tensor(float(((q * q) + 1.0))))))))

def phase_power(phase):
    if (1.0 * phase) == 1.0:
        return 12.0
    return 30.0

def phase_end(phase):
    if (1.0 * phase) == 0.0:
        return 60.0
    if (1.0 * phase) == 1.0:
        return 1860.0
    return 1920.0

def basis_values(xi):
    q = xi
    if xi < 0.0:
        q = (0.0 * xi)
    if xi > 1.0:
        q = (1.0 + (0.0 * xi))
    span = 0
    if q >= (1.0 / 3.0):
        span = 1
    if q >= (2.0 / 3.0):
        span = 2
    u = ((3.0 * q) - span)
    return (torch.stack([torch.as_tensor(1.0), torch.as_tensor(u), torch.as_tensor((u * u)), torch.as_tensor(((u * u) * u))]) @ spline_coefficients[int(span)])

def basis_derivative(xi):
    span = 0
    if xi >= (1.0 / 3.0):
        span = 1
    if xi >= (2.0 / 3.0):
        span = 2
    u = ((3.0 * xi) - span)
    return (torch.stack([torch.as_tensor(0.0), torch.as_tensor(3.0), torch.as_tensor((6.0 * u)), torch.as_tensor(((9.0 * u) * u))]) @ spline_coefficients[int(span)])

def spline_basis(ce):
    size = get_1d_array_length(ce)
    result = torch.zeros(int(size), int(6), device=DEVICE)
    q = ((3.0 * (ce - 200.0)) / 2300.0)
    u0 = minimum(maximum(q, 0.0), 1.0)
    u1 = minimum(maximum((q - 1.0), 0.0), 1.0)
    u2 = minimum(maximum((q - 2.0), 0.0), 1.0)
    for k in range(int(0), int(6)):
        result[:, int(k)] = (((spline_coefficients[int(0), int(0), int(k)] + (u0 * (spline_coefficients[int(0), int(1), int(k)] + (u0 * (spline_coefficients[int(0), int(2), int(k)] + (u0 * spline_coefficients[int(0), int(3), int(k)])))))) + (u1 * (spline_coefficients[int(1), int(1), int(k)] + (u1 * (spline_coefficients[int(1), int(2), int(k)] + (u1 * spline_coefficients[int(1), int(3), int(k)])))))) + (u2 * (spline_coefficients[int(2), int(1), int(k)] + (u2 * (spline_coefficients[int(2), int(2), int(k)] + (u2 * spline_coefficients[int(2), int(3), int(k)]))))))
    return result

def property_curve(basis, coefficients):
    size = get_2d_array_num_rows(basis)
    column_shape = [spline_size, one_column]
    vector_shape = [size]
    column = torch.reshape(coefficients, (column_shape,) if isinstance(column_shape, int) else column_shape)
    values = (basis @ column)
    return torch.reshape(values, (vector_shape,) if isinstance(vector_shape, int) else vector_shape)

def modification_matrix():
    result = torch.zeros(int(6), int(6), device=DEVICE)
    b = torch.zeros(int(6), device=DEVICE)
    d = torch.zeros(int(6), device=DEVICE)
    xi = 0.0
    for span in range(int(0), int(3)):
        for point in range(int(0), int(4)):
            xi = ((span + (0.5 * (quadrature_nodes[int(point)] + 1.0))) / 3.0)
            b = basis_values(xi)
            d = basis_derivative(xi)
            for row in range(int(0), int(6)):
                result[int(row), :] = (result[int(row), :] + ((quadrature_weights[int(point)] / 18.0) * ((b[int(row)] * b) + (((roughness_length ** 2) * d[int(row)]) * d))))
    return result

def property_budget(theta):
    return ((((theta[int(0):int(6)] @ budget_matrix) @ theta[int(0):int(6)]) + ((theta[int(6):int(12)] @ budget_matrix) @ theta[int(6):int(12)])) + ((theta[int(12)] ** 2) / 3.0))

def project_budget(v):
    identity = budget_identity
    hessian = budget_hessian
    target = ((modification_budget ** 2) * (1.0 - 8e-07))
    result = v
    low = 0.0
    high = 1.0
    middle = 0.0
    candidate = v
    if property_budget(v) <= target:
        return torch.cat([v, torch.tensor([1.0], device=DEVICE)])
    result = gaussian_solve((identity + ((2.0 * high) * hessian)), v)
    for bracket in range(int(0), int(32)):
        if property_budget(result) > target:
            high = (2.0 * high)
            result = gaussian_solve((identity + ((2.0 * high) * hessian)), v)
    if property_budget(result) > target:
        return torch.cat([result, torch.tensor([0.0], device=DEVICE)])
    for search in range(int(0), int(32)):
        middle = (0.5 * (low + high))
        candidate = gaussian_solve((identity + ((2.0 * middle) * hessian)), v)
        if property_budget(candidate) > target:
            low = middle
        else:
            high = middle
            result = candidate
    return torch.cat([result, torch.tensor([1.0], device=DEVICE)])

def project_design(v):
    x = v
    y = minimum(maximum(v, (-1.0)), 1.0)
    old = v
    box_correction = torch.zeros(int(n_parameters), device=DEVICE)
    budget_correction = torch.zeros(int(n_parameters), device=DEVICE)
    ellipse = torch.zeros(int((n_parameters + 1)), device=DEVICE)
    change = 1.0
    live = 1.0
    status = 0.0
    if property_budget(y) <= ((modification_budget ** 2) * (1.0 - 8e-07)):
        return torch.cat([y, torch.tensor([1.0], device=DEVICE)])
    for cycle in range(int(0), int(512)):
        if live > 0.0:
            old = x
            y = minimum(maximum((x + box_correction), (-1.0)), 1.0)
            box_correction = ((x + box_correction) - y)
            ellipse = project_budget((y + budget_correction))
            x = ellipse[int(0):int(n_parameters)]
            budget_correction = ((y + budget_correction) - x)
            change = max(max_norm((x - old), n_parameters), max_norm((x - y), n_parameters))
            if ellipse[int(n_parameters)] < 1.0:
                live = 0.0
            if change <= 2e-07:
                live = 0.0
                status = 1.0
    x = minimum(maximum(x, (-1.0)), 1.0)
    if property_budget(x) > ((modification_budget ** 2) * (1.0 - 8e-07)):
        x = (x * torch.sqrt((((modification_budget ** 2) * (1.0 - 8e-07)) / property_budget(x)) if isinstance((((modification_budget ** 2) * (1.0 - 8e-07)) / property_budget(x)), torch.Tensor) else torch.tensor(float((((modification_budget ** 2) * (1.0 - 8e-07)) / property_budget(x))))))
    return torch.cat([x, torch.stack([torch.as_tensor(status)])])

def constrained_stationarity(theta, gradient):
    hessian = budget_hessian
    projection = project_design((theta - gradient))
    budget_gradient = torch.zeros(int(n_parameters), device=DEVICE)
    for row in range(int(0), int(n_parameters)):
        budget_gradient[int(row)] = (2.0 * torch.sum((hessian[int(row), :] * theta) if isinstance((hessian[int(row), :] * theta), torch.Tensor) else torch.tensor(float((hessian[int(row), :] * theta)))))
    free_gradient = (gradient + torch.zeros(int(n_parameters), device=DEVICE))
    free_budget = (budget_gradient + torch.zeros(int(n_parameters), device=DEVICE))
    multiplier = 0.0
    slack = ((modification_budget ** 2) - property_budget(theta))
    residual = torch.zeros(int(n_parameters), device=DEVICE)
    for parameter in range(int(0), int(n_parameters)):
        if torch.abs(theta[int(parameter)] if isinstance(theta[int(parameter)], torch.Tensor) else torch.tensor(float(theta[int(parameter)]))) >= (1.0 - 1e-06):
            free_gradient[int(parameter)] = 0.0
            free_budget[int(parameter)] = 0.0
    if slack <= 1e-06:
        if (free_budget @ free_budget) > 1e-12:
            multiplier = max(0.0, ((-(free_gradient @ free_budget)) / (free_budget @ free_budget)))
    residual = (theta - minimum(maximum(((theta - gradient) - (multiplier * budget_gradient)), (-1.0)), 1.0))
    return torch.stack([torch.as_tensor(max_norm((theta - projection[int(0):int(n_parameters)]), n_parameters)), torch.as_tensor(multiplier), torch.as_tensor(slack), torch.as_tensor(max_norm(residual, n_parameters)), torch.as_tensor(torch.abs((multiplier * slack) if isinstance((multiplier * slack), torch.Tensor) else torch.tensor(float((multiplier * slack))))), torch.as_tensor(projection[int(n_parameters)])])

def design_diffusivity(ce, theta):
    bounded = minimum(maximum(ce, 200.0), 2500.0)
    basis = spline_basis(bounded)
    return (electrolyte_diffusivity(bounded) * torch.exp((torch.log(2.0 if isinstance(2.0, torch.Tensor) else torch.tensor(float(2.0))) * property_curve(basis, theta[int(0):int(6)])) if isinstance((torch.log(2.0 if isinstance(2.0, torch.Tensor) else torch.tensor(float(2.0))) * property_curve(basis, theta[int(0):int(6)])), torch.Tensor) else torch.tensor(float((torch.log(2.0 if isinstance(2.0, torch.Tensor) else torch.tensor(float(2.0))) * property_curve(basis, theta[int(0):int(6)]))))))

def design_conductivity(ce, theta):
    bounded = minimum(maximum(ce, 200.0), 2500.0)
    basis = spline_basis(bounded)
    return (electrolyte_conductivity(bounded) * torch.exp((torch.log(2.0 if isinstance(2.0, torch.Tensor) else torch.tensor(float(2.0))) * property_curve(basis, theta[int(6):int(12)])) if isinstance((torch.log(2.0 if isinstance(2.0, torch.Tensor) else torch.tensor(float(2.0))) * property_curve(basis, theta[int(6):int(12)])), torch.Tensor) else torch.tensor(float((torch.log(2.0 if isinstance(2.0, torch.Tensor) else torch.tensor(float(2.0))) * property_curve(basis, theta[int(6):int(12)]))))))

def design_transference(theta):
    return (t_plus + (0.1 * theta[int(12)]))

def positive_profile(z):
    return torch.cat([z[int(nx_n):int(((nx_n + nx_p) - 1))], torch.tensor([0.0], device=DEVICE)])

def ocp_p_pair(q):
    a = (15.789 * (q - 0.3117))
    delta = ((0.1418 * (q - 0.3117)) - 0.00477924)
    ta = tanh(a)
    tb = tanh((a + delta))
    return (((-0.1484) * ta) + ((17.5842 * tanh(delta)) * (1.0 - (ta * tb))))

def ocp_p_change(theta):
    q = maximum(minimum(theta, (1.0 - 1e-10)), 1e-10)
    return (((((((-0.809) * (q - positive_initial_stoich)) - (0.0428 * (tanh((18.5138 * (q - 0.5542))) - positive_small_reference[int(0)]))) + ocp_p_pair(q)) - positive_pair_reference[int(0)]) + ocp_barrier(theta)) - positive_barrier_reference[int(0)])

def reaction_overpotential_n(x, z):
    cs = (S_n @ x[int(0):int(ns_n)])
    return (((z[int(0):int(nx_n)] - z[int((nx_n + nx_p)):int(((nx_n + nx_p) + nx_n))]) + eta_n_reference) + (u_n_initial - ocp_n((cs / cmax_n))))

def reaction_overpotential_p(x, z):
    cs = (S_p @ x[int(ns_n):int((ns_n + ns_p))])
    return ((((positive_profile(z) - z[int((((nx_n + nx_p) + nx_n) + nx_s)):int(nv)]) + z[int(((nx_n + nx_p) - 1))]) + eta_p_reference) - ocp_p_change((cs / cmax_p)))

def reaction_n(t, x, z):
    cs = (S_n @ x[int(0):int(ns_n)])
    ce = x[int((ns_n + ns_p)):int(((ns_n + ns_p) + nx_n))]
    T = cell_temperature(x)
    return ((2.0 * exchange_n_thermal(ce, cs, T)) * sinh(((faraday * reaction_overpotential_n(x, z)) / ((2.0 * gas_constant) * T))))

def reaction_p(t, x, z):
    cs = (S_p @ x[int(ns_n):int((ns_n + ns_p))])
    ce = x[int((((ns_n + ns_p) + nx_n) + nx_s)):int(((ns_n + ns_p) + ne))]
    T = cell_temperature(x)
    return ((2.0 * exchange_p_thermal(ce, cs, T)) * sinh(((faraday * reaction_overpotential_p(x, z)) / ((2.0 * gas_constant) * T))))

def electrolyte_current(t, x, z, theta):
    ce = x[int((ns_n + ns_p)):int(((ns_n + ns_p) + ne))]
    phi_e = z[int((nx_n + nx_p)):int(nv)]
    conductivity = (A_e @ (transport * design_conductivity(ce, theta)))
    chi = ((((2.0 * (1.0 - design_transference(theta))) * gas_constant) * cell_temperature(x)) / (faraday * maximum(ce, 0.01)))
    return (conductivity * ((harmonic_edges(chi, H_e, ne) * gradient_values(ce, width, ne)) - gradient_values(phi_e, width, ne)))

def dfn_rhs(t, x, z, theta):
    jn = reaction_n(t, x, z)
    jp = reaction_p(t, x, z)
    ce = x[int((ns_n + ns_p)):int(((ns_n + ns_p) + ne))]
    aj = torch.cat([(a_n * jn), separator_zero, (a_p * jp)])
    flux = (((-harmonic_edges((transport * design_diffusivity(ce, theta)), H_e, ne)) * gradient_values(ce, width, ne)) + ((design_transference(theta) * electrolyte_current(t, x, z, theta)) / faraday))
    dcn = ((K_n @ (x[int(0):int(ns_n)] - initial_n)) + (B_n @ jn))
    dcp = ((K_p @ (x[int(ns_n):int((ns_n + ns_p))] - initial_p)) + (B_p @ jp))
    dce = (((-divergence_values(flux, width, ne)) + (aj / faraday)) / eps)
    return torch.cat([dcn, dcp, dce, torch.stack([torch.as_tensor(thermal_rate(x, z, theta)), torch.as_tensor((z[int(nv)] / 3600.0))])])

def dfn_algebraic(t, x, z, theta):
    jn = reaction_n(t, x, z)
    jp = reaction_p(t, x, z)
    aj = torch.cat([(a_n * jn), separator_zero, (a_p * jp)])
    phin = z[int(0):int(nx_n)]
    phip = positive_profile(z)
    gradn = gradient_values(phin, width_n, nx_n)
    gradn[int(0)] = ((2.0 * phin[int(0)]) / width_n[int(0)])
    isn = ((-sigma_n) * gradn)
    boundary = positive_boundary
    isp = (((-sigma_p) * gradient_values(phip, width_p, nx_p)) + ((z[int(nv)] / area) * boundary))
    gn = ((length_total ** 2) * (divergence_values(isn, width_n, nx_n) + (a_n * jn)))
    gp = ((length_total ** 2) * (divergence_values(isp, width_p, nx_p) + (a_p * jp)))
    ge = ((length_total ** 2) * (divergence_values(electrolyte_current(t, x, z, theta), width, ne) - aj))
    return torch.cat([gn, gp, ge])

def dfn_voltage(t, x, z):
    return ((ref_p + z[int(((nx_n + nx_p) - 1))]) - (0.5 * z[int(((nx_n + nx_p) - 2))]))

def physical_potentials(z):
    return torch.cat([z[int(0):int(nx_n)], ((ref_p + z[int(((nx_n + nx_p) - 1))]) + positive_profile(z)), (ref_e + z[int((nx_n + nx_p)):int(nv)])])

def surface_n(x):
    return (S_n @ x[int(0):int(ns_n)])

def surface_p(x):
    return (S_p @ x[int(ns_n):int((ns_n + ns_p))])

def algebraic_residual(x, z, theta, power):
    charge = ((dfn_algebraic(0.0, x, z, theta) / g_scale) * (newton_tolerance / algebraic_tolerance))
    return torch.cat([charge, torch.stack([torch.as_tensor((((z[int(nv)] * dfn_voltage(0.0, x, z)) - power) / 30.0))])])

def smooth_max(values, size):
    shift = values[int(0)]
    for k in range(int(0), int(size)):
        if values[int(k)] > shift:
            shift = values[int(k)]
    return (shift + (torch.log(torch.sum(torch.exp((constraint_sharpness * (values - shift)) if isinstance((constraint_sharpness * (values - shift)), torch.Tensor) else torch.tensor(float((constraint_sharpness * (values - shift))))) if isinstance(torch.exp((constraint_sharpness * (values - shift)) if isinstance((constraint_sharpness * (values - shift)), torch.Tensor) else torch.tensor(float((constraint_sharpness * (values - shift))))), torch.Tensor) else torch.tensor(float(torch.exp((constraint_sharpness * (values - shift)) if isinstance((constraint_sharpness * (values - shift)), torch.Tensor) else torch.tensor(float((constraint_sharpness * (values - shift)))))))) if isinstance(torch.sum(torch.exp((constraint_sharpness * (values - shift)) if isinstance((constraint_sharpness * (values - shift)), torch.Tensor) else torch.tensor(float((constraint_sharpness * (values - shift))))) if isinstance(torch.exp((constraint_sharpness * (values - shift)) if isinstance((constraint_sharpness * (values - shift)), torch.Tensor) else torch.tensor(float((constraint_sharpness * (values - shift))))), torch.Tensor) else torch.tensor(float(torch.exp((constraint_sharpness * (values - shift)) if isinstance((constraint_sharpness * (values - shift)), torch.Tensor) else torch.tensor(float((constraint_sharpness * (values - shift)))))))), torch.Tensor) else torch.tensor(float(torch.sum(torch.exp((constraint_sharpness * (values - shift)) if isinstance((constraint_sharpness * (values - shift)), torch.Tensor) else torch.tensor(float((constraint_sharpness * (values - shift))))) if isinstance(torch.exp((constraint_sharpness * (values - shift)) if isinstance((constraint_sharpness * (values - shift)), torch.Tensor) else torch.tensor(float((constraint_sharpness * (values - shift))))), torch.Tensor) else torch.tensor(float(torch.exp((constraint_sharpness * (values - shift)) if isinstance((constraint_sharpness * (values - shift)), torch.Tensor) else torch.tensor(float((constraint_sharpness * (values - shift))))))))))) / constraint_sharpness))

def solid_stoichiometry(x):
    return torch.cat([(x[int(0):int(ns_n)] / cmax_n), (x[int(ns_n):int((ns_n + ns_p))] / cmax_p), (surface_n(x) / cmax_n), (surface_p(x) / cmax_p)])

def node_constraints(w):
    x = w[int(0):int(nd)]
    z = w[int(nd):int(nb)]
    ce = x[int((ns_n + ns_p)):int(((ns_n + ns_p) + ne))]
    solids = solid_stoichiometry(x)
    return torch.stack([torch.as_tensor((minimum_voltage - dfn_voltage(0.0, x, z))), torch.as_tensor(((z[int(nv)] - 15.0) / 15.0)), torch.as_tensor(smooth_max(((200.0 - ce) / 1000.0), ne)), torch.as_tensor(smooth_max(((ce - 2500.0) / 1000.0), ne)), torch.as_tensor(smooth_max((0.01 - solids), (((ns_n + ns_p) + nx_n) + nx_p))), torch.as_tensor(smooth_max((solids - 0.99), (((ns_n + ns_p) + nx_n) + nx_p)))])

def mission_constraints(theta, tape):
    groups = torch.zeros(int(8), device=DEVICE)
    local = node_constraints(tape[int(1), int(6):int(tape_columns)])
    shift = 0.0
    groups[int(0):int(6)] = local
    for node in range(int(2), int((tape[int(0), int(0)] + 1.0))):
        local = node_constraints(tape[int(node), int(6):int(tape_columns)])
        for family in range(int(0), int(6)):
            shift = max(groups[int(family)], local[int(family)])
            groups[int(family)] = (shift + (torch.log((torch.exp((constraint_sharpness * (groups[int(family)] - shift)) if isinstance((constraint_sharpness * (groups[int(family)] - shift)), torch.Tensor) else torch.tensor(float((constraint_sharpness * (groups[int(family)] - shift))))) + torch.exp((constraint_sharpness * (local[int(family)] - shift)) if isinstance((constraint_sharpness * (local[int(family)] - shift)), torch.Tensor) else torch.tensor(float((constraint_sharpness * (local[int(family)] - shift)))))) if isinstance((torch.exp((constraint_sharpness * (groups[int(family)] - shift)) if isinstance((constraint_sharpness * (groups[int(family)] - shift)), torch.Tensor) else torch.tensor(float((constraint_sharpness * (groups[int(family)] - shift))))) + torch.exp((constraint_sharpness * (local[int(family)] - shift)) if isinstance((constraint_sharpness * (local[int(family)] - shift)), torch.Tensor) else torch.tensor(float((constraint_sharpness * (local[int(family)] - shift)))))), torch.Tensor) else torch.tensor(float((torch.exp((constraint_sharpness * (groups[int(family)] - shift)) if isinstance((constraint_sharpness * (groups[int(family)] - shift)), torch.Tensor) else torch.tensor(float((constraint_sharpness * (groups[int(family)] - shift))))) + torch.exp((constraint_sharpness * (local[int(family)] - shift)) if isinstance((constraint_sharpness * (local[int(family)] - shift)), torch.Tensor) else torch.tensor(float((constraint_sharpness * (local[int(family)] - shift))))))))) / constraint_sharpness))
    groups[int(6)] = ((tape[int(tape[int(0), int(0)]), int(((6 + nd) - 1))] / 5.0) - 0.8)
    groups[int(7)] = ((property_budget(theta) / (modification_budget ** 2)) - 1.0)
    return groups

def mission_merit(theta, tape, groups):
    return (heat_objective(theta, tape) + (penalty_weight * torch.sum((maximum(groups, 0.0) ** 2) if isinstance((maximum(groups, 0.0) ** 2), torch.Tensor) else torch.tensor(float((maximum(groups, 0.0) ** 2))))))

def local_penalty(u, groups, weights):
    local = node_constraints((u * solver_scale))
    return (torch.sum((weights * torch.exp((constraint_sharpness * (local - groups[int(0):int(6)])) if isinstance((constraint_sharpness * (local - groups[int(0):int(6)])), torch.Tensor) else torch.tensor(float((constraint_sharpness * (local - groups[int(0):int(6)])))))) if isinstance((weights * torch.exp((constraint_sharpness * (local - groups[int(0):int(6)])) if isinstance((constraint_sharpness * (local - groups[int(0):int(6)])), torch.Tensor) else torch.tensor(float((constraint_sharpness * (local - groups[int(0):int(6)])))))), torch.Tensor) else torch.tensor(float((weights * torch.exp((constraint_sharpness * (local - groups[int(0):int(6)])) if isinstance((constraint_sharpness * (local - groups[int(0):int(6)])), torch.Tensor) else torch.tensor(float((constraint_sharpness * (local - groups[int(0):int(6)]))))))))) / constraint_sharpness)

def budget_penalty(theta):
    violation = ((property_budget(theta) / (modification_budget ** 2)) - 1.0)
    positive = maximum(torch.stack([torch.as_tensor(violation)]), 0.0)
    return (penalty_weight * (positive[int(0)] ** 2))

def mission_metrics(theta, tape):
    result = torch.tensor([0.0, 1000000.0, (-1000000.0), 1000000.0, (-1000000.0), 1000000.0, (-1000000.0), 0.0, 0.0, 0.0, 0.0, 1.0, 1000000.0, (-1000000.0)], device=DEVICE)
    w = torch.zeros(int(nb), device=DEVICE)
    x = torch.zeros(int(nd), device=DEVICE)
    z = torch.zeros(int(na), device=DEVICE)
    solids = torch.zeros(int((((ns_n + ns_p) + nx_n) + nx_p)), device=DEVICE)
    voltage = 0.0
    coefficient_norm = max_norm(theta, n_parameters)
    budget_limit = (modification_budget ** 2)
    voltage_limit = minimum_voltage
    inventory = (inventory_weights @ x0)
    result[int(0)] = tape[int(tape[int(0), int(0)]), int(((6 + nd) - 1))]
    result[int(7)] = property_budget(theta)
    for node in range(int(1), int((tape[int(0), int(0)] + 1.0))):
        w = tape[int(node), int(6):int(tape_columns)]
        x = w[int(0):int(nd)]
        z = w[int(nd):int(nb)]
        voltage = dfn_voltage(0.0, x, z)
        solids = solid_stoichiometry(x)
        result[int(1)] = min(result[int(1)], voltage)
        result[int(2)] = max(result[int(2)], z[int(nv)])
        result[int(12)] = min(result[int(12)], z[int(nv)])
        result[int(13)] = max(result[int(13)], voltage)
        for cell in range(int(0), int(ne)):
            result[int(3)] = min(result[int(3)], x[int(((ns_n + ns_p) + cell))])
            result[int(4)] = max(result[int(4)], x[int(((ns_n + ns_p) + cell))])
        for cell in range(int(0), int((((ns_n + ns_p) + nx_n) + nx_p))):
            result[int(5)] = min(result[int(5)], solids[int(cell)])
            result[int(6)] = max(result[int(6)], solids[int(cell)])
        result[int(8)] = max(result[int(8)], (torch.abs(((inventory_weights @ x) - inventory) if isinstance(((inventory_weights @ x) - inventory), torch.Tensor) else torch.tensor(float(((inventory_weights @ x) - inventory)))) / inventory))
        result[int(9)] = max(result[int(9)], max_norm((dfn_algebraic(0.0, x, z, theta) / g_scale), nv))
        result[int(10)] = max(result[int(10)], (torch.abs(((z[int(nv)] * voltage) - tape[int(node), int(5)]) if isinstance(((z[int(nv)] * voltage) - tape[int(node), int(5)]), torch.Tensor) else torch.tensor(float(((z[int(nv)] * voltage) - tape[int(node), int(5)])))) / tape[int(node), int(5)]))
        if finite_values(w, nb) < 1.0:
            result[int(11)] = 0.0
    if tape[int(0), int(1)] < 1.0:
        result[int(11)] = 0.0
    if torch.abs((tape[int(tape[int(0), int(0)]), int(0)] - 1920.0) if isinstance((tape[int(tape[int(0), int(0)]), int(0)] - 1920.0), torch.Tensor) else torch.tensor(float((tape[int(tape[int(0), int(0)]), int(0)] - 1920.0)))) > 0.001:
        result[int(11)] = 0.0
    if result[int(0)] > 4.0:
        result[int(11)] = 0.0
    if result[int(1)] < voltage_limit:
        result[int(11)] = 0.0
    if result[int(2)] > 15.0:
        result[int(11)] = 0.0
    if result[int(12)] <= 0.0:
        result[int(11)] = 0.0
    if result[int(3)] < 200.0:
        result[int(11)] = 0.0
    if result[int(4)] > 2500.0:
        result[int(11)] = 0.0
    if result[int(5)] < 0.01:
        result[int(11)] = 0.0
    if result[int(6)] > 0.99:
        result[int(11)] = 0.0
    if result[int(7)] > budget_limit:
        result[int(11)] = 0.0
    if coefficient_norm > 1.0:
        result[int(11)] = 0.0
    if result[int(8)] > 0.0001:
        result[int(11)] = 0.0
    if result[int(9)] > 0.002:
        result[int(11)] = 0.0
    if result[int(10)] > 1e-05:
        result[int(11)] = 0.0
    return result

def optimize_mission(theta0, grid, initial_tape):
    output = torch.zeros(int((optimization_steps + 2)), int(history_columns), device=DEVICE)
    theta = theta0
    best = theta0
    trial = theta0
    gradient = torch.zeros(int(n_parameters), device=DEVICE)
    projected = torch.zeros(int(n_parameters), device=DEVICE)
    adjoint = torch.zeros(int((n_parameters + 2)), device=DEVICE)
    trial_adjoint = torch.zeros(int((n_parameters + 2)), device=DEVICE)
    stationary = torch.zeros(int(6), device=DEVICE)
    trial_stationary = torch.zeros(int(6), device=DEVICE)
    cached_gradient = 0.0
    displacement = torch.zeros(int(n_parameters), device=DEVICE)
    gradient_change = torch.zeros(int(n_parameters), device=DEVICE)
    secant_curvature = 0.0
    projection = torch.zeros(int((n_parameters + 1)), device=DEVICE)
    tape = initial_tape
    candidate = torch.zeros(int(max_nodes), int(tape_columns), device=DEVICE)
    groups = mission_constraints(theta, tape)
    trial_groups = torch.zeros(int(8), device=DEVICE)
    metrics = mission_metrics(theta, tape)
    trial_metrics = torch.zeros(int(14), device=DEVICE)
    heat_metrics = thermal_metrics(theta, tape)
    merit = mission_merit(theta, tape, groups)
    trial_merit = 0.0
    best_peak = heat_objective(theta, tape)
    best_merit = merit
    merit_floor = merit
    step = starting_step
    projected_norm = 0.0
    predicted_change = 0.0
    resolution = merit_resolution
    tolerance = stationarity_tolerance
    accepted = 0.0
    updates = 0.0
    forwards = 0.0
    backwards = 0.0
    live = metrics[int(11)]
    status = (-1.0)
    output[int(1), :] = torch.cat([theta, torch.stack([torch.as_tensor(0.0), torch.as_tensor(heat_metrics[int(0)]), torch.as_tensor(merit), torch.as_tensor(step), torch.as_tensor(0.0), torch.as_tensor(metrics[int(11)]), torch.as_tensor(metrics[int(7)]), torch.as_tensor(1.0)])])
    if live > 0.0:
        status = 1.0
    for iteration in range(int(0), int(optimization_steps)):
        if live > 0.0:
            if cached_gradient < 1.0:
                adjoint = heat_adjoint(theta, tape, detach(groups), 1.0)
                backwards = (backwards + 1.0)
            cached_gradient = 0.0
            if adjoint[int(n_parameters)] < 1.0:
                live = 0.0
                status = (-2.0)
            if live > 0.0:
                gradient = detach(adjoint[int(0):int(n_parameters)])
                stationary = constrained_stationarity(theta, gradient)
                projected_norm = max(stationary[int(0)], stationary[int(3)])
                if stationary[int(5)] < 1.0:
                    live = 0.0
                    status = (-3.0)
                if projected_norm <= tolerance:
                    live = 0.0
                    status = (-4.0)
                    if metrics[int(11)] > 0.0:
                        if max_norm(maximum(groups, 0.0), 8) == 0.0:
                            status = 2.0
            if live > 0.0:
                accepted = 0.0
                for search in range(int(0), int(9)):
                    if accepted < 1.0:
                        if live > 0.0:
                            projection = project_design(detach((theta - (step * gradient))))
                            trial = detach(projection[int(0):int(n_parameters)])
                            if projection[int(n_parameters)] < 1.0:
                                diagnostic_label = 'Rejected projection trial: [step, projection status]'
                                print(diagnostic_label)
                                print(torch.stack([torch.as_tensor(step), torch.as_tensor(projection[int(n_parameters)])]))
                            else:
                                predicted_change = (gradient @ (trial - theta))
                                if max_norm((trial - theta), n_parameters) <= 2e-07:
                                    live = 0.0
                                    status = 4.0
                                if predicted_change < 0.0:
                                    if live > 0.0:
                                        candidate = replay_mission(trial, grid)
                                        forwards = (forwards + 1.0)
                                        if candidate[int(0), int(1)] > 0.0:
                                            trial_groups = mission_constraints(trial, candidate)
                                            trial_merit = mission_merit(trial, candidate, trial_groups)
                                            if trial_merit < merit:
                                                if trial_merit <= (merit + (0.0001 * predicted_change)):
                                                    accepted = 1.0
                                            if accepted < 1.0:
                                                if trial_merit <= (merit_floor + resolution):
                                                    trial_metrics = mission_metrics(trial, candidate)
                                                    if trial_metrics[int(11)] > 0.0:
                                                        if max_norm(maximum(trial_groups, 0.0), 8) == 0.0:
                                                            trial_adjoint = heat_adjoint(trial, candidate, detach(trial_groups), 1.0)
                                                            backwards = (backwards + 1.0)
                                                            if trial_adjoint[int(n_parameters)] > 0.0:
                                                                trial_stationary = constrained_stationarity(trial, detach(trial_adjoint[int(0):int(n_parameters)]))
                                                                if trial_stationary[int(5)] > 0.0:
                                                                    if max(trial_stationary[int(0)], trial_stationary[int(3)]) < (0.99 * projected_norm):
                                                                        accepted = 1.0
                                                                    if max(trial_stationary[int(0)], trial_stationary[int(3)]) <= tolerance:
                                                                        accepted = 1.0
                                                                    if accepted > 0.0:
                                                                        adjoint = trial_adjoint
                                                                        cached_gradient = 1.0
                            if accepted < 1.0:
                                step = (step / 2.0)
                if accepted > 0.0:
                    if cached_gradient < 1.0:
                        adjoint = heat_adjoint(trial, candidate, detach(trial_groups), 1.0)
                        backwards = (backwards + 1.0)
                        cached_gradient = 1.0
                    displacement = (trial - theta)
                    gradient_change = (adjoint[int(0):int(n_parameters)] - gradient)
                    theta = trial
                    tape = candidate
                    groups = trial_groups
                    merit = trial_merit
                    metrics = mission_metrics(theta, tape)
                    heat_metrics = thermal_metrics(theta, tape)
                    updates = (updates + 1.0)
                    output[int((updates + 1)), :] = torch.cat([theta, torch.stack([torch.as_tensor(updates), torch.as_tensor(heat_metrics[int(0)]), torch.as_tensor(merit), torch.as_tensor(step), torch.as_tensor(projected_norm), torch.as_tensor(metrics[int(11)]), torch.as_tensor(metrics[int(7)]), torch.as_tensor(1.0)])])
                    print(torch.stack([torch.as_tensor(updates), torch.as_tensor(heat_metrics[int(0)]), torch.as_tensor(merit), torch.as_tensor(step), torch.as_tensor(projected_norm), torch.as_tensor(metrics[int(11)]), torch.as_tensor(metrics[int(7)])]))
                    step = min(maximum_design_step, (1.5 * step))
                    if cached_gradient > 0.0:
                        secant_curvature = (displacement @ gradient_change)
                        if secant_curvature > 1e-12:
                            step = min(maximum_design_step, max(0.1, ((displacement @ displacement) / secant_curvature)))
                    if metrics[int(11)] > 0.0:
                        merit_floor = min(merit_floor, merit)
                        if merit <= (merit_floor + resolution):
                            best = theta
                            best_peak = heat_objective(theta, tape)
                            best_merit = merit
                else:
                    if live > 0.0:
                        live = 0.0
                        status = 3.0
    output[int(0), :] = torch.cat([best, torch.stack([torch.as_tensor(status), torch.as_tensor(updates), torch.as_tensor(best_peak), torch.as_tensor(best_merit), torch.as_tensor(projected_norm), torch.as_tensor(forwards), torch.as_tensor(backwards), torch.as_tensor(metrics[int(11)])])])
    return output

def property_table(theta):
    result = torch.zeros(int(41), int(5), device=DEVICE)
    ce = 200.0
    basis = torch.zeros(int(6), device=DEVICE)
    D = torch.zeros(int(1), device=DEVICE)
    kappa = torch.zeros(int(1), device=DEVICE)
    for row in range(int(0), int(41)):
        ce = (200.0 + ((2300.0 * row) / 40.0))
        basis = basis_values(((ce - 200.0) / 2300.0))
        D = electrolyte_diffusivity(torch.stack([torch.as_tensor(ce)]))
        kappa = electrolyte_conductivity(torch.stack([torch.as_tensor(ce)]))
        result[int(row), :] = torch.stack([torch.as_tensor(ce), torch.as_tensor(D[int(0)]), torch.as_tensor((D[int(0)] * torch.exp((torch.log(2.0 if isinstance(2.0, torch.Tensor) else torch.tensor(float(2.0))) * torch.sum((basis * theta[int(0):int(6)]) if isinstance((basis * theta[int(0):int(6)]), torch.Tensor) else torch.tensor(float((basis * theta[int(0):int(6)]))))) if isinstance((torch.log(2.0 if isinstance(2.0, torch.Tensor) else torch.tensor(float(2.0))) * torch.sum((basis * theta[int(0):int(6)]) if isinstance((basis * theta[int(0):int(6)]), torch.Tensor) else torch.tensor(float((basis * theta[int(0):int(6)]))))), torch.Tensor) else torch.tensor(float((torch.log(2.0 if isinstance(2.0, torch.Tensor) else torch.tensor(float(2.0))) * torch.sum((basis * theta[int(0):int(6)]) if isinstance((basis * theta[int(0):int(6)]), torch.Tensor) else torch.tensor(float((basis * theta[int(0):int(6)])))))))))), torch.as_tensor(kappa[int(0)]), torch.as_tensor((kappa[int(0)] * torch.exp((torch.log(2.0 if isinstance(2.0, torch.Tensor) else torch.tensor(float(2.0))) * torch.sum((basis * theta[int(6):int(12)]) if isinstance((basis * theta[int(6):int(12)]), torch.Tensor) else torch.tensor(float((basis * theta[int(6):int(12)]))))) if isinstance((torch.log(2.0 if isinstance(2.0, torch.Tensor) else torch.tensor(float(2.0))) * torch.sum((basis * theta[int(6):int(12)]) if isinstance((basis * theta[int(6):int(12)]), torch.Tensor) else torch.tensor(float((basis * theta[int(6):int(12)]))))), torch.Tensor) else torch.tensor(float((torch.log(2.0 if isinstance(2.0, torch.Tensor) else torch.tensor(float(2.0))) * torch.sum((basis * theta[int(6):int(12)]) if isinstance((basis * theta[int(6):int(12)]), torch.Tensor) else torch.tensor(float((basis * theta[int(6):int(12)]))))))))))])
    return result

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

def cell_temperature(x):
    return (temperature + x[int(temperature_index)])

def exchange_n_thermal(ce, cs, T):
    q = maximum(minimum(cs, cmax_n), 0.0)
    return ((((6.48e-07 * torch.exp(((35000.0 / gas_constant) * ((1.0 / 298.15) - (1.0 / T))) if isinstance(((35000.0 / gas_constant) * ((1.0 / 298.15) - (1.0 / T))), torch.Tensor) else torch.tensor(float(((35000.0 / gas_constant) * ((1.0 / 298.15) - (1.0 / T))))))) * torch.sqrt(maximum(ce, 0.0) if isinstance(maximum(ce, 0.0), torch.Tensor) else torch.tensor(float(maximum(ce, 0.0))))) * torch.sqrt(q if isinstance(q, torch.Tensor) else torch.tensor(float(q)))) * torch.sqrt((cmax_n - q) if isinstance((cmax_n - q), torch.Tensor) else torch.tensor(float((cmax_n - q)))))

def exchange_p_thermal(ce, cs, T):
    q = maximum(minimum(cs, cmax_p), 0.0)
    return ((((3.42e-06 * torch.exp(((17800.0 / gas_constant) * ((1.0 / 298.15) - (1.0 / T))) if isinstance(((17800.0 / gas_constant) * ((1.0 / 298.15) - (1.0 / T))), torch.Tensor) else torch.tensor(float(((17800.0 / gas_constant) * ((1.0 / 298.15) - (1.0 / T))))))) * torch.sqrt(maximum(ce, 0.0) if isinstance(maximum(ce, 0.0), torch.Tensor) else torch.tensor(float(maximum(ce, 0.0))))) * torch.sqrt(q if isinstance(q, torch.Tensor) else torch.tensor(float(q)))) * torch.sqrt((cmax_p - q) if isinstance((cmax_p - q), torch.Tensor) else torch.tensor(float((cmax_p - q)))))

def edge_average(v, cells):
    return (0.5 * (v[int(0):int(cells)] + v[int(1):int((cells + 1))]))

def heat_components(x, z, theta):
    grad_n = gradient_values(z[int(0):int(nx_n)], width_n, nx_n)
    grad_n[int(0)] = ((2.0 * z[int(0)]) / width_n[int(0)])
    grad_p = gradient_values(positive_profile(z), width_p, nx_p)
    grad_p[int(nx_p)] = ((-z[int(nv)]) / (area * sigma_p))
    gn = edge_average(grad_n, nx_n)
    gp = edge_average(grad_p, nx_p)
    solid = (area * (torch.sum(((width_n * sigma_n) * (gn ** 2)) if isinstance(((width_n * sigma_n) * (gn ** 2)), torch.Tensor) else torch.tensor(float(((width_n * sigma_n) * (gn ** 2))))) + torch.sum(((width_p * sigma_p) * (gp ** 2)) if isinstance(((width_p * sigma_p) * (gp ** 2)), torch.Tensor) else torch.tensor(float(((width_p * sigma_p) * (gp ** 2)))))))
    grad_e = gradient_values(z[int((nx_n + nx_p)):int(nv)], width, ne)
    ionic = electrolyte_current(0.0, x, z, theta)
    electrolyte = ((-area) * torch.sum(((width * edge_average(ionic, ne)) * edge_average(grad_e, ne)) if isinstance(((width * edge_average(ionic, ne)) * edge_average(grad_e, ne)), torch.Tensor) else torch.tensor(float(((width * edge_average(ionic, ne)) * edge_average(grad_e, ne))))))
    reaction = (area * (torch.sum((((width_n * a_n) * reaction_n(0.0, x, z)) * reaction_overpotential_n(x, z)) if isinstance((((width_n * a_n) * reaction_n(0.0, x, z)) * reaction_overpotential_n(x, z)), torch.Tensor) else torch.tensor(float((((width_n * a_n) * reaction_n(0.0, x, z)) * reaction_overpotential_n(x, z))))) + torch.sum((((width_p * a_p) * reaction_p(0.0, x, z)) * reaction_overpotential_p(x, z)) if isinstance((((width_p * a_p) * reaction_p(0.0, x, z)) * reaction_overpotential_p(x, z)), torch.Tensor) else torch.tensor(float((((width_p * a_p) * reaction_p(0.0, x, z)) * reaction_overpotential_p(x, z)))))))
    collectors = (((z[int(nv)] ** 2) / area) * ((collector_length_n / collector_sigma_n) + (collector_length_p / collector_sigma_p)))
    return torch.stack([torch.as_tensor(solid), torch.as_tensor(electrolyte), torch.as_tensor(reaction), torch.as_tensor(collectors), torch.as_tensor(0.0), torch.as_tensor((((solid + electrolyte) + reaction) + collectors))])

def thermal_rate(x, z, theta):
    heat = heat_components(x, z, theta)
    return ((heat[int(5)] - ((cooling_coefficient * cooling_area) * x[int(temperature_index)])) / thermal_capacity)

def local_heat_state(u, theta):
    w = (u * solver_scale)
    heat = heat_components(w[int(0):int(nd)], w[int(nd):int(nb)], theta)
    return (heat[int(5)] / heat_scale)

def local_heat_parameter(theta, u, n_parameters=None, nb=None):
    if n_parameters is None:
        n_parameters = int(theta.shape[0])
    if nb is None:
        nb = int(u.shape[0])
    return local_heat_state(u, theta)

def heat_interval_exponential(v, shift):
    d = (heat_sharpness * (v[int(1)] - v[int(0)]))
    if torch.abs(d if isinstance(d, torch.Tensor) else torch.tensor(float(d))) < 0.05:
        return (torch.exp((heat_sharpness * (v[int(0)] - shift)) if isinstance((heat_sharpness * (v[int(0)] - shift)), torch.Tensor) else torch.tensor(float((heat_sharpness * (v[int(0)] - shift))))) * (((((1.0 + (d / 2.0)) + ((d ** 2) / 6.0)) + ((d ** 3) / 24.0)) + ((d ** 4) / 120.0)) + ((d ** 5) / 720.0)))
    if d > 0.0:
        return ((torch.exp((heat_sharpness * (v[int(1)] - shift)) if isinstance((heat_sharpness * (v[int(1)] - shift)), torch.Tensor) else torch.tensor(float((heat_sharpness * (v[int(1)] - shift))))) * (1.0 - torch.exp((-d) if isinstance((-d), torch.Tensor) else torch.tensor(float((-d)))))) / d)
    return ((torch.exp((heat_sharpness * (v[int(0)] - shift)) if isinstance((heat_sharpness * (v[int(0)] - shift)), torch.Tensor) else torch.tensor(float((heat_sharpness * (v[int(0)] - shift))))) * (1.0 - torch.exp(d if isinstance(d, torch.Tensor) else torch.tensor(float(d))))) / (-d))

def mission_heat_values(theta, tape):
    values = torch.zeros(int(max_nodes), device=DEVICE)
    w = torch.zeros(int(nb), device=DEVICE)
    heat = torch.zeros(int(6), device=DEVICE)
    for node in range(int(1), int((tape[int(0), int(0)] + 1.0))):
        w = tape[int(node), int(6):int(tape_columns)]
        heat = heat_components(w[int(0):int(nd)], w[int(nd):int(nb)], theta)
        values[int(node)] = (heat[int(5)] / heat_scale)
    return values

def heat_objective_weights(theta, tape, objective):
    values = mission_heat_values(theta, tape)
    weights = torch.zeros(int(max_nodes), device=DEVICE)
    pair = torch.zeros(int(2), device=DEVICE)
    derivative = torch.zeros(int(2), device=DEVICE)
    last = tape[int(0), int(0)]
    duration = max(1e-09, (tape[int(last), int(0)] - tape[int(1), int(0)]))
    for node in range(int(2), int((last + 1.0))):
        if tape[int(node), int(1)] > 0.0:
            pair = torch.stack([torch.as_tensor(values[int((node - 1))]), torch.as_tensor(values[int(node)])])
            derivative = ((tape[int(node), int(1)] / (duration * heat_sharpness)) * compute_grad(lambda _dpair: heat_interval_exponential(_dpair, objective), pair))
            weights[int((node - 1))] = (weights[int((node - 1))] + derivative[int(0)])
            weights[int(node)] = (weights[int(node)] + derivative[int(1)])
    if torch.sum(weights if isinstance(weights, torch.Tensor) else torch.tensor(float(weights))) > 0.0:
        weights = (weights / torch.sum(weights if isinstance(weights, torch.Tensor) else torch.tensor(float(weights))))
    return weights

def heat_objective(theta, tape):
    last = tape[int(0), int(0)]
    values = mission_heat_values(theta, tape)
    shift = (-1000000.0)
    total = 0.0
    duration = max(1e-09, (tape[int(last), int(0)] - tape[int(1), int(0)]))
    for node in range(int(1), int((last + 1.0))):
        shift = max(shift, values[int(node)])
    for node in range(int(2), int((last + 1.0))):
        if tape[int(node), int(1)] > 0.0:
            total = (total + (tape[int(node), int(1)] * heat_interval_exponential(torch.stack([torch.as_tensor(values[int((node - 1))]), torch.as_tensor(values[int(node)])]), shift)))
    return (shift + (torch.log((total / duration) if isinstance((total / duration), torch.Tensor) else torch.tensor(float((total / duration)))) / heat_sharpness))

def thermal_metrics(theta, tape):
    result = torch.stack([torch.as_tensor((-1000000.0)), torch.as_tensor(temperature), torch.as_tensor(0.0), torch.as_tensor(0.0), torch.as_tensor(0.0), torch.as_tensor(0.0), torch.as_tensor(0.0), torch.as_tensor(0.0), torch.as_tensor(temperature), torch.as_tensor(0.0), torch.as_tensor(0.0), torch.as_tensor(0.0)])
    w = torch.zeros(int(nb), device=DEVICE)
    x = torch.zeros(int(nd), device=DEVICE)
    heat = torch.zeros(int(6), device=DEVICE)
    increment_heat = 0.0
    increment_cooling = 0.0
    removed = 0.0
    last = tape[int(0), int(0)]
    for node in range(int(1), int((last + 1))):
        w = tape[int(node), int(6):int(tape_columns)]
        x = w[int(0):int(nd)]
        heat = heat_components(x, w[int(nd):int(nb)], theta)
        result[int(1)] = max(result[int(1)], cell_temperature(x))
        if heat[int(5)] > result[int(0)]:
            result[int(0)] = heat[int(5)]
            result[int(7)] = tape[int(node), int(0)]
            result[int(9):int(12)] = heat[int(0):int(3)]
        if tape[int(node), int(1)] > 0.0:
            removed = ((cooling_coefficient * cooling_area) * x[int(temperature_index)])
            increment_heat = (((tape[int(node), int(1)] * heat[int(5)]) + (tape[int(node), int(4)] * increment_heat)) / tape[int(node), int(2)])
            increment_cooling = (((tape[int(node), int(1)] * removed) + (tape[int(node), int(4)] * increment_cooling)) / tape[int(node), int(2)])
            result[int(3)] = (result[int(3)] + increment_heat)
            result[int(4)] = (result[int(4)] + increment_cooling)
        else:
            increment_heat = 0.0
            increment_cooling = 0.0
    result[int(2)] = tape[int(last), int(((6 + nd) - 1))]
    result[int(5)] = (thermal_capacity * (tape[int(last), int((6 + temperature_index))] - tape[int(1), int((6 + temperature_index))]))
    result[int(6)] = torch.abs(((result[int(3)] - result[int(4)]) - result[int(5)]) if isinstance(((result[int(3)] - result[int(4)]) - result[int(5)]), torch.Tensor) else torch.tensor(float(((result[int(3)] - result[int(4)]) - result[int(5)]))))
    result[int(8)] = (temperature + tape[int(last), int((6 + temperature_index))])
    return result

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

def initial_residual(u, x, t, theta, power):
    return algebraic_residual(x, (u * z_scale), theta, power)

def initial_jacobian(u, x, t, theta, power):
    shape = [na, na]
    return torch.reshape(compute_grad(lambda _du: initial_residual(_du, x, t, theta, power), u), (shape,) if isinstance(shape, int) else shape)

def consistent_potentials(x, guess, t, theta, power):
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
            r = initial_residual(u, x, t, theta, power)
            if finite_values(r, na) > 0.0:
                rnorm = max_norm(r, na)
                if rnorm <= tolerance:
                    success = 1.0
                    active = 0.0
                else:
                    if iteration < last_iteration:
                        J = initial_jacobian(u, x, t, theta, power)
                        iterations = (iterations + 1.0)
                        delta = gaussian_solve(J, (-r))
                        if finite_values(delta, na) > 0.0:
                            alpha = 1.0
                            found = 0.0
                            for search in range(int(0), int(16)):
                                if found < 1.0:
                                    trial = initial_residual((u + (alpha * delta)), x, t, theta, power)
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

def step_residual(u, a, history, t, h, alpha, theta, power):
    y = (u * solver_scale)
    increment = y[int(0):int(nd)]
    z = y[int(nd):int(nb)]
    x = (a + increment)
    return torch.cat([((((alpha * increment) + history) - (h * dfn_rhs((t + h), x, z, theta))) / x_scale), algebraic_residual(x, z, theta, power)])

def step_jacobian(u, a, history, t, h, alpha, theta, power):
    shape = [nb, nb]
    return torch.reshape(compute_grad(lambda _du: step_residual(_du, a, history, t, h, alpha, theta, power), u), (shape,) if isinstance(shape, int) else shape)

def step_newton(a, b, history, t, h, alpha, theta, power):
    tolerance = newton_tolerance
    state_tolerance = state_correction_tolerance
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
            r = step_residual(u, a, history, t, h, alpha, theta, power)
            if finite_values(r, nb) > 0.0:
                rnorm = max_norm(r, nb)
                if rnorm <= tolerance:
                    success = 1.0
                    active = 0.0
                else:
                    if iteration < last_iteration:
                        J = step_jacobian(u, a, history, t, h, alpha, theta, power)
                        iterations = (iterations + 1.0)
                        delta = gaussian_solve(J, (-r))
                        if finite_values(delta, nb) > 0.0:
                            gnorm = max_norm(r[int(nd):int(nb)], na)
                            correction = max_norm(delta[int(0):int(nd)], nd)
                            if gnorm <= tolerance:
                                if correction > state_tolerance:
                                    defect = torch.cat([r[int(0):int(nd)], torch.zeros(int(na), device=DEVICE)])
                                    tangent = gaussian_solve(J, (-defect))
                                    if finite_values(tangent, nb) > 0.0:
                                        correction = max_norm(tangent[int(0):int(nd)], nd)
                                    else:
                                        active = 0.0
                                if correction <= state_tolerance:
                                    success = 1.0
                                    active = 0.0
                            if active > 0.0:
                                u = (u + delta)
                        else:
                            active = 0.0
            else:
                active = 0.0
    return torch.cat([(u * solver_scale), torch.stack([torch.as_tensor(success), torch.as_tensor(iterations)])])

def bdf_coefficients(h, previous_h, started):
    ratio = 0.0
    gamma = 0.0
    alpha = 1.0
    if started > 0.0:
        ratio = (h / previous_h)
        gamma = ((ratio ** 2) / (1.0 + ratio))
        alpha = ((1.0 + (2.0 * ratio)) / (1.0 + ratio))
    return torch.stack([torch.as_tensor(alpha), torch.as_tensor((alpha + gamma)), torch.as_tensor(gamma)])

def bdf_step(t, a, b, h, previous, previous_h, started, theta, power):
    alpha = 1.0
    ratio = 0.0
    history = torch.zeros(int(nd), device=DEVICE)
    if started > 0.0:
        ratio = (h / previous_h)
        alpha = ((1.0 + (2.0 * ratio)) / (1.0 + ratio))
        history = (((ratio ** 2) / (1.0 + ratio)) * (previous - a))
    result = step_newton(a, b, history, t, h, alpha, theta, power)
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

def attempt_step(t, x, z, h, previous, previous_h, started, theta, power):
    full = bdf_step(t, x, z, h, previous, previous_h, started, theta, power)
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
        half = bdf_step(t, x, z, (h / 2.0), previous, previous_h, started, theta, power)
        count = (count + half[int((nb + 1))])
        if half[int(nb)] > 0.0:
            small = bdf_step((t + (h / 2.0)), half[int(0):int(nd)], half[int(nd):int(nb)], (h / 2.0), x, (h / 2.0), 1.0, theta, power)
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

def adaptive_mission(theta, limit):
    output = torch.zeros(int(max_nodes), int(tape_columns), device=DEVICE)
    x = x0
    z = z0
    previous = x0
    initial = consistent_potentials(x, z, 0.0, theta, 30.0)
    result = torch.zeros(int((nb + 3)), device=DEVICE)
    coefficients = torch.tensor([1.0, 1.0, 0.0], device=DEVICE)
    saved = 1
    live = initial[int(na)]
    status = (-1.0)
    t = 0.0
    stop = 0.0
    power = 30.0
    h = 0.1
    hh = 0.1
    previous_h = 0.1
    started = 0.0
    accepted = 0.0
    rejected = 0.0
    corrections = initial[int((na + 1))]
    retries = 0
    q_increment = 0.0
    q_sum = 0.0
    q_compensation = 0.0
    q_addend = 0.0
    q_total = 0.0
    z = initial[int(0):int(na)]
    output[int(1), int(0):int(6)] = torch.tensor([0.0, 0.0, 1.0, 1.0, 0.0, 30.0], device=DEVICE)
    output[int(1), int(6):int(tape_columns)] = torch.cat([x, z])
    for phase in range(int(0), int(3)):
        if live > 0.0:
            power = phase_power(phase)
            stop = phase_end(phase)
            if (1.0 * phase) > 0.0:
                initial = consistent_potentials(x, z, t, theta, power)
                corrections = (corrections + initial[int((na + 1))])
                live = initial[int(na)]
                if live > 0.0:
                    saved = (saved + 1)
                    z = initial[int(0):int(na)]
                    output[int(saved), int(0):int(6)] = torch.stack([torch.as_tensor(t), torch.as_tensor(0.0), torch.as_tensor(1.0), torch.as_tensor(1.0), torch.as_tensor(0.0), torch.as_tensor(power)])
                    output[int(saved), int(6):int(tape_columns)] = torch.cat([x, z])
            h = 0.1
            previous_h = 0.1
            previous = x
            started = 0.0
            q_increment = 0.0
            retries = 0
            for attempt in range(int(0), int(3000)):
                if live > 0.0:
                    if t < stop:
                        hh = h
                        if hh > limit:
                            hh = limit
                        if hh > (stop - t):
                            hh = (stop - t)
                        if hh < 1e-09:
                            live = 0.0
                            status = (-2.0)
                        if live > 0.0:
                            result = attempt_step(t, x, z, hh, previous, previous_h, started, theta, power)
                            corrections = (corrections + result[int((nb + 2))])
                            if result[int(nb)] > 0.0:
                                if (1.0 * saved) < (max_nodes - 1):
                                    coefficients = bdf_coefficients(hh, previous_h, started)
                                    saved = (saved + 1)
                                    t = (t + hh)
                                    output[int(saved), int(0):int(6)] = torch.cat([torch.stack([torch.as_tensor(t), torch.as_tensor(hh)]), coefficients, torch.stack([torch.as_tensor(power)])])
                                    previous = x
                                    x = detach(result[int(0):int(nd)])
                                    z = detach(result[int(nd):int(nb)])
                                    q_increment = ((((hh * z[int(nv)]) / 3600.0) + (coefficients[int(2)] * q_increment)) / coefficients[int(0)])
                                    q_addend = (q_increment - q_compensation)
                                    q_total = (q_sum + q_addend)
                                    q_compensation = ((q_total - q_sum) - q_addend)
                                    q_sum = q_total
                                    x[int((nd - 1))] = q_sum
                                    output[int(saved), int(6):int(tape_columns)] = torch.cat([x, z])
                                    previous_h = hh
                                    started = 1.0
                                    accepted = (accepted + 1.0)
                                    retries = 0
                                    h = (hh * step_factor(result[int((nb + 1))]))
                                else:
                                    live = 0.0
                                    status = (-4.0)
                            else:
                                rejected = (rejected + 1.0)
                                retries = (retries + 1)
                                h = (hh / 2.0)
                            if retries > 15:
                                live = 0.0
                                status = (-3.0)
            if t < stop:
                live = 0.0
                if status == (-1.0):
                    status = (-5.0)
    if live > 0.0:
        status = 1.0
    output[int(0), int(0):int(5)] = torch.stack([torch.as_tensor(saved), torch.as_tensor(status), torch.as_tensor(accepted), torch.as_tensor(rejected), torch.as_tensor(corrections)])
    return output

def replay_mission(theta, grid):
    output = torch.zeros(int(max_nodes), int(tape_columns), device=DEVICE)
    output[:, int(0):int(6)] = grid[:, int(0):int(6)]
    x = x0
    z = z0
    older = x0
    history = torch.zeros(int(nd), device=DEVICE)
    initial = consistent_potentials(x, z, 0.0, theta, grid[int(1), int(5)])
    result = torch.zeros(int((nb + 2)), device=DEVICE)
    live = initial[int(na)]
    corrections = initial[int((na + 1))]
    accepted = 0.0
    completed = 1
    h = 0.0
    alpha = 1.0
    gamma = 0.0
    q_increment = 0.0
    q_sum = 0.0
    q_compensation = 0.0
    q_addend = 0.0
    q_total = 0.0
    if grid[int(0), int(1)] < 1.0:
        live = 0.0
    z = initial[int(0):int(na)]
    output[int(1), int(6):int(tape_columns)] = torch.cat([x, z])
    for node in range(int(2), int((grid[int(0), int(0)] + 1.0))):
        if live > 0.0:
            h = grid[int(node), int(1)]
            if h == 0.0:
                initial = consistent_potentials(x, z, grid[int(node), int(0)], theta, grid[int(node), int(5)])
                corrections = (corrections + initial[int((na + 1))])
                live = initial[int(na)]
                z = initial[int(0):int(na)]
                q_increment = 0.0
            else:
                alpha = grid[int(node), int(2)]
                gamma = grid[int(node), int(4)]
                history = (gamma * (older - x))
                history[int((nd - 1))] = ((-gamma) * q_increment)
                result = step_newton(x, z, history, grid[int((node - 1)), int(0)], h, alpha, theta, grid[int(node), int(5)])
                corrections = (corrections + result[int((nb + 1))])
                live = result[int(nb)]
                older = x
                x = detach((x + result[int(0):int(nd)]))
                z = detach(result[int(nd):int(nb)])
                if live > 0.0:
                    q_increment = ((((h * z[int(nv)]) / 3600.0) + (gamma * q_increment)) / alpha)
                    q_addend = (q_increment - q_compensation)
                    q_total = (q_sum + q_addend)
                    q_compensation = ((q_total - q_sum) - q_addend)
                    q_sum = q_total
                    x[int((nd - 1))] = q_sum
                    accepted = (accepted + 1.0)
            if live > 0.0:
                completed = node
                output[int(node), int(6):int(tape_columns)] = torch.cat([x, z])
    output[int(0), int(0)] = completed
    output[int(0), int(1)] = live
    output[int(0), int(2)] = accepted
    output[int(0), int(3)] = 0.0
    output[int(0), int(4)] = corrections
    return output

def refine_grid(grid):
    output = torch.zeros(int(max_nodes), int(tape_columns), device=DEVICE)
    output[int(1), :] = grid[int(1), :]
    saved = 1
    h = 0.0
    previous_h = 0.1
    started = 0.0
    coefficients = torch.tensor([1.0, 1.0, 0.0], device=DEVICE)
    live = grid[int(0), int(1)]
    time = 0.0
    for node in range(int(2), int((grid[int(0), int(0)] + 1.0))):
        if live > 0.0:
            if grid[int(node), int(1)] == 0.0:
                if (1.0 * saved) < (max_nodes - 1):
                    saved = (saved + 1)
                    output[int(saved), int(0):int(6)] = grid[int(node), int(0):int(6)]
                    started = 0.0
                else:
                    live = 0.0
            else:
                h = (grid[int(node), int(1)] / 2.0)
                for half in range(int(0), int(2)):
                    if (1.0 * saved) < (max_nodes - 1):
                        coefficients = bdf_coefficients(h, previous_h, started)
                        saved = (saved + 1)
                        time = (grid[int(node), int(0)] - h)
                        if (1.0 * half) > 0.0:
                            time = grid[int(node), int(0)]
                        output[int(saved), int(0):int(6)] = torch.cat([torch.stack([torch.as_tensor(time), torch.as_tensor(h)]), coefficients, torch.stack([torch.as_tensor(grid[int(node), int(5)])])])
                        started = 1.0
                        previous_h = h
                    else:
                        live = 0.0
    output[int(0), int(0)] = saved
    output[int(0), int(1)] = live
    return output

def residual_contraction(theta, multiplier, u, a, history, t, h, alpha, power):
    return (multiplier @ step_residual(u, a, history, t, h, alpha, theta, power))

def heat_adjoint(theta, tape, groups, penalized):
    output = torch.zeros(int((n_parameters + 2)), device=DEVICE)
    rhs = torch.zeros(int(max_nodes), int(nb), device=DEVICE)
    weights = (((penalized * 2.0) * penalty_weight) * maximum(groups[int(0):int(6)], 0.0))
    path_active = torch.sum(weights if isinstance(weights, torch.Tensor) else torch.tensor(float(weights)))
    last = tape[int(0), int(0)]
    node = last
    u = torch.zeros(int(nb), device=DEVICE)
    scaled_state = torch.zeros(int(nb), device=DEVICE)
    a = x0
    history = torch.zeros(int(nd), device=DEVICE)
    w = torch.zeros(int(nb), device=DEVICE)
    multiplier = torch.zeros(int(nb), device=DEVICE)
    J = torch.zeros(int(nb), int(nb), device=DEVICE)
    transposed = torch.zeros(int(nb), int(nb), device=DEVICE)
    gradient = (penalized * compute_grad(lambda _dtheta: budget_penalty(_dtheta), theta))
    objective = detach(heat_objective(theta, tape))
    objective_weights = detach(heat_objective_weights(theta, tape, objective))
    heat_weight = 0.0
    live = tape[int(0), int(1)]
    h = 0.0
    alpha = 1.0
    gamma = 0.0
    power = 30.0
    rhs[int(last), int((nd - 1))] = (((penalized * 2.0) * penalty_weight) * max(groups[int(6)], 0.0))
    for backward in range(int(0), int(last)):
        if live > 0.0:
            node = (last - backward)
            w = tape[int(node), int(6):int(tape_columns)]
            a = x0
            history = torch.zeros(int(nd), device=DEVICE)
            h = tape[int(node), int(1)]
            alpha = tape[int(node), int(2)]
            gamma = tape[int(node), int(4)]
            power = tape[int(node), int(5)]
            if (1.0 * node) > 1.0:
                a = tape[int((node - 1)), int(6):int((6 + nd))]
            if gamma > 0.0:
                history = (gamma * (tape[int((node - 2)), int(6):int((6 + nd))] - a))
            u = (torch.cat([(w[int(0):int(nd)] - a), w[int(nd):int(nb)]]) / solver_scale)
            scaled_state = (w / solver_scale)
            heat_weight = objective_weights[int(node)]
            rhs[int(node), :] = (rhs[int(node), :] + (heat_weight * compute_grad(lambda _dscaled_state: local_heat_state(_dscaled_state, theta), scaled_state)))
            gradient = (gradient + (heat_weight * compute_grad(lambda _dtheta: local_heat_parameter(_dtheta, scaled_state), theta)))
            if path_active > 0.0:
                rhs[int(node), :] = (rhs[int(node), :] + compute_grad(lambda _dscaled_state: local_penalty(_dscaled_state, groups, weights), scaled_state))
            J = step_jacobian(u, a, history, (tape[int(node), int(0)] - h), h, alpha, theta, power)
            for column in range(int(0), int(nb)):
                transposed[int(column), :] = J[:, int(column)]
            multiplier = detach(gaussian_solve(transposed, rhs[int(node), :]))
            if finite_values(multiplier, nb) > 0.0:
                gradient = (gradient - compute_grad(lambda _dtheta: residual_contraction(_dtheta, multiplier, u, a, history, (tape[int(node), int(0)] - h), h, alpha, power), theta))
                output[int((n_parameters + 1))] = (output[int((n_parameters + 1))] + 1.0)
                if (1.0 * node) > 1.0:
                    rhs[int((node - 1)), int(0):int(nd)] = (rhs[int((node - 1)), int(0):int(nd)] + (tape[int(node), int(3)] * multiplier[int(0):int(nd)]))
                if gamma > 0.0:
                    rhs[int((node - 2)), int(0):int(nd)] = (rhs[int((node - 2)), int(0):int(nd)] - (gamma * multiplier[int(0):int(nd)]))
            else:
                live = 0.0
    if finite_values(gradient, n_parameters) < 1.0:
        live = 0.0
    output[int(0):int(n_parameters)] = gradient
    output[int(n_parameters)] = live
    return output

def resolve_heat_grid(grid, high_step_limit):
    output = torch.zeros(int(max_nodes), int(tape_columns), device=DEVICE)
    output[int(1), :] = grid[int(1), :]
    saved = 1
    pieces = 1
    h = 0.0
    previous_h = 0.1
    started = 0.0
    live = grid[int(0), int(1)]
    coefficients = torch.tensor([1.0, 1.0, 0.0], device=DEVICE)
    time = 0.0
    for node in range(int(2), int((grid[int(0), int(0)] + 1.0))):
        if live > 0.0:
            if grid[int(node), int(1)] == 0.0:
                if (1.0 * saved) < (max_nodes - 1):
                    saved = (saved + 1)
                    output[int(saved), int(0):int(6)] = grid[int(node), int(0):int(6)]
                    started = 0.0
                else:
                    live = 0.0
            else:
                pieces = 1
                if grid[int(node), int(5)] > 12.0:
                    for division in range(int(0), int(63)):
                        if grid[int(node), int(1)] > (high_step_limit * pieces):
                            pieces = (pieces + 1)
                h = (grid[int(node), int(1)] / pieces)
                for piece in range(int(0), int(pieces)):
                    if live > 0.0:
                        if (1.0 * saved) < (max_nodes - 1):
                            coefficients = bdf_coefficients(h, previous_h, started)
                            saved = (saved + 1)
                            time = ((grid[int(node), int(0)] - grid[int(node), int(1)]) + ((piece + 1) * h))
                            output[int(saved), int(0):int(6)] = torch.cat([torch.stack([torch.as_tensor(time), torch.as_tensor(h)]), coefficients, torch.stack([torch.as_tensor(grid[int(node), int(5)])])])
                            started = 1.0
                            previous_h = h
                        else:
                            live = 0.0
    output[int(0), int(0)] = saved
    output[int(0), int(1)] = live
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
nd = (((ns_n + ns_p) + ne) + 2)
nv = ((nx_n + nx_p) + ne)
na = (nv + 1)
nb = (nd + na)
n_parameters = 13
spline_size = 6
one_column = 1
history_columns = (n_parameters + 8)
n_solids = (((ns_n + ns_p) + nx_n) + nx_p)
max_nodes = 1200
tape_columns = (nb + 6)
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
x0 = torch.cat([cn0, cp0, ce0, torch.tensor([0.0, 0.0], device=DEVICE)])
un0 = ocp_n(torch.stack([torch.as_tensor((initial_n / cmax_n))]))
u_n_initial = un0[int(0)]
up0 = ocp_p(torch.stack([torch.as_tensor((initial_p / cmax_p))]))
u_p_initial = up0[int(0)]
positive_initial_stoich = (initial_p / cmax_p)
positive_pair_reference = ocp_p_pair(torch.stack([torch.as_tensor(positive_initial_stoich)]))
positive_small_reference = tanh((18.5138 * (torch.stack([torch.as_tensor(positive_initial_stoich)]) - 0.5542)))
positive_barrier_reference = ocp_barrier(torch.stack([torch.as_tensor(positive_initial_stoich)]))
i0n_ref = exchange_n(torch.stack([torch.as_tensor(initial_e)]), torch.stack([torch.as_tensor(initial_n)]))
i0p_ref = exchange_p(torch.stack([torch.as_tensor(initial_e)]), torch.stack([torch.as_tensor(initial_p)]))
eta_n_reference = ((((2.0 * gas_constant) * temperature) / faraday) * asinh_scalar((current_nominal / ((((2.0 * area) * length_n) * a_n) * i0n_ref[int(0)]))))
eta_p_reference = ((((2.0 * gas_constant) * temperature) / faraday) * asinh_scalar(((-current_nominal) / ((((2.0 * area) * length_p) * a_p) * i0p_ref[int(0)]))))
ref_p = (((u_p_initial - u_n_initial) + eta_p_reference) - eta_n_reference)
ref_e = ((-u_n_initial) - eta_n_reference)
z0 = torch.cat([torch.zeros(int(nv), device=DEVICE), torch.tensor([7.5], device=DEVICE)])
sxn = torch.stack([(cmax_n + (0.0 * k)) for _fi_k in range(int(ns_n)) for k in [torch.tensor(float(_fi_k), device=DEVICE)]])
sxp = torch.stack([(cmax_p + (0.0 * k)) for _fi_k in range(int(ns_p)) for k in [torch.tensor(float(_fi_k), device=DEVICE)]])
sxe = torch.stack([(initial_e + (0.0 * k)) for _fi_k in range(int(ne)) for k in [torch.tensor(float(_fi_k), device=DEVICE)]])
x_scale = torch.cat([sxn, sxp, sxe, torch.tensor([10.0, 5.0], device=DEVICE)])
szn = torch.stack([(0.0001 + (0.0 * k)) for _fi_k in range(int(nx_n)) for k in [torch.tensor(float(_fi_k), device=DEVICE)]])
szp = torch.stack([(0.001 + (0.0 * k)) for _fi_k in range(int(nx_p)) for k in [torch.tensor(float(_fi_k), device=DEVICE)]])
szp[int((nx_p - 1))] = 0.05
sze = torch.stack([(0.05 + (0.0 * k)) for _fi_k in range(int(ne)) for k in [torch.tensor(float(_fi_k), device=DEVICE)]])
z_scale = torch.cat([szn, szp, sze, torch.tensor([5.0], device=DEVICE)])
solver_scale = torch.cat([x_scale, z_scale])
gsn = torch.stack([((((length_total ** 2) * 5.0) / (area * width_n[int(k)])) + (0.0 * k)) for _fi_k in range(int(nx_n)) for k in [torch.tensor(float(_fi_k), device=DEVICE)]])
gsp = torch.stack([((((length_total ** 2) * 5.0) / (area * width_p[int(k)])) + (0.0 * k)) for _fi_k in range(int(nx_p)) for k in [torch.tensor(float(_fi_k), device=DEVICE)]])
gse = torch.stack([((((length_total ** 2) * 5.0) / (area * width[int(k)])) + (0.0 * k)) for _fi_k in range(int(ne)) for k in [torch.tensor(float(_fi_k), device=DEVICE)]])
g_scale = torch.cat([gsn, gsp, gse])
wn = particle_inventory(nx_n, nr_n, radius_n, (length_n / nx_n), active_n)
wp = particle_inventory(nx_p, nr_p, radius_p, (length_p / nx_p), active_p)
inventory_weights = torch.cat([wn, wp, (eps * width), torch.tensor([0.0, 0.0], device=DEVICE)])
cooling_coefficient = 10.0
cooling_area = 0.00531
cell_volume = 2.42e-05
collector_length_n = 1.2e-05
collector_length_p = 1.6e-05
collector_sigma_n = 58411000.0
collector_sigma_p = 36914000.0
thermal_length = ((((collector_length_n + length_n) + length_s) + length_p) + collector_length_p)
thermal_capacity = ((cell_volume * ((((((collector_length_n * 8960.0) * 385.0) + ((length_n * 1657.0) * 700.0)) + ((length_s * 397.0) * 700.0)) + ((length_p * 3262.0) * 700.0)) + ((collector_length_p * 2700.0) * 897.0))) / thermal_length)
temperature_index = ((ns_n + ns_p) + ne)
heat_scale = 1.0
heat_sharpness = 1000.0
maximum_step = 10.0
high_power_step = 0.625
optimization_steps = 200
starting_step = 100.0
maximum_design_step = 10000.0
stationarity_tolerance = 5e-05
penalty_weight = 100.0
constraint_sharpness = 1000.0
modification_budget = 0.25
roughness_length = 0.05
merit_resolution = 1e-05
minimum_voltage = 3.0
spline_coefficients = torch.tensor([[[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [(-3.0), 3.0, 0.0, 0.0, 0.0, 0.0], [3.0, (-4.5), 1.5, 0.0, 0.0, 0.0], [(-1.0), 1.75, ((-11.0) / 12.0), (1.0 / 6.0), 0.0, 0.0]], [[0.0, 0.25, (7.0 / 12.0), (1.0 / 6.0), 0.0, 0.0], [0.0, (-0.75), 0.25, 0.5, 0.0, 0.0], [0.0, 0.75, (-1.25), 0.5, 0.0, 0.0], [0.0, (-0.25), (7.0 / 12.0), ((-7.0) / 12.0), 0.25, 0.0]], [[0.0, 0.0, (1.0 / 6.0), (7.0 / 12.0), 0.25, 0.0], [0.0, 0.0, (-0.5), (-0.25), 0.75, 0.0], [0.0, 0.0, 0.5, (-1.25), 0.75, 0.0], [0.0, 0.0, ((-1.0) / 6.0), (11.0 / 12.0), (-1.75), 1.0]]], device=DEVICE)
quadrature_nodes = torch.tensor([(-0.8611363115940526), (-0.3399810435848563), 0.3399810435848563, 0.8611363115940526], device=DEVICE)
quadrature_weights = torch.tensor([0.3478548451374539, 0.6521451548625461, 0.6521451548625461, 0.3478548451374539], device=DEVICE)
budget_matrix = modification_matrix()
budget_identity = torch.zeros(int(n_parameters), int(n_parameters), device=DEVICE)
budget_hessian = torch.zeros(int(n_parameters), int(n_parameters), device=DEVICE)
for row in range(int(0), int(n_parameters)):
    budget_identity[int(row), int(row)] = 1.0
for row in range(int(0), int(6)):
    budget_hessian[int(row), int(0):int(6)] = budget_matrix[int(row), :]
    budget_hessian[int((row + 6)), int(6):int(12)] = budget_matrix[int(row), :]
budget_hessian[int(12), int(12)] = (1.0 / 3.0)
newton_tolerance = 2e-06
state_correction_tolerance = 5e-06
algebraic_tolerance = 0.002
relative_tolerance = 0.0001
absolute_tolerance = 1e-06
algebraic_tolerance = 0.0002
state_correction_tolerance = 1e-06
theta_baseline = torch.zeros(int(n_parameters), device=DEVICE)
diagnostic_label = 'Baseline thermal mission: adaptive grid'
print(print(diagnostic_label))
adaptive_baseline_grid = adaptive_mission(theta_baseline, maximum_step)
mission_grid = resolve_heat_grid(adaptive_baseline_grid, high_power_step)
diagnostic_label = 'Baseline thermal mission: fixed-grid replay'
print(print(diagnostic_label))
baseline_trajectory = replay_mission(theta_baseline, mission_grid)
baseline_metrics = mission_metrics(theta_baseline, baseline_trajectory)
baseline_heat = thermal_metrics(theta_baseline, baseline_trajectory)
diagnostic_label = 'Lumped model: thermal capacity J/K, cooling coefficient W/m2/K, cooling area m2'
print(print(diagnostic_label))
print(print(torch.stack([torch.as_tensor(thermal_capacity), torch.as_tensor(cooling_coefficient), torch.as_tensor(cooling_area)])))
diagnostic_label = 'Baseline: electrical feasibility metrics, then thermal metrics'
print(print(diagnostic_label))
print(print(baseline_metrics))
print(print(baseline_heat))
optimization_history = torch.zeros(int((optimization_steps + 2)), int(history_columns), device=DEVICE)
theta_best = theta_baseline
if baseline_metrics[int(11)] > 0.0:
    diagnostic_label = 'Native peak-heat optimization: [update,peak W,merit,step,incoming stationarity,feasible,budget]'
    print(diagnostic_label)
    optimization_history = optimize_mission(theta_baseline, mission_grid, baseline_trajectory)
    theta_best = optimization_history[int(0), int(0):int(n_parameters)]
else:
    diagnostic_label = 'BASELINE INFEASIBLE: optimization skipped; inspect the diagnostics.'
    print(diagnostic_label)
diagnostic_label = 'Optimization summary, then design coefficients'
print(print(diagnostic_label))
print(print(optimization_history[int(0), :]))
print(print(theta_best))
best_trajectory = replay_mission(theta_best, mission_grid)
best_metrics = mission_metrics(theta_best, best_trajectory)
best_heat = thermal_metrics(theta_best, best_trajectory)
best_constraints = mission_constraints(theta_best, best_trajectory)
best_adjoint = heat_adjoint(theta_best, best_trajectory, best_constraints, 0.0)
stationarity_diagnostics = constrained_stationarity(theta_best, detach(best_adjoint[int(0):int(n_parameters)]))
converged = 0.0
if optimization_history[int(0), int(n_parameters)] == 2.0:
    if best_metrics[int(11)] > 0.0:
        if stationarity_diagnostics[int(0)] <= stationarity_tolerance:
            if stationarity_diagnostics[int(3)] <= stationarity_tolerance:
                if stationarity_diagnostics[int(5)] > 0.0:
                    converged = 1.0
diagnostic_label = 'Best: electrical metrics, thermal metrics, and first-order stationarity'
print(print(diagnostic_label))
print(print(best_metrics))
print(print(best_heat))
print(print(stationarity_diagnostics))
property_targets = property_table(theta_best)
diagnostic_label = 'Time verification: halved-grid forward and adjoint solves'
print(print(diagnostic_label))
refined_mission_grid = refine_grid(mission_grid)
refined_baseline_trajectory = replay_mission(theta_baseline, refined_mission_grid)
refined_best_trajectory = replay_mission(theta_best, refined_mission_grid)
refined_baseline_metrics = mission_metrics(theta_baseline, refined_baseline_trajectory)
refined_best_metrics = mission_metrics(theta_best, refined_best_trajectory)
refined_baseline_heat = thermal_metrics(theta_baseline, refined_baseline_trajectory)
refined_best_heat = thermal_metrics(theta_best, refined_best_trajectory)
refined_best_constraints = mission_constraints(theta_best, refined_best_trajectory)
refined_best_adjoint = heat_adjoint(theta_best, refined_best_trajectory, refined_best_constraints, 0.0)
refined_stationarity_diagnostics = constrained_stationarity(theta_best, detach(refined_best_adjoint[int(0):int(n_parameters)]))
diagnostic_label = 'Time verification: independent adaptive forward solve'
print(print(diagnostic_label))
verification_trajectory = adaptive_mission(theta_best, (maximum_step / 2.0))
verification_metrics = mission_metrics(theta_best, verification_trajectory)
verification_heat = thermal_metrics(theta_best, verification_trajectory)
refinement_errors = torch.stack([torch.as_tensor(torch.abs((baseline_heat[int(0)] - refined_baseline_heat[int(0)]) if isinstance((baseline_heat[int(0)] - refined_baseline_heat[int(0)]), torch.Tensor) else torch.tensor(float((baseline_heat[int(0)] - refined_baseline_heat[int(0)]))))), torch.as_tensor(torch.abs((best_heat[int(0)] - refined_best_heat[int(0)]) if isinstance((best_heat[int(0)] - refined_best_heat[int(0)]), torch.Tensor) else torch.tensor(float((best_heat[int(0)] - refined_best_heat[int(0)]))))), torch.as_tensor(torch.abs((baseline_heat[int(1)] - refined_baseline_heat[int(1)]) if isinstance((baseline_heat[int(1)] - refined_baseline_heat[int(1)]), torch.Tensor) else torch.tensor(float((baseline_heat[int(1)] - refined_baseline_heat[int(1)]))))), torch.as_tensor(torch.abs((best_heat[int(1)] - refined_best_heat[int(1)]) if isinstance((best_heat[int(1)] - refined_best_heat[int(1)]), torch.Tensor) else torch.tensor(float((best_heat[int(1)] - refined_best_heat[int(1)]))))), torch.as_tensor(torch.abs((best_metrics[int(1)] - refined_best_metrics[int(1)]) if isinstance((best_metrics[int(1)] - refined_best_metrics[int(1)]), torch.Tensor) else torch.tensor(float((best_metrics[int(1)] - refined_best_metrics[int(1)]))))), torch.as_tensor(max(torch.abs((best_metrics[int(3)] - refined_best_metrics[int(3)]) if isinstance((best_metrics[int(3)] - refined_best_metrics[int(3)]), torch.Tensor) else torch.tensor(float((best_metrics[int(3)] - refined_best_metrics[int(3)])))), torch.abs((best_metrics[int(4)] - refined_best_metrics[int(4)]) if isinstance((best_metrics[int(4)] - refined_best_metrics[int(4)]), torch.Tensor) else torch.tensor(float((best_metrics[int(4)] - refined_best_metrics[int(4)]))))))])
verification_errors = torch.stack([torch.as_tensor(torch.abs((best_heat[int(0)] - verification_heat[int(0)]) if isinstance((best_heat[int(0)] - verification_heat[int(0)]), torch.Tensor) else torch.tensor(float((best_heat[int(0)] - verification_heat[int(0)]))))), torch.as_tensor(torch.abs((best_heat[int(1)] - verification_heat[int(1)]) if isinstance((best_heat[int(1)] - verification_heat[int(1)]), torch.Tensor) else torch.tensor(float((best_heat[int(1)] - verification_heat[int(1)]))))), torch.as_tensor(torch.abs((best_metrics[int(1)] - verification_metrics[int(1)]) if isinstance((best_metrics[int(1)] - verification_metrics[int(1)]), torch.Tensor) else torch.tensor(float((best_metrics[int(1)] - verification_metrics[int(1)]))))), torch.as_tensor(max(torch.abs((best_metrics[int(3)] - verification_metrics[int(3)]) if isinstance((best_metrics[int(3)] - verification_metrics[int(3)]), torch.Tensor) else torch.tensor(float((best_metrics[int(3)] - verification_metrics[int(3)])))), torch.abs((best_metrics[int(4)] - verification_metrics[int(4)]) if isinstance((best_metrics[int(4)] - verification_metrics[int(4)]), torch.Tensor) else torch.tensor(float((best_metrics[int(4)] - verification_metrics[int(4)]))))))])
run_verified = (((converged * refined_best_metrics[int(11)]) * refined_baseline_metrics[int(11)]) * verification_metrics[int(11)])
if refined_stationarity_diagnostics[int(0)] > stationarity_tolerance:
    run_verified = 0.0
if refined_stationarity_diagnostics[int(3)] > stationarity_tolerance:
    run_verified = 0.0
if refined_best_adjoint[int(n_parameters)] < 1.0:
    run_verified = 0.0
if refined_stationarity_diagnostics[int(5)] < 1.0:
    run_verified = 0.0
if max(refinement_errors[int(0)], refinement_errors[int(1)]) > 0.005:
    run_verified = 0.0
if max(refinement_errors[int(2)], refinement_errors[int(3)]) > 0.01:
    run_verified = 0.0
if refinement_errors[int(4)] > 0.001:
    run_verified = 0.0
if refinement_errors[int(5)] > 10.0:
    run_verified = 0.0
if verification_errors[int(0)] > 0.005:
    run_verified = 0.0
if verification_errors[int(1)] > 0.01:
    run_verified = 0.0
if verification_errors[int(2)] > 0.001:
    run_verified = 0.0
if verification_errors[int(3)] > 10.0:
    run_verified = 0.0
if (refined_baseline_heat[int(0)] - refined_best_heat[int(0)]) <= (5.0 * max(refinement_errors[int(0)], refinement_errors[int(1)])):
    run_verified = 0.0
diagnostic_label = 'Halved-grid baseline/best thermal metrics, stationarity, forward errors'
print(print(diagnostic_label))
print(print(refined_baseline_heat))
print(print(refined_best_heat))
print(print(refined_stationarity_diagnostics))
print(print(refinement_errors))
diagnostic_label = 'Independent adaptive thermal metrics and forward errors'
print(print(diagnostic_label))
print(print(verification_heat))
print(print(verification_errors))
if run_verified > 0.0:
    diagnostic_label = 'RUN VERIFIED: first-order constrained convergence and time checks passed.'
    print(diagnostic_label)
else:
    diagnostic_label = 'RUN INCOMPLETE: convergence or forward verification did not pass; inspect diagnostics.'
    print(diagnostic_label)
print(print(torch.stack([torch.as_tensor(run_verified), torch.as_tensor(converged)])))