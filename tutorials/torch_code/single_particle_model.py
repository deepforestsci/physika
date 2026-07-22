import torch
import torch.nn as nn
import torch.optim as optim
from physika.runtime import DEVICE

from physika.runtime import physika_print
from physika.runtime import compute_grad

# === Functions ===
def zero_1d_array(len):
    results = torch.stack([(i * 0) for _fi_i in range(int(len)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
    return results

def get_1d_array_length(x):
    total = 0
    temp = 0
    for i in range(len(x)):
        temp = x[int(i)]
        total = total + 1
    return total

def clip_stoichiometry(theta):
    x = theta
    if x < 0.002:
        x = 0.002
    if x > 0.995:
        x = 0.995
    return x

def clip_concentration(c_s, c_s_max):
    c = c_s
    if c < 1.0:
        c = 1.0
    if c > (c_s_max - 1.0):
        c = (c_s_max - 1.0)
    return c

def tanh_scalar(z):
    return ((torch.exp(z if isinstance(z, torch.Tensor) else torch.tensor(float(z))) - torch.exp((-z) if isinstance((-z), torch.Tensor) else torch.tensor(float((-z))))) / (torch.exp(z if isinstance(z, torch.Tensor) else torch.tensor(float(z))) + torch.exp((-z) if isinstance((-z), torch.Tensor) else torch.tensor(float((-z))))))

def ocp_negative(theta):
    x = clip_stoichiometry(theta)
    return (((((1.9793 * torch.exp(((-39.3631) * x) if isinstance(((-39.3631) * x), torch.Tensor) else torch.tensor(float(((-39.3631) * x))))) + 0.2482) - (0.0909 * tanh_scalar((29.8538 * (x - 0.1234))))) - (0.04478 * tanh_scalar((14.9159 * (x - 0.2769))))) - (0.0205 * tanh_scalar((30.4444 * (x - 0.6103)))))

def ocp_positive(theta):
    x = clip_stoichiometry(theta)
    return ((((((-0.809) * x) + 4.4875) - (0.0428 * tanh_scalar((18.5138 * (x - 0.5542))))) - (17.7326 * tanh_scalar((15.789 * (x - 0.3117))))) + (17.5842 * tanh_scalar((15.9308 * (x - 0.312)))))

def exchange_current(c_s_surface, c_s_max, k0):
    c_s = clip_concentration(c_s_surface, c_s_max)
    return (k0 * torch.sqrt(((c_e * c_s) * (c_s_max - c_s)) if isinstance(((c_e * c_s) * (c_s_max - c_s)), torch.Tensor) else torch.tensor(float(((c_e * c_s) * (c_s_max - c_s))))))

def asinh_scalar(z):
    return torch.log((z + torch.sqrt(((z ** 2) + 1.0) if isinstance(((z ** 2) + 1.0), torch.Tensor) else torch.tensor(float(((z ** 2) + 1.0))))) if isinstance((z + torch.sqrt(((z ** 2) + 1.0) if isinstance(((z ** 2) + 1.0), torch.Tensor) else torch.tensor(float(((z ** 2) + 1.0))))), torch.Tensor) else torch.tensor(float((z + torch.sqrt(((z ** 2) + 1.0) if isinstance(((z ** 2) + 1.0), torch.Tensor) else torch.tensor(float(((z ** 2) + 1.0))))))))

def bv_overpotential(reaction_current, i0):
    z = (reaction_current / (2.0 * i0))
    return ((((2.0 * gas_constant) * temperature) / faraday) * asinh_scalar(z))

def negative_surface_flux(current):
    area_density = ((3.0 * eps_s_n) / radius_n)
    return (current / (((faraday * electrode_area) * thickness_n) * area_density))

def positive_surface_flux(current, eps_s_p):
    area_density = ((3.0 * eps_s_p) / radius_p)
    return ((-current) / (((faraday * electrode_area) * thickness_p) * area_density))

def particle_rhs(c, radius, diffusivity, j_out):
    dc = zero_1d_array(nr)
    dr = (radius / nr)
    shell_volume_factor = 0.0
    outward_gradient = 0.0
    inward_gradient = 0.0
    diffusion_in = 0.0
    first_cell = 0
    last_cell = (nr - 1)
    for i in range(int(0), int(nr)):
        shell_volume_factor = (((i + 1) ** 3) - (i ** 3))
        if i == first_cell:
            dc[int(i)] = (((3.0 * diffusivity) * (c[int(1)] - c[int(0)])) / (dr ** 2))
        else:
            if i < last_cell:
                outward_gradient = (((i + 1) ** 2) * (c[int((i + 1))] - c[int(i)]))
                inward_gradient = ((i ** 2) * (c[int(i)] - c[int((i - 1))]))
                dc[int(i)] = (((3.0 * diffusivity) * (outward_gradient - inward_gradient)) / ((dr ** 2) * shell_volume_factor))
            else:
                diffusion_in = ((((i ** 2) * diffusivity) * (c[int(i)] - c[int((i - 1))])) / dr)
                dc[int(i)] = ((3.0 * (((-((i + 1) ** 2)) * j_out) - diffusion_in)) / (dr * shell_volume_factor))
    return dc

def rk4_particle(c, radius, diffusivity, j_out, step_dt):
    k1 = particle_rhs(c, radius, diffusivity, j_out)
    k2 = particle_rhs((c + ((0.5 * step_dt) * k1)), radius, diffusivity, j_out)
    k3 = particle_rhs((c + ((0.5 * step_dt) * k2)), radius, diffusivity, j_out)
    k4 = particle_rhs((c + (step_dt * k3)), radius, diffusivity, j_out)
    return (c + ((step_dt * (((k1 + (2.0 * k2)) + (2.0 * k3)) + k4)) / 6.0))

def terminal_voltage(c_n, c_p, eps_s_p, current):
    c_ns = clip_concentration(c_n[int((nr - 1))], c_n_max)
    c_ps = clip_concentration(c_p[int((nr - 1))], c_p_max)
    theta_n = (c_ns / c_n_max)
    theta_p = (c_ps / c_p_max)
    j_n = negative_surface_flux(current)
    j_p = positive_surface_flux(current, eps_s_p)
    i0_n = exchange_current(c_ns, c_n_max, k0_n)
    i0_p = exchange_current(c_ps, c_p_max, k0_p)
    eta_n = bv_overpotential((faraday * j_n), i0_n)
    eta_p = bv_overpotential((faraday * j_p), i0_p)
    return ((((ocp_positive(theta_p) - ocp_negative(theta_n)) + eta_p) - eta_n) - (current * contact_resistance))

def spm_solver(eps_s_p, c_rate, substeps_per_sample, step_dt):
    c_n = torch.stack([torch.as_tensor(c_n_initial), torch.as_tensor(c_n_initial), torch.as_tensor(c_n_initial), torch.as_tensor(c_n_initial), torch.as_tensor(c_n_initial)])
    c_p = torch.stack([torch.as_tensor(c_p_initial), torch.as_tensor(c_p_initial), torch.as_tensor(c_p_initial), torch.as_tensor(c_p_initial), torch.as_tensor(c_p_initial)])
    voltage = zero_1d_array(num_samples)
    current = (c_rate * nominal_capacity_ah)
    j_n = negative_surface_flux(current)
    j_p = positive_surface_flux(current, eps_s_p)
    voltage[int(0)] = terminal_voltage(c_n, c_p, eps_s_p, current)
    for sample in range(int(0), int((num_samples - 1))):
        for substep in range(int(0), int(substeps_per_sample)):
            c_n = rk4_particle(c_n, radius_n, diffusivity_n, j_n, step_dt)
            c_p = rk4_particle(c_p, radius_p, diffusivity_p, j_p, step_dt)
        voltage[int((sample + 1))] = terminal_voltage(c_n, c_p, eps_s_p, current)
    return voltage

def validation_protocol_solver(eps_s_p, step_dt):
    c_n = torch.stack([torch.as_tensor(c_n_initial), torch.as_tensor(c_n_initial), torch.as_tensor(c_n_initial), torch.as_tensor(c_n_initial), torch.as_tensor(c_n_initial)])
    c_p = torch.stack([torch.as_tensor(c_p_initial), torch.as_tensor(c_p_initial), torch.as_tensor(c_p_initial), torch.as_tensor(c_p_initial), torch.as_tensor(c_p_initial)])
    voltage = zero_1d_array(validation_num_samples)
    current = validation_current_a[int(0)]
    j_n = negative_surface_flux(current)
    j_p = positive_surface_flux(current, eps_s_p)
    first_transition = 8
    second_transition = 17
    voltage[int(0)] = terminal_voltage(c_n, c_p, eps_s_p, current)
    for sample in range(int(0), int((validation_num_samples - 1))):
        current = validation_current_a[int((sample + 1))]
        j_n = negative_surface_flux(current)
        j_p = positive_surface_flux(current, eps_s_p)
        if sample != first_transition:
            if sample != second_transition:
                for substep in range(int(0), int(validation_substeps_per_sample)):
                    c_n = rk4_particle(c_n, radius_n, diffusivity_n, j_n, step_dt)
                    c_p = rk4_particle(c_p, radius_p, diffusivity_p, j_p, step_dt)
        voltage[int((sample + 1))] = terminal_voltage(c_n, c_p, eps_s_p, current)
    return voltage

def rmse(a, b):
    loss = 0.0
    n = get_1d_array_length(a)
    for i in range(int(0), int(n)):
        loss = loss + ((a[int(i)] - b[int(i)]) ** 2)
    return torch.sqrt((loss / n) if isinstance((loss / n), torch.Tensor) else torch.tensor(float((loss / n))))

def calculate_loss(eps_fit):
    predicted = spm_solver(eps_fit, train_c_rate, train_substeps_per_sample, rk4_step_s)
    return torch.mean(((predicted - dfn_train_voltage) ** 2) if isinstance(((predicted - dfn_train_voltage) ** 2), torch.Tensor) else torch.tensor(float(((predicted - dfn_train_voltage) ** 2))))

# === Program ===
faraday = 96485.33212
gas_constant = 8.31446261815324
temperature = 298.15
c_e = 1000.0
electrode_area = 0.1027
nominal_capacity_ah = 5.0
contact_resistance = 0.0
thickness_n = 8.52e-05
thickness_p = 7.56e-05
radius_n = 5.86e-06
radius_p = 5.22e-06
diffusivity_n = 3.3e-14
diffusivity_p = 4e-15
eps_s_n = 0.75
c_n_initial = 29866.0
c_p_initial = 17038.0
c_n_max = 33133.0
c_p_max = 63104.0
k0_n = 6.48e-07
k0_p = 3.42e-06
nr = 5
num_samples = 21
train_c_rate = 0.1
train_substeps_per_sample = 90
rk4_step_s = 20.0
validation_num_samples = 27
validation_substeps_per_sample = 45
validation_time_h = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.0, 4.25, 4.5, 4.75, 5.0, 5.25, 5.5, 5.75, 6.0], device=DEVICE)
validation_c_rate_profile = torch.tensor([0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05], device=DEVICE)
validation_current_a = (validation_c_rate_profile * nominal_capacity_ah)
dfn_train_voltage = torch.tensor([4.158395248, 4.084192602, 4.071392813, 4.049195473, 4.006860924, 3.961241395, 3.916527071, 3.86177453, 3.814408034, 3.772015162, 3.725484224, 3.682248062, 3.645685812, 3.609974847, 3.565901908, 3.513874271, 3.468832454, 3.424676253, 3.317194985, 3.146746296, 2.789451932], device=DEVICE)
initial_eps_s_p = 0.3
chen2020_eps_s_p = 0.665
eps_s_p = torch.tensor(initial_eps_s_p, requires_grad=True)
learning_rate = 0.86
learning_rate_decay = 0.25
epochs = 5
initial_train_voltage = spm_solver(initial_eps_s_p, train_c_rate, train_substeps_per_sample, rk4_step_s)
initial_train_rmse = rmse(initial_train_voltage, dfn_train_voltage)
loss_history = zero_1d_array((epochs + 1))
loss_history[int(0)] = (initial_train_rmse ** 2)
for epoch in range(int(0), int(epochs)):
    g = compute_grad(calculate_loss, eps_s_p)
    eps_s_p = (eps_s_p - (learning_rate * g))
    loss_history[int((epoch + 1))] = calculate_loss(eps_s_p)
    learning_rate = (learning_rate * learning_rate_decay)
fitted_train_voltage = spm_solver(eps_s_p, train_c_rate, train_substeps_per_sample, rk4_step_s)
fitted_train_rmse = rmse(fitted_train_voltage, dfn_train_voltage)
pybamm_validation_eps_s_p = 0.6471111178398132
pybamm_validation_voltage = torch.tensor([4.142996394, 4.067772943, 4.057471567, 4.027229404, 3.981657244, 3.937880704, 3.895608468, 3.838549294, 3.788632364, 3.81322811, 3.833002643, 3.833640604, 3.833695039, 3.833696392, 3.833696305, 3.833696268, 3.833696266, 3.833696264, 3.827385441, 3.811877071, 3.800818412, 3.789583321, 3.778002657, 3.766121504, 3.754073902, 3.742039259, 3.730207631], device=DEVICE)
validation_voltage = validation_protocol_solver(eps_s_p, rk4_step_s)
validation_rmse = rmse(validation_voltage, pybamm_validation_voltage)
physika_print(physika_print(eps_s_p))
physika_print(physika_print(initial_train_rmse))
physika_print(physika_print(fitted_train_rmse))
physika_print(physika_print(validation_rmse))
physika_print(physika_print(fitted_train_voltage))
physika_print(physika_print(validation_voltage))
physika_print(physika_print(loss_history))