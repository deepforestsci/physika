import torch
import torch.nn as nn
import torch.optim as optim
from physika.runtime import DEVICE

from physika.runtime import print
from physika.runtime import compute_grad

# === Functions ===
def calculate_model_energies_eV(energy_spacing_eV):
    model_energies_eV = torch.stack([(n * 0.0) for _fi_n in range(int(N_levels)) for n in [torch.tensor(float(_fi_n), device=DEVICE)]])
    for n in range(int(0), int(N_levels)):
        model_energies_eV[int(n)] = (energy_spacing_eV * ((n * 1.0) + 0.5))
    return model_energies_eV

def calculate_energy_spacing_loss(energy_spacing_eV):
    model_energies_eV = calculate_model_energies_eV(energy_spacing_eV)
    total_loss = 0.0
    energy_error_eV = 0.0
    for n in range(int(0), int(N_levels)):
        energy_error_eV = (model_energies_eV[int(n)] - reference_energies_eV[int(n)])
        total_loss = (total_loss + (energy_error_eV ** 2))
    return (total_loss / (N_levels * 1.0))

def adam(parameter, gradient_value, first_moment, second_moment, step, learning_rate):
    beta_1 = 0.9
    beta_2 = 0.999
    epsilon = 1e-06
    new_first_moment = ((beta_1 * first_moment) + ((1.0 - beta_1) * gradient_value))
    new_second_moment = ((beta_2 * second_moment) + ((1.0 - beta_2) * (gradient_value ** 2)))
    corrected_first_moment = (new_first_moment / (1.0 - (beta_1 ** step)))
    corrected_second_moment = (new_second_moment / (1.0 - (beta_2 ** step)))
    new_parameter = (parameter - ((learning_rate * corrected_first_moment) / (torch.sqrt(torch.as_tensor(corrected_second_moment).float()) + epsilon)))
    return torch.stack([torch.as_tensor(new_parameter), torch.as_tensor(new_first_moment), torch.as_tensor(new_second_moment), torch.as_tensor((step + 1.0))])

# === Program ===
ℏ_SI = 1.054571817e-34
joule_per_electronvolt = 1.602176634e-19
atomic_mass_unit = 1.6605390666e-27
π = 3.141592653589793
meter_to_angstrom = 10000000000.0
carbon_mass_amu = 12.0
oxygen_mass_amu = 15.99491461957
reduced_mass_amu = ((carbon_mass_amu * oxygen_mass_amu) / (carbon_mass_amu + oxygen_mass_amu))
reduced_mass = (reduced_mass_amu * atomic_mass_unit)
N_levels = 5
reference_energies_eV = torch.stack([torch.as_tensor(0.134509), torch.as_tensor(0.403527), torch.as_tensor(0.672545), torch.as_tensor(0.941563), torch.as_tensor(1.210581)])
learned_energy_spacing_eV = torch.tensor(0.0, requires_grad=True)
first_moment = 0.0
second_moment = 0.0
optimizer_step = 1.0
learning_rate = 0.01
epochs = 1
energy_spacing_gradient = 0.0
adam_result = torch.stack([torch.as_tensor(learned_energy_spacing_eV), torch.as_tensor(first_moment), torch.as_tensor(second_moment), torch.as_tensor(optimizer_step)])
for epoch in range(int(0), int(epochs)):
    energy_spacing_gradient = compute_grad(calculate_energy_spacing_loss, learned_energy_spacing_eV)
    adam_result = adam(learned_energy_spacing_eV, energy_spacing_gradient, first_moment, second_moment, optimizer_step, learning_rate)
    learned_energy_spacing_eV = adam_result[int(0)]
    first_moment = adam_result[int(1)]
    second_moment = adam_result[int(2)]
    optimizer_step = adam_result[int(3)]
learned_angular_frequency = ((learned_energy_spacing_eV * joule_per_electronvolt) / ℏ_SI)
learned_force_constant = (reduced_mass * (learned_angular_frequency ** 2))
oscillator_length_m = torch.sqrt(torch.as_tensor((ℏ_SI / (reduced_mass * learned_angular_frequency))).float())
x_max_m = (10.0 * oscillator_length_m)
N_grid = 601
dx_m = ((2.0 * x_max_m) / (float((N_grid - 1)) * 1.0))
position_m = torch.stack([((-x_max_m) + (i * dx_m)) for _fi_i in range(int(N_grid)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
position_A = torch.stack([(position_m[int(i)] * meter_to_angstrom) for _fi_i in range(int(N_grid)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
wavefunctions = torch.stack([torch.stack([((n + i) * 0.0) for _fi_i in range(int(N_grid)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]]) for _fi_n in range(int(N_levels)) for n in [torch.tensor(float(_fi_n), device=DEVICE)]])
mass_frequency_over_hbar = ((reduced_mass * learned_angular_frequency) / ℏ_SI)
normalization_constant = ((mass_frequency_over_hbar / π) ** 0.25)
for i in range(int(0), int(N_grid)):
    wavefunctions[int(0), int(i)] = (normalization_constant * torch.exp((((-0.5) * mass_frequency_over_hbar) * (position_m[int(i)] ** 2)) if isinstance((((-0.5) * mass_frequency_over_hbar) * (position_m[int(i)] ** 2)), torch.Tensor) else torch.tensor(float((((-0.5) * mass_frequency_over_hbar) * (position_m[int(i)] ** 2))))))
one_level = 1
if N_levels > one_level:
    for i in range(int(0), int(N_grid)):
        wavefunctions[int(1), int(i)] = ((torch.sqrt((2.0 * mass_frequency_over_hbar) if isinstance((2.0 * mass_frequency_over_hbar), torch.Tensor) else torch.tensor(float((2.0 * mass_frequency_over_hbar)))) * position_m[int(i)]) * wavefunctions[int(0), int(i)])
two_levels = 2
n_real = 0.0
next_n_real = 0.0
first_coefficient = 0.0
second_coefficient = 0.0
if N_levels > two_levels:
    for n in range(int(1), int((N_levels - 1))):
        n_real = (n * 1.0)
        next_n_real = (n_real + 1.0)
        first_coefficient = torch.sqrt((2.0 / next_n_real) if isinstance((2.0 / next_n_real), torch.Tensor) else torch.tensor(float((2.0 / next_n_real))))
        second_coefficient = torch.sqrt((n_real / next_n_real) if isinstance((n_real / next_n_real), torch.Tensor) else torch.tensor(float((n_real / next_n_real))))
        for i in range(int(0), int(N_grid)):
            wavefunctions[int((n + 1)), int(i)] = ((((first_coefficient * torch.sqrt(mass_frequency_over_hbar if isinstance(mass_frequency_over_hbar, torch.Tensor) else torch.tensor(float(mass_frequency_over_hbar)))) * position_m[int(i)]) * wavefunctions[int(n), int(i)]) - (second_coefficient * wavefunctions[int((n - 1)), int(i)]))
kinetic_coefficient_J_m2 = ((ℏ_SI ** 2) / (2.0 * reduced_mass))
second_derivative = 0.0
hamiltonian_wavefunctions = torch.stack([torch.stack([((n + i) * 0.0) for _fi_i in range(int(N_grid)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]]) for _fi_n in range(int(N_levels)) for n in [torch.tensor(float(_fi_n), device=DEVICE)]])
kinetic_part_J = 0.0
potential_part_J = 0.0
potential_J = torch.stack([(((0.5 * reduced_mass) * (learned_angular_frequency ** 2)) * (position_m[int(i)] ** 2)) for _fi_i in range(int(N_grid)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
potential_eV = torch.stack([(potential_J[int(i)] / joule_per_electronvolt) for _fi_i in range(int(N_grid)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
for n in range(int(0), int(N_levels)):
    for i in range(int(1), int((N_grid - 1))):
        second_derivative = (((wavefunctions[int(n), int((i + 1))] - (2.0 * wavefunctions[int(n), int(i)])) + wavefunctions[int(n), int((i - 1))]) / (dx_m * dx_m))
        kinetic_part_J = ((-kinetic_coefficient_J_m2) * second_derivative)
        potential_part_J = (potential_J[int(i)] * wavefunctions[int(n), int(i)])
        hamiltonian_wavefunctions[int(n), int(i)] = ((kinetic_part_J + potential_part_J) / joule_per_electronvolt)
energy_numerator = 0.0
normalization_integral = 0.0
numerical_energies_eV = torch.stack([(n * 0.0) for _fi_n in range(int(N_levels)) for n in [torch.tensor(float(_fi_n), device=DEVICE)]])
for n in range(int(0), int(N_levels)):
    energy_numerator = 0.0
    normalization_integral = 0.0
    for i in range(int(1), int((N_grid - 1))):
        energy_numerator = (energy_numerator + ((wavefunctions[int(n), int(i)] * hamiltonian_wavefunctions[int(n), int(i)]) * dx_m))
        normalization_integral = (normalization_integral + ((wavefunctions[int(n), int(i)] * wavefunctions[int(n), int(i)]) * dx_m))
    numerical_energies_eV[int(n)] = (energy_numerator / normalization_integral)
print(print(learned_energy_spacing_eV))
print(print(learned_angular_frequency))
print(print(learned_force_constant))