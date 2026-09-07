import torch
import torch.nn as nn
import torch.optim as optim
from physika.runtime import DEVICE

from physika.runtime import print
from physika.runtime import compute_grad

# === Functions ===
def zero_array(length):
    values = torch.stack([(index * 0.0) for _fi_index in range(int(length)) for index in [torch.tensor(float(_fi_index), device=DEVICE)]])
    return values

def morse_model_potential(dissociation_eV, α_inverse_angstrom, equilibrium_angstrom):
    model_potential = zero_array(N_reference_points)
    for reference_index in range(int(0), int(N_reference_points)):
        model_potential[int(reference_index)] = (dissociation_eV * ((1.0 - torch.exp(((-α_inverse_angstrom) * (reference_bond_distance_angstrom[int(reference_index)] - equilibrium_angstrom)) if isinstance(((-α_inverse_angstrom) * (reference_bond_distance_angstrom[int(reference_index)] - equilibrium_angstrom)), torch.Tensor) else torch.tensor(float(((-α_inverse_angstrom) * (reference_bond_distance_angstrom[int(reference_index)] - equilibrium_angstrom)))))) ** 2))
    return model_potential

def morse_parameter_loss(dissociation_eV, α_inverse_angstrom, equilibrium_angstrom):
    predicted = morse_model_potential(dissociation_eV, α_inverse_angstrom, equilibrium_angstrom)
    total_loss = 0.0
    error = 0.0
    for reference_index in range(int(0), int(N_reference_points)):
        error = (predicted[int(reference_index)] - reference_potential_eV[int(reference_index)])
        total_loss = (total_loss + (error ** 2))
    return (total_loss / (N_reference_points * 1.0))

def loss_with_respect_to_dissociation(dissociation_eV):
    return morse_parameter_loss(dissociation_eV, learned_α_inverse_angstrom, learned_equilibrium_distance_angstrom)

def loss_with_respect_to_α(α_inverse_angstrom):
    return morse_parameter_loss(learned_dissociation_energy_eV, α_inverse_angstrom, learned_equilibrium_distance_angstrom)

def loss_with_respect_to_equilibrium(equilibrium_angstrom):
    return morse_parameter_loss(learned_dissociation_energy_eV, learned_α_inverse_angstrom, equilibrium_angstrom)

def adam(parameter, gradient_value, first_moment, second_moment, step, learning_rate):
    β_1 = 0.9
    β_2 = 0.999
    ε = 1e-08
    new_first_moment = ((β_1 * first_moment) + ((1.0 - β_1) * gradient_value))
    new_second_moment = ((β_2 * second_moment) + ((1.0 - β_2) * (gradient_value ** 2)))
    corrected_first_moment = (new_first_moment / (1.0 - (β_1 ** step)))
    corrected_second_moment = (new_second_moment / (1.0 - (β_2 ** step)))
    new_parameter = (parameter - ((learning_rate * corrected_first_moment) / (torch.sqrt(torch.as_tensor(corrected_second_moment).float()) + ε)))
    return torch.stack([torch.as_tensor(new_parameter), torch.as_tensor(new_first_moment), torch.as_tensor(new_second_moment), torch.as_tensor((step + 1.0))])

def zero_matrix(rows, columns):
    values = torch.stack([torch.stack([((index + column_index) * 0.0) for _fi_column_index in range(int(columns)) for column_index in [torch.tensor(float(_fi_column_index), device=DEVICE)]]) for _fi_index in range(int(rows)) for index in [torch.tensor(float(_fi_index), device=DEVICE)]])
    return values

def linspace(start, end, number):
    values = zero_array(number)
    spacing = ((end - start) / (number - 1))
    for index in range(int(0), int(number)):
        values[int(index)] = (start + (index * spacing))
    return values

def integrate(values, grid_spacing, number, m=None):
    if m is None:
        m = int(values.shape[0])
    integral = 0.0
    return (lambda _acc: ([_acc := (lambda integral=_acc: (lambda integral=(integral + (values[int(index)] * grid_spacing)): integral)())() for index in range(int((number - 0)))], _acc)[1])(integral)

def dot_product(first, second, number, m=None, n=None):
    if m is None:
        m = int(first.shape[0])
    if n is None:
        n = int(second.shape[0])
    value = 0.0
    return (lambda _acc: ([_acc := (lambda value=_acc: (lambda value=(value + (first[int(index)] * second[int(index)])): value)())() for index in range(int((number - 0)))], _acc)[1])(value)

def normalize_vector(values, number):
    result = zero_array(number)
    norm = torch.sqrt(dot_product(values, values, number) if isinstance(dot_product(values, values, number), torch.Tensor) else torch.tensor(float(dot_product(values, values, number))))
    for index in range(int(0), int(number)):
        result[int(index)] = (values[int(index)] / norm)
    return result

def apply_hamiltonian(wavefunction, potential, kinetic_coefficient, number):
    result = zero_array(number)
    result[int(0)] = ((((2.0 * kinetic_coefficient) + potential[int(0)]) * wavefunction[int(0)]) - (kinetic_coefficient * wavefunction[int(1)]))
    for index in range(int(1), int((number - 1))):
        result[int(index)] = ((((-kinetic_coefficient) * wavefunction[int((index - 1))]) + (((2.0 * kinetic_coefficient) + potential[int(index)]) * wavefunction[int(index)])) - (kinetic_coefficient * wavefunction[int((index + 1))]))
    result[int((number - 1))] = (((-kinetic_coefficient) * wavefunction[int((number - 2))]) + (((2.0 * kinetic_coefficient) + potential[int((number - 1))]) * wavefunction[int((number - 1))]))
    return result

def jacobi_diagonalize():
    jacobi_not_converged = 0
    jacobi_converged = 1
    jacobi_converged_local = jacobi_not_converged
    for index in range(int(0), int(Krylov_dimension)):
        Ritz_vectors[int(index), int(index)] = 1.0
        for column_index in range(int(0), int(Krylov_dimension)):
            projected_work_matrix[int(index), int(column_index)] = projected_hamiltonian[int(index), int(column_index)]
    for rotation in range(int(0), int(jacobi_maximum_rotations)):
        jacobi_p = 0
        jacobi_q = 1
        jacobi_largest = torch.abs(projected_work_matrix[int(0), int(1)] if isinstance(projected_work_matrix[int(0), int(1)], torch.Tensor) else torch.tensor(float(projected_work_matrix[int(0), int(1)])))
        for index in range(int(0), int(Krylov_dimension)):
            for column_index in range(int((index + 1)), int(Krylov_dimension)):
                if torch.abs(projected_work_matrix[int(index), int(column_index)] if isinstance(projected_work_matrix[int(index), int(column_index)], torch.Tensor) else torch.tensor(float(projected_work_matrix[int(index), int(column_index)]))) > jacobi_largest:
                    jacobi_largest = torch.abs(projected_work_matrix[int(index), int(column_index)] if isinstance(projected_work_matrix[int(index), int(column_index)], torch.Tensor) else torch.tensor(float(projected_work_matrix[int(index), int(column_index)])))
                    jacobi_p = index
                    jacobi_q = column_index
        if jacobi_converged_local == jacobi_not_converged:
            if jacobi_largest <= jacobi_tolerance:
                jacobi_converged_local = jacobi_converged
        if jacobi_largest > jacobi_tolerance:
            jacobi_angle = (0.5 * torch.atan2((2.0 * projected_work_matrix[int(jacobi_p), int(jacobi_q)]), (projected_work_matrix[int(jacobi_q), int(jacobi_q)] - projected_work_matrix[int(jacobi_p), int(jacobi_p)])))
            jacobi_cosine = torch.cos(jacobi_angle if isinstance(jacobi_angle, torch.Tensor) else torch.tensor(float(jacobi_angle)))
            jacobi_sine = torch.sin(jacobi_angle if isinstance(jacobi_angle, torch.Tensor) else torch.tensor(float(jacobi_angle)))
            jacobi_app = (projected_work_matrix[int(jacobi_p), int(jacobi_p)] + 0.0)
            jacobi_aqq = (projected_work_matrix[int(jacobi_q), int(jacobi_q)] + 0.0)
            jacobi_apq = (projected_work_matrix[int(jacobi_p), int(jacobi_q)] + 0.0)
            for basis_index in range(int(0), int(Krylov_dimension)):
                if basis_index != jacobi_p:
                    if basis_index != jacobi_q:
                        jacobi_akp = (projected_work_matrix[int(basis_index), int(jacobi_p)] + 0.0)
                        jacobi_akq = (projected_work_matrix[int(basis_index), int(jacobi_q)] + 0.0)
                        projected_work_matrix[int(basis_index), int(jacobi_p)] = ((jacobi_cosine * jacobi_akp) - (jacobi_sine * jacobi_akq))
                        projected_work_matrix[int(jacobi_p), int(basis_index)] = projected_work_matrix[int(basis_index), int(jacobi_p)]
                        projected_work_matrix[int(basis_index), int(jacobi_q)] = ((jacobi_sine * jacobi_akp) + (jacobi_cosine * jacobi_akq))
                        projected_work_matrix[int(jacobi_q), int(basis_index)] = projected_work_matrix[int(basis_index), int(jacobi_q)]
            projected_work_matrix[int(jacobi_p), int(jacobi_p)] = ((((jacobi_cosine * jacobi_cosine) * jacobi_app) - (((2.0 * jacobi_sine) * jacobi_cosine) * jacobi_apq)) + ((jacobi_sine * jacobi_sine) * jacobi_aqq))
            projected_work_matrix[int(jacobi_q), int(jacobi_q)] = ((((jacobi_sine * jacobi_sine) * jacobi_app) + (((2.0 * jacobi_sine) * jacobi_cosine) * jacobi_apq)) + ((jacobi_cosine * jacobi_cosine) * jacobi_aqq))
            projected_work_matrix[int(jacobi_p), int(jacobi_q)] = 0.0
            projected_work_matrix[int(jacobi_q), int(jacobi_p)] = 0.0
            for basis_index in range(int(0), int(Krylov_dimension)):
                jacobi_vkp = (Ritz_vectors[int(basis_index), int(jacobi_p)] + 0.0)
                jacobi_vkq = (Ritz_vectors[int(basis_index), int(jacobi_q)] + 0.0)
                Ritz_vectors[int(basis_index), int(jacobi_p)] = ((jacobi_cosine * jacobi_vkp) - (jacobi_sine * jacobi_vkq))
                Ritz_vectors[int(basis_index), int(jacobi_q)] = ((jacobi_sine * jacobi_vkp) + (jacobi_cosine * jacobi_vkq))
    for index in range(int(0), int(Krylov_dimension)):
        Ritz_values[int(index)] = projected_work_matrix[int(index), int(index)]
    for index in range(int(0), int((Krylov_dimension - 1))):
        jacobi_minimum = index
        for column_index in range(int((index + 1)), int(Krylov_dimension)):
            if Ritz_values[int(column_index)] < Ritz_values[int(jacobi_minimum)]:
                jacobi_minimum = column_index
        if jacobi_minimum != index:
            jacobi_temporary_value = (Ritz_values[int(index)] + 0.0)
            Ritz_values[int(index)] = Ritz_values[int(jacobi_minimum)]
            Ritz_values[int(jacobi_minimum)] = jacobi_temporary_value
            for basis_index in range(int(0), int(Krylov_dimension)):
                jacobi_temporary_vector = (Ritz_vectors[int(basis_index), int(index)] + 0.0)
                Ritz_vectors[int(basis_index), int(index)] = Ritz_vectors[int(basis_index), int(jacobi_minimum)]
                Ritz_vectors[int(basis_index), int(jacobi_minimum)] = jacobi_temporary_vector
    return jacobi_converged_local

# === Program ===
atomic_mass_unit = 1.6605390666e-27
ℏ = 1.054571817e-34
planck_constant = 6.62607015e-34
speed_of_light_cm = 29979245800.0
electron_volt = 1.602176634e-19
inverse_angstrom_to_inverse_meter = 10000000000.0
mass_H_u = 1.00784
mass_Cl_u = 35.45
mass_H = (mass_H_u * atomic_mass_unit)
mass_Cl = (mass_Cl_u * atomic_mass_unit)
μ_HCl = ((mass_H * mass_Cl) / (mass_H + mass_Cl))
N_reference_points = 13
reference_bond_distance_angstrom = torch.stack([torch.as_tensor(1.1), torch.as_tensor(1.2746), torch.as_tensor(1.6), torch.as_tensor(2.15), torch.as_tensor(2.65), torch.as_tensor(3.2), torch.as_tensor(3.75), torch.as_tensor(3.95), torch.as_tensor(4.25), torch.as_tensor(4.55), torch.as_tensor(4.8), torch.as_tensor(5.0), torch.as_tensor(5.3)])
reference_potential_eV = torch.stack([torch.as_tensor(0.68888), torch.as_tensor(0.0), torch.as_tensor(0.97847), torch.as_tensor(3.22469), torch.as_tensor(4.2397), torch.as_tensor(4.54743), torch.as_tensor(4.60134), torch.as_tensor(4.60678), torch.as_tensor(4.61086), torch.as_tensor(4.61274), torch.as_tensor(4.61355), torch.as_tensor(4.61396), torch.as_tensor(4.61434)])
learned_dissociation_energy_eV = torch.tensor(3.5, requires_grad=True)
learned_α_inverse_angstrom = torch.tensor(1.5, requires_grad=True)
learned_equilibrium_distance_angstrom = torch.tensor(1.35, requires_grad=True)
dissociation_first_moment = 0.0
dissociation_second_moment = 0.0
α_first_moment = 0.0
α_second_moment = 0.0
equilibrium_first_moment = 0.0
equilibrium_second_moment = 0.0
optimizer_step = 1.0
learning_rate_dissociation = 0.005
learning_rate_α = 0.001
learning_rate_equilibrium = 0.0005
learning_epochs = 1
dissociation_gradient = 0.0
α_gradient = 0.0
equilibrium_gradient = 0.0
dissociation_adam_result = torch.stack([torch.as_tensor(learned_dissociation_energy_eV), torch.as_tensor(0.0), torch.as_tensor(0.0), torch.as_tensor(optimizer_step)])
α_adam_result = torch.stack([torch.as_tensor(learned_α_inverse_angstrom), torch.as_tensor(0.0), torch.as_tensor(0.0), torch.as_tensor(optimizer_step)])
equilibrium_adam_result = torch.stack([torch.as_tensor(learned_equilibrium_distance_angstrom), torch.as_tensor(0.0), torch.as_tensor(0.0), torch.as_tensor(optimizer_step)])
loss_history = zero_array(learning_epochs)
for epoch in range(int(0), int(learning_epochs)):
    dissociation_gradient = compute_grad(loss_with_respect_to_dissociation, learned_dissociation_energy_eV)
    α_gradient = compute_grad(loss_with_respect_to_α, learned_α_inverse_angstrom)
    equilibrium_gradient = compute_grad(loss_with_respect_to_equilibrium, learned_equilibrium_distance_angstrom)
    dissociation_adam_result = adam(learned_dissociation_energy_eV, dissociation_gradient, dissociation_first_moment, dissociation_second_moment, optimizer_step, learning_rate_dissociation)
    α_adam_result = adam(learned_α_inverse_angstrom, α_gradient, α_first_moment, α_second_moment, optimizer_step, learning_rate_α)
    equilibrium_adam_result = adam(learned_equilibrium_distance_angstrom, equilibrium_gradient, equilibrium_first_moment, equilibrium_second_moment, optimizer_step, learning_rate_equilibrium)
    learned_dissociation_energy_eV = dissociation_adam_result[int(0)]
    dissociation_first_moment = dissociation_adam_result[int(1)]
    dissociation_second_moment = dissociation_adam_result[int(2)]
    learned_α_inverse_angstrom = α_adam_result[int(0)]
    α_first_moment = α_adam_result[int(1)]
    α_second_moment = α_adam_result[int(2)]
    learned_equilibrium_distance_angstrom = equilibrium_adam_result[int(0)]
    equilibrium_first_moment = equilibrium_adam_result[int(1)]
    equilibrium_second_moment = equilibrium_adam_result[int(2)]
    loss_history[int(epoch)] = morse_parameter_loss(learned_dissociation_energy_eV, learned_α_inverse_angstrom, learned_equilibrium_distance_angstrom)
    optimizer_step = (optimizer_step + 1.0)
final_learning_loss = morse_parameter_loss(learned_dissociation_energy_eV, learned_α_inverse_angstrom, learned_equilibrium_distance_angstrom)
learned_reference_potential_eV = morse_model_potential(learned_dissociation_energy_eV, learned_α_inverse_angstrom, learned_equilibrium_distance_angstrom)
dissociation_energy_eV = learned_dissociation_energy_eV
dissociation_energy_J = (dissociation_energy_eV * electron_volt)
equilibrium_distance = (learned_equilibrium_distance_angstrom * 1e-10)
morse_α = (learned_α_inverse_angstrom * inverse_angstrom_to_inverse_meter)
N_grid = 100
N_levels = 2
block_size = 2
Krylov_dimension = 40
r_min = 5e-11
r_max = 2e-10
grid_spacing = ((r_max - r_min) / float((N_grid + 1)))
r_start = (r_min + grid_spacing)
r_end = (r_max - grid_spacing)
bond_distance = linspace(r_start, r_end, N_grid)
distance_from_equilibrium = (bond_distance - equilibrium_distance)
morse_exponential = torch.exp(((-morse_α) * distance_from_equilibrium) if isinstance(((-morse_α) * distance_from_equilibrium), torch.Tensor) else torch.tensor(float(((-morse_α) * distance_from_equilibrium))))
morse_difference = (1.0 - morse_exponential)
potential_J = ((dissociation_energy_J * morse_difference) * morse_difference)
potential_eV = (potential_J / electron_volt)
kinetic_coefficient_J = ((ℏ / grid_spacing) * (ℏ / ((2.0 * μ_HCl) * grid_spacing)))
kinetic_coefficient_eV = (kinetic_coefficient_J / electron_volt)
trial_width = 2e-11
gaussian_argument = (distance_from_equilibrium / trial_width)
gaussian_envelope = torch.exp(((-gaussian_argument) * gaussian_argument) if isinstance(((-gaussian_argument) * gaussian_argument), torch.Tensor) else torch.tensor(float(((-gaussian_argument) * gaussian_argument))))
Krylov_basis = zero_matrix(Krylov_dimension, N_grid)
H_Krylov = zero_matrix(Krylov_dimension, N_grid)
projected_hamiltonian = zero_matrix(Krylov_dimension, Krylov_dimension)
projected_work_matrix = zero_matrix(Krylov_dimension, Krylov_dimension)
Ritz_vectors = zero_matrix(Krylov_dimension, Krylov_dimension)
Ritz_values = zero_array(Krylov_dimension)
jacobi_tolerance = 1e-07
jacobi_maximum_rotations = 1
jacobi_p = 0
jacobi_q = 1
candidate = zero_array(N_grid)
overlap = 0.0
candidate_norm = 0.0
for index in range(int(0), int(N_grid)):
    Krylov_basis[int(0), int(index)] = gaussian_envelope[int(index)]
Krylov_basis[int(0)] = normalize_vector(Krylov_basis[int(0)], N_grid)
for q_index in range(int(1), int(Krylov_dimension)):
    if q_index < block_size:
        for index in range(int(0), int(N_grid)):
            candidate[int(index)] = (gaussian_argument[int(index)] * Krylov_basis[int((q_index - 1)), int(index)])
    else:
        candidate = apply_hamiltonian(Krylov_basis[int((q_index - block_size))], potential_eV, kinetic_coefficient_eV, N_grid)
    for orthogonalization_pass in range(int(0), int(2)):
        for lower in range(int(0), int(q_index)):
            overlap = dot_product(Krylov_basis[int(lower)], candidate, N_grid)
            for index in range(int(0), int(N_grid)):
                candidate[int(index)] = (candidate[int(index)] - (overlap * Krylov_basis[int(lower), int(index)]))
    candidate_norm = torch.sqrt(dot_product(candidate, candidate, N_grid) if isinstance(dot_product(candidate, candidate, N_grid), torch.Tensor) else torch.tensor(float(dot_product(candidate, candidate, N_grid))))
    for index in range(int(0), int(N_grid)):
        Krylov_basis[int(q_index), int(index)] = (candidate[int(index)] / candidate_norm)
for q_index in range(int(0), int(Krylov_dimension)):
    H_Krylov[int(q_index)] = apply_hamiltonian(Krylov_basis[int(q_index)], potential_eV, kinetic_coefficient_eV, N_grid)
for row in range(int(0), int(Krylov_dimension)):
    for column in range(int(row), int(Krylov_dimension)):
        projected_hamiltonian[int(row), int(column)] = dot_product(Krylov_basis[int(row)], H_Krylov[int(column)], N_grid)
        projected_hamiltonian[int(column), int(row)] = projected_hamiltonian[int(row), int(column)]
print(jacobi_diagonalize())
vibrational_energies_eV = zero_array(N_levels)
ψ_raw = zero_matrix(N_levels, N_grid)
for n in range(int(0), int(N_levels)):
    vibrational_energies_eV[int(n)] = Ritz_values[int(n)]
    for index in range(int(0), int(N_grid)):
        for q_index in range(int(0), int(Krylov_dimension)):
            ψ_raw[int(n), int(index)] = (ψ_raw[int(n), int(index)] + (Krylov_basis[int(q_index), int(index)] * Ritz_vectors[int(q_index), int(n)]))
    ψ_raw[int(n)] = normalize_vector(ψ_raw[int(n)], N_grid)
normalization_factor = zero_array(N_levels)
ψ = zero_matrix(N_levels, N_grid)
for n in range(int(0), int(N_levels)):
    normalization_factor[int(n)] = torch.sqrt(integrate((ψ_raw[int(n)] * ψ_raw[int(n)]), grid_spacing, N_grid) if isinstance(integrate((ψ_raw[int(n)] * ψ_raw[int(n)]), grid_spacing, N_grid), torch.Tensor) else torch.tensor(float(integrate((ψ_raw[int(n)] * ψ_raw[int(n)]), grid_spacing, N_grid))))
    for index in range(int(0), int(N_grid)):
        ψ[int(n), int(index)] = (ψ_raw[int(n), int(index)] / normalization_factor[int(n)])
transition_eV = (vibrational_energies_eV[int(1)] - vibrational_energies_eV[int(0)])
transition_J = (transition_eV * electron_volt)
wavenumber = (transition_J / (planck_constant * speed_of_light_cm))
wavelength_micrometer = (10000.0 / wavenumber)
print(print(learned_dissociation_energy_eV))
print(print(learned_α_inverse_angstrom))
print(print(learned_equilibrium_distance_angstrom))
print(print(final_learning_loss))
print(print(vibrational_energies_eV[int(0)]))
print(print(vibrational_energies_eV[int(1)]))
print(print(transition_eV))
print(print(wavenumber))
print(print(wavelength_micrometer))