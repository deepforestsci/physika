import torch
import torch.nn as nn
import torch.optim as optim
from physika.runtime import DEVICE

from physika.runtime import print

# === Functions ===
def zero_1d_array(n):
    results = torch.stack([(i * 0.0) for _fi_i in range(int(n)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
    return results

def sum_1d_array(arr, n):
    total = 0.0
    for i in range(int(0), int(n)):
        total = (total + arr[int(i)])
    return total

def line_average(arr, n):
    s = sum_1d_array(arr, n)
    return (s / n)

def volume_average(arr, vp, n):
    weighted_sum = 0.0
    vol_sum = 0.0
    for i in range(int(0), int(n)):
        weighted_sum = (weighted_sum + (arr[int(i)] * vp[int(i)]))
        vol_sum = (vol_sum + vp[int(i)])
    return (weighted_sum / vol_sum)

def solve_thomas(sub, diag, sup, rhs, n):
    cp = zero_1d_array(n)
    dp = zero_1d_array(n)
    cp[int(0)] = (sup[int(0)] / diag[int(0)])
    dp[int(0)] = (rhs[int(0)] / diag[int(0)])
    last = (n - 1)
    denom = 0.0
    for i in range(int(1), int(n)):
        denom = (diag[int(i)] - (sub[int((i - 1))] * cp[int((i - 1))]))
        if i < last:
            cp[int(i)] = (sup[int(i)] / denom)
        dp[int(i)] = ((rhs[int(i)] - (sub[int((i - 1))] * dp[int((i - 1))])) / denom)
    x = zero_1d_array(n)
    x[int(last)] = dp[int(last)]
    idx = 0
    next_idx = 0
    for k in range(int(1), int(n)):
        idx = ((n - 1) - k)
        next_idx = (idx + 1)
        x[int(idx)] = (dp[int(idx)] - (cp[int(idx)] * x[int(next_idx)]))
    return x

# === Program ===
n_cells = 50
n_faces = 51
n_sub = 49
drho = (1.0 / n_cells)
rho = torch.stack([((i + 0.5) * drho) for _fi_i in range(int(n_cells)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
rho_faces = torch.stack([(i * drho) for _fi_i in range(int(n_faces)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
v_prime = rho
T_edge = 0.2
w = 0.2
S_e0 = 145.0
S_i0 = 225.0
S_e = torch.stack([(S_e0 * torch.exp((-(((rho[int(i)] - 0.11) / w) ** 2)) if isinstance((-(((rho[int(i)] - 0.11) / w) ** 2)), torch.Tensor) else torch.tensor(float((-(((rho[int(i)] - 0.11) / w) ** 2)))))) for _fi_i in range(int(n_cells)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
S_i = torch.stack([(S_i0 * torch.exp((-(((rho[int(i)] - 0.11) / w) ** 2)) if isinstance((-(((rho[int(i)] - 0.11) / w) ** 2)), torch.Tensor) else torch.tensor(float((-(((rho[int(i)] - 0.11) / w) ** 2)))))) for _fi_i in range(int(n_cells)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
chi_e_const = 1.0
chi_i_const = 2.0
flux_factor = (1.0 / (drho * drho))
flux_term_e = torch.stack([((rho_faces[int(i)] * chi_e_const) * flux_factor) for _fi_i in range(int(n_faces)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
flux_term_i = torch.stack([((rho_faces[int(i)] * chi_i_const) * flux_factor) for _fi_i in range(int(n_faces)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
edge_coeff_e = (2.0 * flux_term_e[int(n_cells)])
edge_coeff_i = (2.0 * flux_term_i[int(n_cells)])
dt = 0.05
n_steps = 200
sub_e = torch.stack([(-flux_term_e[int((i + 1))]) for _fi_i in range(int(n_sub)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
sup_e = torch.stack([(-flux_term_e[int((i + 1))]) for _fi_i in range(int(n_sub)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
diag_e = torch.stack([(((v_prime[int(i)] / dt) + flux_term_e[int(i)]) + flux_term_e[int((i + 1))]) for _fi_i in range(int(n_cells)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
diag_e[int(0)] = ((v_prime[int(0)] / dt) + flux_term_e[int(1)])
diag_e[int((n_cells - 1))] = (((v_prime[int((n_cells - 1))] / dt) + flux_term_e[int((n_cells - 1))]) + edge_coeff_e)
sub_i = torch.stack([(-flux_term_i[int((i + 1))]) for _fi_i in range(int(n_sub)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
sup_i = torch.stack([(-flux_term_i[int((i + 1))]) for _fi_i in range(int(n_sub)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
diag_i = torch.stack([(((v_prime[int(i)] / dt) + flux_term_i[int(i)]) + flux_term_i[int((i + 1))]) for _fi_i in range(int(n_cells)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
diag_i[int(0)] = ((v_prime[int(0)] / dt) + flux_term_i[int(1)])
diag_i[int((n_cells - 1))] = (((v_prime[int((n_cells - 1))] / dt) + flux_term_i[int((n_cells - 1))]) + edge_coeff_i)
T_e = torch.stack([(((8.0 - 0.2) * (1.0 - rho[int(i)])) + T_edge) for _fi_i in range(int(n_cells)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
T_i = torch.stack([(((8.0 - 0.2) * (1.0 - rho[int(i)])) + T_edge) for _fi_i in range(int(n_cells)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
initial_line_avg_Te = line_average(T_e, n_cells)
initial_line_avg_Ti = line_average(T_i, n_cells)
nu_ei = 0.15
rhs_e = zero_1d_array(n_cells)
rhs_i = zero_1d_array(n_cells)
Q_ei = zero_1d_array(n_cells)
for step in range(int(0), int(n_steps)):
    Q_ei = (nu_ei * (T_e - T_i))
    rhs_e = ((((v_prime / dt) * T_e) + (v_prime * S_e)) - Q_ei)
    rhs_e[int((n_cells - 1))] = (rhs_e[int((n_cells - 1))] + (edge_coeff_e * T_edge))
    T_e = solve_thomas(sub_e, diag_e, sup_e, rhs_e, n_cells)
    rhs_i = ((((v_prime / dt) * T_i) + (v_prime * S_i)) + Q_ei)
    rhs_i[int((n_cells - 1))] = (rhs_i[int((n_cells - 1))] + (edge_coeff_i * T_edge))
    T_i = solve_thomas(sub_i, diag_i, sup_i, rhs_i, n_cells)
final_Te_core = T_e[int(0)]
final_Ti_core = T_i[int(0)]
final_Te_edge = T_e[int((n_cells - 1))]
final_Ti_edge = T_i[int((n_cells - 1))]
final_line_avg_Te = line_average(T_e, n_cells)
final_line_avg_Ti = line_average(T_i, n_cells)
final_vol_avg_Te = volume_average(T_e, v_prime, n_cells)
final_vol_avg_Ti = volume_average(T_i, v_prime, n_cells)
print(initial_line_avg_Te)
print(final_Te_core)
print(final_Ti_core)
print(final_line_avg_Te)
print(final_line_avg_Ti)
print(final_vol_avg_Te)
print(final_vol_avg_Ti)
T_pred = solve_thomas(sub_e, diag_e, sup_e, rhs_e, n_cells)
T_corr = solve_thomas(sub_e, diag_e, sup_e, rhs_e, n_cells)
pc_residual = torch.abs((T_corr[int(0)] - T_pred[int(0)]) if isinstance((T_corr[int(0)] - T_pred[int(0)]), torch.Tensor) else torch.tensor(float((T_corr[int(0)] - T_pred[int(0)]))))
print(pc_residual)