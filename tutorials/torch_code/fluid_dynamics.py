import torch
import torch.nn as nn
import torch.optim as optim
from physika.runtime import DEVICE

from physika.runtime import print

# === Functions ===
def knudsen(mfp, L):
    return (mfp / L)

def pressure_ideal_gas(ρ, Rs, T):
    return ((ρ * Rs) * T)

def speed_of_sound(γ, Rs, T):
    return torch.sqrt(((γ * Rs) * T) if isinstance(((γ * Rs) * T), torch.Tensor) else torch.tensor(float(((γ * Rs) * T))))

def mach(v, c):
    return (v / c)

def reynolds(ρ, v, L, μ):
    return (((ρ * v) * L) / μ)

def ρ_expand(t, x, ρ0, a):
    return (ρ0 * torch.exp((0.0 - (a * t)) if isinstance((0.0 - (a * t)), torch.Tensor) else torch.tensor(float((0.0 - (a * t))))))

def u_expand(t, x, a):
    return (a * x)

def mass_flux(t, x, ρ0, a):
    return (ρ_expand(t, x, ρ0, a) * u_expand(t, x, a))

def continuity_residual(t, x, ρ0, a):
    dρ_dt = compute_grad(lambda _dt: ρ_expand(_dt, x, ρ0, a), t)
    dflux_dx = compute_grad(lambda _dx: mass_flux(t, _dx, ρ0, a), x)
    return (dρ_dt + dflux_dx)

def φ_expand(t, x, a):
    return (x * torch.exp((0.0 - (a * t)) if isinstance((0.0 - (a * t)), torch.Tensor) else torch.tensor(float((0.0 - (a * t))))))

def material_derivative(t, x, a):
    dφ_dt = compute_grad(lambda _dt: φ_expand(_dt, x, a), t)
    dφ_dx = compute_grad(lambda _dx: φ_expand(t, _dx, a), x)
    return (dφ_dt + (u_expand(t, x, a) * dφ_dx))

def u_linear(x, y, s, Ω):
    return ((s * x) - (Ω * y))

def v_linear(x, y, s, Ω):
    return ((Ω * x) + (s * y))

def divergence(x, y, s, Ω):
    return (compute_grad(lambda _dx: u_linear(_dx, y, s, Ω), x) + compute_grad(lambda _dy: v_linear(x, _dy, s, Ω), y))

def vorticity(x, y, s, Ω):
    return (compute_grad(lambda _dx: v_linear(_dx, y, s, Ω), x) - compute_grad(lambda _dy: u_linear(x, _dy, s, Ω), y))

def u_poiseuille(y, G, μ, H):
    return (((G / (2.0 * μ)) * y) * (H - y))

def du_dy_poiseuille(y, G, μ, H):
    return compute_grad(lambda _dy: u_poiseuille(_dy, G, μ, H), y)

def d2u_dy2_poiseuille(y, G, μ, H, h):
    return ((du_dy_poiseuille((y + h), G, μ, H) - du_dy_poiseuille((y - h), G, μ, H)) / (2.0 * h))

def p_poiseuille(x, p0, G):
    return (p0 - (G * x))

def shear_stress(y, G, μ, H):
    return (μ * du_dy_poiseuille(y, G, μ, H))

def navier_stokes_residual(x, y, p0, G, μ, H, h):
    dp_dx = compute_grad(lambda _dx: p_poiseuille(_dx, p0, G), x)
    return ((0.0 - dp_dx) + (μ * d2u_dy2_poiseuille(y, G, μ, H, h)))

def p_vortex(x, y, p0, ρ, Ω):
    return (p0 + ((((0.5 * ρ) * Ω) * Ω) * ((x * x) + (y * y))))

def euler_residual_x(x, y, p0, ρ, Ω):
    u = u_linear(x, y, 0.0, Ω)
    v = v_linear(x, y, 0.0, Ω)
    du_dx = compute_grad(lambda _dx: u_linear(_dx, y, 0.0, Ω), x)
    du_dy = compute_grad(lambda _dy: u_linear(x, _dy, 0.0, Ω), y)
    dp_dx = compute_grad(lambda _dx: p_vortex(_dx, y, p0, ρ, Ω), x)
    return ((ρ * ((u * du_dx) + (v * du_dy))) + dp_dx)

def euler_residual_y(x, y, p0, ρ, Ω):
    u = u_linear(x, y, 0.0, Ω)
    v = v_linear(x, y, 0.0, Ω)
    dv_dx = compute_grad(lambda _dx: v_linear(_dx, y, 0.0, Ω), x)
    dv_dy = compute_grad(lambda _dy: v_linear(x, _dy, 0.0, Ω), y)
    dp_dy = compute_grad(lambda _dy: p_vortex(x, _dy, p0, ρ, Ω), y)
    return ((ρ * ((u * dv_dx) + (v * dv_dy))) + dp_dy)

def dynamic_pressure(ρ, v):
    return (((0.5 * ρ) * v) * v)

def stagnation_pressure(p, ρ, v):
    return (p + dynamic_pressure(ρ, v))

def bernoulli_head(p, ρ, v, g, z):
    return ((p + (((0.5 * ρ) * v) * v)) + ((ρ * g) * z))

def venturi_velocity(v1, A1, A2):
    return ((v1 * A1) / A2)

def venturi_pressure(p1, ρ, v1, A1, A2):
    v2 = venturi_velocity(v1, A1, A2)
    return (p1 + ((0.5 * ρ) * ((v1 * v1) - (v2 * v2))))

def stagnation_pressure_compressible(p, γ, Ma):
    return (p * ((1.0 + (((0.5 * (γ - 1.0)) * Ma) * Ma)) ** (γ / (γ - 1.0))))

def stokes_drag(μ, r_s, v):
    return ((((6.0 * 3.14159) * μ) * r_s) * v)

def terminal_velocity(ρ_s, ρ_f, g, r_s, μ):
    return ((((((2.0 / 9.0) * (ρ_s - ρ_f)) * g) * r_s) * r_s) / μ)

def probe_velocity(t, U, A, ω_f):
    return (U + (A * torch.sin((ω_f * t) if isinstance((ω_f * t), torch.Tensor) else torch.tensor(float((ω_f * t))))))

def time_mean(U, A, ω_f, dt, n_samples):
    total = 0.0
    for i in range(int(0), int(n_samples)):
        total = (total + probe_velocity((i * dt), U, A, ω_f))
    return (total / n_samples)

def turbulent_kinetic_energy(U, A, ω_f, dt, n_samples):
    u_mean = time_mean(U, A, ω_f, dt, n_samples)
    total = 0.0
    u_prime = 0.0
    for i in range(int(0), int(n_samples)):
        u_prime = (probe_velocity((i * dt), U, A, ω_f) - u_mean)
        total = (total + (u_prime * u_prime))
    return ((0.5 * total) / n_samples)

# === Program ===
ρ_air, Rs_air, T_air, γ_air, μ_air = 1.225, 287.05, 288.15, 1.4, 1.81e-05
mfp_air, L_body, v_air = 6.8e-08, 1.0, 30.0
Kn = knudsen(mfp_air, L_body)
print(print(Kn))
p_air = pressure_ideal_gas(ρ_air, Rs_air, T_air)
print(print(p_air))
c_air = speed_of_sound(γ_air, Rs_air, T_air)
print(print(c_air))
Ma = mach(v_air, c_air)
print(print(Ma))
Re = reynolds(ρ_air, v_air, L_body, μ_air)
print(print(Re))
t0, x0, ρ0, a = 0.5, 2.0, 1.0, 0.3
res_continuity = continuity_residual(t0, x0, ρ0, a)
print(print(res_continuity))
Dφ_Dt = material_derivative(t0, x0, a)
print(print(Dφ_Dt))
xp, yp = 1.0, 2.0
div_vortex = divergence(xp, yp, 0.0, 1.5)
print(print(div_vortex))
ω_vortex = vorticity(xp, yp, 0.0, 1.5)
print(print(ω_vortex))
div_source = divergence(xp, yp, 0.8, 0.0)
print(print(div_source))
ω_source = vorticity(xp, yp, 0.8, 0.0)
print(print(ω_source))
μ_water, H_gap, G_drive, p_in, h_fd = 0.001, 0.01, 0.1, 101325.0, 0.0001
x_probe, y_probe = 0.5, 0.003
u_center = u_poiseuille((H_gap / 2.0), G_drive, μ_water, H_gap)
print(print(u_center))
τ_wall = shear_stress(0.0, G_drive, μ_water, H_gap)
print(print(τ_wall))
res_ns = navier_stokes_residual(x_probe, y_probe, p_in, G_drive, μ_water, H_gap, h_fd)
print(print(res_ns))
res_euler_x = euler_residual_x(xp, yp, p_in, 1000.0, 1.5)
print(print(res_euler_x))
res_euler_y = euler_residual_y(xp, yp, p_in, 1000.0, 1.5)
print(print(res_euler_y))
ρ_water, g0 = 1000.0, 9.81
v_pipe, A_pipe, A_throat = 2.0, 0.01, 0.004
v_throat = venturi_velocity(v_pipe, A_pipe, A_throat)
print(print(v_throat))
p_throat = venturi_pressure(p_in, ρ_water, v_pipe, A_pipe, A_throat)
print(print(p_throat))
head_pipe = bernoulli_head(p_in, ρ_water, v_pipe, g0, 0.0)
print(print(head_pipe))
head_throat = bernoulli_head(p_throat, ρ_water, v_throat, g0, 0.0)
print(print(head_throat))
p0_incompressible = stagnation_pressure(p_air, ρ_air, v_air)
print(print(p0_incompressible))
p0_compressible = stagnation_pressure_compressible(p_air, γ_air, Ma)
print(print(p0_compressible))
v_fast = (0.8 * c_air)
p0_incompressible_fast = stagnation_pressure(p_air, ρ_air, v_fast)
print(print(p0_incompressible_fast))
p0_compressible_fast = stagnation_pressure_compressible(p_air, γ_air, 0.8)
print(print(p0_compressible_fast))
π = 3.14159
ρ_sand, r_grain = 2650.0, 5e-05
v_settle = terminal_velocity(ρ_sand, ρ_water, g0, r_grain, μ_water)
print(print(v_settle))
Re_grain = reynolds(ρ_water, v_settle, (2.0 * r_grain), μ_water)
print(print(Re_grain))
F_drag = stokes_drag(μ_water, r_grain, v_settle)
print(print(F_drag))
F_weight = ((((((((ρ_sand - ρ_water) * g0) * 4.0) / 3.0) * π) * r_grain) * r_grain) * r_grain)
print(print(F_weight))
u_mean_probe, A_probe, ω_probe, dt_probe = 10.0, 1.5, (2.0 * π), 0.01
N_probe = 1000
u_bar = time_mean(u_mean_probe, A_probe, ω_probe, dt_probe, N_probe)
print(print(u_bar))
k_turb = turbulent_kinetic_energy(u_mean_probe, A_probe, ω_probe, dt_probe, N_probe)
print(print(k_turb))