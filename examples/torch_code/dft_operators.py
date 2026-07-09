import torch
import torch.nn as nn
import torch.optim as optim

from physika.runtime import physika_print

# === Functions ===
def op_O(W, Ω):
    return (Ω * W)

def op_O_mat(W, Ω):
    return (Ω * W)

def op_L(W, G2c, Ω):
    return (((-Ω) * G2c) * W)

def op_L_mat(W, G2c, Ω):
    Ns = len(W)
    out = torch.stack([op_L(W[int(i)], G2c, Ω) for _fi_i in range(int(Ns)) for i in [torch.tensor(float(_fi_i))]])
    return out

def op_Linv(W, G2, Ω):
    nonzero = (torch.gt(G2, 0.0) * 1.0)
    safe_G2 = (G2 + (1.0 - nonzero))
    return (((W / safe_G2) / (-Ω)) * nonzero)

def op_Linv_mat(W, G2, Ω):
    Ns = len(W)
    out = torch.stack([op_Linv(W[int(i)], G2, Ω) for _fi_i in range(int(Ns)) for i in [torch.tensor(float(_fi_i))]])
    return out

def op_J(W, s1, s2, s3):
    n = ((s1 * s2) * s3)
    cube = torch.reshape(W, (int(s1), int(s2), int(s3),))
    spec = torch.fft.fftn(cube)
    return (torch.reshape(spec, (int(n),)) / n)

def op_J_mat(W, s1, s2, s3):
    Ns = len(W)
    out = torch.stack([op_J(W[int(i)], s1, s2, s3) for _fi_i in range(int(Ns)) for i in [torch.tensor(float(_fi_i))]])
    return out

def op_I(W, s1, s2, s3):
    n = ((s1 * s2) * s3)
    cube = torch.reshape(W, (int(s1), int(s2), int(s3),))
    field = torch.fft.ifftn(cube)
    return (torch.reshape(field, (int(n),)) * n)

def op_I_mat(W, s1, s2, s3):
    Ns = len(W)
    out = torch.stack([op_I(W[int(i)], s1, s2, s3) for _fi_i in range(int(Ns)) for i in [torch.tensor(float(_fi_i))]])
    return out

def op_Idag(W, active, s1, s2, s3):
    n = ((s1 * s2) * s3)
    F = op_J(W, s1, s2, s3)
    return (torch.masked_select(F, active) * n)

def op_Idag_mat(W, active, s1, s2, s3):
    Ns = len(W)
    out = torch.stack([op_Idag(W[int(i)], active, s1, s2, s3) for _fi_i in range(int(Ns)) for i in [torch.tensor(float(_fi_i))]])
    return out

def op_Jdag(W, s1, s2, s3):
    n = ((s1 * s2) * s3)
    return (op_I(W, s1, s2, s3) / n)

def op_Jdag_mat(W, s1, s2, s3):
    n = ((s1 * s2) * s3)
    return (op_I_mat(W, s1, s2, s3) / n)

# === Program ===