import torch
import torch.nn as nn
import torch.optim as optim
from physika.runtime import DEVICE

from physika.runtime import print
from physika.runtime import compute_grad

# === Functions ===
def augment(M, k):
    aug = torch.zeros(int(k), int((2 * k)), device=DEVICE)
    for i in range(int(0), int(k)):
        for j in range(int(0), int((2 * k))):
            if j < k:
                aug[int(i), int(j)] = M[int(i), int(j)]
            else:
                if (j - k) == i:
                    aug[int(i), int(j)] = 1.0
    return aug

def swap_matrix(ra, rb, k):
    P = torch.zeros(int(k), int(k), device=DEVICE)
    for i in range(int(0), int(k)):
        P[int(i), int(i)] = 1.0
    P[int(ra), int(ra)] = 0.0
    P[int(rb), int(rb)] = 0.0
    P[int(ra), int(rb)] = 1.0
    P[int(rb), int(ra)] = 1.0
    return P

def elim_matrix(aug, r, c, k):
    piv = aug[int(r), int(c)]
    E = torch.zeros(int(k), int(k), device=DEVICE)
    for i in range(int(0), int(k)):
        E[int(i), int(i)] = 1.0
    E[int(r), int(r)] = (1.0 / piv)
    for i in range(int(0), int(k)):
        if i != r:
            E[int(i), int(r)] = (0.0 - (aug[int(i), int(c)] / piv))
    return E

def gje_naive(M, k):
    aug = augment(M, k)
    r = 0
    for c in range(int(0), int(k)):
        if r < k:
            found = 0.0
            cand = r
            if aug[int(r), int(c)] != 0:
                found = 1.0
            for i in range(int((r + 1)), int(k)):
                if found < 0.5:
                    if aug[int(i), int(c)] != 0:
                        cand = i
                        found = 1.0
            if found > 0.5:
                Pm = swap_matrix(r, cand, k)
                aug = (Pm @ aug)
                Em = elim_matrix(aug, r, c, k)
                aug = (Em @ aug)
                r = (r + 1)
    return aug

def gje_pp(M, k, ptol):
    aug = augment(M, k)
    r = 0
    for c in range(int(0), int(k)):
        if r < k:
            p = r
            pmax = torch.abs(aug[int(r), int(c)] if isinstance(aug[int(r), int(c)], torch.Tensor) else torch.tensor(float(aug[int(r), int(c)])))
            for i in range(int((r + 1)), int(k)):
                if torch.abs(aug[int(i), int(c)] if isinstance(aug[int(i), int(c)], torch.Tensor) else torch.tensor(float(aug[int(i), int(c)]))) > pmax:
                    pmax = torch.abs(aug[int(i), int(c)] if isinstance(aug[int(i), int(c)], torch.Tensor) else torch.tensor(float(aug[int(i), int(c)])))
                    p = i
            if pmax > ptol:
                Pm = swap_matrix(r, p, k)
                aug = (Pm @ aug)
                Em = elim_matrix(aug, r, c, k)
                aug = (Em @ aug)
                r = (r + 1)
    return aug

def identity_defect(aug, k):
    total = 0.0
    for i in range(int(0), int(k)):
        for j in range(int(0), int(k)):
            if i == j:
                total = total + torch.abs((aug[int(i), int(j)] - 1.0) if isinstance((aug[int(i), int(j)] - 1.0), torch.Tensor) else torch.tensor(float((aug[int(i), int(j)] - 1.0))))
            else:
                total = total + torch.abs(aug[int(i), int(j)] if isinstance(aug[int(i), int(j)], torch.Tensor) else torch.tensor(float(aug[int(i), int(j)])))
    return total

def extract_inv(aug, k):
    out = torch.zeros(int(k), int(k), device=DEVICE)
    out[:, :] = aug[:, int(k):int((2 * k))]
    return out

def nan_matrix(k):
    out = torch.zeros(int(k), int(k), device=DEVICE)
    for i in range(int(0), int(k)):
        for j in range(int(0), int(k)):
            out[int(i), int(j)] = torch.sqrt((0.0 - 1.0) if isinstance((0.0 - 1.0), torch.Tensor) else torch.tensor(float((0.0 - 1.0))))
    return out

def inverse_naive(M, k, tol):
    red = gje_naive(M, k)
    return torch.where(torch.as_tensor(identity_defect(red, k) < tol), extract_inv(red, k), nan_matrix(k))

def inverse_pp(M, k, tol, ptol):
    red = gje_pp(M, k, ptol)
    return torch.where(torch.as_tensor(identity_defect(red, k) < tol), extract_inv(red, k), nan_matrix(k))

def total_naive(M, k, tol):
    return torch.sum(inverse_naive(M, k, tol) if isinstance(inverse_naive(M, k, tol), torch.Tensor) else torch.tensor(float(inverse_naive(M, k, tol))))

def total_pp(M, k, tol, ptol):
    return torch.sum(inverse_pp(M, k, tol, ptol) if isinstance(inverse_pp(M, k, tol, ptol), torch.Tensor) else torch.tensor(float(inverse_pp(M, k, tol, ptol))))

# === Program ===
n = 3
tol = 0.001
ptol = 1e-06
A = torch.as_tensor(torch.tensor([[5.0, 7.0, 9.0], [4.0, 3.0, 8.0], [7.0, 5.0, 6.0]], device=DEVICE)).requires_grad_(True).to(DEVICE)
print(A)
print(inverse_naive(A, n, tol))
print(inverse_pp(A, n, tol, ptol))
B = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], device=DEVICE)
print(B)
print(inverse_naive(B, n, tol))
print(inverse_pp(B, n, tol, ptol))
n2 = 2
P = torch.as_tensor(torch.tensor([[0.0, 1.0], [1.0, 0.0]], device=DEVICE)).requires_grad_(True).to(DEVICE)
print(P)
print(inverse_naive(P, n2, tol))
print(inverse_pp(P, n2, tol, ptol))
S = torch.tensor([[1e-20, 1.0], [1.0, 2.0]], device=DEVICE)
print(S)
print(inverse_naive(S, n2, tol))
print(inverse_pp(S, n2, tol, ptol))
print(compute_grad(lambda _dA: total_naive(_dA, n, tol), A))
print(compute_grad(lambda _dA: total_pp(_dA, n, tol, ptol), A))
print(compute_grad(lambda _dP: total_naive(_dP, n2, tol), P))
print(compute_grad(lambda _dP: total_pp(_dP, n2, tol, ptol), P))