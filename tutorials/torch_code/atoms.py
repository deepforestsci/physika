import torch
import torch.nn as nn
import torch.optim as optim
from physika.runtime import DEVICE

from physika.runtime import print

# === Functions ===
def molecule(numbers, positions, masses, box):
    k = len(numbers)
    zeros1 = torch.stack([(numbers[int(a)] * 0.0) for _fi_a in range(int(k)) for a in [torch.tensor(float(_fi_a), device=DEVICE)]])
    zeros3 = torch.stack([torch.stack([(positions[int(a), int(c)] * 0.0) for _fi_c in range(int(3)) for c in [torch.tensor(float(_fi_c), device=DEVICE)]]) for _fi_a in range(int(k)) for a in [torch.tensor(float(_fi_a), device=DEVICE)]])
    cell = torch.tensor([[box, 0.0, 0.0], [0.0, box, 0.0], [0.0, 0.0, box]], device=DEVICE)
    return Atoms(numbers, positions, masses, zeros1, zeros3, zeros1, zeros1, cell)

# === Classes ===
class Atom(nn.Module):
    def __init__(self, atoms, index):
        super().__init__()
        self.add_module('atoms', atoms)
        self.index = torch.as_tensor(index).float()
        self.learnable_params = [self.index]

    def idx(self):
        this = self
        return self.index

    def number(self):
        this = self
        return self.atoms.atomic_number(self.index)

    def mass(self):
        this = self
        return self.atoms.mass(self.index)

    def position(self):
        this = self
        return self.atoms.position(self.index)

    def x(self):
        this = self
        p = self.atoms.position(self.index)
        return p[int(0)]

    def y(self):
        this = self
        p = self.atoms.position(self.index)
        return p[int(1)]

    def z(self):
        this = self
        p = self.atoms.position(self.index)
        return p[int(2)]

    def momentum(self):
        this = self
        return self.atoms.momentum(self.index)

    def tag(self):
        this = self
        return self.atoms.tag(self.index)

    def magmom(self):
        this = self
        return self.atoms.magmom(self.index)

    def charge(self):
        this = self
        return self.atoms.charge(self.index)

    def distance_to(self, b):
        this = self
        b = torch.as_tensor(b, device=DEVICE).float()
        return self.atoms.distance(self.index, b)

    @property
    def params(self):
        return list(self.parameters())

    def update(self, lr, grads):
        with torch.no_grad():
            for p, g in zip(self.parameters(), grads):
                if g is not None:
                    p -= lr * g

class Atoms(nn.Module):
    def __init__(self, numbers, positions, masses, tags, momenta, magmoms, charges, cell):
        super().__init__()
        self.numbers = torch.as_tensor(numbers).float()
        self.positions = torch.as_tensor(positions).float()
        self.masses = torch.as_tensor(masses).float()
        self.tags = torch.as_tensor(tags).float()
        self.momenta = torch.as_tensor(momenta).float()
        self.magmoms = torch.as_tensor(magmoms).float()
        self.charges = torch.as_tensor(charges).float()
        self.cell = torch.as_tensor(cell).float()
        self.learnable_params = [self.numbers, self.positions, self.masses, self.tags, self.momenta, self.magmoms, self.charges, self.cell]

    def count(self):
        this = self
        return (len(self.numbers) * 1.0)

    def atomic_number(self, a):
        this = self
        a = torch.as_tensor(a, device=DEVICE).float()
        nums = self.numbers
        return nums[int(a)]

    def mass(self, a):
        this = self
        a = torch.as_tensor(a, device=DEVICE).float()
        m = self.masses
        return m[int(a)]

    def position(self, a):
        this = self
        a = torch.as_tensor(a, device=DEVICE).float()
        pos = self.positions
        return pos[int(a)]

    def momentum(self, a):
        this = self
        a = torch.as_tensor(a, device=DEVICE).float()
        mom = self.momenta
        return mom[int(a)]

    def tag(self, a):
        this = self
        a = torch.as_tensor(a, device=DEVICE).float()
        t = self.tags
        return t[int(a)]

    def magmom(self, a):
        this = self
        a = torch.as_tensor(a, device=DEVICE).float()
        mm = self.magmoms
        return mm[int(a)]

    def charge(self, a):
        this = self
        a = torch.as_tensor(a, device=DEVICE).float()
        q = self.charges
        return q[int(a)]

    def atom(self, idx):
        this = self
        idx = torch.as_tensor(idx, device=DEVICE).float()
        return Atom(self, idx)

    def total_mass(self):
        this = self
        m = self.masses
        return torch.sum(m if isinstance(m, torch.Tensor) else torch.tensor(float(m)))

    def total_charge(self):
        this = self
        q = self.charges
        return torch.sum(q if isinstance(q, torch.Tensor) else torch.tensor(float(q)))

    def kinetic_energy(self):
        this = self
        mom = self.momenta
        mass = self.masses
        k = len(mass)
        total = 0.0
        for a in range(int(0), int(k)):
            total = total + (torch.sum((mom[int(a)] * mom[int(a)]) if isinstance((mom[int(a)] * mom[int(a)]), torch.Tensor) else torch.tensor(float((mom[int(a)] * mom[int(a)])))) / (2.0 * mass[int(a)]))
        return total

    def center_of_mass(self):
        this = self
        pos = self.positions
        mass = self.masses
        k = len(pos)
        acc = torch.stack([(c * 0.0) for _fi_c in range(int(3)) for c in [torch.tensor(float(_fi_c), device=DEVICE)]])
        for a in range(int(0), int(k)):
            acc = (acc + (pos[int(a)] * mass[int(a)]))
        return (acc * (1.0 / torch.sum(mass if isinstance(mass, torch.Tensor) else torch.tensor(float(mass)))))

    def distance(self, a, b):
        this = self
        a = torch.as_tensor(a, device=DEVICE).float()
        b = torch.as_tensor(b, device=DEVICE).float()
        pos = self.positions
        d = (pos[int(a)] - pos[int(b)])
        return torch.sqrt(torch.sum((d * d) if isinstance((d * d), torch.Tensor) else torch.tensor(float((d * d)))) if isinstance(torch.sum((d * d) if isinstance((d * d), torch.Tensor) else torch.tensor(float((d * d)))), torch.Tensor) else torch.tensor(float(torch.sum((d * d) if isinstance((d * d), torch.Tensor) else torch.tensor(float((d * d)))))))

    def all_distances(self):
        this = self
        pos = self.positions
        k = len(pos)
        result = torch.stack([torch.stack([((a + b) * 0.0) for _fi_b in range(int(k)) for b in [torch.tensor(float(_fi_b), device=DEVICE)]]) for _fi_a in range(int(k)) for a in [torch.tensor(float(_fi_a), device=DEVICE)]])
        for a in range(int(0), int(k)):
            for b in range(int(0), int(k)):
                d = (pos[int(a)] - pos[int(b)])
                result[int(a), int(b)] = torch.sqrt(torch.sum((d * d) if isinstance((d * d), torch.Tensor) else torch.tensor(float((d * d)))) if isinstance(torch.sum((d * d) if isinstance((d * d), torch.Tensor) else torch.tensor(float((d * d)))), torch.Tensor) else torch.tensor(float(torch.sum((d * d) if isinstance((d * d), torch.Tensor) else torch.tensor(float((d * d)))))))
        return result

    def count_species(self, z):
        this = self
        z = torch.as_tensor(z, device=DEVICE).float()
        nums = self.numbers
        k = len(nums)
        total = 0.0
        for a in range(int(0), int(k)):
            if nums[int(a)] == z:
                total = total + 1.0
        return total

    def volume(self):
        this = self
        c = self.cell
        m0 = ((c[int(1), int(1)] * c[int(2), int(2)]) - (c[int(1), int(2)] * c[int(2), int(1)]))
        m1 = ((c[int(1), int(0)] * c[int(2), int(2)]) - (c[int(1), int(2)] * c[int(2), int(0)]))
        m2 = ((c[int(1), int(0)] * c[int(2), int(1)]) - (c[int(1), int(1)] * c[int(2), int(0)]))
        det = (((c[int(0), int(0)] * m0) - (c[int(0), int(1)] * m1)) + (c[int(0), int(2)] * m2))
        return torch.abs(det if isinstance(det, torch.Tensor) else torch.tensor(float(det)))

    def translate(self, shift):
        this = self
        shift = torch.as_tensor(shift, device=DEVICE).float()
        pos = self.positions
        k = len(pos)
        moved = torch.stack([torch.stack([(pos[int(a), int(c)] + shift[int(c)]) for _fi_c in range(int(3)) for c in [torch.tensor(float(_fi_c), device=DEVICE)]]) for _fi_a in range(int(k)) for a in [torch.tensor(float(_fi_a), device=DEVICE)]])
        return Atoms(self.numbers, moved, self.masses, self.tags, self.momenta, self.magmoms, self.charges, self.cell)

    def centered(self):
        this = self
        com = self.center_of_mass()
        return self.translate((com * (-1.0)))

    @property
    def params(self):
        return list(self.parameters())

    def update(self, lr, grads):
        with torch.no_grad():
            for p, g in zip(self.parameters(), grads):
                if g is not None:
                    p -= lr * g

# === Program ===
w_numbers = torch.tensor([8.0, 1.0, 1.0], device=DEVICE)
w_positions = torch.tensor([[0.0, 0.0, 0.0], [0.9584, 0.0, 0.0], [(-0.2399), 0.9279, 0.0]], device=DEVICE)
w_masses = torch.tensor([15.999, 1.008, 1.008], device=DEVICE)
w_tags = torch.tensor([0.0, 1.0, 1.0], device=DEVICE)
w_momenta = torch.tensor([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [(-0.2), 0.0, 0.0]], device=DEVICE)
w_magmoms = torch.tensor([0.0, 0.0, 0.0], device=DEVICE)
w_charges = torch.tensor([(-0.834), 0.417, 0.417], device=DEVICE)
w_cell = torch.tensor([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]], device=DEVICE)
water = Atoms(w_numbers, w_positions, w_masses, w_tags, w_momenta, w_magmoms, w_charges, w_cell)
print(water.count())
print(water.total_mass())
print(water.total_charge())
print(water.kinetic_energy())
print(water.center_of_mass())
print(water.distance(0.0, 1.0))
print(water.all_distances())
o = water.atom(0.0)
print(o.idx())
print(o.number())
print(o.mass())
print(o.position())
print(o.charge())
h1 = water.atom(1.0)
print(h1.number())
print(h1.x())
print(h1.momentum())
print(h1.tag())
print(h1.magmom())
print(h1.charge())
print(h1.distance_to(2.0))
shift = torch.tensor([1.0, 2.0, 3.0], device=DEVICE)
moved = water.translate(shift)
print(moved.center_of_mass())
centred = water.centered()
print(centred.center_of_mass())
co_numbers = torch.tensor([6.0, 8.0], device=DEVICE)
co_positions = torch.tensor([[0.0, 0.0, 0.0], [1.128, 0.0, 0.0]], device=DEVICE)
co_masses = torch.tensor([12.011, 15.999], device=DEVICE)
co = molecule(co_numbers, co_positions, co_masses, 8.0)
print(co.atom(0.0).charge())
print(co.atom(1.0).tag())
print(co.kinetic_energy())
nacl_numbers = torch.tensor([11.0, 17.0], device=DEVICE)
nacl_positions = torch.tensor([[0.0, 0.0, 0.0], [2.82, 2.82, 2.82]], device=DEVICE)
nacl_masses = torch.tensor([22.99, 35.45], device=DEVICE)
nacl_tags = torch.tensor([0.0, 0.0], device=DEVICE)
nacl_momenta = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], device=DEVICE)
nacl_magmoms = torch.tensor([0.0, 0.0], device=DEVICE)
nacl_charges = torch.tensor([1.0, (-1.0)], device=DEVICE)
nacl_cell = torch.tensor([[0.0, 2.82, 2.82], [2.82, 0.0, 2.82], [2.82, 2.82, 0.0]], device=DEVICE)
nacl = Atoms(nacl_numbers, nacl_positions, nacl_masses, nacl_tags, nacl_momenta, nacl_magmoms, nacl_charges, nacl_cell)
print(nacl.count_species(11.0))
print(nacl.count_species(17.0))
print(nacl.total_charge())
print(nacl.distance(0.0, 1.0))
print(nacl.volume())