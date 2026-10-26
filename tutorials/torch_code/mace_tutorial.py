import torch
import torch.nn as nn
import torch.optim as optim
from physika.runtime import DEVICE

from physika.runtime import print

# === Functions ===
def one_hot(numbers, table):
    num_atoms, num_species = len(numbers), len(table)
    encoding = torch.zeros(int(num_atoms), int(num_species), device=DEVICE)
    for atom in range(int(0), int(num_atoms)):
        for species in range(int(0), int(num_species)):
            if numbers[int(atom)] == table[int(species)]:
                encoding[int(atom), int(species)] = 1.0
    return encoding

def neighbor_search(positions, cutoff):
    num_atoms = len(positions)
    vector, distance = torch.zeros(int(3), device=DEVICE), 0.0
    num_edges = 0
    for sender in range(int(0), int(num_atoms)):
        for receiver in range(int(0), int(num_atoms)):
            vector = (positions[int(receiver)] - positions[int(sender)])
            distance = torch.sqrt(torch.sum((vector * vector) if isinstance((vector * vector), torch.Tensor) else torch.tensor(float((vector * vector)))) if isinstance(torch.sum((vector * vector) if isinstance((vector * vector), torch.Tensor) else torch.tensor(float((vector * vector)))), torch.Tensor) else torch.tensor(float(torch.sum((vector * vector) if isinstance((vector * vector), torch.Tensor) else torch.tensor(float((vector * vector)))))))
            if distance > 0.0:
                if distance < cutoff:
                    num_edges = num_edges + 1
    edge_index = torch.zeros(int(2), int(num_edges), device=DEVICE)
    edge = 0
    for sender in range(int(0), int(num_atoms)):
        for receiver in range(int(0), int(num_atoms)):
            vector = (positions[int(receiver)] - positions[int(sender)])
            distance = torch.sqrt(torch.sum((vector * vector) if isinstance((vector * vector), torch.Tensor) else torch.tensor(float((vector * vector)))) if isinstance(torch.sum((vector * vector) if isinstance((vector * vector), torch.Tensor) else torch.tensor(float((vector * vector)))), torch.Tensor) else torch.tensor(float(torch.sum((vector * vector) if isinstance((vector * vector), torch.Tensor) else torch.tensor(float((vector * vector)))))))
            if distance > 0.0:
                if distance < cutoff:
                    edge_index[:, int(edge)] = torch.stack([torch.as_tensor(sender), torch.as_tensor(receiver)])
                    edge = edge + 1
    return edge_index

def node_embedding(node_attrs, w):
    return ((node_attrs @ w) / torch.sqrt(num_elements if isinstance(num_elements, torch.Tensor) else torch.tensor(float(num_elements))))

def edge_vectors(positions, edge_index):
    num_edges = len(edge_index[int(0)])
    vectors = torch.zeros(int(num_edges), int(3), device=DEVICE)
    for edge in range(int(0), int(num_edges)):
        sender, receiver = edge_index[:, int(edge)]
        vectors[int(edge)] = (positions[int(receiver)] - positions[int(sender)])
    return vectors

def edge_lengths(vectors):
    x, y, z = vectors[:, int(0)], vectors[:, int(1)], vectors[:, int(2)]
    return torch.sqrt((((x * x) + (y * y)) + (z * z)) if isinstance((((x * x) + (y * y)) + (z * z)), torch.Tensor) else torch.tensor(float((((x * x) + (y * y)) + (z * z)))))

def radial_embedding(lengths):
    num_edges = len(lengths)
    edge_feats = torch.zeros(int(num_edges), int(num_bessel), device=DEVICE)
    x = (lengths / r_cut)
    f_cut = (((1.0 - ((((p_env + 1.0) * (p_env + 2.0)) / 2.0) * (x ** p_env))) + ((p_env * (p_env + 2.0)) * (x ** (p_env + 1.0)))) - (((p_env * (p_env + 1.0)) / 2.0) * (x ** (p_env + 2.0))))
    for n in range(int(0), int(num_bessel)):
        edge_feats[:, int(n)] = (((torch.sqrt((2.0 / r_cut) if isinstance((2.0 / r_cut), torch.Tensor) else torch.tensor(float((2.0 / r_cut)))) * torch.sin(((((n + 1.0) * π) * lengths) / r_cut) if isinstance(((((n + 1.0) * π) * lengths) / r_cut), torch.Tensor) else torch.tensor(float(((((n + 1.0) * π) * lengths) / r_cut))))) / lengths) * f_cut)
    return edge_feats

def Y0(vectors, lengths):
    num_edges = len(lengths)
    results = torch.zeros(int(num_edges), int(1), device=DEVICE)
    results[:, int(0)] = 1.0
    return results

def Y1(vectors, lengths):
    num_edges = len(lengths)
    results = torch.zeros(int(num_edges), int(3), device=DEVICE)
    x, y, z = (vectors[:, int(0)] / lengths), (vectors[:, int(1)] / lengths), (vectors[:, int(2)] / lengths)
    results[:, int(0)] = (torch.sqrt(3.0 if isinstance(3.0, torch.Tensor) else torch.tensor(float(3.0))) * x)
    results[:, int(1)] = (torch.sqrt(3.0 if isinstance(3.0, torch.Tensor) else torch.tensor(float(3.0))) * y)
    results[:, int(2)] = (torch.sqrt(3.0 if isinstance(3.0, torch.Tensor) else torch.tensor(float(3.0))) * z)
    return results

def Y2(vectors, lengths):
    num_edges = len(lengths)
    results = torch.zeros(int(num_edges), int(5), device=DEVICE)
    x, y, z = (vectors[:, int(0)] / lengths), (vectors[:, int(1)] / lengths), (vectors[:, int(2)] / lengths)
    results[:, int(0)] = ((torch.sqrt(15.0 if isinstance(15.0, torch.Tensor) else torch.tensor(float(15.0))) * x) * z)
    results[:, int(1)] = ((torch.sqrt(15.0 if isinstance(15.0, torch.Tensor) else torch.tensor(float(15.0))) * x) * y)
    results[:, int(2)] = (torch.sqrt(5.0 if isinstance(5.0, torch.Tensor) else torch.tensor(float(5.0))) * ((y * y) - (0.5 * ((x * x) + (z * z)))))
    results[:, int(3)] = ((torch.sqrt(15.0 if isinstance(15.0, torch.Tensor) else torch.tensor(float(15.0))) * y) * z)
    results[:, int(4)] = ((torch.sqrt(15.0 if isinstance(15.0, torch.Tensor) else torch.tensor(float(15.0))) / 2.0) * ((z * z) - (x * x)))
    return results

def Y3(vectors, lengths):
    num_edges = len(lengths)
    results = torch.zeros(int(num_edges), int(7), device=DEVICE)
    x, y, z = (vectors[:, int(0)] / lengths), (vectors[:, int(1)] / lengths), (vectors[:, int(2)] / lengths)
    results[:, int(0)] = (torch.sqrt((35.0 / 8.0) if isinstance((35.0 / 8.0), torch.Tensor) else torch.tensor(float((35.0 / 8.0)))) * ((((3.0 * x) * z) * z) - ((x * x) * x)))
    results[:, int(1)] = (((torch.sqrt(105.0 if isinstance(105.0, torch.Tensor) else torch.tensor(float(105.0))) * x) * y) * z)
    results[:, int(2)] = ((torch.sqrt((21.0 / 8.0) if isinstance((21.0 / 8.0), torch.Tensor) else torch.tensor(float((21.0 / 8.0)))) * x) * ((((4.0 * y) * y) - (x * x)) - (z * z)))
    results[:, int(3)] = (((torch.sqrt(7.0 if isinstance(7.0, torch.Tensor) else torch.tensor(float(7.0))) / 2.0) * y) * ((((2.0 * y) * y) - ((3.0 * x) * x)) - ((3.0 * z) * z)))
    results[:, int(4)] = ((torch.sqrt((21.0 / 8.0) if isinstance((21.0 / 8.0), torch.Tensor) else torch.tensor(float((21.0 / 8.0)))) * z) * ((((4.0 * y) * y) - (x * x)) - (z * z)))
    results[:, int(5)] = (((torch.sqrt(105.0 if isinstance(105.0, torch.Tensor) else torch.tensor(float(105.0))) / 2.0) * y) * ((z * z) - (x * x)))
    results[:, int(6)] = (torch.sqrt((35.0 / 8.0) if isinstance((35.0 / 8.0), torch.Tensor) else torch.tensor(float((35.0 / 8.0)))) * (((z * z) * z) - (((3.0 * x) * x) * z)))
    return results

def spherical_harmonics(vectors, lengths):
    num_edges = len(lengths)
    edge_attrs = torch.zeros(int(num_edges), int(16), device=DEVICE)
    edge_attrs[:, int(0):int(1)] = Y0(vectors, lengths)
    edge_attrs[:, int(1):int(4)] = Y1(vectors, lengths)
    edge_attrs[:, int(4):int(9)] = Y2(vectors, lengths)
    edge_attrs[:, int(9):int(16)] = Y3(vectors, lengths)
    return edge_attrs

def linear_up(node_feats, w):
    return ((node_feats @ w) / torch.sqrt(num_channels if isinstance(num_channels, torch.Tensor) else torch.tensor(float(num_channels))))

def silu(x):
    return ((silu_norm * x) / (1.0 + torch.exp((-x) if isinstance((-x), torch.Tensor) else torch.tensor(float((-x))))))

def radial_mlp(edge_feats, w1, w2, w3, w4):
    h1 = silu(((edge_feats @ w1) / torch.sqrt(num_bessel if isinstance(num_bessel, torch.Tensor) else torch.tensor(float(num_bessel)))))
    h2 = silu(((h1 @ w2) / torch.sqrt(radial_hidden if isinstance(radial_hidden, torch.Tensor) else torch.tensor(float(radial_hidden)))))
    h3 = silu(((h2 @ w3) / torch.sqrt(radial_hidden if isinstance(radial_hidden, torch.Tensor) else torch.tensor(float(radial_hidden)))))
    return ((h3 @ w4) / torch.sqrt(radial_hidden if isinstance(radial_hidden, torch.Tensor) else torch.tensor(float(radial_hidden))))

def conv_tp(node_feats_up, edge_attrs, tp_weights, edge_index):
    num_edges = len(edge_attrs)
    mji = torch.zeros(int(num_edges), int(num_channels), int(16), device=DEVICE)
    for edge in range(int(0), int(num_edges)):
        sender, receiver = edge_index[:, int(edge)]
        h = node_feats_up[int(sender)]
        s0, s1, s2, s3 = (h * tp_weights[int(edge), int(0):int(128)]), (h * tp_weights[int(edge), int(128):int(256)]), (h * tp_weights[int(edge), int(256):int(384)]), (h * tp_weights[int(edge), int(384):int(512)])
        mji[int(edge), :, int(0)] = (s0 * edge_attrs[int(edge), int(0)])
        for m in range(int(0), int(3)):
            mji[int(edge), :, int((1 + m))] = (s1 * edge_attrs[int(edge), int((1 + m))])
        for m in range(int(0), int(5)):
            mji[int(edge), :, int((4 + m))] = (s2 * edge_attrs[int(edge), int((4 + m))])
        for m in range(int(0), int(7)):
            mji[int(edge), :, int((9 + m))] = (s3 * edge_attrs[int(edge), int((9 + m))])
    return mji

def neighbor_sum(mji, edge_index, num_atoms):
    num_edges = len(mji)
    message = torch.zeros(int(num_atoms), int(num_channels), int(16), device=DEVICE)
    for edge in range(int(0), int(num_edges)):
        sender, receiver = edge_index[:, int(edge)]
        message[int(receiver)] = (message[int(receiver)] + mji[int(edge)])
    return message

def linear(message, w0, w1, w2, w3):
    num_atoms = len(message)
    out = torch.zeros(int(num_atoms), int(num_channels), int(16), device=DEVICE)
    out[:, :, int(0):int(1)] = ((w0 @ message[:, :, int(0):int(1)]) / torch.sqrt(num_channels if isinstance(num_channels, torch.Tensor) else torch.tensor(float(num_channels))))
    out[:, :, int(1):int(4)] = ((w1 @ message[:, :, int(1):int(4)]) / torch.sqrt(num_channels if isinstance(num_channels, torch.Tensor) else torch.tensor(float(num_channels))))
    out[:, :, int(4):int(9)] = ((w2 @ message[:, :, int(4):int(9)]) / torch.sqrt(num_channels if isinstance(num_channels, torch.Tensor) else torch.tensor(float(num_channels))))
    out[:, :, int(9):int(16)] = ((w3 @ message[:, :, int(9):int(16)]) / torch.sqrt(num_channels if isinstance(num_channels, torch.Tensor) else torch.tensor(float(num_channels))))
    return (out / avg_num_neighbors)

def B_features_1(A, U):
    num_atoms = len(A)
    num_paths = len(U[int(0)])
    B1 = torch.zeros(int(num_atoms), int(num_channels), int(num_paths), device=DEVICE)
    for atom in range(int(0), int(num_atoms)):
        B1[int(atom)] = (A[int(atom)] @ U)
    return B1

def B_features_2(A, U):
    num_atoms = len(A)
    num_paths = len(U[int(0), int(0)])
    B2 = torch.zeros(int(num_atoms), int(num_channels), int(num_paths), device=DEVICE)
    for atom in range(int(0), int(num_atoms)):
        for i in range(int(0), int(16)):
            M = (A[int(atom)] @ U[int(i)])
            for p in range(int(0), int(num_paths)):
                B2[int(atom), :, int(p)] = (B2[int(atom), :, int(p)] + (A[int(atom), :, int(i)] * M[:, int(p)]))
    return B2

def B_features_3(A, U):
    num_atoms = len(A)
    num_paths = len(U[int(0), int(0), int(0)])
    B3 = torch.zeros(int(num_atoms), int(num_channels), int(num_paths), device=DEVICE)
    for atom in range(int(0), int(num_atoms)):
        for i in range(int(0), int(16)):
            for j in range(int(0), int(16)):
                M = (A[int(atom)] @ U[int(i), int(j)])
                for p in range(int(0), int(num_paths)):
                    B3[int(atom), :, int(p)] = (B3[int(atom), :, int(p)] + ((A[int(atom), :, int(i)] * A[int(atom), :, int(j)]) * M[:, int(p)]))
    return B3

def weighted_sum(B1, B2, B3, node_attrs, wp1, wp2, wp3):
    num_atoms = len(B1)
    m = torch.zeros(int(num_atoms), int(num_channels), device=DEVICE)
    for p in range(int(0), int(len(B1[int(0), int(0)]))):
        m = (m + ((node_attrs @ wp1[:, int(p), :]) * B1[:, :, int(p)]))
    for p in range(int(0), int(len(B2[int(0), int(0)]))):
        m = (m + ((node_attrs @ wp2[:, int(p), :]) * B2[:, :, int(p)]))
    for p in range(int(0), int(len(B3[int(0), int(0)]))):
        m = (m + ((node_attrs @ wp3[:, int(p), :]) * B3[:, :, int(p)]))
    return m

def scalar_product(A, node_attrs):
    return weighted_sum(B_features_1(A, U1), B_features_2(A, U2), B_features_3(A, U3), node_attrs, w_prod1, w_prod2, w_prod3)

def vector_product(A, node_attrs):
    num_atoms = len(A)
    out = torch.zeros(int(num_atoms), int(num_channels), int(3), device=DEVICE)
    for m in range(int(0), int(3)):
        out[:, :, int(m)] = weighted_sum(B_features_1(A, U1v[int(m)]), B_features_2(A, U2v[int(m)]), B_features_3(A, U3v[int(m)]), node_attrs, w_prod1v, w_prod2v, w_prod3v)
    return out

def skip_tp(node_feats, node_attrs, w):
    num_atoms = len(node_feats)
    sc = torch.zeros(int(num_atoms), int(num_channels), device=DEVICE)
    for atom in range(int(0), int(num_atoms)):
        for z in range(int(0), int(num_elements)):
            if node_attrs[int(atom), int(z)] == 1.0:
                sc[int(atom)] = ((node_feats[int(atom)] @ w[:, int(z), :]) / torch.sqrt((num_channels * num_elements) if isinstance((num_channels * num_elements), torch.Tensor) else torch.tensor(float((num_channels * num_elements)))))
    return sc

def node_update(m, sc, wp):
    return (((m @ wp) / torch.sqrt(num_channels if isinstance(num_channels, torch.Tensor) else torch.tensor(float(num_channels)))) + sc)

def node_update_vectors(mv, wp):
    num_atoms = len(mv)
    out = torch.zeros(int(num_atoms), int(num_channels), int(3), device=DEVICE)
    for atom in range(int(0), int(num_atoms)):
        out[int(atom)] = ((wp @ mv[int(atom)]) / torch.sqrt(num_channels if isinstance(num_channels, torch.Tensor) else torch.tensor(float(num_channels))))
    return out

def readout(node_feats, w):
    return ((node_feats @ w) / torch.sqrt(num_channels if isinstance(num_channels, torch.Tensor) else torch.tensor(float(num_channels))))

def linear_up_vectors(node_vecs, w):
    num_atoms = len(node_vecs)
    out = torch.zeros(int(num_atoms), int(num_channels), int(3), device=DEVICE)
    for atom in range(int(0), int(num_atoms)):
        out[int(atom)] = ((w @ node_vecs[int(atom)]) / torch.sqrt(num_channels if isinstance(num_channels, torch.Tensor) else torch.tensor(float(num_channels))))
    return out

def neighbor_sum_second(mji, edge_index, num_atoms):
    num_edges = len(mji)
    message = torch.zeros(int(num_atoms), int(num_channels), int(40), device=DEVICE)
    for edge in range(int(0), int(num_edges)):
        sender, receiver = edge_index[:, int(edge)]
        message[int(receiver)] = (message[int(receiver)] + mji[int(edge)])
    return message

def conv_tp_second(up, up_vecs, edge_attrs, tp_weights, edge_index):
    num_edges = len(edge_attrs)
    mji = torch.zeros(int(num_edges), int(num_channels), int(40), device=DEVICE)
    y1 = torch.zeros(int(3), int(1), device=DEVICE)
    y2 = torch.zeros(int(5), int(1), device=DEVICE)
    y3 = torch.zeros(int(7), int(1), device=DEVICE)
    for edge in range(int(0), int(num_edges)):
        sender, receiver = edge_index[:, int(edge)]
        s, v = up[int(sender)], up_vecs[int(sender)]
        y1[:, int(0)] = edge_attrs[int(edge), int(1):int(4)]
        y2[:, int(0)] = edge_attrs[int(edge), int(4):int(9)]
        y3[:, int(0)] = edge_attrs[int(edge), int(9):int(16)]
        d = (v @ y1)
        mji[int(edge), :, int(0)] = ((tp_weights[int(edge), int(0):int(128)] * s) * edge_attrs[int(edge), int(0)])
        mji[int(edge), :, int(1)] = ((tp_weights[int(edge), int(128):int(256)] * d[:, int(0)]) / torch.sqrt(3.0 if isinstance(3.0, torch.Tensor) else torch.tensor(float(3.0))))
        for m in range(int(0), int(3)):
            a = (v @ (cg_121[:, :, int(m)] @ y2))
            mji[int(edge), :, int((2 + m))] = ((tp_weights[int(edge), int(256):int(384)] * s) * y1[int(m), int(0)])
            mji[int(edge), :, int((5 + m))] = ((tp_weights[int(edge), int(384):int(512)] * v[:, int(m)]) * edge_attrs[int(edge), int(0)])
            mji[int(edge), :, int((8 + m))] = ((tp_weights[int(edge), int(512):int(640)] * torch.sqrt(3.0 if isinstance(3.0, torch.Tensor) else torch.tensor(float(3.0)))) * a[:, int(0)])
        for m in range(int(0), int(5)):
            b = (v @ (cg_112[:, :, int(m)] @ y1))
            c = (v @ (cg_132[:, :, int(m)] @ y3))
            mji[int(edge), :, int((11 + m))] = ((tp_weights[int(edge), int(640):int(768)] * s) * y2[int(m), int(0)])
            mji[int(edge), :, int((16 + m))] = ((tp_weights[int(edge), int(768):int(896)] * torch.sqrt(5.0 if isinstance(5.0, torch.Tensor) else torch.tensor(float(5.0)))) * b[:, int(0)])
            mji[int(edge), :, int((21 + m))] = ((tp_weights[int(edge), int(896):int(1024)] * torch.sqrt(5.0 if isinstance(5.0, torch.Tensor) else torch.tensor(float(5.0)))) * c[:, int(0)])
        for m in range(int(0), int(7)):
            d3 = (v @ (cg_123[:, :, int(m)] @ y2))
            mji[int(edge), :, int((26 + m))] = ((tp_weights[int(edge), int(1024):int(1152)] * s) * y3[int(m), int(0)])
            mji[int(edge), :, int((33 + m))] = ((tp_weights[int(edge), int(1152):int(1280)] * torch.sqrt(7.0 if isinstance(7.0, torch.Tensor) else torch.tensor(float(7.0)))) * d3[:, int(0)])
    return mji

def linear_second(message, w0, w1, w2, w3, w4, w5, w6, w7, w8, w9):
    num_atoms = len(message)
    out = torch.zeros(int(num_atoms), int(num_channels), int(16), device=DEVICE)
    out[:, :, int(0):int(1)] = (((w0 @ message[:, :, int(0):int(1)]) + (w1 @ message[:, :, int(1):int(2)])) / torch.sqrt(256.0 if isinstance(256.0, torch.Tensor) else torch.tensor(float(256.0))))
    out[:, :, int(1):int(4)] = ((((w2 @ message[:, :, int(2):int(5)]) + (w3 @ message[:, :, int(5):int(8)])) + (w4 @ message[:, :, int(8):int(11)])) / torch.sqrt(384.0 if isinstance(384.0, torch.Tensor) else torch.tensor(float(384.0))))
    out[:, :, int(4):int(9)] = ((((w5 @ message[:, :, int(11):int(16)]) + (w6 @ message[:, :, int(16):int(21)])) + (w7 @ message[:, :, int(21):int(26)])) / torch.sqrt(384.0 if isinstance(384.0, torch.Tensor) else torch.tensor(float(384.0))))
    out[:, :, int(9):int(16)] = (((w8 @ message[:, :, int(26):int(33)]) + (w9 @ message[:, :, int(33):int(40)])) / torch.sqrt(256.0 if isinstance(256.0, torch.Tensor) else torch.tensor(float(256.0))))
    return (out / avg_num_neighbors)

def readout2(node_feats, w1, w2):
    hidden = silu(((node_feats @ w1) / torch.sqrt(num_channels if isinstance(num_channels, torch.Tensor) else torch.tensor(float(num_channels)))))
    return ((hidden @ w2) / torch.sqrt(16.0 if isinstance(16.0, torch.Tensor) else torch.tensor(float(16.0))))

def mace_energy(positions, numbers):
    num_atoms = len(positions)
    node_attrs = one_hot(numbers, z_table)
    edge_index = neighbor_search(positions, r_cut)
    vectors = edge_vectors(positions, edge_index)
    lengths = edge_lengths(vectors)
    edge_feats, edge_attrs = radial_embedding(lengths), spherical_harmonics(vectors, lengths)
    node_feats0 = node_embedding(node_attrs, w_embed)
    node_feats_up = linear_up(node_feats0, w_up)
    tp_weights = radial_mlp(edge_feats, w_r1, w_r2, w_r3, w_r4)
    mji = conv_tp(node_feats_up, edge_attrs, tp_weights, edge_index)
    message = linear(neighbor_sum(mji, edge_index, num_atoms), w_0, w_1, w_2, w_3)
    sc = skip_tp(node_feats0, node_attrs, w_sc)
    node_feats1 = node_update(scalar_product(message, node_attrs), sc, w_p)
    node_vecs1 = node_update_vectors(vector_product(message, node_attrs), w_pv)
    node_energies = readout(node_feats1, w_readout)
    node_feats_up2 = linear_up(node_feats1, w2_up)
    node_vecs_up2 = linear_up_vectors(node_vecs1, w2_up_v)
    tp_weights2 = radial_mlp(edge_feats, w2_r1, w2_r2, w2_r3, w2_r4)
    mji2 = conv_tp_second(node_feats_up2, node_vecs_up2, edge_attrs, tp_weights2, edge_index)
    message2 = linear_second(neighbor_sum_second(mji2, edge_index, num_atoms), w2_0, w2_1, w2_2, w2_3, w2_4, w2_5, w2_6, w2_7, w2_8, w2_9)
    sc2 = skip_tp(node_feats1, node_attrs, w2_sc)
    node_feats2 = node_update(weighted_sum(B_features_1(message2, U1), B_features_2(message2, U2), B_features_3(message2, U3), node_attrs, w2_prod1, w2_prod2, w2_prod3), sc2, w2_p)
    node_energies2 = readout2(node_feats2, w2_ro1, w2_ro2)
    atom_energies = (node_attrs @ atomic_energies)
    energy = ((torch.sum(atom_energies if isinstance(atom_energies, torch.Tensor) else torch.tensor(float(atom_energies))) + (scale * (torch.sum(node_energies if isinstance(node_energies, torch.Tensor) else torch.tensor(float(node_energies))) + torch.sum(node_energies2 if isinstance(node_energies2, torch.Tensor) else torch.tensor(float(node_energies2)))))) + (shift * num_atoms))
    return energy

# === Program ===
water_positions = torch.tensor([[0.0, 0.0, 0.0], [1.3035, 0.121, (-0.0361)], [0.0387, 1.0143, (-0.0409)]], device=DEVICE)
water_numbers = torch.tensor([8.0, 1.0, 1.0], device=DEVICE)
methane_positions = torch.tensor([[0.0, 0.0, 0.0], [0.629, 0.629, 0.629], [(-0.629), (-0.629), 0.629], [(-0.629), 0.629, (-0.629)], [0.629, (-0.629), (-0.629)]], device=DEVICE)
methane_numbers = torch.tensor([6.0, 1.0, 1.0, 1.0, 1.0], device=DEVICE)
formaldehyde_positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.205], [0.0, 0.94, (-0.587)], [0.0, (-0.94), (-0.587)]], device=DEVICE)
formaldehyde_numbers = torch.tensor([6.0, 8.0, 1.0, 1.0], device=DEVICE)
reference_energies = torch.tensor([(-12.655106), (-23.912676), (-21.853748)], device=DEVICE)
scale, shift = 0.8041538754478097, 0.16409696359187365
z_table = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0, 64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0, 80.0, 81.0, 82.0, 83.0, 84.0, 85.0, 86.0, 87.0, 88.0, 89.0], device=DEVICE)
r_cut = 6.0
positions = water_positions
node_attrs = one_hot(water_numbers, z_table)
edge_index = neighbor_search(positions, r_cut)
print(print(positions))
print(print(edge_index))
num_elements, num_channels = 89, 128
w_embed = (torch.exp(torch.zeros(int(89), int(128), device=DEVICE) if isinstance(torch.zeros(int(89), int(128), device=DEVICE), torch.Tensor) else torch.tensor(float(torch.zeros(int(89), int(128), device=DEVICE)))) / 128.0)
π, num_bessel, p_env = 3.141592653589793, 10, 5.0
vectors = edge_vectors(positions, edge_index)
lengths = edge_lengths(vectors)
edge_feats = radial_embedding(lengths)
edge_attrs = spherical_harmonics(vectors, lengths)
print(print(edge_feats))
print(print(edge_attrs))
w_up = (torch.exp(torch.zeros(int(128), int(128), device=DEVICE) if isinstance(torch.zeros(int(128), int(128), device=DEVICE), torch.Tensor) else torch.tensor(float(torch.zeros(int(128), int(128), device=DEVICE)))) / 128.0)
silu_norm = 1.679176792
radial_hidden = 64
w_r1 = (torch.exp(torch.zeros(int(10), int(64), device=DEVICE) if isinstance(torch.zeros(int(10), int(64), device=DEVICE), torch.Tensor) else torch.tensor(float(torch.zeros(int(10), int(64), device=DEVICE)))) / 128.0)
w_r2 = (torch.exp(torch.zeros(int(64), int(64), device=DEVICE) if isinstance(torch.zeros(int(64), int(64), device=DEVICE), torch.Tensor) else torch.tensor(float(torch.zeros(int(64), int(64), device=DEVICE)))) / 128.0)
w_r3 = (torch.exp(torch.zeros(int(64), int(64), device=DEVICE) if isinstance(torch.zeros(int(64), int(64), device=DEVICE), torch.Tensor) else torch.tensor(float(torch.zeros(int(64), int(64), device=DEVICE)))) / 128.0)
w_r4 = (torch.exp(torch.zeros(int(64), int(512), device=DEVICE) if isinstance(torch.zeros(int(64), int(512), device=DEVICE), torch.Tensor) else torch.tensor(float(torch.zeros(int(64), int(512), device=DEVICE)))) / 128.0)
avg_num_neighbors = 61.964672446250916
w_0 = (torch.exp(torch.zeros(int(128), int(128), device=DEVICE) if isinstance(torch.zeros(int(128), int(128), device=DEVICE), torch.Tensor) else torch.tensor(float(torch.zeros(int(128), int(128), device=DEVICE)))) / 128.0)
w_1 = (torch.exp(torch.zeros(int(128), int(128), device=DEVICE) if isinstance(torch.zeros(int(128), int(128), device=DEVICE), torch.Tensor) else torch.tensor(float(torch.zeros(int(128), int(128), device=DEVICE)))) / 128.0)
w_2 = (torch.exp(torch.zeros(int(128), int(128), device=DEVICE) if isinstance(torch.zeros(int(128), int(128), device=DEVICE), torch.Tensor) else torch.tensor(float(torch.zeros(int(128), int(128), device=DEVICE)))) / 128.0)
w_3 = (torch.exp(torch.zeros(int(128), int(128), device=DEVICE) if isinstance(torch.zeros(int(128), int(128), device=DEVICE), torch.Tensor) else torch.tensor(float(torch.zeros(int(128), int(128), device=DEVICE)))) / 128.0)
U1 = (torch.exp(torch.zeros(int(16), int(1), device=DEVICE) if isinstance(torch.zeros(int(16), int(1), device=DEVICE), torch.Tensor) else torch.tensor(float(torch.zeros(int(16), int(1), device=DEVICE)))) / 128.0)
U2 = (torch.exp(torch.zeros(int(16), int(16), int(4), device=DEVICE) if isinstance(torch.zeros(int(16), int(16), int(4), device=DEVICE), torch.Tensor) else torch.tensor(float(torch.zeros(int(16), int(16), int(4), device=DEVICE)))) / 128.0)
U3 = (torch.exp(torch.zeros(int(16), int(16), int(16), int(23), device=DEVICE) if isinstance(torch.zeros(int(16), int(16), int(16), int(23), device=DEVICE), torch.Tensor) else torch.tensor(float(torch.zeros(int(16), int(16), int(16), int(23), device=DEVICE)))) / 128.0)
U1v = (torch.exp(torch.zeros(int(3), int(16), int(1), device=DEVICE) if isinstance(torch.zeros(int(3), int(16), int(1), device=DEVICE), torch.Tensor) else torch.tensor(float(torch.zeros(int(3), int(16), int(1), device=DEVICE)))) / 128.0)
U2v = (torch.exp(torch.zeros(int(3), int(16), int(16), int(6), device=DEVICE) if isinstance(torch.zeros(int(3), int(16), int(16), int(6), device=DEVICE), torch.Tensor) else torch.tensor(float(torch.zeros(int(3), int(16), int(16), int(6), device=DEVICE)))) / 128.0)
U3v = (torch.exp(torch.zeros(int(3), int(16), int(16), int(16), int(51), device=DEVICE) if isinstance(torch.zeros(int(3), int(16), int(16), int(16), int(51), device=DEVICE), torch.Tensor) else torch.tensor(float(torch.zeros(int(3), int(16), int(16), int(16), int(51), device=DEVICE)))) / 128.0)
w_prod1 = (torch.exp(torch.zeros(int(89), int(1), int(128), device=DEVICE) if isinstance(torch.zeros(int(89), int(1), int(128), device=DEVICE), torch.Tensor) else torch.tensor(float(torch.zeros(int(89), int(1), int(128), device=DEVICE)))) / 128.0)
w_prod2 = (torch.exp(torch.zeros(int(89), int(4), int(128), device=DEVICE) if isinstance(torch.zeros(int(89), int(4), int(128), device=DEVICE), torch.Tensor) else torch.tensor(float(torch.zeros(int(89), int(4), int(128), device=DEVICE)))) / 128.0)
w_prod3 = (torch.exp(torch.zeros(int(89), int(23), int(128), device=DEVICE) if isinstance(torch.zeros(int(89), int(23), int(128), device=DEVICE), torch.Tensor) else torch.tensor(float(torch.zeros(int(89), int(23), int(128), device=DEVICE)))) / 128.0)
w_prod1v = (torch.exp(torch.zeros(int(89), int(1), int(128), device=DEVICE) if isinstance(torch.zeros(int(89), int(1), int(128), device=DEVICE), torch.Tensor) else torch.tensor(float(torch.zeros(int(89), int(1), int(128), device=DEVICE)))) / 128.0)
w_prod2v = (torch.exp(torch.zeros(int(89), int(6), int(128), device=DEVICE) if isinstance(torch.zeros(int(89), int(6), int(128), device=DEVICE), torch.Tensor) else torch.tensor(float(torch.zeros(int(89), int(6), int(128), device=DEVICE)))) / 128.0)
w_prod3v = (torch.exp(torch.zeros(int(89), int(51), int(128), device=DEVICE) if isinstance(torch.zeros(int(89), int(51), int(128), device=DEVICE), torch.Tensor) else torch.tensor(float(torch.zeros(int(89), int(51), int(128), device=DEVICE)))) / 128.0)
w_sc = (torch.exp(torch.zeros(int(128), int(89), int(128), device=DEVICE) if isinstance(torch.zeros(int(128), int(89), int(128), device=DEVICE), torch.Tensor) else torch.tensor(float(torch.zeros(int(128), int(89), int(128), device=DEVICE)))) / 128.0)
w_p = (torch.exp(torch.zeros(int(128), int(128), device=DEVICE) if isinstance(torch.zeros(int(128), int(128), device=DEVICE), torch.Tensor) else torch.tensor(float(torch.zeros(int(128), int(128), device=DEVICE)))) / 128.0)
w_pv = (torch.exp(torch.zeros(int(128), int(128), device=DEVICE) if isinstance(torch.zeros(int(128), int(128), device=DEVICE), torch.Tensor) else torch.tensor(float(torch.zeros(int(128), int(128), device=DEVICE)))) / 128.0)
w_readout = (torch.exp(torch.zeros(int(128), int(1), device=DEVICE) if isinstance(torch.zeros(int(128), int(1), device=DEVICE), torch.Tensor) else torch.tensor(float(torch.zeros(int(128), int(1), device=DEVICE)))) / 128.0)
w2_up = (torch.exp(torch.zeros(int(128), int(128), device=DEVICE) if isinstance(torch.zeros(int(128), int(128), device=DEVICE), torch.Tensor) else torch.tensor(float(torch.zeros(int(128), int(128), device=DEVICE)))) / 128.0)
w2_up_v = (torch.exp(torch.zeros(int(128), int(128), device=DEVICE) if isinstance(torch.zeros(int(128), int(128), device=DEVICE), torch.Tensor) else torch.tensor(float(torch.zeros(int(128), int(128), device=DEVICE)))) / 128.0)
w2_r1 = (torch.exp(torch.zeros(int(10), int(64), device=DEVICE) if isinstance(torch.zeros(int(10), int(64), device=DEVICE), torch.Tensor) else torch.tensor(float(torch.zeros(int(10), int(64), device=DEVICE)))) / 128.0)
w2_r2 = (torch.exp(torch.zeros(int(64), int(64), device=DEVICE) if isinstance(torch.zeros(int(64), int(64), device=DEVICE), torch.Tensor) else torch.tensor(float(torch.zeros(int(64), int(64), device=DEVICE)))) / 128.0)
w2_r3 = (torch.exp(torch.zeros(int(64), int(64), device=DEVICE) if isinstance(torch.zeros(int(64), int(64), device=DEVICE), torch.Tensor) else torch.tensor(float(torch.zeros(int(64), int(64), device=DEVICE)))) / 128.0)
w2_r4 = (torch.exp(torch.zeros(int(64), int(1280), device=DEVICE) if isinstance(torch.zeros(int(64), int(1280), device=DEVICE), torch.Tensor) else torch.tensor(float(torch.zeros(int(64), int(1280), device=DEVICE)))) / 128.0)
cg_121 = (torch.exp(torch.zeros(int(3), int(5), int(3), device=DEVICE) if isinstance(torch.zeros(int(3), int(5), int(3), device=DEVICE), torch.Tensor) else torch.tensor(float(torch.zeros(int(3), int(5), int(3), device=DEVICE)))) / 128.0)
cg_112 = (torch.exp(torch.zeros(int(3), int(3), int(5), device=DEVICE) if isinstance(torch.zeros(int(3), int(3), int(5), device=DEVICE), torch.Tensor) else torch.tensor(float(torch.zeros(int(3), int(3), int(5), device=DEVICE)))) / 128.0)
cg_132 = (torch.exp(torch.zeros(int(3), int(7), int(5), device=DEVICE) if isinstance(torch.zeros(int(3), int(7), int(5), device=DEVICE), torch.Tensor) else torch.tensor(float(torch.zeros(int(3), int(7), int(5), device=DEVICE)))) / 128.0)
cg_123 = (torch.exp(torch.zeros(int(3), int(5), int(7), device=DEVICE) if isinstance(torch.zeros(int(3), int(5), int(7), device=DEVICE), torch.Tensor) else torch.tensor(float(torch.zeros(int(3), int(5), int(7), device=DEVICE)))) / 128.0)
w2_0 = (torch.exp(torch.zeros(int(128), int(128), device=DEVICE) if isinstance(torch.zeros(int(128), int(128), device=DEVICE), torch.Tensor) else torch.tensor(float(torch.zeros(int(128), int(128), device=DEVICE)))) / 128.0)
w2_1 = (torch.exp(torch.zeros(int(128), int(128), device=DEVICE) if isinstance(torch.zeros(int(128), int(128), device=DEVICE), torch.Tensor) else torch.tensor(float(torch.zeros(int(128), int(128), device=DEVICE)))) / 128.0)
w2_2 = (torch.exp(torch.zeros(int(128), int(128), device=DEVICE) if isinstance(torch.zeros(int(128), int(128), device=DEVICE), torch.Tensor) else torch.tensor(float(torch.zeros(int(128), int(128), device=DEVICE)))) / 128.0)
w2_3 = (torch.exp(torch.zeros(int(128), int(128), device=DEVICE) if isinstance(torch.zeros(int(128), int(128), device=DEVICE), torch.Tensor) else torch.tensor(float(torch.zeros(int(128), int(128), device=DEVICE)))) / 128.0)
w2_4 = (torch.exp(torch.zeros(int(128), int(128), device=DEVICE) if isinstance(torch.zeros(int(128), int(128), device=DEVICE), torch.Tensor) else torch.tensor(float(torch.zeros(int(128), int(128), device=DEVICE)))) / 128.0)
w2_5 = (torch.exp(torch.zeros(int(128), int(128), device=DEVICE) if isinstance(torch.zeros(int(128), int(128), device=DEVICE), torch.Tensor) else torch.tensor(float(torch.zeros(int(128), int(128), device=DEVICE)))) / 128.0)
w2_6 = (torch.exp(torch.zeros(int(128), int(128), device=DEVICE) if isinstance(torch.zeros(int(128), int(128), device=DEVICE), torch.Tensor) else torch.tensor(float(torch.zeros(int(128), int(128), device=DEVICE)))) / 128.0)
w2_7 = (torch.exp(torch.zeros(int(128), int(128), device=DEVICE) if isinstance(torch.zeros(int(128), int(128), device=DEVICE), torch.Tensor) else torch.tensor(float(torch.zeros(int(128), int(128), device=DEVICE)))) / 128.0)
w2_8 = (torch.exp(torch.zeros(int(128), int(128), device=DEVICE) if isinstance(torch.zeros(int(128), int(128), device=DEVICE), torch.Tensor) else torch.tensor(float(torch.zeros(int(128), int(128), device=DEVICE)))) / 128.0)
w2_9 = (torch.exp(torch.zeros(int(128), int(128), device=DEVICE) if isinstance(torch.zeros(int(128), int(128), device=DEVICE), torch.Tensor) else torch.tensor(float(torch.zeros(int(128), int(128), device=DEVICE)))) / 128.0)
w2_prod1 = (torch.exp(torch.zeros(int(89), int(1), int(128), device=DEVICE) if isinstance(torch.zeros(int(89), int(1), int(128), device=DEVICE), torch.Tensor) else torch.tensor(float(torch.zeros(int(89), int(1), int(128), device=DEVICE)))) / 128.0)
w2_prod2 = (torch.exp(torch.zeros(int(89), int(4), int(128), device=DEVICE) if isinstance(torch.zeros(int(89), int(4), int(128), device=DEVICE), torch.Tensor) else torch.tensor(float(torch.zeros(int(89), int(4), int(128), device=DEVICE)))) / 128.0)
w2_prod3 = (torch.exp(torch.zeros(int(89), int(23), int(128), device=DEVICE) if isinstance(torch.zeros(int(89), int(23), int(128), device=DEVICE), torch.Tensor) else torch.tensor(float(torch.zeros(int(89), int(23), int(128), device=DEVICE)))) / 128.0)
w2_sc = (torch.exp(torch.zeros(int(128), int(89), int(128), device=DEVICE) if isinstance(torch.zeros(int(128), int(89), int(128), device=DEVICE), torch.Tensor) else torch.tensor(float(torch.zeros(int(128), int(89), int(128), device=DEVICE)))) / 128.0)
w2_p = (torch.exp(torch.zeros(int(128), int(128), device=DEVICE) if isinstance(torch.zeros(int(128), int(128), device=DEVICE), torch.Tensor) else torch.tensor(float(torch.zeros(int(128), int(128), device=DEVICE)))) / 128.0)
w2_ro1 = (torch.exp(torch.zeros(int(128), int(16), device=DEVICE) if isinstance(torch.zeros(int(128), int(16), device=DEVICE), torch.Tensor) else torch.tensor(float(torch.zeros(int(128), int(16), device=DEVICE)))) / 128.0)
w2_ro2 = (torch.exp(torch.zeros(int(16), int(1), device=DEVICE) if isinstance(torch.zeros(int(16), int(1), device=DEVICE), torch.Tensor) else torch.tensor(float(torch.zeros(int(16), int(1), device=DEVICE)))) / 128.0)
atomic_energies = (torch.exp(torch.zeros(int(89), int(1), device=DEVICE) if isinstance(torch.zeros(int(89), int(1), device=DEVICE), torch.Tensor) else torch.tensor(float(torch.zeros(int(89), int(1), device=DEVICE)))) / 128.0)
water_energy = mace_energy(water_positions, water_numbers)
methane_energy = mace_energy(methane_positions, methane_numbers)
formaldehyde_energy = mace_energy(formaldehyde_positions, formaldehyde_numbers)
predictions = torch.stack([torch.as_tensor(water_energy), torch.as_tensor(methane_energy), torch.as_tensor(formaldehyde_energy)])
print(print(predictions))
print(print(reference_energies))
print(print((predictions - reference_energies)))