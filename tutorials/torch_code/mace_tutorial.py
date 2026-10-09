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
    vector = torch.zeros(int(3), device=DEVICE)
    distance = 0.0
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

def count_edges(edge_index):
    total = 0
    for edge in range(int(0), int(len(edge_index[int(0)]))):
        sender, receiver = edge_index[:, int(edge)]
        if sender != receiver:
            total = total + 1
    return total

def batch_edge_index(batch):
    num_samples = len(batch)
    result = torch.zeros(int(num_samples), int(2), int(6), device=DEVICE)
    num_edges = 0
    for i in range(int(0), int(num_samples)):
        edges = neighbor_search(batch[int(i)], r_cut)
        num_edges = len(edges[int(0)])
        result[int(i), :, int(0):int(num_edges)] = edges
    return result

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
    num_edges = len(vectors)
    lengths = torch.zeros(int(num_edges), device=DEVICE)
    for edge in range(int(0), int(num_edges)):
        lengths[int(edge)] = torch.sqrt(torch.sum((vectors[int(edge)] * vectors[int(edge)]) if isinstance((vectors[int(edge)] * vectors[int(edge)]), torch.Tensor) else torch.tensor(float((vectors[int(edge)] * vectors[int(edge)])))) if isinstance(torch.sum((vectors[int(edge)] * vectors[int(edge)]) if isinstance((vectors[int(edge)] * vectors[int(edge)]), torch.Tensor) else torch.tensor(float((vectors[int(edge)] * vectors[int(edge)])))), torch.Tensor) else torch.tensor(float(torch.sum((vectors[int(edge)] * vectors[int(edge)]) if isinstance((vectors[int(edge)] * vectors[int(edge)]), torch.Tensor) else torch.tensor(float((vectors[int(edge)] * vectors[int(edge)])))))))
    return lengths

def radial_embedding(lengths):
    num_edges = len(lengths)
    edge_feats = torch.zeros(int(num_edges), int(num_bessel), device=DEVICE)
    f_cut = 0.0
    for edge in range(int(0), int(num_edges)):
        r, x = lengths[int(edge)], (lengths[int(edge)] / r_cut)
        if x < 1.0:
            f_cut = (((1.0 - ((((p_env + 1.0) * (p_env + 2.0)) / 2.0) * (x ** p_env))) + ((p_env * (p_env + 2.0)) * (x ** (p_env + 1.0)))) - (((p_env * (p_env + 1.0)) / 2.0) * (x ** (p_env + 2.0))))
            for n in range(int(0), int(num_bessel)):
                edge_feats[int(edge), int(n)] = (((torch.sqrt((2.0 / r_cut) if isinstance((2.0 / r_cut), torch.Tensor) else torch.tensor(float((2.0 / r_cut)))) * torch.sin(((((n + 1.0) * π) * r) / r_cut) if isinstance(((((n + 1.0) * π) * r) / r_cut), torch.Tensor) else torch.tensor(float(((((n + 1.0) * π) * r) / r_cut))))) / r) * f_cut)
    return edge_feats

def Y0(vectors, lengths):
    num_edges = len(lengths)
    results = torch.zeros(int(num_edges), int(1), device=DEVICE)
    for edge in range(int(0), int(num_edges)):
        results[int(edge), int(0)] = 1.0
    return results

def Y1(vectors, lengths):
    num_edges = len(lengths)
    results = torch.zeros(int(num_edges), int(3), device=DEVICE)
    for edge in range(int(0), int(num_edges)):
        x, y, z = (vectors[int(edge)] / lengths[int(edge)])
        results[int(edge)] = torch.stack([torch.as_tensor((torch.sqrt(3.0 if isinstance(3.0, torch.Tensor) else torch.tensor(float(3.0))) * x)), torch.as_tensor((torch.sqrt(3.0 if isinstance(3.0, torch.Tensor) else torch.tensor(float(3.0))) * y)), torch.as_tensor((torch.sqrt(3.0 if isinstance(3.0, torch.Tensor) else torch.tensor(float(3.0))) * z))])
    return results

def Y2(vectors, lengths):
    num_edges = len(lengths)
    results = torch.zeros(int(num_edges), int(5), device=DEVICE)
    for edge in range(int(0), int(num_edges)):
        x, y, z = (vectors[int(edge)] / lengths[int(edge)])
        results[int(edge)] = torch.stack([torch.as_tensor(((torch.sqrt(15.0 if isinstance(15.0, torch.Tensor) else torch.tensor(float(15.0))) * x) * z)), torch.as_tensor(((torch.sqrt(15.0 if isinstance(15.0, torch.Tensor) else torch.tensor(float(15.0))) * x) * y)), torch.as_tensor((torch.sqrt(5.0 if isinstance(5.0, torch.Tensor) else torch.tensor(float(5.0))) * ((y * y) - (0.5 * ((x * x) + (z * z)))))), torch.as_tensor(((torch.sqrt(15.0 if isinstance(15.0, torch.Tensor) else torch.tensor(float(15.0))) * y) * z)), torch.as_tensor(((torch.sqrt(15.0 if isinstance(15.0, torch.Tensor) else torch.tensor(float(15.0))) / 2.0) * ((z * z) - (x * x))))])
    return results

def spherical_harmonics(vectors, lengths):
    num_edges = len(lengths)
    edge_attrs = torch.zeros(int(num_edges), int(9), device=DEVICE)
    edge_attrs[:, int(0):int(1)] = Y0(vectors, lengths)
    edge_attrs[:, int(1):int(4)] = Y1(vectors, lengths)
    edge_attrs[:, int(4):int(9)] = Y2(vectors, lengths)
    return edge_attrs

def batch_edge_feats(batch, edge_indices):
    num_samples = len(batch)
    result = torch.zeros(int(num_samples), int(6), int(8), device=DEVICE)
    num_edges = 0
    for i in range(int(0), int(num_samples)):
        num_edges = count_edges(edge_indices[int(i)])
        vecs = edge_vectors(batch[int(i)], edge_indices[int(i), :, int(0):int(num_edges)])
        result[int(i), int(0):int(num_edges), :] = radial_embedding(edge_lengths(vecs))
    return result

def batch_edge_attrs(batch, edge_indices):
    num_samples = len(batch)
    result = torch.zeros(int(num_samples), int(6), int(9), device=DEVICE)
    num_edges = 0
    for i in range(int(0), int(num_samples)):
        num_edges = count_edges(edge_indices[int(i)])
        vecs = edge_vectors(batch[int(i)], edge_indices[int(i), :, int(0):int(num_edges)])
        result[int(i), int(0):int(num_edges), :] = spherical_harmonics(vecs, edge_lengths(vecs))
    return result

def linear_up(node_feats, w):
    num_atoms = len(node_feats)
    node_feats_up = torch.zeros(int(num_atoms), int(num_channels), device=DEVICE)
    for atom in range(int(0), int(num_atoms)):
        for channel in range(int(0), int(num_channels)):
            node_feats_up[int(atom), int(channel)] = (torch.sum((node_feats[int(atom)] * w[:, int(channel)]) if isinstance((node_feats[int(atom)] * w[:, int(channel)]), torch.Tensor) else torch.tensor(float((node_feats[int(atom)] * w[:, int(channel)])))) / torch.sqrt(num_channels if isinstance(num_channels, torch.Tensor) else torch.tensor(float(num_channels))))
    return node_feats_up

def silu(x):
    return ((silu_norm * x) / (1.0 + torch.exp((-x) if isinstance((-x), torch.Tensor) else torch.tensor(float((-x))))))

def radial_mlp(edge_feats, w1, w2, w3, w4):
    h1 = silu(((edge_feats @ w1) / torch.sqrt(num_bessel if isinstance(num_bessel, torch.Tensor) else torch.tensor(float(num_bessel)))))
    h2 = silu(((h1 @ w2) / torch.sqrt(radial_hidden if isinstance(radial_hidden, torch.Tensor) else torch.tensor(float(radial_hidden)))))
    h3 = silu(((h2 @ w3) / torch.sqrt(radial_hidden if isinstance(radial_hidden, torch.Tensor) else torch.tensor(float(radial_hidden)))))
    return ((h3 @ w4) / torch.sqrt(radial_hidden if isinstance(radial_hidden, torch.Tensor) else torch.tensor(float(radial_hidden))))

def tensor_product_reduce(cg, a, b, dim1, dim2, dim3):
    results = torch.zeros(int(dim3), device=DEVICE)
    acc = 0.0
    for k in range(int(0), int(dim3)):
        acc = 0.0
        for i in range(int(0), int(dim1)):
            for j in range(int(0), int(dim2)):
                acc = acc + ((cg[int(i), int(j), int(k)] * a[int(i)]) * b[int(j)])
        results[int(k)] = acc
    return results

def conv_tp(node_feats_up, edge_attrs, tp_weights, edge_index):
    num_edges = len(edge_attrs)
    mji = torch.zeros(int(num_edges), int(num_channels), int(9), device=DEVICE)
    for edge in range(int(0), int(num_edges)):
        sender, receiver = edge_index[:, int(edge)]
        for channel in range(int(0), int(num_channels)):
            h = torch.stack([torch.as_tensor(node_feats_up[int(sender), int(channel)])])
            mji[int(edge), int(channel), int(0):int(1)] = (tp_weights[int(edge), int(channel)] * tensor_product_reduce(cg_000, edge_attrs[int(edge), int(0):int(1)], h, 1, 1, 1))
            mji[int(edge), int(channel), int(1):int(4)] = (tp_weights[int(edge), int((num_channels + channel))] * tensor_product_reduce(cg_101, edge_attrs[int(edge), int(1):int(4)], h, 3, 1, 3))
            mji[int(edge), int(channel), int(4):int(9)] = (tp_weights[int(edge), int(((2 * num_channels) + channel))] * tensor_product_reduce(cg_202, edge_attrs[int(edge), int(4):int(9)], h, 5, 1, 5))
    return mji

def neighbor_sum(mji, edge_index, num_atoms):
    num_edges = len(mji)
    message = torch.zeros(int(num_atoms), int(num_channels), int(9), device=DEVICE)
    for edge in range(int(0), int(num_edges)):
        sender, receiver = edge_index[:, int(edge)]
        message[int(receiver)] = (message[int(receiver)] + mji[int(edge)])
    return message

def linear(message, w0, w1, w2):
    num_atoms = len(message)
    out = torch.zeros(int(num_atoms), int(num_channels), int(9), device=DEVICE)
    for atom in range(int(0), int(num_atoms)):
        out[int(atom), :, int(0):int(1)] = ((w0 @ message[int(atom), :, int(0):int(1)]) / torch.sqrt(num_channels if isinstance(num_channels, torch.Tensor) else torch.tensor(float(num_channels))))
        out[int(atom), :, int(1):int(4)] = ((w1 @ message[int(atom), :, int(1):int(4)]) / torch.sqrt(num_channels if isinstance(num_channels, torch.Tensor) else torch.tensor(float(num_channels))))
        out[int(atom), :, int(4):int(9)] = ((w2 @ message[int(atom), :, int(4):int(9)]) / torch.sqrt(num_channels if isinstance(num_channels, torch.Tensor) else torch.tensor(float(num_channels))))
    return (out / avg_num_neighbors)

def B_features_L0(A):
    num_atoms = len(A)
    B0 = torch.zeros(int(num_atoms), int(num_channels), int(4), device=DEVICE)
    for atom in range(int(0), int(num_atoms)):
        for channel in range(int(0), int(num_channels)):
            a0, a1, a2 = A[int(atom), int(channel), int(0):int(1)], A[int(atom), int(channel), int(1):int(4)], A[int(atom), int(channel), int(4):int(9)]
            B0[int(atom), int(channel), int(0):int(1)] = a0
            B0[int(atom), int(channel), int(1):int(2)] = tensor_product_reduce(cg_000, a0, a0, 1, 1, 1)
            B0[int(atom), int(channel), int(2):int(3)] = tensor_product_reduce(cg_110, a1, a1, 3, 3, 1)
            B0[int(atom), int(channel), int(3):int(4)] = tensor_product_reduce(cg_220, a2, a2, 5, 5, 1)
    return B0

def B_features_L1(A):
    num_atoms = len(A)
    B1 = torch.zeros(int(num_atoms), int(num_channels), int(3), int(3), device=DEVICE)
    for atom in range(int(0), int(num_atoms)):
        for channel in range(int(0), int(num_channels)):
            a0, a1, a2 = A[int(atom), int(channel), int(0):int(1)], A[int(atom), int(channel), int(1):int(4)], A[int(atom), int(channel), int(4):int(9)]
            B1[int(atom), int(channel), int(0)] = a1
            B1[int(atom), int(channel), int(1)] = tensor_product_reduce(cg_011, a0, a1, 1, 3, 3)
            B1[int(atom), int(channel), int(2)] = tensor_product_reduce(cg_121, a1, a2, 3, 5, 3)
    return B1

def weighted_sum(B0, B1, node_attrs, wp0, wp1):
    num_atoms = len(B0)
    m = torch.zeros(int(num_atoms), int(num_channels), int(4), device=DEVICE)
    for atom in range(int(0), int(num_atoms)):
        w0, w1 = torch.zeros(int(4), int(8), device=DEVICE), torch.zeros(int(3), int(8), device=DEVICE)
        for z in range(int(0), int(num_elements)):
            w0 = (w0 + (node_attrs[int(atom), int(z)] * wp0[int(z)]))
            w1 = (w1 + (node_attrs[int(atom), int(z)] * wp1[int(z)]))
        for channel in range(int(0), int(num_channels)):
            m[int(atom), int(channel), int(0):int(1)] = torch.stack([torch.as_tensor(torch.sum((w0[:, int(channel)] * B0[int(atom), int(channel)]) if isinstance((w0[:, int(channel)] * B0[int(atom), int(channel)]), torch.Tensor) else torch.tensor(float((w0[:, int(channel)] * B0[int(atom), int(channel)])))))])
            m[int(atom), int(channel), int(1):int(4)] = (w1[:, int(channel)] @ B1[int(atom), int(channel)])
    return m

def skip_tp(node_feats, node_attrs, w):
    num_atoms = len(node_feats)
    sc = torch.zeros(int(num_atoms), int(num_channels), device=DEVICE)
    acc = 0.0
    for atom in range(int(0), int(num_atoms)):
        for channel in range(int(0), int(num_channels)):
            acc = 0.0
            for z in range(int(0), int(num_elements)):
                acc = acc + (node_attrs[int(atom), int(z)] * torch.sum((node_feats[int(atom)] * w[:, int(z), int(channel)]) if isinstance((node_feats[int(atom)] * w[:, int(z), int(channel)]), torch.Tensor) else torch.tensor(float((node_feats[int(atom)] * w[:, int(z), int(channel)])))))
            sc[int(atom), int(channel)] = (acc / torch.sqrt((num_channels * num_elements) if isinstance((num_channels * num_elements), torch.Tensor) else torch.tensor(float((num_channels * num_elements)))))
    return sc

def node_update(m, sc, wp0, wp1):
    num_atoms = len(m)
    node_feats1 = torch.zeros(int(num_atoms), int(num_channels), int(4), device=DEVICE)
    for atom in range(int(0), int(num_atoms)):
        for channel in range(int(0), int(num_channels)):
            node_feats1[int(atom), int(channel), int(0)] = ((torch.sum((wp0[int(channel)] * m[int(atom), :, int(0)]) if isinstance((wp0[int(channel)] * m[int(atom), :, int(0)]), torch.Tensor) else torch.tensor(float((wp0[int(channel)] * m[int(atom), :, int(0)])))) / torch.sqrt(num_channels if isinstance(num_channels, torch.Tensor) else torch.tensor(float(num_channels)))) + sc[int(atom), int(channel)])
        node_feats1[int(atom), :, int(1):int(4)] = ((wp1 @ m[int(atom), :, int(1):int(4)]) / torch.sqrt(num_channels if isinstance(num_channels, torch.Tensor) else torch.tensor(float(num_channels))))
    return node_feats1

def readout(node_feats, w):
    num_atoms = len(node_feats)
    node_energies = torch.zeros(int(num_atoms), device=DEVICE)
    for atom in range(int(0), int(num_atoms)):
        node_energies[int(atom)] = (torch.sum((w * node_feats[int(atom), :, int(0)]) if isinstance((w * node_feats[int(atom), :, int(0)]), torch.Tensor) else torch.tensor(float((w * node_feats[int(atom), :, int(0)])))) / torch.sqrt(num_channels if isinstance(num_channels, torch.Tensor) else torch.tensor(float(num_channels))))
    return node_energies

def linear_up2(node_feats, w0, w1):
    num_atoms = len(node_feats)
    node_feats_up = torch.zeros(int(num_atoms), int(num_channels), int(4), device=DEVICE)
    for atom in range(int(0), int(num_atoms)):
        for channel in range(int(0), int(num_channels)):
            node_feats_up[int(atom), int(channel), int(0)] = (torch.sum((node_feats[int(atom), :, int(0)] * w0[:, int(channel)]) if isinstance((node_feats[int(atom), :, int(0)] * w0[:, int(channel)]), torch.Tensor) else torch.tensor(float((node_feats[int(atom), :, int(0)] * w0[:, int(channel)])))) / torch.sqrt(num_channels if isinstance(num_channels, torch.Tensor) else torch.tensor(float(num_channels))))
            node_feats_up[int(atom), int(channel), int(1):int(4)] = ((w1[:, int(channel)] @ node_feats[int(atom), :, int(1):int(4)]) / torch.sqrt(num_channels if isinstance(num_channels, torch.Tensor) else torch.tensor(float(num_channels))))
    return node_feats_up

def radial_mlp2(edge_feats, w1, w2, w3, w4):
    h1 = silu(((edge_feats @ w1) / torch.sqrt(num_bessel if isinstance(num_bessel, torch.Tensor) else torch.tensor(float(num_bessel)))))
    h2 = silu(((h1 @ w2) / torch.sqrt(radial_hidden if isinstance(radial_hidden, torch.Tensor) else torch.tensor(float(radial_hidden)))))
    h3 = silu(((h2 @ w3) / torch.sqrt(radial_hidden if isinstance(radial_hidden, torch.Tensor) else torch.tensor(float(radial_hidden)))))
    return ((h3 @ w4) / torch.sqrt(radial_hidden if isinstance(radial_hidden, torch.Tensor) else torch.tensor(float(radial_hidden))))

def conv_tp2(node_feats_up, edge_attrs, tp_weights, edge_index):
    num_edges = len(edge_attrs)
    mji = torch.zeros(int(num_edges), int(num_channels), int(21), device=DEVICE)
    for edge in range(int(0), int(num_edges)):
        sender, receiver = edge_index[:, int(edge)]
        y0, y1, y2 = edge_attrs[int(edge), int(0):int(1)], edge_attrs[int(edge), int(1):int(4)], edge_attrs[int(edge), int(4):int(9)]
        for channel in range(int(0), int(num_channels)):
            h0, h1 = torch.stack([torch.as_tensor(node_feats_up[int(sender), int(channel), int(0)])]), node_feats_up[int(sender), int(channel), int(1):int(4)]
            mji[int(edge), int(channel), int(0):int(1)] = (tp_weights[int(edge), int(channel)] * tensor_product_reduce(cg_000, h0, y0, 1, 1, 1))
            mji[int(edge), int(channel), int(1):int(2)] = (tp_weights[int(edge), int((num_channels + channel))] * tensor_product_reduce(cg_110, h1, y1, 3, 3, 1))
            mji[int(edge), int(channel), int(2):int(5)] = (tp_weights[int(edge), int(((2 * num_channels) + channel))] * tensor_product_reduce(cg_011, h0, y1, 1, 3, 3))
            mji[int(edge), int(channel), int(5):int(8)] = (tp_weights[int(edge), int(((3 * num_channels) + channel))] * tensor_product_reduce(cg_101, h1, y0, 3, 1, 3))
            mji[int(edge), int(channel), int(8):int(11)] = (tp_weights[int(edge), int(((4 * num_channels) + channel))] * tensor_product_reduce(cg_121, h1, y2, 3, 5, 3))
            mji[int(edge), int(channel), int(11):int(16)] = (tp_weights[int(edge), int(((5 * num_channels) + channel))] * tensor_product_reduce(cg_022, h0, y2, 1, 5, 5))
            mji[int(edge), int(channel), int(16):int(21)] = (tp_weights[int(edge), int(((6 * num_channels) + channel))] * tensor_product_reduce(cg_112, h1, y1, 3, 3, 5))
    return mji

def neighbor_sum2(mji, edge_index, num_atoms):
    num_edges = len(mji)
    summed = torch.zeros(int(num_atoms), int(num_channels), int(21), device=DEVICE)
    for edge in range(int(0), int(num_edges)):
        sender, receiver = edge_index[:, int(edge)]
        summed[int(receiver)] = (summed[int(receiver)] + mji[int(edge)])
    return summed

def linear2(summed, w):
    num_atoms = len(summed)
    message = torch.zeros(int(num_atoms), int(num_channels), int(9), device=DEVICE)
    for atom in range(int(0), int(num_atoms)):
        message[int(atom), :, int(0):int(1)] = (((w[int(0)] @ summed[int(atom), :, int(0):int(1)]) + (w[int(1)] @ summed[int(atom), :, int(1):int(2)])) / torch.sqrt((2.0 * num_channels) if isinstance((2.0 * num_channels), torch.Tensor) else torch.tensor(float((2.0 * num_channels)))))
        message[int(atom), :, int(1):int(4)] = ((((w[int(2)] @ summed[int(atom), :, int(2):int(5)]) + (w[int(3)] @ summed[int(atom), :, int(5):int(8)])) + (w[int(4)] @ summed[int(atom), :, int(8):int(11)])) / torch.sqrt((3.0 * num_channels) if isinstance((3.0 * num_channels), torch.Tensor) else torch.tensor(float((3.0 * num_channels)))))
        message[int(atom), :, int(4):int(9)] = (((w[int(5)] @ summed[int(atom), :, int(11):int(16)]) + (w[int(6)] @ summed[int(atom), :, int(16):int(21)])) / torch.sqrt((2.0 * num_channels) if isinstance((2.0 * num_channels), torch.Tensor) else torch.tensor(float((2.0 * num_channels)))))
    return (message / avg_num_neighbors)

def weighted_sum2(B0, node_attrs, wp):
    num_atoms = len(B0)
    m = torch.zeros(int(num_atoms), int(num_channels), device=DEVICE)
    w0 = torch.zeros(int(4), int(8), device=DEVICE)
    for atom in range(int(0), int(num_atoms)):
        w0 = torch.zeros(int(4), int(8), device=DEVICE)
        for z in range(int(0), int(num_elements)):
            w0 = (w0 + (node_attrs[int(atom), int(z)] * wp[int(z)]))
        for channel in range(int(0), int(num_channels)):
            m[int(atom), int(channel)] = torch.sum((w0[:, int(channel)] * B0[int(atom), int(channel)]) if isinstance((w0[:, int(channel)] * B0[int(atom), int(channel)]), torch.Tensor) else torch.tensor(float((w0[:, int(channel)] * B0[int(atom), int(channel)]))))
    return m

def node_update2(m, sc, wp):
    num_atoms = len(m)
    node_feats = torch.zeros(int(num_atoms), int(num_channels), device=DEVICE)
    for atom in range(int(0), int(num_atoms)):
        for channel in range(int(0), int(num_channels)):
            node_feats[int(atom), int(channel)] = ((torch.sum((wp[int(channel)] * m[int(atom)]) if isinstance((wp[int(channel)] * m[int(atom)]), torch.Tensor) else torch.tensor(float((wp[int(channel)] * m[int(atom)])))) / torch.sqrt(num_channels if isinstance(num_channels, torch.Tensor) else torch.tensor(float(num_channels)))) + sc[int(atom), int(channel)])
    return node_feats

def readout2(node_feats, w1, w2):
    num_atoms = len(node_feats)
    hidden = silu(((node_feats @ w1) / torch.sqrt(num_channels if isinstance(num_channels, torch.Tensor) else torch.tensor(float(num_channels)))))
    node_energies = torch.zeros(int(num_atoms), device=DEVICE)
    for atom in range(int(0), int(num_atoms)):
        node_energies[int(atom)] = (torch.sum((hidden[int(atom)] * w2) if isinstance((hidden[int(atom)] * w2), torch.Tensor) else torch.tensor(float((hidden[int(atom)] * w2)))) / torch.sqrt(16.0 if isinstance(16.0, torch.Tensor) else torch.tensor(float(16.0))))
    return node_energies

def mse(pred, target):
    diff = (pred - target)
    result = (diff * diff)
    return result

# === Classes ===
class MACEModel(nn.Module):
    def __init__(self, w_embed, w_up, w_r1, w_r2, w_r3, w_r4, w_0, w_1, w_2, w_prod0, w_prod1, w_sc, w_p0, w_p1, w_readout, w2_up0, w2_up1, w2_r1, w2_r2, w2_r3, w2_r4, w2_msg, w2_prod0, w2_sc, w2_p, w2_ro1, w2_ro2):
        super().__init__()
        self.w_embed = nn.Parameter(torch.as_tensor(w_embed))
        self.w_up = nn.Parameter(torch.as_tensor(w_up))
        self.w_r1 = nn.Parameter(torch.as_tensor(w_r1))
        self.w_r2 = nn.Parameter(torch.as_tensor(w_r2))
        self.w_r3 = nn.Parameter(torch.as_tensor(w_r3))
        self.w_r4 = nn.Parameter(torch.as_tensor(w_r4))
        self.w_0 = nn.Parameter(torch.as_tensor(w_0))
        self.w_1 = nn.Parameter(torch.as_tensor(w_1))
        self.w_2 = nn.Parameter(torch.as_tensor(w_2))
        self.w_prod0 = nn.Parameter(torch.as_tensor(w_prod0))
        self.w_prod1 = nn.Parameter(torch.as_tensor(w_prod1))
        self.w_sc = nn.Parameter(torch.as_tensor(w_sc))
        self.w_p0 = nn.Parameter(torch.as_tensor(w_p0))
        self.w_p1 = nn.Parameter(torch.as_tensor(w_p1))
        self.w_readout = nn.Parameter(torch.as_tensor(w_readout))
        self.w2_up0 = nn.Parameter(torch.as_tensor(w2_up0))
        self.w2_up1 = nn.Parameter(torch.as_tensor(w2_up1))
        self.w2_r1 = nn.Parameter(torch.as_tensor(w2_r1))
        self.w2_r2 = nn.Parameter(torch.as_tensor(w2_r2))
        self.w2_r3 = nn.Parameter(torch.as_tensor(w2_r3))
        self.w2_r4 = nn.Parameter(torch.as_tensor(w2_r4))
        self.w2_msg = nn.Parameter(torch.as_tensor(w2_msg))
        self.w2_prod0 = nn.Parameter(torch.as_tensor(w2_prod0))
        self.w2_sc = nn.Parameter(torch.as_tensor(w2_sc))
        self.w2_p = nn.Parameter(torch.as_tensor(w2_p))
        self.w2_ro1 = nn.Parameter(torch.as_tensor(w2_ro1))
        self.w2_ro2 = nn.Parameter(torch.as_tensor(w2_ro2))
        self.learnable_params = [self.w_embed, self.w_up, self.w_r1, self.w_r2, self.w_r3, self.w_r4, self.w_0, self.w_1, self.w_2, self.w_prod0, self.w_prod1, self.w_sc, self.w_p0, self.w_p1, self.w_readout, self.w2_up0, self.w2_up1, self.w2_r1, self.w2_r2, self.w2_r3, self.w2_r4, self.w2_msg, self.w2_prod0, self.w2_sc, self.w2_p, self.w2_ro1, self.w2_ro2]

    def forward(self, node_attrs, edge_index, edge_feats, edge_attrs):
        this = self
        node_attrs = torch.as_tensor(node_attrs, device=DEVICE).float()
        edge_index = torch.as_tensor(edge_index, device=DEVICE).float()
        edge_feats = torch.as_tensor(edge_feats, device=DEVICE).float()
        edge_attrs = torch.as_tensor(edge_attrs, device=DEVICE).float()
        node_feats0 = node_embedding(node_attrs, self.w_embed)
        node_feats_up = linear_up(node_feats0, self.w_up)
        tp_weights = radial_mlp(edge_feats, self.w_r1, self.w_r2, self.w_r3, self.w_r4)
        mji = conv_tp(node_feats_up, edge_attrs, tp_weights, edge_index)
        message = linear(neighbor_sum(mji, edge_index, 3), self.w_0, self.w_1, self.w_2)
        B0 = B_features_L0(message)
        B1 = B_features_L1(message)
        m = weighted_sum(B0, B1, node_attrs, self.w_prod0, self.w_prod1)
        sc = skip_tp(node_feats0, node_attrs, self.w_sc)
        node_feats1 = node_update(m, sc, self.w_p0, self.w_p1)
        node_energies = readout(node_feats1, self.w_readout)
        node_feats_up2 = linear_up2(node_feats1, self.w2_up0, self.w2_up1)
        tp_weights2 = radial_mlp2(edge_feats, self.w2_r1, self.w2_r2, self.w2_r3, self.w2_r4)
        mji2 = conv_tp2(node_feats_up2, edge_attrs, tp_weights2, edge_index)
        message2 = linear2(neighbor_sum2(mji2, edge_index, 3), self.w2_msg)
        B0_2 = B_features_L0(message2)
        m2 = weighted_sum2(B0_2, node_attrs, self.w2_prod0)
        sc2 = skip_tp(node_feats1[:, :, int(0)], node_attrs, self.w2_sc)
        node_feats2 = node_update2(m2, sc2, self.w2_p)
        node_energies2 = readout2(node_feats2, self.w2_ro1, self.w2_ro2)
        energy = (torch.sum(node_energies if isinstance(node_energies, torch.Tensor) else torch.tensor(float(node_energies))) + torch.sum(node_energies2 if isinstance(node_energies2, torch.Tensor) else torch.tensor(float(node_energies2))))
        return energy

    def loss_sample(self, sample):
        this = self
        pred = self(node_attrs, train_edge_index[int(sample)], train_edge_feats[int(sample)], train_edge_attrs[int(sample)])
        target = ((train_energies[int(sample)] - energy_mean) / energy_std)
        result = mse(pred, target)
        return result

    def error_sample(self, sample):
        this = self
        scaled = self(node_attrs, test_edge_index[int(sample)], test_edge_feats[int(sample)], test_edge_attrs[int(sample)])
        result = ((energy_mean + (energy_std * scaled)) - test_energies[int(sample)])
        return result

    def train(self, epochs, lr):
        this = self
        lr = torch.as_tensor(lr, device=DEVICE).float()
        last_loss = 0
        current_loss = 0
        for epoch in range(int(0), int(epochs)):
            for sample in range(int(0), int(num_train)):
                for rep in range(int(0), int(1)):
                    current_loss = self.loss_sample(sample)
                    learnable_grads = compute_grad(current_loss, self.learnable_params)
                    self.update_params(lr, learnable_grads)
                    last_loss = current_loss
        return last_loss

    def evaluate(self):
        this = self
        total_error = 0
        current_error = 0
        for sample in range(int(0), int(num_test)):
            for rep in range(int(0), int(1)):
                current_error = self.error_sample(sample)
                total_error = (total_error + (current_error * current_error))
        result = torch.sqrt((total_error / num_test) if isinstance((total_error / num_test), torch.Tensor) else torch.tensor(float((total_error / num_test))))
        return result

    def update_params(self, lr, learnable_grads):
        this = self
        lr = torch.as_tensor(lr, device=DEVICE).float()
        with torch.no_grad():
            self.w_embed.copy_((self.w_embed - (lr * learnable_grads[int(0)])))
        with torch.no_grad():
            self.w_up.copy_((self.w_up - (lr * learnable_grads[int(1)])))
        with torch.no_grad():
            self.w_r1.copy_((self.w_r1 - (lr * learnable_grads[int(2)])))
        with torch.no_grad():
            self.w_r2.copy_((self.w_r2 - (lr * learnable_grads[int(3)])))
        with torch.no_grad():
            self.w_r3.copy_((self.w_r3 - (lr * learnable_grads[int(4)])))
        with torch.no_grad():
            self.w_r4.copy_((self.w_r4 - (lr * learnable_grads[int(5)])))
        with torch.no_grad():
            self.w_0.copy_((self.w_0 - (lr * learnable_grads[int(6)])))
        with torch.no_grad():
            self.w_1.copy_((self.w_1 - (lr * learnable_grads[int(7)])))
        with torch.no_grad():
            self.w_2.copy_((self.w_2 - (lr * learnable_grads[int(8)])))
        with torch.no_grad():
            self.w_prod0.copy_((self.w_prod0 - (lr * learnable_grads[int(9)])))
        with torch.no_grad():
            self.w_prod1.copy_((self.w_prod1 - (lr * learnable_grads[int(10)])))
        with torch.no_grad():
            self.w_sc.copy_((self.w_sc - (lr * learnable_grads[int(11)])))
        with torch.no_grad():
            self.w_p0.copy_((self.w_p0 - (lr * learnable_grads[int(12)])))
        with torch.no_grad():
            self.w_p1.copy_((self.w_p1 - (lr * learnable_grads[int(13)])))
        with torch.no_grad():
            self.w_readout.copy_((self.w_readout - (lr * learnable_grads[int(14)])))
        with torch.no_grad():
            self.w2_up0.copy_((self.w2_up0 - (lr * learnable_grads[int(15)])))
        with torch.no_grad():
            self.w2_up1.copy_((self.w2_up1 - (lr * learnable_grads[int(16)])))
        with torch.no_grad():
            self.w2_r1.copy_((self.w2_r1 - (lr * learnable_grads[int(17)])))
        with torch.no_grad():
            self.w2_r2.copy_((self.w2_r2 - (lr * learnable_grads[int(18)])))
        with torch.no_grad():
            self.w2_r3.copy_((self.w2_r3 - (lr * learnable_grads[int(19)])))
        with torch.no_grad():
            self.w2_r4.copy_((self.w2_r4 - (lr * learnable_grads[int(20)])))
        with torch.no_grad():
            self.w2_msg.copy_((self.w2_msg - (lr * learnable_grads[int(21)])))
        with torch.no_grad():
            self.w2_prod0.copy_((self.w2_prod0 - (lr * learnable_grads[int(22)])))
        with torch.no_grad():
            self.w2_sc.copy_((self.w2_sc - (lr * learnable_grads[int(23)])))
        with torch.no_grad():
            self.w2_p.copy_((self.w2_p - (lr * learnable_grads[int(24)])))
        with torch.no_grad():
            self.w2_ro1.copy_((self.w2_ro1 - (lr * learnable_grads[int(25)])))
        with torch.no_grad():
            self.w2_ro2.copy_((self.w2_ro2 - (lr * learnable_grads[int(26)])))

# === Program ===
num_train = 8
num_test = 4
train_positions = torch.tensor([[[0.0, 0.0, 0.0], [1.3035, 0.121, (-0.0361)], [0.0387, 1.0143, (-0.0409)]], [[0.0, 0.0, 0.0], [0.5566, 0.1041, (-1.2624)], [(-0.4404), 0.7846, 0.8458]], [[0.0, 0.0, 0.0], [(-0.9748), 0.9579, 0.2162], [0.8859, 0.0659, (-0.309)]], [[0.0, 0.0, 0.0], [(-0.6634), 0.3849, (-0.4699)], [(-0.2285), (-0.3502), 1.2694]], [[0.0, 0.0, 0.0], [(-0.8672), (-0.2552), (-1.0005)], [(-0.2882), 0.8669, 0.9533]], [[0.0, 0.0, 0.0], [0.9919, (-0.2859), (-0.1822)], [(-0.5818), 1.0325, (-0.0912)]], [[0.0, 0.0, 0.0], [1.0513, (-0.3912), 0.607], [0.2607, 0.9788, (-0.3458)]], [[0.0, 0.0, 0.0], [1.1268, (-0.4287), (-0.8342)], [0.026, 1.3507, (-0.4679)]]], device=DEVICE)
train_energies = torch.tensor([(-12.4103), (-10.7186), (-11.8415), (-12.2896), (-10.8181), (-12.6558), (-12.6144), (-9.8413)], device=DEVICE)
test_positions = torch.tensor([[[0.0, 0.0, 0.0], [1.1014, 0.3095, 0.1934], [(-0.4569), 1.3549, (-0.1436)]], [[0.0, 0.0, 0.0], [0.611, (-0.8965), (-0.6491)], [0.5299, 0.687, 0.5715]], [[0.0, 0.0, 0.0], [1.2824, 0.0804, (-0.2181)], [(-0.5724), 1.4111, (-0.2409)]], [[0.0, 0.0, 0.0], [(-0.6127), 0.4966, 0.8616], [0.4113, (-1.2876), 0.3875]]], device=DEVICE)
test_energies = torch.tensor([(-11.3831), (-12.5956), (-10.0866), (-11.4604)], device=DEVICE)
energy_mean = (torch.sum(train_energies if isinstance(train_energies, torch.Tensor) else torch.tensor(float(train_energies))) / num_train)
energy_std = torch.sqrt((torch.sum(((train_energies - energy_mean) * (train_energies - energy_mean)) if isinstance(((train_energies - energy_mean) * (train_energies - energy_mean)), torch.Tensor) else torch.tensor(float(((train_energies - energy_mean) * (train_energies - energy_mean))))) / num_train) if isinstance((torch.sum(((train_energies - energy_mean) * (train_energies - energy_mean)) if isinstance(((train_energies - energy_mean) * (train_energies - energy_mean)), torch.Tensor) else torch.tensor(float(((train_energies - energy_mean) * (train_energies - energy_mean))))) / num_train), torch.Tensor) else torch.tensor(float((torch.sum(((train_energies - energy_mean) * (train_energies - energy_mean)) if isinstance(((train_energies - energy_mean) * (train_energies - energy_mean)), torch.Tensor) else torch.tensor(float(((train_energies - energy_mean) * (train_energies - energy_mean))))) / num_train))))
r_cut = 2.0
atomic_numbers = torch.tensor([8.0, 1.0, 1.0], device=DEVICE)
z_table = torch.tensor([1.0, 8.0], device=DEVICE)
positions = train_positions[int(0)]
node_attrs = one_hot(atomic_numbers, z_table)
edge_index = neighbor_search(positions, r_cut)
print(print(positions))
print(print(node_attrs))
print(print(edge_index))
train_edge_index = batch_edge_index(train_positions)
test_edge_index = batch_edge_index(test_positions)
num_elements = 2
num_channels = 8
w_embed = torch.stack([torch.distributions.Normal(0.0, 1.0).rsample((int(8),)) for _fi_z in range(int(num_elements)) for z in [torch.tensor(float(_fi_z), device=DEVICE)]])
π = 3.141592653589793
num_bessel = 8
p_env = 6.0
vectors = edge_vectors(positions, edge_index)
lengths = edge_lengths(vectors)
edge_feats = radial_embedding(lengths)
edge_attrs = spherical_harmonics(vectors, lengths)
print(print(edge_feats))
print(print(edge_attrs))
train_edge_feats = batch_edge_feats(train_positions, train_edge_index)
test_edge_feats = batch_edge_feats(test_positions, test_edge_index)
train_edge_attrs = batch_edge_attrs(train_positions, train_edge_index)
test_edge_attrs = batch_edge_attrs(test_positions, test_edge_index)
w_up = torch.stack([torch.distributions.Normal(0.0, 1.0).rsample((int(8),)) for _fi_c in range(int(num_channels)) for c in [torch.tensor(float(_fi_c), device=DEVICE)]])
silu_norm = 1.679176792
radial_hidden = 64
w_r1 = torch.stack([torch.distributions.Normal(0.0, 1.0).rsample((int(64),)) for _fi_a in range(int(num_bessel)) for a in [torch.tensor(float(_fi_a), device=DEVICE)]])
w_r2 = torch.stack([torch.distributions.Normal(0.0, 1.0).rsample((int(64),)) for _fi_a in range(int(radial_hidden)) for a in [torch.tensor(float(_fi_a), device=DEVICE)]])
w_r3 = torch.stack([torch.distributions.Normal(0.0, 1.0).rsample((int(64),)) for _fi_a in range(int(radial_hidden)) for a in [torch.tensor(float(_fi_a), device=DEVICE)]])
w_r4 = torch.stack([torch.distributions.Normal(0.0, 1.0).rsample((int(24),)) for _fi_a in range(int(radial_hidden)) for a in [torch.tensor(float(_fi_a), device=DEVICE)]])
cg_000 = torch.tensor([[[1.0]]], device=DEVICE)
cg_101 = torch.tensor([[[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]], [[0.0, 0.0, 1.0]]], device=DEVICE)
cg_202 = torch.tensor([[[1.0, 0.0, 0.0, 0.0, 0.0]], [[0.0, 1.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 1.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 1.0, 0.0]], [[0.0, 0.0, 0.0, 0.0, 1.0]]], device=DEVICE)
avg_num_neighbors = 2.0
w_0 = torch.stack([torch.distributions.Normal(0.0, 1.0).rsample((int(8),)) for _fi_c in range(int(num_channels)) for c in [torch.tensor(float(_fi_c), device=DEVICE)]])
w_1 = torch.stack([torch.distributions.Normal(0.0, 1.0).rsample((int(8),)) for _fi_c in range(int(num_channels)) for c in [torch.tensor(float(_fi_c), device=DEVICE)]])
w_2 = torch.stack([torch.distributions.Normal(0.0, 1.0).rsample((int(8),)) for _fi_c in range(int(num_channels)) for c in [torch.tensor(float(_fi_c), device=DEVICE)]])
cg_110 = torch.tensor([[[0.57735], [0.0], [0.0]], [[0.0], [0.57735], [0.0]], [[0.0], [0.0], [0.57735]]], device=DEVICE)
cg_220 = torch.tensor([[[0.447214], [0.0], [0.0], [0.0], [0.0]], [[0.0], [0.447214], [0.0], [0.0], [0.0]], [[0.0], [0.0], [0.447214], [0.0], [0.0]], [[0.0], [0.0], [0.0], [0.447214], [0.0]], [[0.0], [0.0], [0.0], [0.0], [0.447214]]], device=DEVICE)
cg_011 = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]], device=DEVICE)
cg_121 = torch.tensor([[[0.0, 0.0, 0.547723], [0.0, 0.547723, 0.0], [(-0.316228), 0.0, 0.0], [0.0, 0.0, 0.0], [(-0.547723), 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.547723, 0.0, 0.0], [0.0, 0.632456, 0.0], [0.0, 0.0, 0.547723], [0.0, 0.0, 0.0]], [[0.547723, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, (-0.316228)], [0.0, 0.547723, 0.0], [0.0, 0.0, 0.547723]]], device=DEVICE)
w_prod0 = torch.stack([torch.stack([torch.distributions.Normal(0.0, 0.25).rsample((int(8),)) for _fi_q in range(int(4)) for q in [torch.tensor(float(_fi_q), device=DEVICE)]]) for _fi_z in range(int(num_elements)) for z in [torch.tensor(float(_fi_z), device=DEVICE)]])
w_prod1 = torch.stack([torch.stack([torch.distributions.Normal(0.0, 0.333).rsample((int(8),)) for _fi_q in range(int(3)) for q in [torch.tensor(float(_fi_q), device=DEVICE)]]) for _fi_z in range(int(num_elements)) for z in [torch.tensor(float(_fi_z), device=DEVICE)]])
w_sc = torch.stack([torch.stack([torch.distributions.Normal(0.0, 1.0).rsample((int(8),)) for _fi_z in range(int(num_elements)) for z in [torch.tensor(float(_fi_z), device=DEVICE)]]) for _fi_c in range(int(num_channels)) for c in [torch.tensor(float(_fi_c), device=DEVICE)]])
w_p0 = torch.stack([torch.distributions.Normal(0.0, 1.0).rsample((int(8),)) for _fi_c in range(int(num_channels)) for c in [torch.tensor(float(_fi_c), device=DEVICE)]])
w_p1 = torch.stack([torch.distributions.Normal(0.0, 1.0).rsample((int(8),)) for _fi_c in range(int(num_channels)) for c in [torch.tensor(float(_fi_c), device=DEVICE)]])
w_readout = torch.distributions.Normal(0.0, 1.0).rsample((int(8),))
w2_up0 = torch.stack([torch.distributions.Normal(0.0, 1.0).rsample((int(8),)) for _fi_c in range(int(num_channels)) for c in [torch.tensor(float(_fi_c), device=DEVICE)]])
w2_up1 = torch.stack([torch.distributions.Normal(0.0, 1.0).rsample((int(8),)) for _fi_c in range(int(num_channels)) for c in [torch.tensor(float(_fi_c), device=DEVICE)]])
w2_r1 = torch.stack([torch.distributions.Normal(0.0, 1.0).rsample((int(64),)) for _fi_a in range(int(num_bessel)) for a in [torch.tensor(float(_fi_a), device=DEVICE)]])
w2_r2 = torch.stack([torch.distributions.Normal(0.0, 1.0).rsample((int(64),)) for _fi_a in range(int(radial_hidden)) for a in [torch.tensor(float(_fi_a), device=DEVICE)]])
w2_r3 = torch.stack([torch.distributions.Normal(0.0, 1.0).rsample((int(64),)) for _fi_a in range(int(radial_hidden)) for a in [torch.tensor(float(_fi_a), device=DEVICE)]])
w2_r4 = torch.stack([torch.distributions.Normal(0.0, 1.0).rsample((int(56),)) for _fi_a in range(int(radial_hidden)) for a in [torch.tensor(float(_fi_a), device=DEVICE)]])
cg_022 = torch.tensor([[[1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0]]], device=DEVICE)
cg_112 = torch.tensor([[[0.0, 0.0, (-0.408248), 0.0, (-0.707107)], [0.0, 0.707107, 0.0, 0.0, 0.0], [0.707107, 0.0, 0.0, 0.0, 0.0]], [[0.0, 0.707107, 0.0, 0.0, 0.0], [0.0, 0.0, 0.816497, 0.0, 0.0], [0.0, 0.0, 0.0, 0.707107, 0.0]], [[0.707107, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.707107, 0.0], [0.0, 0.0, (-0.408248), 0.0, 0.707107]]], device=DEVICE)
w2_msg = torch.stack([torch.stack([torch.distributions.Normal(0.0, 1.0).rsample((int(8),)) for _fi_c in range(int(num_channels)) for c in [torch.tensor(float(_fi_c), device=DEVICE)]]) for _fi_p in range(int(7)) for p in [torch.tensor(float(_fi_p), device=DEVICE)]])
w2_prod0 = torch.stack([torch.stack([torch.distributions.Normal(0.0, 0.25).rsample((int(8),)) for _fi_q in range(int(4)) for q in [torch.tensor(float(_fi_q), device=DEVICE)]]) for _fi_z in range(int(num_elements)) for z in [torch.tensor(float(_fi_z), device=DEVICE)]])
w2_sc = torch.stack([torch.stack([torch.distributions.Normal(0.0, 1.0).rsample((int(8),)) for _fi_z in range(int(num_elements)) for z in [torch.tensor(float(_fi_z), device=DEVICE)]]) for _fi_c in range(int(num_channels)) for c in [torch.tensor(float(_fi_c), device=DEVICE)]])
w2_p = torch.stack([torch.distributions.Normal(0.0, 1.0).rsample((int(8),)) for _fi_c in range(int(num_channels)) for c in [torch.tensor(float(_fi_c), device=DEVICE)]])
w2_ro1 = torch.stack([torch.distributions.Normal(0.0, 1.0).rsample((int(16),)) for _fi_a in range(int(num_channels)) for a in [torch.tensor(float(_fi_a), device=DEVICE)]])
w2_ro2 = torch.distributions.Normal(0.0, 1.0).rsample((int(16),))
mace_object = MACEModel(w_embed, w_up, w_r1, w_r2, w_r3, w_r4, w_0, w_1, w_2, w_prod0, w_prod1, w_sc, w_p0, w_p1, w_readout, w2_up0, w2_up1, w2_r1, w2_r2, w2_r3, w2_r4, w2_msg, w2_prod0, w2_sc, w2_p, w2_ro1, w2_ro2).to(DEVICE)
example_energy = mace_object(node_attrs, edge_index, edge_feats, edge_attrs)
print(print(example_energy))
lr = 0.05
epochs = 1
rmse_before = mace_object.evaluate()
print(print(rmse_before))
final_loss = mace_object.train(epochs, lr)
print(print(final_loss))
rmse_after = mace_object.evaluate()
print(print(rmse_after))
test_predictions = torch.zeros(int(num_test), device=DEVICE)
for sample in range(int(0), int(num_test)):
    test_predictions[int(sample)] = (test_energies[int(sample)] + mace_object.error_sample(sample))
print(print(test_predictions))