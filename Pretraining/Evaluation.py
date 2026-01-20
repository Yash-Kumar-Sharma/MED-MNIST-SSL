import torch
import math

# ---------------------------------------------
# Helper: contingency matrix
# ---------------------------------------------
def _contingency(labels_true, labels_pred, device=None, dtype=torch.float32):
    labels_true = labels_true.long()
    labels_pred = labels_pred.long()
    device = device or labels_true.device
    unique_true, true_idx = torch.unique(labels_true, return_inverse=True)
    unique_pred, pred_idx = torch.unique(labels_pred, return_inverse=True)
    n_true = unique_true.size(0)
    n_pred = unique_pred.size(0)
    contingency = torch.zeros((n_true, n_pred), device=device, dtype=dtype)
    indices = true_idx * n_pred + pred_idx
    ones = torch.ones_like(indices, dtype=dtype, device=device)
    contingency.view(-1).scatter_add_(0, indices, ones)
    return contingency, unique_true, unique_pred

# ---------------------------------------------
# Adjusted Rand Index (ARI)
# ---------------------------------------------
def ari_torch(labels_true, labels_pred, eps=1e-8):
    contingency, _, _ = _contingency(labels_true, labels_pred, device=labels_true.device)
    n = labels_true.size(0)
    nij = contingency
    nij_sum = (nij * (nij - 1)).sum() / 2.0
    a = nij.sum(dim=1)
    b = nij.sum(dim=0)
    ai = (a * (a - 1)).sum() / 2.0
    bj = (b * (b - 1)).sum() / 2.0
    total_pairs = n * (n - 1) / 2.0
    expected_index = (ai * bj) / (total_pairs + eps)
    max_index = 0.5 * (ai + bj)
    ari = (nij_sum - expected_index) / (max_index - expected_index + eps)
    return ari.item()

# ---------------------------------------------
# Mutual Information (MI helper)
# ---------------------------------------------
def mutual_information_from_contingency(contingency, eps=1e-12):
    n = contingency.sum()
    P = contingency / (n + eps)
    Pa = P.sum(dim=1, keepdim=True)
    Pb = P.sum(dim=0, keepdim=True)
    outer = Pa @ Pb
    mask = P > 0
    MI = (P[mask] * (torch.log(P[mask] + eps) - torch.log(outer[mask] + eps))).sum()
    return MI, Pa.squeeze(1), Pb.squeeze(0)

# ---------------------------------------------
# NMI
# ---------------------------------------------
def nmi_torch(labels_true, labels_pred, eps=1e-12):
    contingency, _, _ = _contingency(labels_true, labels_pred, device=labels_true.device)
    MI, Pa, Pb = mutual_information_from_contingency(contingency, eps=eps)
    H_true = -(Pa[Pa>0] * torch.log(Pa[Pa>0] + eps)).sum()
    H_pred = -(Pb[Pb>0] * torch.log(Pb[Pb>0] + eps)).sum()
    nmi = (2.0 * MI) / (H_true + H_pred + eps)
    return nmi.item()

# ---------------------------------------------
# AMI (approximation)
# ---------------------------------------------
def ami_torch(labels_true, labels_pred, eps=1e-8):
    contingency, _, _ = _contingency(labels_true, labels_pred, device=labels_true.device)
    MI, Pa, Pb = mutual_information_from_contingency(contingency, eps=1e-12)
    H_true = -(Pa[Pa>0] * torch.log(Pa[Pa>0] + eps)).sum()
    H_pred = -(Pb[Pb>0] * torch.log(Pb[Pb>0] + eps)).sum()
    expected_MI = (H_true * H_pred) / (torch.log(torch.tensor(labels_true.size(0) + 1.0, device=labels_true.device)) + eps)
    ami = (MI - expected_MI) / (0.5 * (H_true + H_pred) - expected_MI + eps)
    return ami.item()

# ---------------------------------------------
# V-measure
# ---------------------------------------------
def v_measure_torch(labels_true, labels_pred, eps=1e-12):
    contingency, _, _ = _contingency(labels_true, labels_pred, device=labels_true.device)
    MI, Pa, Pb = mutual_information_from_contingency(contingency, eps=eps)
    H_true = -(Pa[Pa>0] * torch.log(Pa[Pa>0] + eps)).sum()
    H_pred = -(Pb[Pb>0] * torch.log(Pb[Pb>0] + eps)).sum()
    homogeneity = 1.0 - ((H_true - MI) / (H_true + eps))
    completeness = 1.0 - ((H_pred - MI) / (H_pred + eps))
    v_measure = 2.0 * (homogeneity * completeness) / (homogeneity + completeness + eps)
    return v_measure.item()

# ---------------------------------------------
# FMI (Fowlkes-Mallows Index)
# ---------------------------------------------
def fmi_torch(labels_true, labels_pred, eps=1e-12):
    contingency, _, _ = _contingency(labels_true, labels_pred, device=labels_true.device)
    nij = contingency
    TP = (nij * (nij - 1)).sum() / 2.0
    a = nij.sum(dim=1)
    b = nij.sum(dim=0)
    FP_plus_TP = (b * (b - 1)).sum() / 2.0
    FN_plus_TP = (a * (a - 1)).sum() / 2.0
    denom = torch.sqrt((FP_plus_TP) * (FN_plus_TP) + eps)
    fmi = TP / (denom + eps)
    return fmi.item()

# ---------------------------------------------
# Alignment (SimCLR)
# ---------------------------------------------
def alignment(z1, z2):
    diff = z1 - z2
    return (diff.pow(2).sum(dim=1).mean()).item()

# ---------------------------------------------
# Uniformity (SimCLR)
# ---------------------------------------------
def uniformity(z, t=2.0):
    z = z / (z.norm(dim=1, keepdim=True) + 1e-12)
    pdist_sq = torch.cdist(z, z, p=2).pow(2)
    n = z.size(0)
    mask = ~torch.eye(n, dtype=torch.bool, device=z.device)
    vals = torch.exp(-t * pdist_sq[mask])
    return torch.log(vals.mean() + 1e-12).item()

# ---------------------------------------------
# DMI (Determinant Mutual Information)
# ---------------------------------------------
def dmi_from_contingency(contingency, eps=1e-12):
    n = contingency.sum()
    M = contingency / (n + eps)
    if M.shape[0] <= M.shape[1]:
        mat = M @ M.t()
    else:
        mat = M.t() @ M
    Id = torch.eye(mat.size(0), device=mat.device, dtype=mat.dtype)
    mat = Id + mat
    sign, logabsdet = torch.slogdet(mat)
    if sign <= 0:
        return (logabsdet * sign).item()
    return logabsdet.item()

def dmi_torch(labels_true, labels_pred):
    contingency, _, _ = _contingency(labels_true, labels_pred, device=labels_true.device)
    return dmi_from_contingency(contingency)

# ---------------------------------------------
# Weighted / Entropy-balanced NMI
# ---------------------------------------------
def weighted_nmi_torch(labels_true, labels_pred, eps=1e-12):
    contingency, _, _ = _contingency(labels_true, labels_pred, device=labels_true.device)
    n = contingency.sum()
    P = contingency / (n + eps)
    Pa = P.sum(dim=1)
    Pb = P.sum(dim=0)
    w = 1.0 / (Pa + eps)
    w = w / w.sum()
    MI = 0.0
    for i in range(P.shape[0]):
        row = P[i]
        outer = (Pa[i] * Pb)
        mask = row > 0
        if mask.any():
            MI = MI + w[i] * (row[mask] * (torch.log(row[mask] + eps) - torch.log(outer[mask] + eps))).sum()
    H_true_w = -(w * Pa * torch.log(Pa + eps)).sum()
    H_pred = -(Pb[Pb > 0] * torch.log(Pb[Pb > 0] + eps)).sum()
    w_nmi = (2.0 * MI) / (H_true_w + H_pred + eps)
    return w_nmi.item()

# ---------------------------------------------
# Silhouette Coefficient
# ---------------------------------------------
def silhouette_score_torch_mean_std(X, labels):
    """
    Compute Silhouette Coefficient mean and std for embeddings X with cluster labels.
    X: [N, D] embeddings tensor
    labels: [N] cluster assignments tensor
    Returns: (SC_mean, SC_std)
    """

    device = X.device
    N = X.size(0)

    # Normalize for stability
    X = X / (X.norm(dim=1, keepdim=True) + 1e-8)

    # Pairwise distances
    dist_matrix = torch.cdist(X, X, p=2)

    unique_labels = torch.unique(labels)
    n_clusters = len(unique_labels)

    if n_clusters < 2:
        return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)

    a = torch.zeros(N, device=device)  # intra-cluster distance
    b = torch.zeros(N, device=device)  # nearest-cluster distance

    for lbl in unique_labels:
        mask_i = (labels == lbl)
        cluster_i = dist_matrix[mask_i][:, mask_i]
        if cluster_i.size(0) > 1:
            a[mask_i] = cluster_i.sum(1) / (cluster_i.size(1) - 1)
        else:
            a[mask_i] = 0.0

        other_means = []
        for other_lbl in unique_labels:
            if other_lbl == lbl:
                continue
            mask_j = (labels == other_lbl)
            inter = dist_matrix[mask_i][:, mask_j].mean(1)
            other_means.append(inter)
        b[mask_i] = torch.stack(other_means, dim=1).min(1)[0]

    s = (b - a) / (torch.maximum(a, b) + 1e-8)

    SC_mean = s.mean()
    SC_std = s.std(unbiased=False)

    return SC_mean, SC_std

# ---------------------------------------------
# Example usage
# ---------------------------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N = 5000
    # Imbalanced true labels
    true_counts = [int(N*0.7), int(N*0.2), N - int(N*0.7) - int(N*0.2)]
    labels_true = torch.tensor([0]*true_counts[0] + [1]*true_counts[1] + [2]*true_counts[2], device=device)
    perm = torch.randperm(N, device=device)
    labels_pred = labels_true[perm].clone()
    num_swap = int(0.2 * N)
    swap_idx = torch.randperm(N, device=device)[:num_swap]
    labels_pred[swap_idx] = torch.randint(0, 3, (num_swap,), device=device)

    # Random embeddings for alignment/uniformity
    D = 128
    z1 = torch.randn((N, D), device=device)
    z2 = z1 + 0.1 * torch.randn((N, D), device=device)
    z1 = z1 / (z1.norm(dim=1, keepdim=True) + 1e-12)
    z2 = z2 / (z2.norm(dim=1, keepdim=True) + 1e-12)

    results = {
        'ARI': ari_torch(labels_true, labels_pred),
        'NMI': nmi_torch(labels_true, labels_pred),
        'AMI': ami_torch(labels_true, labels_pred),
        'V-measure': v_measure_torch(labels_true, labels_pred),
        'FMI': fmi_torch(labels_true, labels_pred),
        'Alignment': alignment(z1, z2),
        'Uniformity': uniformity(torch.cat([z1, z2], dim=0)),
        'DMI': dmi_torch(labels_true, labels_pred),
        'Weighted NMI': weighted_nmi_torch(labels_true, labels_pred)
    }

    print("Results on synthetic data:")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")
    
    # Example
    X = torch.randn(300, 16).cuda()
    labels = torch.randint(0, 4, (300,)).cuda()

    SC_mean, SC_std = silhouette_score_torch_mean_std(X, labels)
    print(f"SC_mean: {SC_mean.item():.4f}, SC_std: {SC_std.item():.4f}")

