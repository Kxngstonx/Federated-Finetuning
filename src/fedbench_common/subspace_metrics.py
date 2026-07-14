"""Basis-overlap / cosine-similarity metrics -- FedRot-LoRA and FeDoRA only, per the
experiment plan. Not applicable to the other 5 strategies (fedavg, fedit, fedsvd, ffalora,
flora), which don't have a well-defined "reference direction" to compare a client's basis
against in the same sense.

Uses principal angles (cosines of the singular values of Qa^T @ Qb, where Qa/Qb are orthonormal
bases of the two row-spaces being compared) rather than a naive flattened cosine similarity,
since LoRA's A matrices are r-dimensional subspaces (r>1), not single vectors -- a subspace-level
similarity is the correct generalization of "cosine similarity" here. 1.0 = identical subspaces,
0.0 = orthogonal subspaces.
"""

import math
from itertools import combinations
from typing import Sequence

import numpy as np
import torch


def principal_angle_cosines(subspace_a: torch.Tensor, subspace_b: torch.Tensor) -> torch.Tensor:
    """subspace_a, subspace_b: (r, d) row-space bases (rows need not be orthonormal already --
    QR-orthonormalized here). Returns the r singular values of Qa^T @ Qb in descending order,
    i.e. the cosines of the principal angles between the two row-spaces."""
    qa, _ = torch.linalg.qr(subspace_a.to(torch.float32).T)  # (d, r) orthonormal columns
    qb, _ = torch.linalg.qr(subspace_b.to(torch.float32).T)
    return torch.linalg.svdvals(qa.T @ qb)


def mean_subspace_overlap(subspace_a, subspace_b) -> float:
    """Scalar summary: mean cosine across all principal angles between the two subspaces."""
    a_t = subspace_a if isinstance(subspace_a, torch.Tensor) else torch.from_numpy(np.asarray(subspace_a))
    b_t = subspace_b if isinstance(subspace_b, torch.Tensor) else torch.from_numpy(np.asarray(subspace_b))
    return principal_angle_cosines(a_t, b_t).mean().item()


def overlap_to_misalign_deg(overlap: float) -> float:
    """arccos(overlap) in degrees -- overlap alone is a bad anomaly-detection scale: it saturates
    near 1.0 (cosine is flat there), so e.g. 0.999999 -> 0.998 looks like a tiny 0.0016 drop but
    is actually ~0.08deg -> ~3.2deg, a ~40x change in true angular misalignment. Clamped into
    [-1, 1] first since float error can push a numerically-1.0 overlap slightly past 1."""
    return math.degrees(math.acos(min(max(overlap, -1.0), 1.0)))


def mean_pairwise_subspace_overlap(subspaces: Sequence) -> float:
    """Requirement (FedRot): mean subspace overlap across all client pairs (i, j), i != j. Used
    to compare pre-rotation vs post-rotation A_i/A_j alignment across clients -- see
    strategies/fedrot.py::FedRot.aggregate_fit."""
    pairs = list(combinations(range(len(subspaces)), 2))
    if not pairs:
        return 1.0  # a single client has no pairwise comparison to make
    return float(np.mean([mean_subspace_overlap(subspaces[i], subspaces[j]) for i, j in pairs]))
