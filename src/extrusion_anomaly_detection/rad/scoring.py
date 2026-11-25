import math

import torch
import torch.nn.functional as F
from torch import nn

K = 100
RHO = 2
POOL = "mean"
TOP_PERCENT = 0.01
IMAGE_SIZE = 512


def retrieve_topk(cls_test, cls_bank, k=K):
    """Indices of the k bank images whose [CLS] is most similar to cls_test.

    cls_test: [D], cls_bank: [N, D], both l2-normalized. Returns idx [k].
    """
    sims = cls_bank @ cls_test
    return sims.topk(min(k, cls_bank.shape[0])).indices


def neighborhood_similarity(test, refs, rho=RHO):
    """Best cosine similarity of each test patch within a (2*rho+1)^2 window.

    test: [H, W, D], refs: [K, H, W, D], both l2-normalized.
    For every offset (dh, dw) in [-rho, rho]^2 the test patch at (h, w) is
    compared with refs[:, h + dh, w + dw]; offsets that fall outside the grid
    are skipped. Returns sim [H, W], the max over references and offsets.
    """
    h, w, _ = test.shape
    sim = torch.full((h, w), -math.inf, device=test.device, dtype=test.dtype)

    for dh in range(-rho, rho + 1):
        for dw in range(-rho, rho + 1):
            h0, h1 = max(0, -dh), h - max(0, dh)
            w0, w1 = max(0, -dw), w - max(0, dw)
            if h0 >= h1 or w0 >= w1:
                continue

            t = test[h0:h1, w0:w1]
            r = refs[:, h0 + dh : h1 + dh, w0 + dw : w1 + dw]
            s = torch.einsum("hwd,khwd->khw", t, r).amax(dim=0)

            sim[h0:h1, w0:w1] = torch.maximum(sim[h0:h1, w0:w1], s)

    return sim


def coreset_similarity(test, vectors, positions, idx, rho=RHO):
    """neighborhood_similarity against a coreset instead of whole reference grids.

    test: [H, W, D]; vectors: [m, D] and positions: [m, 3] (image, h, w) of the
    coreset of one layer; idx: retrieved image indices. A test patch at (h, w) is
    compared with the coreset patches of the retrieved images whose position is
    within rho cells. Patches with no candidate fall back to the whole coreset.
    Returns (sim [H, W], number of fallback patches).
    """
    h, w, d = test.shape
    flat = test.reshape(-1, d)

    candidates = torch.isin(positions[:, 0], idx)
    cand_vectors = vectors[candidates]
    cand_pos = positions[candidates]

    grid_h = torch.arange(h, device=test.device).repeat_interleave(w)
    grid_w = torch.arange(w, device=test.device).repeat(h)
    near = ((grid_h[:, None] - cand_pos[None, :, 1]).abs() <= rho) & (
        (grid_w[:, None] - cand_pos[None, :, 2]).abs() <= rho
    )

    sims = flat @ cand_vectors.T
    sims = sims.masked_fill(~near, -math.inf)
    sim = sims.amax(dim=1) if sims.shape[1] > 0 else sims.new_full((h * w,), -math.inf)

    missing = torch.isinf(sim)
    n_missing = int(missing.sum())
    if n_missing:
        sim[missing] = (flat[missing] @ vectors.T).amax(dim=1)

    return sim.reshape(h, w), n_missing


def anomaly_map(patch_scores):
    """Average per-layer patch scores {layer: [H, W]}, upsample to [512, 512]."""
    fused = torch.stack(list(patch_scores.values())).mean(dim=0)
    fused = F.interpolate(
        fused[None, None],
        size=(IMAGE_SIZE, IMAGE_SIZE),
        mode="bilinear",
        align_corners=False,
    )
    return fused[0, 0]


def image_score(amap, pool=POOL):
    """Reduce an anomaly map to a 0-dim tensor: top 1% mean, max or mean."""
    values = amap.flatten()
    if pool == "top1":
        n = max(1, round(values.numel() * TOP_PERCENT))
        return values.topk(n).values.mean()
    if pool == "max":
        return values.max()
    if pool == "mean":
        return values.mean()
    raise ValueError(f"unknown pool: {pool}")


class RADScorer(nn.Module):
    """Score test images against a bank with `cls` [N, D] and either
    `get_patches` (full MemoryBank) or `get_coreset` (CoresetBank).

    The [CLS] bank and the coreset are buffers, so they follow .to(device);
    the full bank stays on disk and its patches are moved when read.
    """

    def __init__(self, bank, k=K, rho=RHO, pool=POOL):
        super().__init__()
        self.bank = bank
        self.k = k
        self.rho = rho
        self.pool = pool
        self.register_buffer("cls_bank", bank.cls.float())

        self.use_coreset = hasattr(bank, "get_coreset")
        if self.use_coreset:
            for layer in bank.layers:
                vectors, positions = bank.get_coreset(layer)
                self.register_buffer(f"vectors_l{layer}", vectors.float())
                self.register_buffer(f"positions_l{layer}", positions)
        self.n_fallback = 0
        self.n_patches = 0

    @torch.inference_mode()
    def score(self, cls, patches):
        """cls: [D], patches: {layer: [H, W, D]}.

        Returns (score as a 0-dim tensor, amap [512, 512]).
        """
        device = self.cls_bank.device
        idx = retrieve_topk(cls.to(device).float(), self.cls_bank, self.k)

        patch_scores = {}
        for layer, test in patches.items():
            test = test.to(device).float()
            if not self.use_coreset:
                refs = self.bank.get_patches(layer, idx.cpu().numpy())
                sim = neighborhood_similarity(test, refs.to(device).float(), self.rho)
            else:
                vectors = getattr(self, f"vectors_l{layer}")
                positions = getattr(self, f"positions_l{layer}")
                sim, n_missing = coreset_similarity(
                    test, vectors, positions, idx, self.rho
                )
                self.n_fallback += n_missing
                self.n_patches += sim.numel()
            patch_scores[layer] = 1.0 - sim

        amap = anomaly_map(patch_scores)
        return image_score(amap, self.pool), amap
