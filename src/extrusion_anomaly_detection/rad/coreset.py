import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from extrusion_anomaly_detection.rad import MODELS_DIR
from extrusion_anomaly_detection.rad.memory_bank import MemoryBank

CORESET_RATIO = 0.01
PROJ_DIM = 128
SEED = 0
CHUNK_IMAGES = 250


def random_projection(dim_in, dim_out=PROJ_DIM, seed=SEED):
    """Fixed Gaussian projection [dim_in, dim_out] (Johnson-Lindenstrauss)."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(dim_in, dim_out, generator=generator) / math.sqrt(dim_out)


def greedy_k_center(points, m, seed=SEED):
    """Greedy k-center (farthest point) selection of m rows of points [M, d].

    Starts from a random point and keeps, for every point, the squared distance
    to its nearest selected center; each step picks the point farthest from the
    selected set. Runs on the device of `points`. Returns idx [m] (cpu, int64).
    """
    n = points.shape[0]
    m = min(m, n)
    device = points.device

    generator = torch.Generator().manual_seed(seed)
    current = torch.randint(n, (1,), generator=generator).to(device)[0]

    sq_norms = points.float().pow(2).sum(dim=1)
    min_dist = torch.full((n,), math.inf, device=device)
    selected = torch.empty(m, dtype=torch.long, device=device)

    for t in tqdm(range(m), desc="k-center", leave=False):
        selected[t] = current
        dots = (points @ points[current]).float()
        dist = sq_norms + sq_norms[current] - 2 * dots
        min_dist = torch.minimum(min_dist, dist)
        # selected points can never be picked again, even with duplicate vectors
        min_dist[current] = -1.0
        current = min_dist.argmax()

    return selected.cpu()


def _image_chunks(n_images, chunk=CHUNK_IMAGES):
    for start in range(0, n_images, chunk):
        yield start, min(start + chunk, n_images)


def build_coreset(
    bank, out_dir, ratio=CORESET_RATIO, proj_dim=PROJ_DIM, seed=SEED, device="cuda"
):
    """Select a k-center coreset of the patches of every layer of `bank`.

    The selection runs on randomly projected patches; the saved vectors are the
    original ones. For each layer this writes coreset_l{layer}.npy [m, D] fp16
    and coreset_l{layer}_pos.npy [m, 3] (image, h, w). The [CLS] bank is kept
    whole, so global retrieval is unchanged.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_images = bank.n_images
    grid_h, grid_w = bank.meta["grid"]
    dim = bank.meta["dim"]
    per_image = grid_h * grid_w
    n_patches = n_images * per_image
    m = max(1, round(n_patches * ratio))

    projection = random_projection(dim, proj_dim, seed).to(device)
    timings = {}

    for layer in bank.layers:
        start_time = time.perf_counter()

        # 1. project every patch, streaming the bank from disk
        projected = torch.empty(n_patches, proj_dim, dtype=torch.float16, device=device)
        for start, end in _image_chunks(n_images):
            patches = bank.get_patches(layer, np.arange(start, end))
            patches = patches.to(device).float().reshape(-1, dim)
            projected[start * per_image : end * per_image] = (
                patches @ projection
            ).half()

        # 2. select the coreset on the projected patches
        selected = greedy_k_center(projected, m, seed).numpy()
        del projected
        selected.sort()

        # 3. gather the original vectors of the selected patches
        vectors = np.empty((m, dim), dtype=np.float16)
        filled = 0
        for start, end in _image_chunks(n_images):
            lo, hi = np.searchsorted(selected, [start * per_image, end * per_image])
            if lo == hi:
                continue
            patches = bank.get_patches(layer, np.arange(start, end)).numpy()
            local = selected[lo:hi] - start * per_image
            vectors[filled : filled + hi - lo] = patches.reshape(-1, dim)[local]
            filled += hi - lo

        image = selected // per_image
        rest = selected % per_image
        positions = np.stack([image, rest // grid_w, rest % grid_w], axis=1)

        np.save(out_dir / f"coreset_l{layer}.npy", vectors)
        np.save(out_dir / f"coreset_l{layer}_pos.npy", positions.astype(np.int32))
        timings[layer] = time.perf_counter() - start_time

    np.save(out_dir / "cls.npy", bank.cls.numpy())

    meta = {
        **{k: bank.meta[k] for k in ("model_id", "layers", "grid", "dim", "n_images")},
        "source_bank": str(bank.directory),
        "ratio": ratio,
        "proj_dim": proj_dim,
        "seed": seed,
        "n_patches": n_patches,
        "n_coreset": m,
        "build_seconds_per_layer": timings,
        "img_paths": bank.img_paths,
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return CoresetBank.open(out_dir)


class CoresetBank:
    """Coreset of patch vectors with their (image, h, w) positions.

    Same [CLS] interface as MemoryBank; patches are read with get_coreset().
    """

    def __init__(self, directory, meta, cls, vectors, positions):
        self.directory = Path(directory)
        self.meta = meta
        self.layers = tuple(meta["layers"])
        self.img_paths = meta["img_paths"]
        self._cls = cls
        self._vectors = vectors
        self._positions = positions

    @classmethod
    def open(cls, directory):
        directory = Path(directory)
        with open(directory / "meta.json") as f:
            meta = json.load(f)

        cls_bank = torch.from_numpy(np.load(directory / "cls.npy"))
        vectors, positions = {}, {}
        for layer in meta["layers"]:
            vectors[layer] = torch.from_numpy(
                np.load(directory / f"coreset_l{layer}.npy")
            )
            positions[layer] = torch.from_numpy(
                np.load(directory / f"coreset_l{layer}_pos.npy")
            ).long()
        return cls(directory, meta, cls_bank, vectors, positions)

    @property
    def cls(self):
        return self._cls

    @property
    def n_images(self):
        return self.meta["n_images"]

    def get_coreset(self, layer):
        """(vectors fp16 [m, D], positions int64 [m, 3] as (image, h, w))."""
        return self._vectors[layer], self._positions[layer]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", type=Path, default=MODELS_DIR / "rad_bank")
    parser.add_argument("--out", type=Path, default=MODELS_DIR / "rad_coreset")
    parser.add_argument("--ratio", type=float, default=CORESET_RATIO)
    parser.add_argument("--proj-dim", type=int, default=PROJ_DIM)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    bank = MemoryBank.open(args.bank)
    coreset = build_coreset(
        bank, args.out, args.ratio, args.proj_dim, args.seed, device=device
    )

    size_mb = sum(f.stat().st_size for f in args.out.iterdir()) / 2**20
    print(
        f"coreset {coreset.meta['n_coreset']} / {coreset.meta['n_patches']} patches "
        f"per layer, {size_mb:.0f} MB in {args.out}"
    )
    for layer, seconds in coreset.meta["build_seconds_per_layer"].items():
        print(f"layer {layer}: {seconds:.0f} s")


if __name__ == "__main__":
    main()
