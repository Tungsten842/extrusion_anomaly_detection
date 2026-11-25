import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from extrusion_anomaly_detection.data.dataset import CaxtonDataset
from extrusion_anomaly_detection.rad import MODELS_DIR, RESULTS_DIR
from extrusion_anomaly_detection.rad.features import FeatureExtractor

BATCH_SIZE = 16
NUM_WORKERS = 2


def collate_images(batch):
    return torch.stack([image for image, _ in batch])


def build_memory_bank(dataset, extractor, out_dir, batch_size=BATCH_SIZE):
    """Extract features for every image of `dataset` and write them to `out_dir`.

    Patches are written batch by batch into fp16 memmaps, so the full bank never
    has to fit in RAM. meta.json is written last and marks the bank as complete.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n = len(dataset)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=NUM_WORKERS,
        collate_fn=collate_images,
    )

    cls_bank = None
    patch_banks = {}
    offset = 0

    for images in tqdm(loader, desc=f"features -> {out_dir}"):
        cls, patches = extractor.extract(images)
        b = cls.shape[0]

        if cls_bank is None:
            cls_bank = np.zeros((n, cls.shape[1]), dtype=np.float32)
            for layer, feats in patches.items():
                patch_banks[layer] = np.lib.format.open_memmap(
                    out_dir / f"patch_l{layer}.npy",
                    mode="w+",
                    dtype=np.float16,
                    shape=(n, *feats.shape[1:]),
                )

        cls_bank[offset : offset + b] = cls.cpu().numpy()
        for layer, feats in patches.items():
            patch_banks[layer][offset : offset + b] = feats.half().cpu().numpy()

        offset += b

    np.save(out_dir / "cls.npy", cls_bank)
    for bank in patch_banks.values():
        bank.flush()

    grid = list(next(iter(patch_banks.values())).shape[1:3])
    meta = {
        "model_id": extractor.model.config._name_or_path,
        "layers": list(extractor.layers),
        "grid": grid,
        "dim": extractor.hidden_size,
        "n_images": n,
        "img_paths": [row["img_path"] for row in dataset.rows],
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return MemoryBank.open(out_dir)


class MemoryBank:
    def __init__(self, directory, meta, cls, patches):
        self.directory = Path(directory)
        self.meta = meta
        self.layers = tuple(meta["layers"])
        self.img_paths = meta["img_paths"]
        self._cls = cls
        self._patches = patches

    @classmethod
    def open(cls, directory):
        directory = Path(directory)
        with open(directory / "meta.json") as f:
            meta = json.load(f)

        cls_bank = torch.from_numpy(np.load(directory / "cls.npy"))
        patches = {
            layer: np.load(directory / f"patch_l{layer}.npy", mmap_mode="r")
            for layer in meta["layers"]
        }
        return cls(directory, meta, cls_bank, patches)

    @property
    def cls(self):
        return self._cls

    @property
    def n_images(self):
        return self.meta["n_images"]

    def get_patches(self, layer, idx):
        """Read the patches of images `idx` for one layer: fp16 [K, H, W, D].

        Rows are read with plain file reads instead of memmap fancy indexing:
        page-fault driven memmap reads are ~3x slower once the bank no longer
        fits in the page cache.
        """
        memmap = self._patches[layer]
        row_shape = memmap.shape[1:]
        row_bytes = memmap[0].nbytes

        idx = np.asarray(idx)
        out = np.empty((len(idx), *row_shape), dtype=memmap.dtype)

        # sorted reads are sequential on disk; each row goes to its requested slot
        with open(memmap.filename, "rb", buffering=0) as f:
            for slot in np.argsort(idx, kind="stable"):
                f.seek(memmap.offset + int(idx[slot]) * row_bytes)
                f.readinto(memoryview(out[slot]).cast("B"))

        return torch.from_numpy(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    if args.split == "train":
        out_dir = MODELS_DIR / "rad_bank"
    else:
        out_dir = RESULTS_DIR / f"features_{args.split}"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor = FeatureExtractor(device=device)
    dataset = CaxtonDataset(args.split)

    bank = build_memory_bank(dataset, extractor, out_dir, batch_size=args.batch_size)

    grid_h, grid_w = bank.meta["grid"]
    print(
        f"saved {bank.n_images} images to {out_dir}: layers {list(bank.layers)}, "
        f"{grid_h}x{grid_w} patches of dim {bank.meta['dim']}"
    )


if __name__ == "__main__":
    main()
