import argparse
import json
import time
from pathlib import Path

import polars as pl
import torch
from tqdm import tqdm

from extrusion_anomaly_detection.data.dataset import CaxtonDataset
from extrusion_anomaly_detection.rad import RESULTS_DIR
from extrusion_anomaly_detection.rad.features import FeatureExtractor
from extrusion_anomaly_detection.rad.memory_bank import MemoryBank, build_memory_bank
from extrusion_anomaly_detection.rad.model import open_bank
from extrusion_anomaly_detection.rad.scoring import POOL, RHO, K, RADScorer


def load_features(split, dataset, bank_meta, device):
    """Open the feature cache of `split`, building it first if it is missing,
    incomplete or extracted with a different model/layers than the bank.

    Returns (cache, extraction ms per image or None if the cache was reused).
    """
    cache_dir = RESULTS_DIR / f"features_{split}"

    if (cache_dir / "meta.json").exists():
        cache = MemoryBank.open(cache_dir)
        same_model = (
            cache.meta["model_id"] == bank_meta["model_id"]
            and cache.meta["layers"] == bank_meta["layers"]
        )
        if same_model and cache.n_images == len(dataset):
            return cache, None

    extractor = FeatureExtractor(
        model_id=bank_meta["model_id"], layers=bank_meta["layers"], device=device
    )
    start = time.perf_counter()
    cache = build_memory_bank(dataset, extractor, cache_dir)
    extract_ms = (time.perf_counter() - start) * 1000 / len(dataset)

    del extractor
    if device == "cuda":
        torch.cuda.empty_cache()

    return cache, extract_ms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--bank", type=Path, default=None)
    parser.add_argument("--coreset", action="store_true")
    parser.add_argument("--k", type=int, default=K)
    parser.add_argument("--rho", type=int, default=RHO)
    parser.add_argument("--pool", choices=["top1", "max", "mean"], default=POOL)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    bank = open_bank(args.coreset, args.bank)
    suffix = "_coreset" if args.coreset else ""
    dataset = CaxtonDataset(args.split)
    n_images = len(dataset)

    cache, extract_ms = load_features(args.split, dataset, bank.meta, device)
    assert cache.img_paths == dataset.df["img_path"].to_list(), (
        "feature cache out of sync"
    )

    scorer = RADScorer(bank, k=args.k, rho=args.rho, pool=args.pool).to(device)

    scores = []
    start = time.perf_counter()
    for i in tqdm(range(n_images), desc=f"scoring {args.split}"):
        patches = {layer: cache.get_patches(layer, [i])[0] for layer in cache.layers}
        score, _ = scorer.score(cache.cls[i], patches)
        scores.append(score.item())
    if device == "cuda":
        torch.cuda.synchronize()
    score_ms = (time.perf_counter() - start) * 1000 / n_images

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    scores_df = dataset.df.insert_column(1, pl.Series("score", scores))
    scores_df.write_csv(RESULTS_DIR / f"scores_{args.split}{suffix}.csv")

    run_info = {
        "split": args.split,
        "bank": str(bank.directory),
        "coreset": args.coreset,
        "coreset_ratio": bank.meta.get("ratio"),
        "n_bank_images": bank.n_images,
        "n_images": n_images,
        "model_id": bank.meta["model_id"],
        "layers": bank.meta["layers"],
        "k": args.k,
        "rho": args.rho,
        "pool": args.pool,
        "device": device,
        "extract_ms_per_image": extract_ms,
        "score_ms_per_image": score_ms,
        "coreset_fallback_patches": scorer.n_fallback if args.coreset else None,
        "coreset_fallback_fraction": (
            scorer.n_fallback / scorer.n_patches if args.coreset else None
        ),
        "peak_vram_mb": (
            torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else None
        ),
    }
    with open(RESULTS_DIR / f"run_{args.split}{suffix}.json", "w") as f:
        json.dump(run_info, f, indent=2)

    print(json.dumps(run_info, indent=2))


if __name__ == "__main__":
    main()
