import zipfile
from collections import defaultdict
from pathlib import Path

import polars as pl
from torchcodec.decoders import decode_image
from torchcodec.encoders import JpegEncoder
from torchvision.transforms import v2
from tqdm import tqdm


def process_image(img_path, split, output_dir, transform, archive):
    path = Path(img_path[15:])
    image_name = f"{path.parent.name}/{path.name}"

    image = decode_image(archive.read(image_name))

    image = transform(image)

    output_path = output_dir / split / path.with_suffix(".jpg")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    JpegEncoder(image).to_file(
        output_path,
        quality=95,
    )

    return str(output_path)


def main():
    csv_path = Path("dataset/caxton_dataset_final.csv")
    output_dir = Path("dataset_resized")
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = 43

    transform = v2.Compose(
        [
            v2.Resize(size=512),
            v2.CenterCrop(size=512),
        ]
    )

    df = pl.read_csv(csv_path)

    df = df.with_columns(
        pl.col("img_path").str.split("/").list.get(-2).alias("print_id")
    )

    train_pct = 0.70
    val_pct = 0.15
    test_pct = 0.15

    prints = (
        df["print_id"].unique().sort().sample(fraction=1.0, shuffle=True, seed=seed)
    )
    total_prints = len(prints)

    n_train_prints = int(total_prints * train_pct)
    n_val_prints = int(total_prints * val_pct)
    n_test_prints = int(total_prints * test_pct)

    train_prints = prints.slice(0, n_train_prints).to_list()
    val_prints = prints.slice(n_train_prints, n_val_prints).to_list()
    test_prints = prints.slice(n_train_prints + n_val_prints, n_test_prints).to_list()

    normal_df = df.filter(
        pl.all_horizontal(
            pl.col(["flow_rate_class", "feed_rate_class", "hotend_class"]) == 1
        )
    )

    anomaly_df = df.filter(
        ~pl.all_horizontal(
            pl.col(["flow_rate_class", "feed_rate_class", "hotend_class"]) == 1
        )
    )

    n_train_normal_df = 5000

    n_val_normal_df = 2000
    n_val_anomalies = 200

    n_test_normal_df = 2000
    n_test_anomalies = 200

    train_df = normal_df.filter(pl.col("print_id").is_in(train_prints)).sample(
        n=n_train_normal_df, shuffle=True, seed=seed + 1
    )

    val_df = pl.concat(
        [
            normal_df.filter(pl.col("print_id").is_in(val_prints)).sample(
                n=n_val_normal_df, shuffle=True, seed=seed + 2
            ),
            anomaly_df.filter(pl.col("print_id").is_in(val_prints)).sample(
                n=n_val_anomalies, shuffle=True, seed=seed + 3
            ),
        ]
    ).sample(fraction=1.0, shuffle=True, seed=seed + 4)

    test_df = pl.concat(
        [
            normal_df.filter(pl.col("print_id").is_in(test_prints)).sample(
                n=n_test_normal_df, shuffle=True, seed=seed + 5
            ),
            anomaly_df.filter(pl.col("print_id").is_in(test_prints)).sample(
                n=n_test_anomalies, shuffle=True, seed=seed + 6
            ),
        ]
    ).sample(fraction=1.0, shuffle=True, seed=seed + 7)

    splits = {
        "train": train_df,
        "val": val_df,
        "test": test_df,
    }

    for split, split_df in splits.items():
        output_paths = [
            str(output_dir / split / Path(row["img_path"][15:]).with_suffix(".jpg"))
            for row in split_df.iter_rows(named=True)
        ]
        split_df.with_columns(pl.Series("img_path", output_paths)).write_csv(
            output_dir / f"{split}.csv"
        )

    jobs = defaultdict(list)

    for split, split_df in splits.items():
        for row in split_df.iter_rows(named=True):
            path = Path(row["img_path"][7:])
            zip_path = path.parent.with_suffix(".zip")
            jobs[zip_path].append((row["img_path"], split))

    for zip_path, images in tqdm(jobs.items()):
        with zipfile.ZipFile(zip_path, "r") as archive:
            for img_path, split in images:
                process_image(img_path, split, output_dir, transform, archive)


if __name__ == "__main__":
    main()
