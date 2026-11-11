from pathlib import Path
from typing import Literal

import polars as pl
from torch.utils.data import Dataset
from torchcodec.decoders import decode_image
from tqdm import tqdm


class CaxtonDataset(Dataset):
    def __init__(self, split: Literal["train", "test", "val"] = "train"):
        self.df = pl.read_csv(Path("dataset_resized") / f"{split}.csv")
        self.rows = self.df.to_dicts()

        self.parameter_columns = [
            col for col in self.df.columns if col not in ("img_path")
        ]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]

        image = decode_image(row["img_path"])

        parameters = {col: row[col] for col in self.parameter_columns}

        return image, parameters


dataset = CaxtonDataset()
for i, p in tqdm(dataset):
    print(p)
