import torch
from torch import nn

from extrusion_anomaly_detection.rad import BANK_DIR, CORESET_DIR
from extrusion_anomaly_detection.rad.coreset import CoresetBank
from extrusion_anomaly_detection.rad.features import FeatureExtractor
from extrusion_anomaly_detection.rad.memory_bank import MemoryBank
from extrusion_anomaly_detection.rad.scoring import RADScorer


def open_bank(coreset=True, bank_dir=None):
    """Open the coreset or the full memory bank, from bank_dir or the default path."""
    if coreset:
        return CoresetBank.open(bank_dir or CORESET_DIR)
    return MemoryBank.open(bank_dir or BANK_DIR)


class RADModel(nn.Module):
    """RAD anomaly detector: images -> anomaly scores, higher = more anomalous.

    coreset=True scores against the 1% coreset (~30 ms per image),
    coreset=False against the full memory bank (~1 s per image).
    Hyperparameters are the defaults of scoring.py. The backbone is loaded on
    cuda (fp16) when available; .to() moves the model like any nn.Module.
    """

    def __init__(self, coreset=True):
        super().__init__()
        bank = open_bank(coreset)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.extractor = FeatureExtractor(
            model_id=bank.meta["model_id"], layers=bank.meta["layers"], device=device
        )
        self.scorer = RADScorer(bank).to(device)

    @torch.inference_mode()
    def forward(self, images):
        """images: uint8 [B, 3, H, W], as from CaxtonDataset. Returns scores [B]."""
        cls, patches = self.extractor.extract(images)
        return torch.stack(
            [
                self.scorer.score(
                    cls[i], {layer: p[i] for layer, p in patches.items()}
                )[0]
                for i in range(len(images))
            ]
        )

    def score(self, images):
        """Same as model(images); used by evaluation.evaluate()."""
        return self(images)
