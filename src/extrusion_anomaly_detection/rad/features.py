import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoImageProcessor, AutoModel

MODEL_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"
LAYERS = (4, 7, 10, 12)


class FeatureExtractor(nn.Module):
    """Frozen DINOv3 backbone. Loaded in fp16 on cuda, fp32 on cpu."""

    def __init__(self, model_id=MODEL_ID, layers=LAYERS, device="cuda"):
        super().__init__()
        device = torch.device(device)
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        self.layers = tuple(layers)

        # images arrive already resized to 512 by CaxtonDataset: the processor
        # only rescales and normalizes, it does not resize to its default 224
        self.processor = AutoImageProcessor.from_pretrained(model_id, do_resize=False)
        self.model = AutoModel.from_pretrained(model_id, dtype=dtype)
        self.model.to(device).eval()

        config = self.model.config
        self.patch_size = config.patch_size
        self.hidden_size = config.hidden_size
        # [CLS] + register tokens come before the patch tokens
        self.n_prefix = 1 + getattr(config, "num_register_tokens", 0)

    @torch.inference_mode()
    def extract(self, images):
        """images: uint8 [B, 3, H, W].

        Returns cls [B, D] and patches {layer: [B, H/p, W/p, D]}, all l2-normalized.
        """
        b, _, h, w = images.shape
        grid_h, grid_w = h // self.patch_size, w // self.patch_size

        weight = next(self.model.parameters())
        x = self.processor(images=images, return_tensors="pt").pixel_values
        x = x.to(weight.device, weight.dtype)

        outputs = self.model(pixel_values=x, output_hidden_states=True)

        # hidden_states[0] is the embedding output, hidden_states[l] is block l
        hidden_states = outputs.hidden_states

        cls = F.normalize(outputs.last_hidden_state[:, 0].float(), dim=-1)

        patches = {}
        for layer in self.layers:
            tokens = hidden_states[layer][:, self.n_prefix :]
            tokens = tokens.reshape(b, grid_h, grid_w, self.hidden_size)
            patches[layer] = F.normalize(tokens.float(), dim=-1)

        return cls, patches
