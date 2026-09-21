from typing import List, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import AutoTokenizer, CLIPModel


class CLIPSemanticSimilarity(nn.Module):
    """Frozen CLIP feature extractor used for alignment loss and CLIPScore."""

    image_mean = (0.48145466, 0.4578275, 0.40821073)
    image_std = (0.26862954, 0.26130258, 0.27577711)

    def __init__(self, model_name_or_path: str, image_size: int = 224):
        super().__init__()
        self.model = CLIPModel.from_pretrained(model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.image_size = image_size
        self.model.requires_grad_(False)
        self.model.eval()
        self.register_buffer(
            "mean",
            torch.tensor(self.image_mean).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "std",
            torch.tensor(self.image_std).view(1, 3, 1, 1),
            persistent=False,
        )

    def train(self, mode: bool = True):
        super().train(False)
        self.model.eval()
        return self

    def _prepare_images(self, images: Tensor) -> Tensor:
        images = F.interpolate(
            images.float(),
            size=(self.image_size, self.image_size),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        ).clamp(0, 1)
        return (images - self.mean) / self.std

    def image_features(self, images: Tensor) -> Tensor:
        outputs = self.model.vision_model(
            pixel_values=self._prepare_images(images),
            return_dict=True,
        )
        pooled = outputs.pooler_output
        features = self.model.visual_projection(pooled)
        return F.normalize(features.float(), dim=-1)

    @torch.no_grad()
    def text_features(
        self,
        texts: Union[List[str], str],
        device: torch.device,
    ) -> Tensor:
        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        outputs = self.model.text_model(**tokens, return_dict=True)
        features = self.model.text_projection(outputs.pooler_output)
        return F.normalize(features.float(), dim=-1)

    def similarity(
        self,
        images: Tensor,
        texts: Union[List[str], str],
    ) -> Tensor:
        image_features = self.image_features(images)
        text_features = self.text_features(texts, images.device)
        return (image_features * text_features).sum(dim=-1)

    def image_similarity(self, images_a: Tensor, images_b: Tensor) -> Tensor:
        features_a = self.image_features(images_a)
        with torch.no_grad():
            features_b = self.image_features(images_b)
        return (features_a * features_b).sum(dim=-1)
