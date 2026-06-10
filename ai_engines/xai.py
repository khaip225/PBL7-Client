"""Grad-CAM and attention visualization utilities."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


class GradCAMWrapper(nn.Module):
    """Wraps an encoder + classifier head into a single forward-pass model for Grad-CAM."""

    def __init__(self, encoder, classifier_head):
        super().__init__()
        self.encoder = encoder
        self.features = encoder.features
        self.classifier_head = classifier_head

    def forward(self, x):
        emb = self.encoder(x)
        return self.classifier_head(emb)


class GradCAM:
    """Grad-CAM implementation for DenseNet121-based encoder."""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.feature_maps = None
        self.gradients = None
        self._fh = self.target_layer.register_forward_hook(
            lambda m, inp, out: setattr(self, 'feature_maps', out))
        self._bh = self.target_layer.register_full_backward_hook(
            lambda m, gin, gout: setattr(self, 'gradients', gout[0]))

    def remove(self):
        self._fh.remove()
        self._bh.remove()

    def generate(self, input_tensor, class_idx=0):
        import cv2
        self.model.zero_grad()
        output = self.model(input_tensor)
        if output.dim() == 1:
            output = output.unsqueeze(0)
        output[0, class_idx].backward()
        pooled = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        cam = (pooled * self.feature_maps).sum(dim=1).squeeze(0)
        cam = F.relu(cam).detach().cpu().numpy()
        if cam.max() > 0:
            cam /= cam.max()
        return cam

    def overlay(self, pil_image, heatmap, alpha=0.4):
        import cv2
        img = np.array(pil_image.convert("RGB"))
        h, w = img.shape[:2]
        hm = cv2.resize(heatmap, (w, h))
        hm_color = cv2.applyColorMap(np.uint8(255 * hm), cv2.COLORMAP_JET)
        hm_color = cv2.cvtColor(hm_color, cv2.COLOR_BGR2RGB)
        return Image.fromarray((img * (1 - alpha) + hm_color * alpha).astype(np.uint8))
