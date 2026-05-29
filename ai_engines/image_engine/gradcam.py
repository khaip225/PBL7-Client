"""
Grad-CAM: hiển thị vùng ảnh X-quang mà model "chú ý" khi ra quyết định.
Dùng cho ResNet18 (binary) và sau này EfficientNet (multi-label).
"""

import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.feature_maps = None
        self.gradients = None
        self._hooks()

    def _hooks(self):
        def forward_hook(module, inp, out):
            self.feature_maps = out

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self._fh = self.target_layer.register_forward_hook(forward_hook)
        self._bh = self.target_layer.register_full_backward_hook(backward_hook)

    def remove(self):
        self._fh.remove()
        self._bh.remove()

    def generate(self, input_tensor, class_idx=0):
        """Sinh heatmap cho 1 class cụ thể.

        Args:
            input_tensor: (1, C, H, W) đã normalize, trên GPU/CPU
            class_idx: index của class cần visualize (với binary: luôn là 0)
        Returns:
            heatmap: numpy array (H', W') trong khoảng [0, 1]
        """
        self.model.zero_grad()

        output = self.model(input_tensor)
        if output.dim() == 1:
            output = output.unsqueeze(0)

        score = output[0, class_idx]
        score.backward()

        pooled = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        cam = torch.sum(pooled * self.feature_maps, dim=1).squeeze(0)
        cam = F.relu(cam).detach().cpu().numpy()

        if cam.max() > 0:
            cam = cam / cam.max()
        return cam

    def overlay(self, pil_image, heatmap, alpha=0.4):
        """Chồng heatmap lên ảnh gốc.

        Args:
            pil_image: ảnh X-quang gốc (PIL Image)
            heatmap: numpy array (H', W') [0, 1]
            alpha: độ trong suốt của heatmap (0.4 = 40%)
        Returns:
            PIL Image đã chồng heatmap
        """
        img = np.array(pil_image.convert("RGB"))
        h, w = img.shape[:2]

        heatmap_resized = cv2.resize(heatmap, (w, h))
        heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

        overlay = (img * (1 - alpha) + heatmap_color * alpha).astype(np.uint8)
        return Image.fromarray(overlay)
