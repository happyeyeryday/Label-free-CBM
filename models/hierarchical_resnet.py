import torch
from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor


class HierarchicalResNet(torch.nn.Module):
    """Frozen ResNet50 that exposes GAP pooled layer1-layer4 features."""

    def __init__(self, device=None):
        super().__init__()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        weights = models.ResNet50_Weights.IMAGENET1K_V1
        backbone = models.resnet50(weights=weights)
        backbone.eval()

        for param in backbone.parameters():
            param.requires_grad = False

        self.extractor = create_feature_extractor(
            backbone,
            return_nodes={"layer1": "l1", "layer2": "l2", "layer3": "l3", "layer4": "l4"},
        ).to(self.device)
        self.extractor.eval()

    def forward(self, x):
        feats = self.extractor(x)
        l1 = feats["l1"].mean(dim=(2, 3))
        l2 = feats["l2"].mean(dim=(2, 3))
        l3 = feats["l3"].mean(dim=(2, 3))
        l4 = feats["l4"].mean(dim=(2, 3))
        return {"l1": l1, "l2": l2, "l3": l3, "l4": l4}
