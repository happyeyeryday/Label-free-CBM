import torch
from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor


class HierarchicalResNet(torch.nn.Module):
    """Frozen ResNet50 with spatial-preserving shallow features."""

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
        self.shallow_pool = torch.nn.AdaptiveAvgPool2d((4, 4))

    def forward(self, x):
        feats = self.extractor(x)

        # Preserve spatial distribution for shallow layers.
        l1 = torch.flatten(self.shallow_pool(feats["l1"]), start_dim=1)
        l2 = torch.flatten(self.shallow_pool(feats["l2"]), start_dim=1)

        # Deep layers remain globally pooled.
        l3 = feats["l3"].mean(dim=(2, 3))
        l4 = feats["l4"].mean(dim=(2, 3))
        return {"l1": l1, "l2": l2, "l3": l3, "l4": l4}
