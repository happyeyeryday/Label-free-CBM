# Agent Execution Strategy

## Role: Architect (Backbone & Logic)
- **Responsibility**: Implement the `HierarchicalResNet` wrapper.
- **Key File**: `models/hierarchical_resnet.py`
- **Instructions**: Ensure that the backbone is strictly frozen (`param.requires_grad = False`). Features must be pooled to [Batch, Channel] before the probe.

## Role: Data Engineer (CLIP & Concept alignment)
- **Responsibility**: Generate the target P matrices for CIFAR-10.
- **Key File**: `utils/concept_utils.py`
- **Instructions**: Use the existing `similarity.py` from the project to compute CLIP similarities. Ensure text features are normalized.

## Role: Experiment Runner (Training Loop)
- **Responsibility**: Create a lightweight training script for the probes.
- **Key File**: `run_validation.py`
- **Instructions**: 
    - Use Adam optimizer (lr=1e-3).
    - Loss function = 1 - CosineSimilarity.
    - Run for 50 epochs each to ensure convergence.
    - Save weights in `checkpoints/validation/`.

## Role: Visualizer
- **Responsibility**: Plotting the results.
- **Key File**: `tools/plot_validation.py`
- **Instructions**: Create a 2x2 grid or a grouped bar chart showing the performance gap.