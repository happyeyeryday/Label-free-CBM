# Protocol: Hierarchical Concept Aptitude Validation

## 1. Objective
Validate the hypothesis that shallow CNN layers are more suitable for low-level concepts (color/texture) while deep layers are better for high-level semantics.

## 2. Technical Stack
- Framework: PyTorch 1.13
- Backbone: Frozen ResNet50 (torchvision.models)
- Supervision: CLIP (ViT-B/16) cosine similarity targets
- Dataset: CIFAR-10

## 3. Implementation Tasks
### Task 1: Backbone Feature Extractor
- Modify ResNet50 to extract features from 'layer1' and 'layer4'. 
- Use `torchvision.models.feature_extraction.create_feature_extractor` for clean implementation.
- Add Global Average Pooling (GAP) after each extracted feature.

### Task 2: Linear Probing
- Implement a simple Linear Probe (nn.Linear) that maps feature dimensions (256 for L1, 2048 for L4) to concept dimensions.
- Create two concept target matrices using CLIP:
    - `targets_low`: Based on `data/concept_sets/cifar10_layer1.txt`
    - `targets_high`: Based on `data/concept_sets/cifar10_layer4.txt`

### Task 3: Comparison Matrix
- Run 4 training sessions (independent probes):
    1. L1_Probe -> targets_low
    2. L4_Probe -> targets_low
    3. L1_Probe -> targets_high
    4. L4_Probe -> targets_high
- Metric: Final Cosine Similarity on the Test Set.

## 4. Expected Output
- A CSV file `validation_results.csv` containing final similarities for the 4 cases.
- A visualization script to plot these 4 bars.