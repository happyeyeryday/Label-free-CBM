# Hierarchical Validation Postmortem

## Bottom Line

Earlier suspicious results were not caused by a single bug. They came from a stack of issues:

1. The original metric could be inflated by a strong shared mean direction in CLIP target vectors.
2. The early concept split mixed shallow appearance concepts with deep semantic concepts.
3. The shallow backbone features were over-compressed by global average pooling.
4. The shallow probes had too little capacity, so predicting a near-constant mean concept vector was an easy local optimum.
5. Some automatically generated concept partitions were badly imbalanced, which made comparisons misleading.

The result was a model that looked "good" by raw cosine, while actually failing to produce image-specific concept predictions.

## What Was Actually Wrong Before

### 1. Raw Cosine Was Too Forgiving

The first suspicious scores near `0.99` were not realistic for a frozen backbone with a light probe on this task.

Why this happened:

1. CLIP text concept embeddings are not uniformly spread out.
2. Many concepts share a strong common semantic direction.
3. If the probe outputs a vector close to the average target direction, raw cosine can still look very high.

That means:

1. High raw cosine did not prove sample-specific alignment.
2. A constant or near-constant prediction could score well.

This is why the centered cosine audit was the right correction. Once the common mean direction was removed, the unrealistic scores disappeared.

### 2. Variance Collapse Was Real

The strongest signal in the audit was not the absolute cosine score. It was the very low prediction variance.

When `trained_pred_var_mean` is around `0.000x`, the probe is effectively doing this:

1. Ignore image-specific differences.
2. Output almost the same normalized concept vector for every image.
3. Optimize the average direction instead of recognizing features.

This is the classic "mean vector" failure mode. It is consistent with:

1. Low-capacity probe heads.
2. Targets with a strong shared mean direction.
3. Shallow features that have been spatially over-compressed.

### 3. The Original Low/High Concept Split Was Not Pure

The earlier "low-level" concept set was not truly low-level.

Examples of concepts that should not be treated as shallow:

1. object names
2. role names
3. contextual items
4. environment or activity words

If these appear in the shallow target set, deep layers will win by construction because they are better at object-level semantics.

This explains why deeper layers could outperform shallow layers even on the supposed "low-level" target set.

### 4. GAP Crippled the Shallow Layers

Using global average pooling on `layer1` and `layer2` was a structural mistake for this experiment.

Reason:

1. `layer1` and `layer2` mostly encode local textures, color patches, and small spatial contrasts.
2. Global average pooling removes where those local responses occur and collapses them into a single channel mean.
3. That destroys exactly the kind of micro-pattern information shallow layers are designed to represent.

For standard classification, GAP is often fine. For this experiment, it hides the aptitude we are trying to measure.

Switching `l1` and `l2` to `AdaptiveAvgPool2d((4, 4))` is directionally correct because it preserves coarse spatial layout.

### 5. The Probe Head Was Too Weak

A single linear layer is a harsh bottleneck for mapping:

1. frozen CNN features
2. to CLIP-aligned semantic target vectors
3. across multiple concept granularities

For shallow layers, this can easily fail into the lazy solution:

1. learn one dominant output direction
2. ignore fine feature differences

Adding an MLP to `l1/l2/l3` is reasonable. It increases capacity enough to learn nontrivial mappings without changing the frozen backbone.

### 6. Concept Count Imbalance Distorted the Comparison

Some generated splits produced extreme counts such as:

1. one layer with many augmented concepts
2. another layer with only a handful of concepts

This creates two problems:

1. target difficulty is inconsistent across layers
2. the metric becomes harder to compare directly

Even when the code is correct, a highly imbalanced concept inventory can make the apparent trend misleading.

## Evaluation of the Recent Architecture Changes

### Good Changes

These changes are technically justified:

1. Preserve spatial information in `l1/l2` with `4x4` pooling instead of GAP.
2. Use higher-capacity probes for `l1/l2/l3`.
3. Keep `l4` as a simple linear head.
4. L2-normalize probe outputs and targets before cosine loss.
5. Use centered cosine in the audit.
6. Use a scheduler and nonzero weight decay.

These changes address real failure modes.

### Overstated Claims

Some claims should be treated as hypotheses, not guarantees:

1. "L1 must beat L4 on low-level concepts" is not guaranteed on CIFAR-10.
2. "trained variance must exceed 0.1" is not a universal threshold.

Why:

1. CIFAR-10 is only `32x32`, so deep layers still see almost the whole image and can retain low-level information.
2. Variance magnitude depends on normalization, concept dimensionality, target geometry, and probe architecture.
3. Even after fixing the pipeline, deep layers may remain competitive on some shallow targets.

So the correct target is not a hard numeric threshold. The target is a cleaner aptitude pattern:

1. shallow layers improve on atomic/local concepts
2. deep layers dominate on global semantic concepts
3. each layer shows nontrivial image-dependent variance

## Why the Earlier Results Looked So Wrong

The prior behavior can be explained as the combination below:

1. Mixed concept semantics made the "low" task semantically rich.
2. Raw cosine rewarded the shared mean direction.
3. GAP removed shallow spatial cues.
4. Weak probes collapsed to a mean vector solution.
5. Small or unbalanced concept sets exaggerated instability.

This is why the results were both:

1. numerically high
2. scientifically unconvincing

Those two things are not contradictory. The metric was easy to game.

## What Still Needs Attention

Even after the architecture changes, the experiment can still fail for conceptual reasons.

### 1. The Current Manual `l1` File Is Still Not Purely Atomic

The current `l1` list still includes items like:

1. `a large body`
2. `a slender body`
3. `a short, stocky build`
4. `a flat front and back`

These are more like coarse shape or object-scale descriptors than true atomic visual primitives.

Better `l1` concepts are:

1. colors
2. micro-textures
3. local materials
4. local surface patterns

If `l1` includes global size or body-shape descriptions, deeper layers may still dominate.

### 2. The Current `l4` File Contains Many Context Objects

The current `l4` file includes many nouns like:

1. `a barn`
2. `a bed`
3. `a branch`
4. `a dock`

These are semantically global, but they also introduce dataset co-occurrence bias.

That means:

1. the target can become a context-retrieval task
2. deep layers may exploit class-context shortcuts

This is not necessarily wrong, but it is different from measuring pure object-level semantic abstraction.

### 3. Full 4x4 Evaluation Is More Important Than Single Endpoints

The real scientific check is not only `L1_to_L1` versus `L4_to_L1`.

It is:

1. does each feature layer prefer its own target level
2. does performance move progressively across levels
3. do shallow layers beat deep layers on local concepts more often after the spatial fix

A 4x4 matrix is the right test. A 2x2 endpoint comparison is too easy to misread.

## Recommended Next Checks

Use this checklist for every run:

1. Check `untrained` centered cosine. It should be near zero.
2. Check `trained - untrained`. It should be meaningfully positive.
3. Check `trained_pred_var_mean`. It should not be near zero for all runs.
4. Compare the 4x4 centered similarity matrix, not only raw cosine.
5. Compare concept counts across target levels.
6. Revisit concept purity before tuning the optimizer again.

## Practical Conclusion

Gemini was partly right about the architecture:

1. preserve shallow spatial structure
2. increase shallow probe capacity
3. avoid the lazy mean-vector solution

But the earlier failure was not mainly an optimizer problem.

The deeper cause was experimental design:

1. weak concept taxonomy
2. misleading metric before centering
3. inappropriate pooling for shallow layers

If the next run still fails, the first thing to inspect is not the learning rate. It is the concept files.
