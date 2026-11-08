# Larger generated-bank reference measurements

Extraction120 completed all 3,080 unique inputs (1,540 per language) with the
original v9 encoder and heads in 2,443 seconds. Audit121 independently retokenized
every input and replayed the original heads on CPU. Seven-head argmax and fixed
binary decisions matched. The native encoder was not independently replayed.

Baseline122 measured v9 and the unchanged study75 heads on those features.
The independent audit recounted all 6,160 predictions and 80 panel responses,
then replayed all 3,080 study75 predictions using MLX CPU against the producer's
Torch CPU results. No class or decision changed. This does not establish new
native encoder/GPU parity for study75.

On the fixed operator-plus-both transfer population (768 rows per language),
v9 has 343 Romanian and 274 Ukrainian assertion errors; study75 has 357 and 296.
These are reference errors, not a new model's improvement. They establish
headroom for a separately registered training comparison. No fitting or
checkpoint selection occurred in these studies, and the reserved TEST stayed
closed. These generated labels are engineering proposals, not independent
human gold. The baseline probes are now exposed engineering diagnostics.

The 40-row panel contains 32 aliases of bank rows and eight unique controls;
aliases are not added to unique-input metrics. Both languages contain 288 FIT
rows, but this reference work itself does not admit training.

Reports retain all numeric values and artifact digests with local paths
redacted. The three executable sources are byte-identical to the registered
versions. Redacted registration summaries are not executable. Actual arrays,
private paths and model weights are not included here. Nothing in these
studies qualifies a v10 release; package metadata remains 9.0.0.
