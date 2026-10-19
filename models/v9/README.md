# Optional v9 model distribution

Candidate branches include language-specific Romanian and Ukrainian encoders,
a multilingual entailment encoder, and a composite with a local Gemma
interpreter. Model qualification is tracked in
[V9 stabilization](../../docs/V9_STABILIZATION.md). No stable model profile has
been published yet.

The source checkout now includes:

- `roguard-fetch fetch`: explicitly download a published profile, verify each
  file's size and SHA256, resume completed files and reuse shared model files.
- `roguard-fetch locate`: verify an installed profile without network access.
- `roguard-review conversation.json`: review a declared-language conversation
  using the cached profile, with no implicit model download.
- `roguard-review --language uk --text 'Вітаю.'`: review one local message.

These commands require a published profile, which is withheld until the remaining
candidate checks pass. The current combined runtime was tested on Apple Silicon
with Python 3.12, Torch 2.14, Transformers 5.17 and MLX. It is an optional runtime;
the lightweight package continues to support Python 3.9 and later.

The entailment branch uses the `semantic` dependency extra and has been tested
on CPU and Metal with Python 3.12, Torch 2.14 and Transformers 4.57.6. It reports
separate support and revision-review proposals. These runtime checks establish
implementation parity; the model must also pass the accuracy gates before a
download profile can be published.

A review proposal does not establish that a child is safe or authorize an
external action or notification. Scores are model decision scores, not clinical
probabilities.

## Model terms

Read [NOTICE.txt](NOTICE.txt). The Python source license does not replace the
upstream model licenses. The Gemma components and derivatives are subject to
[Google's Gemma terms](https://ai.google.dev/gemma/terms) and the incorporated
[use policy](https://ai.google.dev/gemma/prohibited_use_policy); copies are included
in [licenses](licenses). The other encoder terms and their source records are
also included there.
The multilingual entailment model's pinned upstream MIT license is preserved
unchanged in [nli-UPSTREAM-LICENSE.txt](licenses/nli-UPSTREAM-LICENSE.txt), with
its source URL and checksum in [sources.json](licenses/sources.json).

## Preparing downloadable assets

Maintainers can use `scripts/prepare_portable_model_download.py` to stage a
completed native or entailment candidate in a new local directory. It checks
the selected artifact, checkpoint and tokenizer identities, copies the required
upstream license records, and writes a download manifest and content-addressed
GitHub release assets. The original pretrained weights are omitted because the
runtime uses the selected fine-tuned checkpoint.

The command performs no upload and does not change the packaged profile catalog.
A staged candidate must still pass the accuracy and installed-runtime checks
before publication. The
[private staging preflight](../../research/v9-stabilization/portable-distribution-preflight-20261019.json)
covers both portable model families; it is distribution evidence, not accuracy
evidence. An isolated wheel with private profiles was built and installed to
verify manifest inclusion, offline cache checks, bundle configuration and the
portable dependency guidance. It contains no weight files. Run the script with
`--help` for its required paths and release URL.
