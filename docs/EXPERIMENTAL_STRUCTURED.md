# Experimental structured predictions

`roguard.experimental_structured` is an explicit research import. It does not change the default model, package version, or model catalog. No qualified residual artifact or calibration artifact is supplied.

`StructuredWrapper` takes one reviewed component through `components={language: component}` and an explicit `active_language`. Its verified closure still binds both languages' selected manifests, specifications, parent assets, selector, and installed source inventory. Romanian and Ukrainian instances of the same frozen wrapper and closure therefore share a pipeline identity. The caller must run language workers sequentially and wait for one process to exit before loading the next backbone. The wrapper does not load models or enforce a global backbone count. A request for an inactive language raises a configuration error; a mixed-language `predict_many` batch is rejected before any prediction.

The default `direct-first` mode preserves supported native inputs, including the original-v9 route above the residual envelope. Memory selection is reserved for actual input-support overflow. The explicit `uniform` mode always applies the frozen memory selector. These modes have distinct pipeline identities and cannot share calibration. Runtime failures propagate; they do not trigger a confidence or error fallback.

The version 2 closure contains only runtime assets: the pinned base and tokenizer files, selector, both languages’ study 75 and original-v9 weights and specifications, selected residual manifests and tensors, and installed Python and native dependency inventories. Research checkpoints, training plans, feature manifests, and selection proofs are not runtime file dependencies. Their four digests remain declared in each residual manifest and therefore remain part of the pipeline identity. The outer research coordinator and language workers must verify those actual research files and their transitive links before and after an evaluation. The runtime receipt explicitly does not establish that research provenance. Asset paths may resolve through Hugging Face snapshot links to regular blobs; sizes and digests are verified, while host paths are excluded from the portable identity. This closure revision changes the pipeline identity and cannot reuse calibration from the earlier closure.

Full file and asset verification runs at batch entry and exit. `predict_many` returns only after successful closing verification. Results obtained inside an explicit `batch()` remain pending until the batch closes. Verification does not claim that disk files were checked after every row or that arbitrary in-memory objects are cryptographically attested.

The experimental result contains the native score and decision, a distribution over the 432 ontology interpretations, original Unicode evidence offsets, and `field_logits`. The latter is the immutable same-call tuple of 19 float32-representable values from the final inference, or `None` when unsupported. If memory selection produces a supported inference, its logits are retained rather than those of the unsupported direct attempt. Evidence describes retained source units; it is not a guarantee of semantic support.

Historical six-field argmax errors use the following raw-logit slices. They must not be replaced by the most probable coupled interpretation, which can differ even when every raw head has a unique interpretation or a deterministic tie rule.

| Field | Slice |
| --- | --- |
| Subject | `0:3` |
| Time | `3:6` |
| Assertion | `6:10` |
| Source | `10:13` |
| Relation | `13:15` |
| Concern | `15:17` |

The final two logits retain the legacy binary head. Residual-route interpretation probabilities use the frozen study 75 couplings. Original-v9 Romanian fallback uses independent semantic fields and its policy score; Ukrainian fallback's legacy binary score is separate from the semantic interpretation mass. These logits do not supply the separate five-class response classifier.

An existing, correctly bound calibration artifact may produce interpretation sets. This module does not fit calibration. Unsupported inputs receive the full ontology set. The module and its synthetic tests do not establish real-world coverage, independent human validation, native candidate parity, or release qualification.
