# Model-authored probe quality gate

Three cached local generators were tried: Qwen3-4B, Phi-4-mini and Mistral-7B. Phi was retried with a lower temperature, higher token limit and repetition penalty. Raw fictional outputs and format-validation outcomes are retained in the JSON records. No real child messages were used.

**None of these generated sets is admitted as labelled training or benchmark ground truth.** JSON validity is insufficient. Inspection found repeated words, incomplete output, unnatural Romanian/Ukrainian, unrelated content, and failures to establish the requested conversational frame. For example, a requested other-person concern became an ordinary question, and a requested quotation became a translation question. The labels in these records are requested generation targets, not validated annotations.

These failed generation attempts demonstrate that using multiple model names does not establish independent, naturalistic or correctly labelled evaluation data. The released context training and evaluation instead use explicit authored fictional families, whose dependence on one author and repeated wrappers is disclosed. The new material has not been independently reviewed by native speakers or child-safety specialists. Older project reviews are not applied to this new task.

Files:

- `model-authored-probes.json`: Qwen3 and initial Phi attempts; temperature 0.7, 260-token limit, seed base 710.
- `phi-probes-retry.json`: Phi retry; generation settings recorded in the file.
- `mistral-probes.json`: Mistral attempt; generation settings recorded in the file.

Generation code: `training/probe_context_v7.py`. The current script implements the retry settings. Failed outputs are not silently relabelled as correct or used to inflate accuracy.
