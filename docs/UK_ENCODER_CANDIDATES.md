# Ukrainian encoder candidates: pinned tokenizer audit

**Status: preparation for a future Ukrainian model study, not a model selection.** The six-category Ukrainian taxonomy still awaits independent fluent-language and child-safety review. No Ukrainian classifier has been trained or validated. The [historical prompt-only probe](UK_MODEL_FAMILY_DIAGNOSTIC.md) failed all 24 complete contrasts, and its cards are consumed and unreviewed.

## Candidate rationale

| Candidate and exact revision | Why include it | Evidence gap |
|---|---|---|
| [Ukrainian RoBERTa](https://huggingface.co/youscan/ukr-roberta-base), `8149bde480a6df9014aa934c3f8af30858dea5a6` | Ukrainian-specific 125M-parameter encoder, Apache-2.0. Its author lists Ukrainian Wikipedia, OSCAR, and sampled social-network mentions as pretraining sources. It is the smaller language-specific challenger. | The card gives no RoGuard D1/S1 results. Its pretraining sources are not evidence of child-safety competence. |
| [Ukrainian ModernBERT](https://huggingface.co/KoichiYasuoka/modernbert-base-ukrainian), `a64c57e5a16e95e27b09227cd335bb234be12ea1` | Another Ukrainian-specific encoder, Apache-2.0; its card describes Ukrainian pretraining and links its source dataset. It tests whether architecture choice matters after the task is reviewable. | The card offers no D1/S1 result or comparison to the other candidates. The local fast tokenizer required `sentencepiece` and `protobuf` in addition to the current training extra. |
| [mmBERT-base](https://huggingface.co/jhu-clsp/mmBERT-base), `c5955035435e2bf121cde7f3c8863ef52ff35d82` | Multilingual 307M-parameter encoder, MIT, and the Romanian experimental baseline family already used by RoGuard. It is the cross-language comparator. | Romanian synthetic-card behavior does not establish Ukrainian behavior. The author-reported multilingual results are not RoGuard results. |

The license labels and architecture/training descriptions above come from the model authors' cards, not from our measurements. Publishing any derived weights remains a separate decision under the master plan.

## Reproducible tokenizer-only check

The [audit program](../eval/audit_uk_tokenizers.py) reads the 48 **historical, consumed, unreviewed** abstract cards in `uk-contrast-0001.jsonl`, checks their fixed SHA-256, and loads only pinned tokenizer/configuration files. It records the hashes of those files and aggregate token lengths in the [saved report](../eval/runs/uk-tokenizer-candidates.json); it prints no card text and downloads no model weights. In a separate research environment, install `.[train]`, `sentencepiece==0.2.2`, and `protobuf==6.33.6`, then run `python eval/audit_uk_tokenizers.py`. Use `--fetch` once if the pinned tokenizer files are absent from the Hugging Face cache. The report records the exact tested package versions; these two extra packages were needed for the Ukrainian ModernBERT tokenizer in the tested environment.

| Candidate | Median tokens, with special tokens | Maximum | Cards exceeding 256 | Unknown tokens |
|---|---:|---:|---:|---:|
| Ukrainian RoBERTa | 24 | 40 | 0/48 | 0 |
| Ukrainian ModernBERT | 26 | 43 | 0/48 | 0 |
| mmBERT-base | 37.5 | 63 | 0/48 | 0 |

This short-card probe shows that **all three tokenizers fit the proposed 256-token input cap on these cards**. A smaller token count does not imply better comprehension, accuracy, fairness, or response quality. The cards do not represent authentic child language, and an unknown-token count of zero is a tokenizer property, not a semantic test.

## Registered decision path for a real comparison

1. Resolve the Ukrainian taxonomy wording with two independent reviewers and freeze its exact bytes. Do not turn their signature into a claim of detection accuracy.
2. Have fluent Ukrainian authors write new, safe abstract cards directly in Ukrainian under the [independent intake protocol](INDEPENDENT_BATCH_INTAKE.md). Have a separate content reviewer inspect them; blind two annotators and adjudicate disagreements. Keep the existing consumed probe out of model selection.
3. Fit the Ukrainian RoBERTa and mmBERT candidates on the **same reviewed training split**, with matched labels, length limit, positive/negative balance, and recorded seeds. Keep ModernBERT as a third preregistered candidate only if its tokenizer and training recipe pass a local smoke run. Select epoch and cutoff on development cards only. Preserve every run, including a miss, and classify an infrastructure failure separately from a model-performance failure.
4. Freeze the selected model, threshold, code, and prediction format before opening a new independent test packet. Report exact negative-to-positive pairs by author, factor, and surface; D1/S1 recall, specificity, precision, false-review load, parse coverage, and uncertainty. Compare against the exact declared-contract checker and trivial review baselines. Run a genuinely new test after each substantive model revision; never repeatedly tune against a consumed test until it passes.

The project can claim a Ukrainian *abstract-card* result only after that sequence and independent review. It cannot claim accurate detection of authentic child disclosures or readability from this study.
