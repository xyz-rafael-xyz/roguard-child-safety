# Contributing

Please follow the [Code of Conduct](CODE_OF_CONDUCT.md) when participating in this project.

Use the [issue chooser](https://github.com/xyz-rafael-xyz/roguard-child-safety/issues/new/choose) for a bug, feature idea, taxonomy/language change, evaluation finding, or usage question. Security vulnerabilities go through the [private route](SECURITY.md). Pull requests begin with a template that asks for the change, relevant verification, and evidence boundaries.

Use only original, abstract, synthetic templates. Do not submit real or realistic disclosure or grooming text, even with names removed. Propose taxonomy changes for language and child-safety review. Generated batches must come from a registered, reproducible generator and pass `training/validate_generated.py` before training or evaluation; historical human approvals remain valid. Keep held-out batches sealed before model tuning and never put a diagnostic or test split into training. Report model revision, exact data and prompt provenance, strict parsing, per-category errors, false review load, and pair consistency. Do not claim live accuracy from symbolic cards.
