# Public source release

This repository is a source-only snapshot of RoGuard at private research commit `f1b2c0bacddb6d319b35b91da09381f5d8f34aa3`. It contains the Romanian and Ukrainian declared-contract checker, taxonomies, synthetic data and research records, examples, and documentation. It begins a new public Git history so earlier private commits cannot expose research artifacts through a clone.

The selected adapter weights and adapter configs are **not** in this repository. They remain in a separate private research archive. The public package does not download them. The public CLI checks caller-declared facts without a model; it is not a detector for real child messages. The model score files and metadata document prior synthetic experiments, including failures, but a reader of this snapshot cannot verify those scores against the withheld weights or independently verify the historical order of private research commits.

Public CI runs the package and contract tests on Python 3.10, 3.11, and 3.12, checks synthetic batch structure, installs a wheel outside the checkout, and scans the public Git tree and history for withheld artifact paths and credential-shaped text. Tests that require selected private weights explicitly report `skipped`. This is a narrower verification claim than the private research CI, which retains the full artifact and frozen-result checks.

The [release-gate record](RELEASE_GATES.md) describes the remaining language-validation and real-world accuracy work. Public availability does not change those gates.
