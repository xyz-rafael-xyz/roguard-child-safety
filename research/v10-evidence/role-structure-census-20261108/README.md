This census checks whether the original 37,766 conversations can distinguish role-aligned prior-token pools from a chronological control with identical pool sizes. Only 936 conversations can differ by structural role order; 36,830 necessarily produce identical masks. Token exclusions may reduce the 936 further.

Every recovered input was matched to the audited feature metadata. Independent verification rechecked all cached token-list digests, every retained census row and all counts, plus abstract role/order and empty-span cases. No tokenizer or model ran. The reports retain the original TRAIN/DEV and source-group denominators.

The public `census.py` differs from the executed private source only in root configuration. Its function ASTs are identical; this exported variant was not rerun. To use it with the retained private input cache, set `ROGUARD_PRIVATE_ROOT`, `ROGUARD_PUBLIC_ROOT` and `ROGUARD_CENSUS_OUTPUT` to existing directories. Use a new empty output directory: files are created exclusively. The reports bind the executed source and input digests.

These are engineering diagnostics. They neither establish model improvement nor qualify a v10 release.
