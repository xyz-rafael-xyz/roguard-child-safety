# RoGuard Local v1.1.1 release scope

This patch expands the corrected-v1 sensitivity audit. It now tests whether changing the proposed recipient's declared role or permission in the matching policy scope changes R1, and whether changing an unpassed support criterion's applicability changes S1. It also continues testing D1 evidence, recipient choice and concern, and applicable S1 pass states. Reports use indexed scope paths and state codes so custom recipient and item values stay out of the sensitivity output. The guided Studio's result panel and role options now follow the selected Romanian or Ukrainian language; historical tabs retain their English controls.

The Studio and CLI still check only caller-declared symbolic facts. No model infers facts from a child's words, and no external action occurs. The corrected taxonomy has owner-attested specialist review at exact digests, but authentic-language model accuracy remains unestablished. The public wheel contains no trained adapter weights.

Install the GitHub release wheel with Python 3.9–3.14 and run `roguard-studio --open`. Use `roguard-v1-explore examples/v1_candidate_ro.json` or its Ukrainian counterpart for the same audit from a terminal. See [Decision Studio](DECISION_STUDIO.md) for the bounded exploration rules and [v1 evidence](V1_EVIDENCE.md) for validation limits.
