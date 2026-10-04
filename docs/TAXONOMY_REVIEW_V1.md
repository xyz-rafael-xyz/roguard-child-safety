# V1 taxonomy review disposition

The project owner supplied a second review memo, `roguard_taxonomy_review.docx`, on 4 October 2026. Its SHA-256 is `944472d966140ec1335a65630158d0b83d0f0aa2211d953633dc812d4a9a52f5`. The memo attributes comments to a Romanian child-protection practitioner, a Ukrainian plain-language linguist, and a child online-safety policy and evaluation specialist. The reviewers' names are withheld; their identities, independence, and credentials have not been verified by this repository. The source memo stays outside Git because it is an owner-supplied review artifact.

The memo reviews the earlier [Romanian v0.2](../taxonomy/taxonomy_ro.md) and [Ukrainian draft v0.1](../taxonomy/taxonomy_uk.md). It reports Romanian D1 unaccepted, R1/A1/P1/S1 conditional, and G1 accepted. It reports Ukrainian D1 unaccepted and the other five categories conditional. Its stated blockers are RC1, RC4, RC5, and RC15. The memo explicitly asks for a corrected version and a new signoff on its exact bytes. Its comments are therefore **revision requests**, not v1 taxonomy approval or model validation.

The declared reviewer coverage is narrower than the v1 release gate: the Romanian practitioner explicitly does not claim a linguist role, the Ukrainian linguist does not claim child-protection expertise, and the policy specialist assessed code structure through translations rather than Romanian or Ukrainian wording. The corrected version still needs a Romanian language review and a Ukrainian child-protection review, as well as exact-byte acceptance of every category by reviewers qualified for their assigned disciplines.

## Implemented in the candidate

| Review items | Change |
|---|---|
| RC1–RC2, RC14 | D1 distinguishes a direct safety statement from an explicit support request, defines the contextual marker, and retains review priority after a retraction or denial. |
| RC3, RC19 | Romanian headings now distinguish evidence support from review priority; both languages define one priority scale. |
| RC4–RC5, RC20 | R1 uses a neutral adult role, an explicit `recipient_may_be_source_of_concern` value, and a required human review for `yes` or `unknown`. Both languages state the historical reference outcomes. |
| RC6, RC15 | S1 has conditional immediate-danger-route and no-pressing-for-details fields, plus an explicit insufficient-contract decision. No emergency number is frozen into the taxonomy. |
| RC7–RC13, RC21 | The requested Romanian and Ukrainian terminology edits were applied in the candidate texts, including the Ukrainian workflow owner phrasing. |
| RC16–RC18 | A1 states the local word-count convention; unavailable research reports are marked private with a public summary; the questioned UNICEF Ukraine citation was removed. |

The corrected [Romanian candidate](../taxonomy/taxonomy_ro_v1_candidate.md) and [Ukrainian candidate](../taxonomy/taxonomy_uk_v1_candidate.md) are separate from the historical taxonomies. The [candidate v1 contract checker](../src/roguard/v1_contract.py) implements the new D1, R1, and S1 fields on caller-declared symbolic cards. It is not a trained Romanian or Ukrainian text reader. The [input schema](../schema/v1-contract-input.schema.json) and [Romanian](../examples/v1_candidate_ro.json) and [Ukrainian](../examples/v1_candidate_uk.json) examples contain no child utterance.

As a cross-check on the design, [Ofsted's safeguarding guidance](https://www.gov.uk/government/publications/ofsted-safeguarding-policy/safeguarding-concerns-guidance-for-inspectors) advises staff against probing for more detail and routes concerns away from a senior leader implicated in an allegation. [NSPCC Learning](https://learning.nspcc.org.uk/child-abuse-and-neglect/recognising-and-responding-to-abuse) advises that concern need not wait for a direct disclosure. These are UK-context sources that motivate the abstract D1, R1, and S1 checks; they do not establish Romanian or Ukrainian legal duties or validate this classifier.

Run `roguard-v1-review --root .` to audit the candidate digests. Its two review records are pending. Each language needs a language review and a child-protection review that accepts **all six categories on the exact candidate digest**. The same person may contribute in one discipline to both languages only if their competence covers both; the machine gate requires two distinct reviewers per language and cannot establish their qualifications. Changes after signoff require another digest and another review. The specialist memo also calls for re-registering the D1/S1 studies after the contract and taxonomy changes. The historical model scores, frozen packets, and approvals stay attached to the earlier definitions and cannot validate this candidate.
