# RoGuard Local video demo transcript

The [46-second captioned demo](../media/roguard-demo.mp4) uses fictional symbolic inputs and the local RoGuard Decision Studio. The [animated README version](../media/roguard-demo-loop.gif) plays automatically and loops without controls. It contains no child messages. The Studio images were captured from the running application; the corrected-v1 CLI rows were formatted from the command's JSON output.

| Time | What appears | What it demonstrates |
|---|---|---|
| 0–4 s | RoGuard Local title | Offline Romanian and Ukrainian declared-contract checks; no external action. |
| 4–12 s | Romanian Studio **Check** | Routing, permission, and review-gate findings from a caller-supplied symbolic contract. The proposed use is `ready_for_human_decision`, which is not approval to act. |
| 12–20 s | Studio **Compare** | Only the proposed recipient changes from `minor` to `guardian`; the declared routing scope does not allow that recipient, so R1 and the proposed-use result change to review. |
| 20–27 s | Studio **Explore changes** | A bounded one-fact sweep tests 10 symbolic variants and reports six decision changes in this example. |
| 27–34 s | Ukrainian Studio **Check** | The same local contract workflow with a Ukrainian-language label and symbolic facts. This does not test Ukrainian child-language understanding. |
| 34–41 s | Corrected-v1 Romanian CLI | `roguard-v1-contract examples/v1_candidate_ro.json` returns D1, R1, and S1 reason codes. The output marks caller facts unverified and external action false. |
| 41–46 s | Limits | RoGuard helps a human inspect declared rules. It has not been validated to detect risk in real child messages. |

To reproduce the Studio scenes from a source checkout, install the package, run `roguard-studio --open`, load the Romanian sample, and select **Check**, **Compare**, and **Explore changes** in turn. Load the Ukrainian sample for its contract check. For the final CLI scene, run `roguard-v1-contract examples/v1_candidate_ro.json` from the checkout. The Studio uses the historical six-category contract rules; the corrected v1 D1/R1/S1 checks are a separate CLI path.
