# RoGuard 10.0.0 animated walkthrough

The README opens with a silent **27-second looping GIF**. It is a designed explanation of the released v10 reviewer and recorded, project-authored synthetic results. It is not a screen recording or an independent human evaluation. No external action is performed.

[Looping GIF](../media/roguard-v10-demo-loop.gif) · [Video with playback controls](../media/roguard-v10-demo.mp4) · [Still poster](../media/roguard-v10-poster.png)

## Transcript and factual basis

| Time | Scene | Factual basis |
| --- | --- | --- |
| 0–4s | A clearer review signal. | V10 reuses the pinned v9 Gemma 4 body and adds a joint feature reviewer. |
| 4–9s | One backbone, two views. | The runtime extracts the classifier's final-token vector and a focal minor-message token mean, then applies four packaged RBF heads: binary review and concern topic in Romanian and Ukrainian. |
| 9–14s | Context changes the call. | The Romanian newspaper-statistics case changed from v9 review to v10 no review. The Ukrainian film-plus-personal-disclosure case remains a review. Both are recorded in challenge O's public-wheel replay. |
| 14–18s | A topic is not an action. | The Ukrainian denied-confinement case has `concern=1` and `binary=0` under the pinned policy. A negative result does not close an earlier concern. |
| 18–23s | Measured comparison. | On the 120-case policy-corrected authored panel, v9 made 24 combined binary and concern output errors and v10 made zero. On the separate 40-case authored challenge O, v9 made nine and v10 made zero. The exact PyPI wheel replayed both populations. |
| 23–27s | Public installation. | Version 10.0.0 is on PyPI. Native model review is validated on Apple Silicon with Python 3.12; the pinned v9 base model is fetched separately. |

The numbers count **output errors**, so one case may contribute both a binary and concern error. The 120-case panel initially mislabeled 20 hypothetical personal concerns; the [correction manifest](../research/v10-evidence/fresh-self-challenges-20261110/gemma4-final-mixed-panel20b-policy-correction-20261110.json) preserves that correction under the preexisting policy. The frozen [challenge history](../research/v10-evidence/fresh-self-challenges-20261110/README.md) also retains fourteen earlier v10 candidates that failed first-pass authored checks. These finite synthetic results do not establish perfect accuracy on real children's messages. Independent case-level human validation remains unavailable.

The [animation facts](../scripts/demo/v10-facts.json) are bound to the [public-wheel verification record](../research/v10-evidence/stable-release-20261110/publication.json), the [public 40-case result](../research/v10-evidence/stable-release-20261110/public-wheel-challenge-o-result.json), and the [public 120-case result](../research/v10-evidence/stable-release-20261110/public-wheel-panel20b-result.json). The wheel SHA-256, source hashes and result hashes are in those files.

## Visual design and reproduction

Midnight navy (`#071321`) anchors the interface. Electric cyan (`#43DDEB`) identifies the final-token path, violet (`#AA88FF`) the focal-message path, coral (`#FF674B`) draws attention to the v9/v10 comparison, and lime (`#C8F36A`) marks the retained topic signal. Cream and cobalt create distinct chapter backgrounds. Meaning is also written in words, never encoded by color alone. Motion is limited to easing, paths, feature pulses, and chart reveals; there is no flashing or invented confidence percentage.

The GIF is **960 × 540 at 20 frames per second**, the MP4 is **1280 × 720**, and the poster is a full-resolution frame. To reproduce them, install `playwright-core` for Node, local Chrome or Chromium, and `imageio-ffmpeg` for Python. Then run:

```sh
node scripts/demo/render-v10.cjs /tmp/roguard-v10-frames
python scripts/demo/encode-v10.py /tmp/roguard-v10-frames
```

Set `CHROME_PATH` for a nonstandard browser location and `PLAYWRIGHT_MODULE` if the Node package is outside its default resolution path. `PREVIEW=1` renders six representative frames. The renderer uses a local canvas and the pinned facts file; it needs no model download or network request.
