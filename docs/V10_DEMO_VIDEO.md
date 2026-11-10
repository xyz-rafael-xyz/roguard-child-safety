# RoGuard 10.0.0 animated walkthrough

The README opens with a silent **27-second looping GIF**. It is a designed explanation of the released v10 reviewer and recorded, project-authored synthetic results. It is not a screen recording or an independent human evaluation. No external action is performed.

[Looping GIF](../media/roguard-v10-demo-apple-loop.gif) · [Video with playback controls](../media/roguard-v10-demo-apple.mp4) · [Still poster](../media/roguard-v10-demo-apple-poster.png)

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

The palette is shared with the Notițe product, chosen to read as calm and trustworthy rather than alarming. Warm paper (`#F3F6F2`) and white alternate as chapter backgrounds, with a deep night (`#111A1C`) finale. Ink (`#1D2B2F`) carries text. Lake blue (`#1E6A86`) identifies the final-token path and Romanian, heather (`#5E4F95`) the focal-message path and Ukrainian, moss (`#3B7A52`) the retained topic signal and v10 results, ember (`#B5412A`) the v9 errors, and apricot (`#FCEBD8`) the no-review action. Meaning is also written in words, never encoded by color alone.

Each chapter carries one idea and one accent color. Elements enter on a critically damped spring (about 2% overshoot), fade in while sharpening from a slight blur, and arrive in a short stagger. Chapters leave by lifting, softening and fading while the background cross-fades to the next one. The Romanian case visibly flips from the v9 call to the v10 call, error bars grow to scale and count up, and the install command types itself. A segmented progress bar shows position in the sequence. There is no flashing or invented confidence percentage.

Type is Inter Display for headlines and Inter for text, rendered locally. The MP4 is **1280 × 720 at 50 frames per second**; the GIF is **960 × 540 at 25 frames per second**, every second frame of the same render; the poster is the frame at 2.8 seconds. To reproduce them, install `playwright-core` for Node, local Chrome or Chromium, the Inter font family, and `imageio-ffmpeg` for Python (`gifsicle` is optional and shrinks the GIF). Then run:

```sh
node scripts/demo/render-v10.cjs /tmp/roguard-v10-frames
python scripts/demo/encode-v10.py /tmp/roguard-v10-frames
```

Set `CHROME_PATH` for a nonstandard browser location and `PLAYWRIGHT_MODULE` if the Node package is outside its default resolution path. `PREVIEW=1` renders six representative frames. Without Inter installed the browser falls back to SF Pro or Helvetica, which shifts glyph widths slightly. The renderer uses a local canvas and the pinned facts file; it needs no model download or network request.
