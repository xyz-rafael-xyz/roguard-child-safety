# RoGuard v6 visual walkthrough

The README opens with a **26-second, silent, continuously looping GIF**. Select it to open the higher-resolution MP4. This is a designed visualization of actual v6 outputs, not a recording of the Studio interface. Every message and artifact in it is fictional. No external action is performed.

## Transcript

| Time | What happens | What it demonstrates |
| --- | --- | --- |
| 0–4s | Romanian and Ukrainian inputs flow through RoGuard toward an application. | Model review and permission checks support the host application; the model has no authority to act. |
| 4–10s | Romanian: “Mă simt singur și aș vrea să vorbesc cu cineva.” Ukrainian: “Мені самотньо і хочеться з кимось поговорити.” Both reveal “Support signal → review.” | The released optional `adapted` backend returned `REVIEW_SUPPORT_SIGNAL` on both existing fictional example files. Neither result triggers automatic notification. |
| 10–16s | Lesson content moves into a revised tutoring export proposal. A private note is excluded. | `repair_demo('ro')` returns `REPAIR_PROPOSED`, selects `lesson`, removes `private_note`, and preserves the host-declared learning requirement. It does not verify semantic sufficiency or authorize execution. |
| 16–21s | A permission grant becomes revoked. The derived summary’s export check changes from `ALLOW` to `BLOCK`. | The ledger retains source ancestry. Revoking the scoped source permission produces `SOURCE_PERMISSION_INACTIVE` for its summary. An allow result is a policy check, not proof of caller authority. |
| 21–26s | Long conversations, encrypted permission state and metadata-only monitoring appear above the installation command. | These v6 capabilities have setup and scope requirements described in [Reliability](RELIABILITY.md). The optional adapted backend needs extra installation and model setup. |

The two successful review examples are **not an accuracy benchmark**. See [v6 experiments and limitations](V6_RELEASE.md) for the small synthetic evaluation and unresolved cases. Monitoring is metadata-only; that is not a claim that every API or application stores no text.

## Visual design

Midnight blue (`#0b1220`) provides the background; mint (`#7ee4c6`) marks permitted flow and review output; lavender (`#b8a5ff`) distinguishes Ukrainian content and the private artifact; amber (`#ffc484`) marks a proposal or revoked permission. Text labels carry every meaning independently of colour.

The animation uses eased entrances and exits, travelling flow particles, a moving lesson card, result reveals, a changing permission state and subtle continuous background motion. It renders at 20 frames per second. The GIF is 960 × 540; the MP4 is 1280 × 720. Both are silent. The MP4 offers playback controls for readers who prefer to pause the animation.

## Evidence and reproduction

[Captured outputs](../scripts/demo/facts.json) were generated from RoGuard 6.0.0 using [the capture script](../scripts/demo/capture.py), the pinned mmBERT base and the packaged v6 adapter. The language inputs come from `examples/conversation_ro.json` and `examples/conversation_uk.json`. All animation text is independently readable in this transcript.

From a checkout with the optional adapted dependencies and model installed:

```sh
PYTHONPATH=src python scripts/demo/capture.py --model-path /path/to/pinned-model
```

To render, install `playwright-core` in a local Node environment and `pillow imageio-ffmpeg` in a Python environment. Set `PLAYWRIGHT_MODULE` to the module path if it is outside Node’s normal module search, and `CHROME_PATH` to a Chrome executable. Then:

```sh
node scripts/demo/render.cjs /tmp/roguard-demo-frames
python scripts/demo/encode.py /tmp/roguard-demo-frames
```

`PREVIEW=1` renders five representative frames. [Animation source](../scripts/demo/animation.html) uses local canvas drawing without external fonts, images or network calls. Re-running model inference on another device can slightly change scores; the visual uses the captured decision labels without inventing confidence percentages.
