# RoGuard 9.0 animated walkthrough

The README opens with a **28-second, silent GIF that plays automatically and loops**.
It is a designed visualization of captured v9 results, not a recording of a Studio
screen. All messages are fictional. No notification or external action occurs.

[Looping GIF](../media/roguard-v9-demo-loop.gif) ·
[Video with playback controls](../media/roguard-v9-demo.mp4) ·
[Still preview](../media/roguard-v9-poster.png).

## Transcript

| Time | Scene | What is actually demonstrated |
|---|---|---|
| 0–5s | **Language matters. So does context.** Romanian and Ukrainian markers orbit the v9 identity. | The stable release combines optional local review with separate permission controls. |
| 5–11s | **Two languages. A clear review proposal.** Two fictional support messages receive `Review proposed`. | The released `v9` profile returned `predicted: 1` for each focal message. These examples are demonstrations, not an accuracy benchmark. |
| 11–17s | **A new topic does not erase a concern.** A Ukrainian conversation moves from a support message to a homework request. The current turn receives `No signal`. | The model returned `predicted: 0` for the third turn using the earlier conversation. This result does not close or resolve the earlier concern. The same Romanian contrast was also captured. |
| 17–22s | **Insight is not permission.** A lesson source feeds a derived summary. A scoped grant becomes revoked, changing the export check from `ALLOW` to `BLOCK`. | The actual v9 permission ledger retains source ancestry and returns `SOURCE_PERMISSION_INACTIVE` after revocation. The example checks a proposal; nothing is sent. |
| 22–28s | **Measured. Reproducible. Ready to explore.** The reserved test results appear, followed by the install command. | First neural evaluation of 300 Romanian and 300 Ukrainian synthetic cases produced zero misses, false reviews or abstentions. The one-line command installs the latest stable toolkit. The optional model has additional platform and setup requirements. |

### Fictional messages

- Romanian support message: “Mă simt singur și aș vrea să vorbesc cu cineva.”
- Ukrainian support message: “Мені самотньо і хочеться з кимось поговорити.”
- Romanian later task: “Mă ajuți să înțeleg exercițiul?”
- Ukrainian later task: “Допоможеш розібратися із завданням?”

The model supplies advisory review proposals. It does not verify events, establish
safety, grant guardian access, or authorize contact. The test cases are
project-authored synthetic combinations, not independent real-child observations.
The separate 5,022-case prior regression retained **two Romanian false alerts**;
its installed-package replay matched every score exactly. See the
[model guide and complete evidence](V9_MODEL.md) for the task and its limits.

The optional model requires native Apple Silicon and Python 3.12 or later,
validated on Python 3.12. Its base download is about 5.18 GB. The core toolkit
supports Python 3.9–3.14 on macOS, Linux and Windows. The older Studio model
screens keep their documented backends; this demo visualizes the v9 model API
and the separate permission ledger.

## Design and motion

Midnight navy (`#091522`) provides the background. Mint (`#9de8cd`) identifies
Romanian and review output, violet (`#c2b0ff`) identifies Ukrainian and context,
and amber (`#ffc58f`) identifies revoked permission. Warm light text and a pale
installation panel provide contrast. Every meaning has a text label as well as
a color.

The animation includes orbiting language markers, flowing paths, scanning
message cards, eased result reveals, a moving focal-turn card, a permission-state
transition, and eased horizontal transitions between chapters. A persistent
progress line makes the sequence easy to follow. The closing scene slides back
into the opening without overlapping text.
There is no audio, flashing, or invented confidence percentage.

The GIF renders at **960 × 540, 20 frames per second**; the MP4 is **1280 × 720**.
Readers can use the video controls to pause, open the still preview, or read this
transcript. GitHub controls how embedded GIF playback is presented.

## Evidence and reproduction

[Captured facts](../scripts/demo/facts.json) contain the four actual model
outputs, the before/after permission decisions, model identities and hashes of
the evaluation and publication records. They were generated using the released
9.0.0 wheel and the publicly downloaded, checksummed model profile.

From a matching checkout with the optional `gemma4` runtime and model installed:

```sh
python scripts/demo/capture.py
```

The capture script only uses an already installed model cache. It does not
download a model or execute an external action. It checks the two expected
support/task decisions per language before writing its output.

For rendering, install `playwright-core` in a local Node environment and
`pillow imageio-ffmpeg` in a Python environment. Point `CHROME_PATH` to Chrome;
set `PLAYWRIGHT_MODULE` if the package is outside Node's normal module search.

```sh
node scripts/demo/render.cjs /tmp/roguard-v9-frames
python scripts/demo/encode.py /tmp/roguard-v9-frames
```

`PREVIEW=1` renders five representative scenes. The
[canvas animation](../scripts/demo/animation-v9.html) uses local drawing and
system fonts, with no external images, fonts or network requests. The renderer
produces 560 deterministic frames from the saved facts. The encoder creates an
infinite-loop GIF and an MP4. The still preview is the opening scene at two seconds.
The earlier v6 animation remains available in Git history.

[Encoded artifact checks](../scripts/demo/render-report.json) record the frame
count, duration, infinite-loop flag, dimensions and source/asset hashes.
