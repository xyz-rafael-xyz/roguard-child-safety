# RoGuard Child Safety

RoGuard is an offline Romanian and Ukrainian child-safety **contract and evaluation toolkit**. Its deterministic checks examine caller-declared facts about disclosure signals, recipient scope, permission, readability limits, support-response fields, and human-review gates. It also includes a small experimental text model that marks possible support signals for human review.

The text model was trained and tested only on project-authored fictional messages. Its synthetic test results do **not** establish accuracy on real children's messages, native-language validity, or safety for autonomous screening. A negative model signal must not close a concern, and a positive signal does not authorize contacting a guardian or anyone else. The software takes no external action.

## Install

```sh
python3 -m pip install --upgrade roguard-child-safety
```

Python 3.9 and newer are supported. Installation downloads the package; use after installation is local and requires no network connection. The import package is `roguard`. The main commands are `roguard` for declared contracts, `roguard-message-screen` for the experimental Romanian/Ukrainian advisory, and `roguard-studio --open` for the local Decision Studio.

See the [README](https://github.com/xyz-rafael-xyz/roguard-child-safety/blob/main/README.md) for examples and the [v3 research report](https://github.com/xyz-rafael-xyz/roguard-child-safety/blob/main/docs/V3_MESSAGE_RELEASE.md) for frozen results and limitations. Report defects through [GitHub Issues](https://github.com/xyz-rafael-xyz/roguard-child-safety/issues).
