# RoGuard

RoGuard is an offline Romanian and Ukrainian child-safety research toolkit. Version 4 adds a local conversation workspace, optional semantic classification, explicit uncertainty, draft-response checks and a separate audit of declared routing, permission and human-review requirements. Reports reference relevant turns without copying message text. The software never contacts a guardian or takes an external action.

## Install and open

```sh
pip install --upgrade roguard-child-safety
roguard-studio --conversation --open
```

The basic package supports Python 3.9–3.14 and includes a small fast model. The stronger semantic backend uses optional dependencies and a one-time, explicit model download:

```sh
pip install --upgrade 'roguard-child-safety[semantic]'
roguard-model fetch
roguard-studio --conversation --open --backend semantic
```

Use Python 3.10+ for the optional backend; the semantic test environment is Python 3.12. After downloading the pinned base, inference is local and requires no network. The Python import is `roguard`. The `roguard-conversation` command supports JSON and batches; `roguard-compare-reports` compares matching case IDs.

The semantic model reached 96.7% Romanian and 96.2% Ukrainian balanced accuracy on a 200-conversation synthetic test. This test contains only 20 phrase families per language and has correlated rows. The fast model is weaker. Neither result establishes accuracy on authentic child language; a negative signal cannot close a concern. Scores are not safety probabilities, response checks are limited phrase rules, and complexity measures do not prove comprehension. Prior specialist taxonomy review does not validate the new models or message banks.

See the [workspace guide](https://github.com/xyz-rafael-xyz/roguard-child-safety/blob/main/docs/CONVERSATION_WORKSPACE.md), [full results](https://github.com/xyz-rafael-xyz/roguard-child-safety/blob/main/docs/V4_RELEASE.md), and [README](https://github.com/xyz-rafael-xyz/roguard-child-safety/blob/main/README.md). Report defects through [GitHub Issues](https://github.com/xyz-rafael-xyz/roguard-child-safety/issues).
