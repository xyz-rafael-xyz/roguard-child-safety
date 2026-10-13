# PyPI distribution and release process

The PyPI distribution is `roguard-child-safety`; the import package and commands remain `roguard` and `roguard-*`. Version 5.0.0 adds the interaction laboratory and permission-aware execution SDK; see [the v5 report](V5_RELEASE.md). The original PyPI setup was introduced in version 3.0.2.

The release workflow builds a source archive and wheel from a matching `vX.Y.Z` tag, validates their metadata and a clean installation, publishes them to PyPI with [Trusted Publishing](https://docs.pypi.org/trusted-publishers/using-a-publisher/), and only then creates the GitHub release with both versioned and stable-name assets. The PyPI job alone receives `id-token: write`; no long-lived package token is stored in GitHub.

For the first upload, the PyPI account owner must register a **pending GitHub Actions publisher** under [PyPI account publishing settings](https://pypi.org/manage/account/publishing/). Use these exact values:

| Field | Value |
|---|---|
| PyPI project name | `roguard-child-safety` |
| GitHub owner | `xyz-rafael-xyz` |
| Repository name | `roguard-child-safety` |
| Workflow filename | `release-assets.yml` |
| Environment name | `pypi` |

The matching GitHub environment already exists. A pending publisher does not reserve the package name; it becomes a regular publisher after the first successful upload. [PyPI's instructions](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/) describe this one-time account action. Do not send an API token or password to a maintainer or store one in the repository.

After the publisher is registered and the `main` CI is green, a maintainer pushes the matching release tag. The workflow must pass before the PyPI project or GitHub release is considered live. Verify the [PyPI project](https://pypi.org/project/roguard-child-safety/) and install in a fresh environment with `python3 -m pip install roguard-child-safety`. Check the installed version and run both fictional CLI examples. Later releases use the same trusted publisher without another account setup step.
