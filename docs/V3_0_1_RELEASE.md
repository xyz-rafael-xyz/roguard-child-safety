# RoGuard Local 3.0.1: latest-release installation

This patch keeps the v3.0.0 model, tests, and evidence unchanged. It adds a stable source archive name to each GitHub release so one pip command follows GitHub's latest stable release automatically:

```sh
python3 -m pip install --upgrade https://github.com/xyz-rafael-xyz/roguard-child-safety/releases/latest/download/roguard-local-latest.tar.gz
```

On Windows, use `py -m pip` instead of `python3 -m pip`. The archive is built from the release tag, and its package metadata holds the actual version. The [release workflow](../.github/workflows/release-assets.yml) checks that the tag matches that version, builds the archive and versioned wheel, installs the archive into a clean environment, and publishes the GitHub release **with both assets already attached**. This source URL is independent of wheel filenames, which change with every version. It requires internet only during installation; the installed program runs offline. See the [v3 study report](V3_MESSAGE_RELEASE.md) for the model's narrow synthetic results and limits.
