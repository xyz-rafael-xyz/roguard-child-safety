# RoGuard Child Safety 3.0.2: PyPI distribution

This packaging release changes the Python distribution name from `roguard-local` to `roguard-child-safety`, enabling `pip install roguard-child-safety` after the first PyPI upload. The import package remains `roguard`. The model weights, contract rules, and [v3 synthetic study](V3_MESSAGE_RELEASE.md) are unchanged.

The [release workflow](../.github/workflows/release-assets.yml) validates the wheel and source archive, publishes through PyPI Trusted Publishing, and creates the GitHub release only after the PyPI upload succeeds. The one-time account setup and exact publisher fields are documented in [PyPI publishing](PYPI_PUBLISHING.md). The historical GitHub installer URL from v3.0.1 remains available as a compatibility asset, while the README recommends the shorter PyPI command.
