# Frozen authored challenge history, 10 November 2026

Each source and freeze manifest is retained beside the corresponding first result. All labels were authored by the project; `independent_human` is false. Counts below sum binary and concern errors, so one case can contribute two errors. These challenges were selected during development and cannot establish unbiased real-world accuracy. An original failure remains recorded even when a later candidate fixes it.

| Challenge | Cases | V9 errors | First V10 errors | Outcome |
| --- | ---: | ---: | ---: | --- |
| A | 40 | 1 | 4 | Failed |
| B | 48 | 8 | 9 | Failed |
| C | 52 | 8 | 8 | Failed |
| D | 48 | 11 | 10 | Failed |
| E | 46 | 3 | 3 | Failed |
| F | 48 | 12 | 3 | Failed |
| G | 40 | 13 | 6 | Failed |
| H | 40 | 9 | 1 | Failed |
| I | 40 | 8 | 5 | Failed |
| J | 40 | 18 | 9 | Failed |
| K | 40 | 10 | 1 | Failed |
| L | 40 | 12 | 2 | Failed |
| M | 40 | 13 | 6 | Failed |
| N | 40 | 15 | 4 | Failed |
| N on v13e | 40 | 15 | 0 | Exposed replay |
| O on v13e | 40 | 9 | 0 | Fresh pass |

Challenge O was authored and frozen before the result of challenge N was inspected. It was scored once against the installed v13e wheel. Challenge N was used to fix v13e, so its later zero-error replay is development evidence. The exact O wheel SHA-256 and model manifest SHA-256 are in its result file. The original N wheel and errors remain in the original result file.

The model treats `concern` as the underlying topic even if denied or fictional, in accordance with the pinned v9 interpretation policy. This is why explicit denial of confinement is `binary=0, concern=1`. These finite tests do not prove error-free behavior on authentic messages. No independent case-level human annotation record was available.
