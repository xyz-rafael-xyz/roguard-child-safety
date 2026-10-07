# Decision Studio and bounded sensitivity audit

Version 0.2 adds a local browser workspace to the declared-contract checker. Install the package, then run `roguard-studio --open` or `python -m roguard.studio --open`. The command prints a `127.0.0.1` URL and opens it if the browser can be launched; otherwise, copy the printed URL into a browser on the same computer. Press Ctrl+C in the terminal to stop the process. No account, model download, cloud service, or JavaScript package is required. The wheel contains the HTML, CSS, and JavaScript files.

The **Check** tab accepts one ordinary [contract input](../schema/contract-input.schema.json) and renders the existing findings and any proposed-use result. The **Compare** tab accepts two complete inputs with the same shape and exactly one changed scalar, using the existing `roguard-contrast` rule. The **Explore changes** tab varies selected declared facts one at a time and shows the variants that change a finding's status or reason codes, or a composed-use decision. Load either symbolic sample to start. The interface displays six Romanian or Ukrainian category names. It uses the historical declared-contract rules; the corrected v1 D1/R1/S1 checks and owner-attested taxonomy review are documented separately.

The same exploration is available without a browser:

```sh
roguard-explore examples/proposed_use_ro.json
roguard-explore --limit 256 examples/contracts_uk.json
```

The explorer generates deterministic variants for D1, G1, P1 and proposed-use boolean declarations; the D1 source role; small A1 count changes; the R1 proposed recipient among declared roles; P1 decision times near events and expiries; and each applicable S1 pass field. It validates each candidate with the ordinary checker, counts invalid candidates separately, and stops after 128 attempts by default (configurable from 1 to 512 on the CLI). `truncated: true` means additional candidates existed. It does **not** explore every imaginable policy, string, event-list edit, or language expression. A missing decision flip does not prove a rule robust, and a found flip does not prove the caller's facts are true. The report identifies paths and mutation kinds without copying custom identifiers or text. Standard S1 field codes are included because they are fixed taxonomy vocabulary.

The Studio binds only to `127.0.0.1` on a randomly chosen port. POST requests require the page's random session token, a matching same-origin header, an exact loopback Host, JSON content type, and a body under 1 MiB. Responses disable caching, declare a restrictive content security policy, and do not echo malformed input values. The service logs no requests and keeps no input files. The interface sends the editor's raw JSON bytes to the Python parser, preserving large decimal spellings that JavaScript numbers would round. Other local processes with access to the machine still require ordinary operating-system isolation; this is a local research workspace, not a multi-user service. Do not expose its port through a proxy or enter real child disclosures into the symbolic examples.

All three views are read-only. A `pass` or `ready_for_human_decision` means only that supplied fictional rules and facts are internally consistent. No person is notified, no data is sent to a guardian, and no account setting changes. The tool does not classify authentic Romanian or Ukrainian child messages. Corrected v1 taxonomy reviews are owner-attested; held-out text-model accuracy validation remains open in [Release gates](RELEASE_GATES.md).
