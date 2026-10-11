"""Select the second fictional-message model before opening its new test."""

import argparse
import json
from pathlib import Path

from training.fit_messages_v3 import fit
from training.generate_messages_v3b import generate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    record = fit(args.train, args.dev, args.output, generator=generate, study="v3b")
    print(json.dumps(record["selected"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
