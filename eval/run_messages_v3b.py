"""Score the second iteration's new fictional-message holdout once."""

import argparse
import json
from pathlib import Path

from eval.run_messages_v3 import evaluate
from training.generate_messages_v3b import generate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-dir", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.selection_dir, args.test, args.output_dir,
                      generator=generate, target_pairs=7, target_recall=7,
                      max_false_reviews=1, study="v3b_natural_message_holdout")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
