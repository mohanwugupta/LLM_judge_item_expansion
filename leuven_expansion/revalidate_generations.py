"""Revalidate preserved v3 raw outputs without issuing model requests."""

from __future__ import annotations

import argparse
import json
from typing import Optional

from leuven_expansion.generate_features import revalidate_generation_outputs


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = _parse_args(argv)
    summary = revalidate_generation_outputs(args.output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
