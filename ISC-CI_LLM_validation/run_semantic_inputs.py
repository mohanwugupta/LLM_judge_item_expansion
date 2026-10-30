#!/usr/bin/env python3
"""Install/check frozen inputs with only the Python standard library."""

import argparse
import json
from pathlib import Path

from semantic_inputs import (
    check_inputs,
    collect_frozen_inputs,
    install_inputs,
    write_bundle,
)


def main():
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--install", action="store_true")
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--build", action="store_true")
    parser.add_argument("--project-root", type=Path, default=project)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=project / "ISC-CI_LLM_validation/configs/semantic_frozen_inputs.json",
    )
    parser.add_argument(
        "--bundle",
        type=Path,
        default=project / "ISC-CI_LLM_validation/inputs/semantic_frozen_inputs.tar.gz",
    )
    args = parser.parse_args()
    try:
        if args.build:
            record = write_bundle(
                args.project_root,
                collect_frozen_inputs(args.project_root),
                args.bundle,
                args.manifest,
            )
            result = {
                "passed": True,
                "files": len(record["files"]),
                "bundle_sha256": record["bundle_sha256"],
                "bundle_bytes": args.bundle.stat().st_size,
            }
        elif args.install:
            result = install_inputs(args.project_root, args.bundle, args.manifest)
        else:
            result = check_inputs(args.project_root, args.manifest)
        print(json.dumps(result, indent=2))
        return 0 if result["passed"] else 2
    except (ValueError, OSError, KeyError) as error:
        print(
            json.dumps(
                {
                    "passed": False,
                    "error": str(error),
                    "scientific_inputs_not_regenerated": True,
                }
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
