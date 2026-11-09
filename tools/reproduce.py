"""Rebuild all five stages in a NEW directory and compare saved CSV results.

Run from the repository root: python tools/reproduce.py --output-dir .validation/reproduction
Only use a trusted local Python environment and trusted saved model artefacts.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        parser.error("Output directory must not already exist; original results are preserved.")
    if not output.is_relative_to(ROOT):
        parser.error("Keep reproduction outputs inside this project.")
    output.mkdir(parents=True)
    data = output / "stage2/water_treatment_stage2.csv"
    commands = [
        ["simulate.py", "--stage", "1", "--output-dir", str(output / "stage1")],
        ["simulate.py", "--output-dir", str(output / "stage2")],
        ["train_baseline.py", "--data", str(data), "--output-dir", str(output / "stage3")],
        ["evaluate_attacks.py", "--data", str(data), "--stage3-dir", str(output / "stage3"),
         "--output-dir", str(output / "stage4")],
        ["evaluate_defence.py", "--data", str(data), "--stage3-dir", str(output / "stage3"),
         "--stage4-dir", str(output / "stage4"), "--output-dir", str(output / "stage5")],
    ]
    for command in commands:
        print(f"Running {command[0]}", flush=True)
        with (output / f"{Path(command[0]).stem}.log").open("w") as log:
            subprocess.run([sys.executable, *command], cwd=ROOT, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
    comparisons = []
    for original in sorted((ROOT / "output").rglob("*")):
        if not original.is_file() or not original.name.endswith((".csv", ".csv.gz")):
            continue
        relative = original.relative_to(ROOT / "output")
        candidate = output / relative if len(relative.parts) > 1 else output / "stage1" / original.name
        before, after = pd.read_csv(original), pd.read_csv(candidate)
        pd.testing.assert_frame_equal(before, after, check_exact=False, atol=1e-12, rtol=0)
        comparisons.append(str(relative))
    (output / "comparison.txt").write_text(
        f"PASS: {len(comparisons)} CSV tables/archives matched (atol=1e-12, rtol=0).\n"
        + "\n".join(comparisons) + "\n"
    )
    print((output / "comparison.txt").read_text())


if __name__ == "__main__":
    main()
