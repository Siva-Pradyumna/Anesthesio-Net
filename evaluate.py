"""
AnesthesioNet-AI Master Evaluation Harness
Allows clinicians and engineers to test, evaluate, and benchmark all developed modules.

Usage:
    python evaluate.py           # Evaluates all completed development steps
    python evaluate.py --step 1   # Evaluates Step 1: Mode A Digital Twin
    python evaluate.py --step 2   # Evaluates Step 2: FSM, Gym Env & HOCBF Safety
"""

import argparse
import sys
from pathlib import Path

# Ensure root path is imported
sys.path.insert(0, str(Path(__file__).resolve().parent))

from anesthesionet.evaluation import evaluate_step1, evaluate_step2, evaluate_step3


def main():
    parser = argparse.ArgumentParser(
        description="AnesthesioNet-AI Master Clinical & Algorithmic Evaluation Suite"
    )
    parser.add_argument(
        "--step",
        type=int,
        choices=[1, 2, 3],
        default=None,
        help="Specify which development step to evaluate (1, 2, or 3). Defaults to all."
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run comprehensive evaluation across all developed stages."
    )

    args = parser.parse_args()

    print("\n" + "#" * 80)
    print("                 ANESTHESIONET-AI: MASTER SYSTEM EVALUATION")
    print("#" * 80)

    overall_results = {}

    if args.step == 1:
        res1 = evaluate_step1()
        overall_results["Step 1 (Digital Twin Physiology)"] = all(res1.values())
    elif args.step == 2:
        res2 = evaluate_step2()
        overall_results["Step 2 (FSM, Gym Env & HOCBF Safety)"] = all(res2.values())
    elif args.step == 3:
        res3 = evaluate_step3()
        overall_results["Step 3 (VitalDB Replay & Signal Filter)"] = all(res3.values())
    else:
        # Default: run all completed steps
        res1 = evaluate_step1()
        res2 = evaluate_step2()
        res3 = evaluate_step3()
        overall_results["Step 1 (Digital Twin Physiology)"] = all(res1.values())
        overall_results["Step 2 (FSM, Gym Env & HOCBF Safety)"] = all(res2.values())
        overall_results["Step 3 (VitalDB Replay & Signal Filter)"] = all(res3.values())

    print("\n" + "=" * 80)
    print("                     EXECUTIVE EVALUATION SUMMARY")
    print("=" * 80)
    for module_name, passed in overall_results.items():
        status_badge = "PASSED [100%]" if passed else "FAILED"
        print(f" • {module_name:<45} : {status_badge}")
    print("=" * 80)
    
    if all(overall_results.values()):
        print(" >> ALL EVALUATION BENCHMARKS SATISFIED — SYSTEM VERIFIED & SECURE <<")
        print("=" * 80 + "\n")
        sys.exit(0)
    else:
        print(" >> SOME EVALUATION BENCHMARKS FAILED — ATTENTION REQUIRED <<")
        print("=" * 80 + "\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
