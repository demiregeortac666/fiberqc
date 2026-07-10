"""
ask.py — natural-language interface (thin CLI over the fiberqc package).

    python ask.py --data rec.ppd --events ev.npy \
                  --question "Is my reward response real or a preprocessing artifact?"

Omit --question for an interactive session.
"""

import argparse
import os
import sys

import fiberqc as fqc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--events", required=True)
    ap.add_argument("--fs", type=float)
    ap.add_argument("--ppd-dir", default="../photometry_preprocessing")
    ap.add_argument("--question")
    a = ap.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Set ANTHROPIC_API_KEY first.")

    rec = fqc.load(a.data, events=a.events, fs=a.fs, ppd_dir=a.ppd_dir)
    print(f"loaded {rec}; running 16 pipelines (~1 min)...")
    result = fqc.multiverse(rec)
    print(f"{result}\n")

    q = a.question or input("Ask about your data> ")
    while q and q.strip():
        print("\n" + result.ask(q) + "\n")
        if a.question:
            break
        q = input("Ask about your data> ")


if __name__ == "__main__":
    main()
