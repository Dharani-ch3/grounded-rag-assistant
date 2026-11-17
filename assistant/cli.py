"""Command line.

gra ask "How many PTO days do I get?" --user ana
gra ask "How many PTO days do I get?" --user ana --mode naive
gra eval                      # tuning set
gra eval --holdout            # held-out set
gra eval --markdown > docs/eval-results.md
gra eval --fail-on-errors     # exit 1 if grounded mode makes any grounding error (used in CI)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .core import Assistant
from .evaluation import load_cases, run_eval, to_markdown


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gra", description="Grounded RAG Assistant")
    sub = parser.add_subparsers(dest="cmd", required=True)

    ask = sub.add_parser("ask", help="Ask a question as a directory user")
    ask.add_argument("question")
    ask.add_argument("--user", default="ana")
    ask.add_argument("--mode", choices=["grounded", "naive"], default="grounded")
    ask.add_argument("--provider", default=None)
    ask.add_argument("--json", action="store_true")

    ev = sub.add_parser("eval", help="Score grounded vs naive mode on the evaluation set")
    ev.add_argument("--holdout", action="store_true", help="Use the held-out questions")
    ev.add_argument("--provider", default=None)
    ev.add_argument("--markdown", action="store_true")
    ev.add_argument("--json", action="store_true")
    ev.add_argument("--fail-on-errors", action="store_true", help="Exit 1 if grounded mode has grounding errors")

    args = parser.parse_args(argv)
    a = Assistant()

    if args.cmd == "ask":
        r = a.ask(args.user, args.question, args.mode, args.provider)
        if args.json:
            print(json.dumps(r, indent=2))
            return 0
        print(r["answer"])
        for c in r["citations"]:
            where = f" › {c['heading']}" if c.get("heading") else ""
            print(f"  [{c['n']}] {c['title']} v{c['version']} ({c['status']}){where}")
        print(f"  route={r['route']['kind']} abstained={r['abstained']} {r['latency_ms']} ms")
        return 0

    name = "eval_holdout.json" if args.holdout else "eval.json"
    report = run_eval(a, args.provider, load_cases(Path(a.settings.knowledge_path) / name))
    if args.json:
        print(json.dumps(report, indent=2))
    elif args.markdown:
        print(to_markdown(report), end="")
    else:
        for mode in ("naive", "grounded"):
            s = report[mode]
            print(
                f"{mode:>9}: accuracy {s['accuracy']:.0%}, grounding errors {s['grounding_errors']}/{s['cases']}  {s['counts']}"
            )
        print(f"grounding errors reduced by {report['grounding_error_reduction']:.0%}")
    if args.fail_on_errors and report["grounded"]["grounding_errors"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
