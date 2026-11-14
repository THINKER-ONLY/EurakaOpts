#!/usr/bin/env python3
"""Submit a candidate to an XPU-OJ contest, wait for the judge, record results."""
import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import oj

TERMINAL = {"Accepted", "WrongAnswer", "TimeLimitExceeded", "RuntimeError",
            "CompileError", "SystemError", "Canceled", "Skipped", "Hacked"}


def report_rows(detail):
    p = detail.get("progress") or {}
    rows = []
    for h, t in (p.get("testcaseResult") or {}).items():
        cm = t.get("checkerMessage", "") or ""
        def g(pat, cast=float):
            m = re.search(pat, cm)
            return cast(m.group(1)) if m else None
        rows.append({
            "display": t.get("displayScore"),
            "ratio": g(r"Score ratio:\s*([\d.]+)"),
            "base_ms": g(r"Baseline:\s*([\d.]+)"),
            "kernel_ms": g(r"User kernel:\s*([\d.]+)"),
            "speedup": g(r"Speedup vs base:\s*([\d.]+)"),
            "config": g(r"Config:\s*(.+)", str),
            "status": t.get("status"),
        })
    rows.sort(key=lambda x: x["ratio"] or 0)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--contest-id", type=int, default=6)
    ap.add_argument("--order", type=int, default=1)
    ap.add_argument("--lang", default="tilelang.maca-c500")
    ap.add_argument("--label", default="")
    ap.add_argument("--ledger", default="/root/lzy/workbench/oj_ledger.jsonl")
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--interval", type=float, default=10)
    a = ap.parse_args()

    code = open(a.file).read()
    content = {"language": a.lang, "code": code, "compileAndRunOptions": {}}
    headers = oj.pow_header("submit_problem")
    st, r = oj.post("contest/play/submit",
                    {"contestId": a.contest_id, "problemOrder": a.order, "content": content},
                    headers)
    print("submit ->", st, r)
    sid = r.get("submissionId") if isinstance(r, dict) else None
    if not sid:
        sys.exit(1)
    print("submissionId", sid)

    deadline = time.time() + a.timeout
    detail = None
    last = None
    while time.time() < deadline:
        try:
            st, d = oj.post("submission/getSubmissionDetail",
                            {"submissionId": str(sid), "locale": "zh_CN"})
        except Exception as e:
            print("poll error:", type(e).__name__, flush=True)
            time.sleep(a.interval)
            continue
        if st // 100 == 2:
            meta = d.get("meta", {})
            s = meta.get("status")
            if s != last:
                print(f"[{time.strftime('%H:%M:%S')}] status={s} "
                      f"display={meta.get('displayScore')}", flush=True)
                last = s
            if s in TERMINAL:
                detail = d
                break
        time.sleep(a.interval)
    if detail is None:
        print("timeout")
        sys.exit(2)

    m = detail["meta"]
    rows = report_rows(detail)
    for x in rows:
        print(f"{x['display']!s:>4} {100*(x['ratio'] or 0):7.2f}% "
              f"base={x['base_ms']} kernel={x['kernel_ms']} x{x['speedup']}  {x['config']}")
    vals = [x["ratio"] for x in rows if x["ratio"] is not None]
    mean = 100 * sum(vals) / len(vals) if vals else None
    print(f"display={m.get('displayScore')} mean_ratio={mean}")

    rec = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "label": a.label or os.path.basename(a.file),
        "file": os.path.abspath(a.file),
        "contest": a.contest_id,
        "submissionId": sid,
        "status": m.get("status"),
        "displayScore": m.get("displayScore"),
        "meanRatio": mean,
        "timeUsed": m.get("timeUsed"),
        "memoryUsed": m.get("memoryUsed"),
        "rows": rows,
    }
    with open(a.ledger, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print("recorded ->", a.ledger)


if __name__ == "__main__":
    main()
