#!/usr/bin/env python3
"""Small XPUOJ (LibreOJ-based) API client for the optimization loop.

Auth: set XPUOJ_TOKEN (API token created in the OJ web UI) or put it in
~/.xpuoj_token.  Never hard-code credentials.

Commands
  session                      show session info (needs token)
  query  [-p ID] [-u USER] [-n N] [-s SKIP]
  detail SUBMISSION_ID
  problems [--keyword K] [--skip 0] [--take 50]
  contests
  contest CONTEST_ID
  submit PROBLEM_ID --lang LANG --file submission.py [--no-pow] [--yes]
  watch  SUBMISSION_ID [--interval 3] [--timeout 900]

Examples
  ./oj.py query -n 10
  ./oj.py submit 123 --lang tilelang --file xpuoj/v060.../submission.py --yes
"""

import argparse
import hashlib
import json
import os
import sys
import time
import http.client
import urllib.error
import urllib.request

BASE = os.environ.get(
    "XPUOJ_BASE",
    "https://sd629vuj4f7uh2cscrbe0.apigateway-cn-beijing.volceapi.com",
)
TOKEN_FILE = os.path.expanduser("~/.xpuoj_token")


def token():
    t = os.environ.get("XPUOJ_TOKEN", "").strip()
    if not t and os.path.exists(TOKEN_FILE):
        t = open(TOKEN_FILE).read().strip()
    return t


def post(path, payload, headers=None, raw=False):
    url = f"{BASE}/api/{path}"
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    if token():
        req.add_header("Authorization", f"Bearer {token()}")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    last = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
                return r.status, data if raw else json.loads(data or b"{}")
        except urllib.error.HTTPError as e:
            data = e.read()
            try:
                parsed = json.loads(data or b"{}")
            except Exception:
                parsed = {"raw": data.decode(errors="replace")}
            return e.code, parsed
        except (urllib.error.URLError, TimeoutError, ConnectionError,
                http.client.IncompleteRead, http.client.HTTPException) as e:
            last = e
            time.sleep(2 * (attempt + 1))
    raise last


def post_multipart(path, fields, files, headers=None):
    """fields: dict[str,str]; files: list of (name, filename, bytes, content_type)."""
    boundary = "----xpuoj" + hashlib.sha256(os.urandom(16)).hexdigest()[:24]
    parts = []
    for k, v in fields.items():
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
        )
    for name, filename, data, ctype in files:
        parts.append(
            (
                f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; '
                f'filename="{filename}"\r\nContent-Type: {ctype}\r\n\r\n'
            ).encode()
        )
        parts.append(data)
        parts.append(b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    body = b"".join(parts)
    req = urllib.request.Request(f"{BASE}/api/{path}", data=body, method="POST")
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    if token():
        req.add_header("Authorization", f"Bearer {token()}")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        data = e.read()
        try:
            return e.code, json.loads(data or b"{}")
        except Exception:
            return e.code, {"raw": data.decode(errors="replace")}


# ---------- proof of work ----------
def solve_pow(random_data, difficulty):
    """sha256(randomData + str(nonce)) has `difficulty` leading zero hex nibbles."""
    n_bytes, odd = difficulty // 2, difficulty % 2
    nonce = 0
    while True:
        d = hashlib.sha256((random_data + str(nonce)).encode()).digest()
        if d[:n_bytes] == b"\x00" * n_bytes and (not odd or (d[n_bytes] >> 4) == 0):
            return nonce, d.hex()
        nonce += 1


def pow_header(action="submit_problem", force=False):
    if not force and not token():
        pass
    st, ch = post("proofOfWork/issueChallenge", {"action": action})
    if st // 100 != 2 or "id" not in ch:
        raise SystemExit(f"issueChallenge failed: {st} {ch}")
    t0 = time.time()
    nonce, response = solve_pow(ch["randomData"], int(ch["difficulty"]))
    print(
        f"[pow] action={action} difficulty={ch['difficulty']} nonce={nonce} "
        f"solved in {time.time()-t0:.2f}s",
        file=sys.stderr,
    )
    return {
        "X-Proof-Of-Work": json.dumps(
            {"id": ch["id"], "nonce": nonce, "response": response}
        )
    }


# ---------- commands ----------
def cmd_login(a):
    user = os.environ.get("XPUOJ_USER") or a.user
    pwd = os.environ.get("XPUOJ_PASS") or a.password
    if not user or not pwd:
        raise SystemExit("set XPUOJ_USER and XPUOJ_PASS (or --user/--password)")
    payload = {"email" if "@" in user else "username": user, "password": pwd}
    headers = pow_header("login")
    st, r = post("auth/login", payload, headers)
    if st // 100 == 2 and r.get("token"):
        with open(TOKEN_FILE, "w") as f:
            f.write(r["token"])
        os.chmod(TOKEN_FILE, 0o600)
        print(f"login ok; token saved to {TOKEN_FILE}")
        print("user:", r.get("username"), "error:", r.get("error"))
    else:
        print(st, json.dumps(r, ensure_ascii=False))
        raise SystemExit(1)


def cmd_session(_a):
    st, r = post("auth/getSessionInfo", {})
    print(st, json.dumps(r, ensure_ascii=False, indent=2)[:3000])


def cmd_query(a):
    p = {"locale": a.locale, "takeCount": a.take}
    if a.skip:
        p["skipCount"] = a.skip
    if a.problem is not None:
        p["problemId"] = a.problem
    if a.user:
        p["submitter"] = a.user
    st, r = post("submission/querySubmission", p)
    if st // 100 != 2:
        print(st, json.dumps(r, ensure_ascii=False))
        return
    subs = r.get("submissions", [])
    print(f"{'id':>8}  {'score':>6}  {'display':>7}  {'status':<14} {'lang':<18} time")
    for s in subs:
        print(
            f"{s['id']:>8}  {s.get('score')!s:>6}  {s.get('displayScore')!s:>7}  "
            f"{s.get('status',''):<14} {s.get('codeLanguage',''):<18} {s.get('submitTime','')}"
        )


def cmd_detail(a):
    st, r = post(
        "submission/getSubmissionDetail",
        {"submissionId": str(a.submission_id), "locale": a.locale},
    )
    print(json.dumps(r, ensure_ascii=False, indent=2))


def cmd_history(a):
    p = {"locale": a.locale, "contestId": a.contest_id, "problemOrder": a.order,
         "takeCount": a.take}
    if a.user:
        p["submitter"] = a.user
    st, r = post("contest/play/querySubmissions", p)
    if st // 100 != 2:
        print(st, json.dumps(r, ensure_ascii=False))
        return
    for s in r.get("submissions", []):
        print(f"{s['id']:>7}  {s.get('status',''):<12} {s.get('displayScore')!s:>6}  "
              f"{s.get('codeLanguage',''):<20} {s.get('submitTime','')}  {s.get('answerSize')}B")


def _report(r):
    import re
    p = r.get("progress") or {}
    print(f"score={p.get('score')} display={p.get('displayScore')} status={p.get('status')}")
    rows = []
    for h, t in (p.get("testcaseResult") or {}).items():
        cm = t.get("checkerMessage", "") or ""
        cfg = re.search(r"Config:\s*(.+)", cm)
        base = re.search(r"Baseline:\s*([\d.]+)", cm)
        ker = re.search(r"User kernel:\s*([\d.]+)", cm)
        sp = re.search(r"Speedup vs base:\s*([\d.]+)", cm)
        sc = re.search(r"Score ratio:\s*([\d.]+)", cm)
        rows.append({
            "display": t.get("displayScore"),
            "ratio": float(sc.group(1)) if sc else None,
            "base": float(base.group(1)) if base else None,
            "kernel": float(ker.group(1)) if ker else None,
            "speedup": float(sp.group(1)) if sp else None,
            "config": cfg.group(1).strip() if cfg else h,
            "status": t.get("status"),
        })
    rows.sort(key=lambda x: x["ratio"] or 0)
    print(f"{'disp':>4} {'ratio%':>7} {'base':>8} {'kernel':>8} {'speedup':>8}  config")
    for x in rows:
        print(f"{x['display']!s:>4} {100*(x['ratio'] or 0):7.2f} {x['base']!s:>8} "
              f"{x['kernel']!s:>8} {x['speedup']!s:>8}  {x['config']}")
    vals = [x["ratio"] for x in rows if x["ratio"] is not None]
    if vals:
        print(f"mean ratio = {100*sum(vals)/len(vals):.3f}%")
    return rows


def cmd_report(a):
    st, r = post("submission/getSubmissionDetail",
                 {"submissionId": str(a.submission_id), "locale": a.locale})
    if st // 100 != 2:
        print(st, json.dumps(r, ensure_ascii=False))
        return
    m = r["meta"]
    print(f"#{m['id']} {m['status']} display={m.get('displayScore')} lang={m['codeLanguage']} "
          f"time={m.get('submitTime')} used={m.get('timeUsed')} mem={m.get('memoryUsed')}")
    _report(r)


def cmd_fetch(a):
    st, r = post("submission/getSubmissionDetail",
                 {"submissionId": str(a.submission_id), "locale": a.locale})
    if st // 100 != 2:
        print(st, json.dumps(r, ensure_ascii=False))
        return
    c = r.get("content") or {}
    code = c.get("code")
    if not code:
        raise SystemExit("no source in this submission (not owner / not public)")
    os.makedirs(a.out, exist_ok=True)
    base = os.path.join(a.out, f"s{a.submission_id}")
    with open(base + ".py", "w") as f:
        f.write(code)
    with open(base + ".json", "w") as f:
        json.dump(r, f, ensure_ascii=False, indent=1)
    print(f"saved {base}.py ({len(code)} bytes) and {base}.json")
    _report(r)


def cmd_problems(a):
    st, r = post(
        "problem/queryProblemSet",
        {
            "locale": a.locale,
            "skipCount": a.skip,
            "takeCount": a.take,
            **({"keyword": a.keyword, "titleOnly": True} if a.keyword else {}),
        },
    )
    print(json.dumps(r, ensure_ascii=False, indent=2)[: a.limit])


def cmd_contests(_a):
    st, r = post("contest/queryContest", {"skipCount": 0, "takeCount": 50})
    print(json.dumps(r, ensure_ascii=False, indent=2)[:4000])


def cmd_contest(a):
    for path, payload in [
        ("contest/getContest", {"id": a.contest_id}),
        ("contest/getContestProblems", {"contestId": a.contest_id}),
        ("contest/play/getContestScoreboard", {"contestId": a.contest_id}),
    ]:
        st, r = post(path, payload)
        print(f"== {path} -> {st}")
        print(json.dumps(r, ensure_ascii=False, indent=2)[:3000])


def cmd_submit(a):
    code = open(a.file, "rb").read()
    content = {"language": a.lang, "code": code.decode(), "compileAndRunOptions": {}}
    if a.contest_id is not None:
        path = "contest/play/submit"
        payload = {
            "contestId": a.contest_id,
            "problemOrder": a.order,
            "content": content,
        }
    else:
        path = "submission/submit"
        payload = {"problemId": a.problem_id, "content": content}
        if a.hide:
            payload["hide"] = True
    headers = {}
    if not a.no_pow:
        headers = pow_header("submit_problem")
    if not a.yes:
        ans = input(
            f"Submit {path} problem={a.problem_id} contest={a.contest_id} "
            f"lang={a.lang} file={a.file} ({len(code)} bytes)? [y/N] "
        )
        if ans.strip().lower() not in ("y", "yes"):
            print("aborted")
            return
    st, r = post(path, payload, headers)
    print(st, json.dumps(r, ensure_ascii=False, indent=2))
    if st // 100 == 2 and r.get("submissionId"):
        print("submissionId:", r["submissionId"])


def cmd_watch(a):
    deadline = time.time() + a.timeout
    last = None
    while time.time() < deadline:
        st, r = post(
            "submission/getSubmissionDetail",
            {"submissionId": str(a.submission_id), "locale": a.locale},
        )
        if st // 100 != 2:
            print(st, json.dumps(r, ensure_ascii=False))
            return
        meta = r.get("meta", {})
        status = meta.get("status")
        if status != last:
            print(
                f"[{time.strftime('%H:%M:%S')}] status={status} score={meta.get('score')} "
                f"display={meta.get('displayScore')}"
            )
            last = status
        if status not in (None, "Waiting", "Pending", "Judging", "Compiling", "Running"):
            print(json.dumps(r, ensure_ascii=False, indent=2))
            return
        time.sleep(a.interval)
    print("timeout waiting for judge")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--locale", default="zh_CN")
    sub = ap.add_subparsers(dest="cmd", required=True)
    lg = sub.add_parser("login")
    lg.add_argument("--user")
    lg.add_argument("--password")
    lg.set_defaults(fn=cmd_login)

    sub.add_parser("session").set_defaults(fn=cmd_session)

    q = sub.add_parser("query")
    q.add_argument("-p", "--problem", type=int)
    q.add_argument("-u", "--user")
    q.add_argument("-n", "--take", type=int, default=20)
    q.add_argument("-s", "--skip", type=int, default=0)
    q.set_defaults(fn=cmd_query)

    d = sub.add_parser("detail")
    d.add_argument("submission_id")
    d.set_defaults(fn=cmd_detail)

    h = sub.add_parser("history")
    h.add_argument("-c", "--contest-id", type=int, required=True)
    h.add_argument("-o", "--order", type=int, default=1)
    h.add_argument("-u", "--user")
    h.add_argument("-n", "--take", type=int, default=30)
    h.set_defaults(fn=cmd_history)

    rp = sub.add_parser("report")
    rp.add_argument("submission_id")
    rp.set_defaults(fn=cmd_report)

    ft = sub.add_parser("fetch")
    ft.add_argument("submission_id")
    ft.add_argument("--out", default="recovered")
    ft.set_defaults(fn=cmd_fetch)

    p = sub.add_parser("problems")
    p.add_argument("--keyword")
    p.add_argument("--skip", type=int, default=0)
    p.add_argument("--take", type=int, default=50)
    p.add_argument("--limit", type=int, default=8000)
    p.set_defaults(fn=cmd_problems)

    sub.add_parser("contests").set_defaults(fn=cmd_contests)

    c = sub.add_parser("contest")
    c.add_argument("contest_id", type=int)
    c.set_defaults(fn=cmd_contest)

    s = sub.add_parser("submit")
    s.add_argument("problem_id", type=int, nargs="?", default=0)
    s.add_argument("--contest-id", type=int)
    s.add_argument("--order", type=int, default=1)
    s.add_argument("--lang", required=True)
    s.add_argument("--file", required=True)
    s.add_argument("--source-file", action="append")
    s.add_argument("--hide", action="store_true", default=True)
    s.add_argument("--no-pow", action="store_true")
    s.add_argument("--yes", action="store_true")
    s.set_defaults(fn=cmd_submit)

    w = sub.add_parser("watch")
    w.add_argument("submission_id")
    w.add_argument("--interval", type=float, default=3)
    w.add_argument("--timeout", type=float, default=900)
    w.set_defaults(fn=cmd_watch)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
