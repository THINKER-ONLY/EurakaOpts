#!/usr/bin/env python3
import argparse, hashlib, importlib.util, json, os, sys, traceback, uuid, zipfile
from pathlib import Path
sys.dont_write_bytecode = True
ROLES = ("baseline", "candidate")
ARGS = (1, 16, 1, 65536, 512, 64)
def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()
def put(path, text):
    with open(path, "x", encoding="utf-8") as f:
        f.write(text)
def scan(d):
    files = sorted(p.relative_to(d).as_posix() for p in d.rglob("*") if p.is_file())
    targets = [f for f in files if "UnrollLoop" in Path(f).name or "SplitHostDevice" in Path(f).name]
    return files, targets
def main():
    ap = argparse.ArgumentParser()
    for k in ("baseline", "candidate", "baseline-sha", "candidate-sha", "output-dir"):
        ap.add_argument("--" + k, required=True)
    a = ap.parse_args()
    out = Path(a.output_dir)
    src = {r: Path(getattr(a, r)) for r in ROLES}
    exp = {r: getattr(a, r + "_sha").lower() for r in ROLES}
    act, herr = {}, {}
    for r in ROLES:
        try:
            act[r], herr[r] = sha(src[r]), None
        except Exception:
            act[r], herr[r] = None, traceback.format_exc()
    bad = {r for r in ROLES if act[r] != exp[r]}
    os.mkdir(out)
    tl = None
    if not bad:
        import tilelang as tl
        tl.disable_cache()
    report = {"compiler_version": getattr(tl, "__version__", None), "baseline": None, "candidate": None,
              "evidence_zip": str(out / "evidence.zip"), "status": ""}
    for r in ROLES:
        d = out / r
        d.mkdir()
        rec = {"source_path": str(src[r]), "source_sha256": act[r], "expected_sha256": exp[r],
               "sha_verified": r not in bad, "configs": None, "compilation": "failed",
               "error": herr[r], "dump_files": [], "target_files": [],
               "kernel_source_file": None, "host_source_file": None}
        if r in bad:
            rec["error"] = herr[r] or "source SHA mismatch"
            report[r] = rec
            continue
        try:
            name = "_dsh_inspect_%s_%s" % (r, uuid.uuid4().hex)
            spec = importlib.util.spec_from_file_location(name, src[r])
            mod = importlib.util.module_from_spec(spec)
            sys.modules[name] = mod
            spec.loader.exec_module(mod)
            jit = mod.flashattn
            cfg = dict(getattr(jit, "pass_configs", {}) or {})
            cfg.update({"tl.enable_dump_ir": True, "tl.dump_ir_path": str(d)})
            jit.pass_configs = cfg
            rec["configs"] = cfg
            kernel = (getattr(mod, "_get_kernel", None) or jit._get_kernel)(*ARGS)
            for kind, meth in (("kernel", "get_kernel_source"), ("host", "get_host_source")):
                name = "%s_%s_source.txt" % (r, kind)
                put(out / name, str(getattr(kernel, meth)()))
                rec[kind + "_source_file"] = name
            rec["compilation"] = "ok"
        except Exception:
            rec["error"] = traceback.format_exc()
        rec["dump_files"], rec["target_files"] = scan(d)
        report[r] = rec
    ok = not any(report[r]["compilation"] != "ok" for r in ROLES) and any(report[r]["dump_files"] for r in ROLES)
    report["status"] = "ok" if ok else "failed"
    rp = out / "report.json"
    put(rp, json.dumps(report, indent=2, sort_keys=True, default=str))
    zp = out / "evidence.zip"
    members = [p for p in sorted(out.rglob("*")) if p.is_file() and p != zp]
    with zipfile.ZipFile(zp, "x", zipfile.ZIP_DEFLATED) as z:
        for p in members:
            z.write(p, p.relative_to(out).as_posix())
    print(json.dumps({"status": report["status"], "report": str(rp), "report_body": report, "evidence_zip": str(zp)}, default=str))
    return 0 if ok else 1

if __name__ == "__main__":
    raise SystemExit(main())
