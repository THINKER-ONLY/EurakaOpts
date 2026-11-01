#!/usr/bin/env python3
"""Minimal MACA diagnostic: source/export/resource inspection only; no GPU execution."""
import argparse, hashlib, importlib.util, json, re, shutil, subprocess, sys, traceback, zipfile
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True
LIMIT = 2 * 1024 * 1024
CONTEXTS = (2048, 8192, 65536)
KEYS = ("register", "vgpr", "sgpr", "private", "scratch", "shared")
LOOP_RE = re.compile(r"\b(for|while)\b|T\.(?:Pipelined|serial|Parallel)\s*\(")
MMA_RE = re.compile(r"\b(?:T\.gemm\w*|mma\w*)\s*\(")

def utc():
    return datetime.now(timezone.utc).isoformat()

def digest(data):
    return hashlib.sha256(data).hexdigest()

def save_report(out, report):
    (out / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")

def load_module(path, role, sha):
    name = f"_diag_{role}_{sha[:12]}"
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def write_artifact(out, filename, getter):
    try:
        value = getter()
        data = (value.decode("utf-8", "replace") if isinstance(value, bytes) else value if isinstance(value, str) else repr(value)).encode("utf-8", "replace")
        (out / filename).write_bytes(data)
        return {"success": True, "path": filename, "bytes": len(data), "sha256": digest(data)}
    except Exception:
        error = traceback.format_exc()
        (out / (filename + ".error.txt")).write_text(error, encoding="utf-8")
        return {"success": False, "path": None, "error": error}

def source_metrics(source):
    lines = source.splitlines()
    loops = [i + 1 for i, line in enumerate(lines) if LOOP_RE.search(line)]
    mmas = [i + 1 for i, line in enumerate(lines) if MMA_RE.search(line)]
    pv = [{"line": i + 1, "text": line.rstrip()} for i, line in enumerate(lines) if "pv" in line.lower()]
    return {"label": "source-level static counts only; not hardware counters or throughput", "loop_count": len(loops), "loop_lines": loops, "mma_count": len(mmas), "mma_lines": mmas, "pv_loop_count": len(pv), "pv_loop_lines": pv}

def parse_resources(text):
    metrics = {key: None for key in KEYS}
    sources = {}
    for line in text.splitlines():
        low = line.lower()
        for key in KEYS:
            if metrics[key] is not None or key not in low:
                continue
            match = re.search(r"(?i)\b[^:=]*" + key + r"[^:=]*[:=]\s*(0x[0-9a-f]+|\d+)", line)
            if match:
                metrics[key] = int(match.group(1), 0); sources[key] = line.strip()
    return {"label": "explicit metadata only; null when absent", "metrics": metrics, "source_lines": sources}

def binary_info(path):
    data = path.read_bytes()
    return {"exists": True, "path": str(path), "size_bytes": len(data), "sha256": digest(data), "magic_hex": data[:8].hex()}

def run_tool(out, name, command):
    tool = shutil.which(command[0])
    if not tool:
        error = f"{command[0]} not found"; (out / f"{name}.txt").write_text(error + "\n", encoding="utf-8")
        return {"available": False, "file": f"{name}.txt", "error": error}, error
    try:
        proc = subprocess.run([tool] + command[1:], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=30, check=False)
        full = proc.stdout or b""; raw = full[:LIMIT]; (out / f"{name}.txt").write_bytes(raw)
        info = {"available": True, "returncode": proc.returncode, "truncated": len(full) > LIMIT, "bytes": len(raw), "file": f"{name}.txt", "label": "read-only; 30s timeout; saved output cap 2 MiB"}
        return info, raw.decode("utf-8", "replace")
    except Exception:
        error = traceback.format_exc(); (out / f"{name}.error.txt").write_text(error, encoding="utf-8")
        return {"available": True, "file": None, "error": error}, ""

def export_binary(kernel, out, role, ctx, entry):
    path = out / f"{role}_{ctx}.mcbin"
    try:
        device = kernel.adapter.rt_mod.imports[0]
        device.write_to_file(str(path), "mcbin")
        if not path.exists() or path.stat().st_size == 0:
            raise FileNotFoundError(f"write_to_file did not create a non-empty binary: {path}")
        entry["export"] = {"success": True, "format": "mcbin", "error": None, **binary_info(path)}
    except Exception:
        entry["export"] = {"success": False, "format": "mcbin", "path": str(path), "error": traceback.format_exc()}
        return
    meta = Path(str(path) + ".meta"); entry["export"]["meta_path"] = str(meta) if meta.exists() else None
    entry["tools"] = []; text = ""
    if meta.exists():
        try: text += meta.read_text(encoding="utf-8", errors="replace") + "\n"
        except Exception: entry["meta_error"] = traceback.format_exc()
    for name, command in (("readelf", ["readelf", "-aW", str(path)]), ("objdump", ["objdump", "-x", str(path)])):
        info, raw = run_tool(out, f"{role}_{ctx}_{name}", command); entry["tools"].append({name: info}); text += raw + "\n"
    entry["resources"] = parse_resources(text)

def run_role(module, role, ctx, out, import_error):
    entry = {"compile": {"success": False, "error": None}, "artifacts": {}, "export": None, "source_metrics": None}
    if module is None:
        entry["compile"]["error"] = import_error; return entry
    try:
        kernel = module._get_kernel(1, 16, 1, ctx, 512, 64); entry["compile"] = {"success": True, "error": None}
    except Exception:
        entry["compile"] = {"success": False, "error": traceback.format_exc()}; return entry
    for filename, label, getter in (
        (f"{role}_{ctx}_kernel_source.txt", "kernel_source", lambda: kernel.get_kernel_source()),
        (f"{role}_{ctx}_host_source.txt", "host_source", lambda: kernel.get_host_source()),
        (f"{role}_{ctx}_device_script.txt", "device_script", lambda: kernel.adapter.device_mod.script())):
        entry["artifacts"][label] = write_artifact(out, filename, getter)
    artifact = entry["artifacts"]["kernel_source"]
    if artifact["success"]: entry["source_metrics"] = source_metrics((out / artifact["path"]).read_text(encoding="utf-8", errors="replace"))
    export_binary(kernel, out, role, ctx, entry); return entry

def update_host_equality(context, out):
    baseline = context["roles"].get("baseline", {}).get("artifacts", {}).get("host_source")
    candidate = context["roles"].get("candidate", {}).get("artifacts", {}).get("host_source")
    if not baseline or not candidate or not baseline.get("success") or not candidate.get("success"):
        context["host_byte_equality"] = {"equal": None, "label": "requires both host sources"}; return
    base_data = (out / baseline["path"]).read_bytes(); cand_data = (out / candidate["path"]).read_bytes()
    context["host_byte_equality"] = {"equal": base_data == cand_data, "baseline_sha256": digest(base_data), "candidate_sha256": digest(cand_data), "baseline_bytes": len(base_data), "candidate_bytes": len(cand_data)}

def zip_evidence(out):
    target = out / "evidence.zip"
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(out.rglob("*")):
            if path.is_file() and path != target:
                archive.write(path, path.relative_to(out).as_posix())

def main():
    parser = argparse.ArgumentParser()
    for name in ("--baseline", "--candidate", "--baseline-sha", "--candidate-sha", "--output-dir"):
        parser.add_argument(name, required=True)
    args = parser.parse_args()
    out = Path(args.output_dir); baseline_path, candidate_path = Path(args.baseline), Path(args.candidate)
    try:
        base_data, cand_data = baseline_path.read_bytes(), candidate_path.read_bytes()
    except Exception:
        print(json.dumps({"ok": False, "error": traceback.format_exc()})); return 2
    base_sha, cand_sha = digest(base_data), digest(cand_data)
    if base_sha.lower() != args.baseline_sha.lower() or cand_sha.lower() != args.candidate_sha.lower():
        print(json.dumps({"ok": False, "error": "sha256 mismatch", "baseline_sha256": base_sha, "candidate_sha256": cand_sha})); return 2
    try:
        out.mkdir(parents=True, exist_ok=False)
    except Exception:
        print(json.dumps({"ok": False, "error": traceback.format_exc()})); return 2
    report = {"tool": "maca_minimal_diagnostic", "gpu_execution": False, "hardware_counters_collected": False, "started_at": utc(),
              "inputs": {"baseline": {"path": str(baseline_path), "expected_sha256": args.baseline_sha, "actual_sha256": base_sha},
                         "candidate": {"path": str(candidate_path), "expected_sha256": args.candidate_sha, "actual_sha256": cand_sha}},
              "environment": {}, "contexts": {}, "import_errors": {}, "errors": []}
    save_report(out, report)
    try:
        try:
            import tilelang, torch
        except Exception:
            report["fatal_error"] = traceback.format_exc(); save_report(out, report)
            print(json.dumps({"ok": False, "error": "import failed", "output_dir": str(out)})); return 1
        report["environment"] = {"compiler": {"name": "tilelang", "version": getattr(tilelang, "__version__", None), "path": getattr(tilelang, "__file__", None)},
                                 "torch": {"version": getattr(torch, "__version__", None), "path": getattr(torch, "__file__", None)},
                                 "python": sys.version, "platform": sys.platform}
        try:
            tilelang.disable_cache(); report["environment"]["disable_cache"] = {"success": True, "error": None}
        except Exception:
            report["environment"]["disable_cache"] = {"success": False, "error": traceback.format_exc()}
            save_report(out, report); print(json.dumps({"ok": False, "error": "disable_cache failed", "output_dir": str(out)})); return 1
        modules = {}
        for role, path, sha in (("baseline", baseline_path, base_sha), ("candidate", candidate_path, cand_sha)):
            try:
                modules[role] = load_module(path, role, sha)
            except Exception:
                report["import_errors"][role] = traceback.format_exc(); modules[role] = None
        save_report(out, report)
        for role, ctx in ((r, c) for r in ("baseline", "candidate") for c in CONTEXTS):
            context = report["contexts"].setdefault(str(ctx), {"context": ctx, "roles": {}, "host_byte_equality": None, "candidate_pv_loop_lines": None})
            entry = run_role(modules[role], role, ctx, out, report["import_errors"].get(role))
            context["roles"][role] = entry
            if role == "candidate" and entry.get("source_metrics") is not None:
                context["candidate_pv_loop_lines"] = entry["source_metrics"]["pv_loop_lines"]
            update_host_equality(context, out); save_report(out, report)
        roles = [(c["context"], role, e) for c in report["contexts"].values() for role, e in c["roles"].items()]
        compile_failures = [{"context": c, "role": r, "error": e["compile"]["error"]} for c, r, e in roles if not e["compile"]["success"]]
        export_failures = [{"context": c, "role": r, "error": e["export"]["error"]} for c, r, e in roles if e.get("export") and not e["export"]["success"]]
        artifact_failures = [{"context": c, "role": r, "artifact": k, "error": v["error"]} for c, r, e in roles for k, v in e.get("artifacts", {}).items() if not v["success"]]
        report["summary"] = {"compile_failures": compile_failures, "export_failures": export_failures, "artifact_failures": artifact_failures, "hardware_counters_collected": False}
        if report["import_errors"] or compile_failures: status, code = "compile_failure", 1
        elif export_failures: status, code = "export_failure", 2
        elif artifact_failures: status, code = "artifact_failure", 3
        else: status, code = "ok", 0
        report["status"] = status; report["finished_at"] = utc(); save_report(out, report)
        print(json.dumps({"ok": code == 0, "status": status, "compile_failures": len(compile_failures), "export_failures": len(export_failures), "artifact_failures": len(artifact_failures), "output_dir": str(out), "report": str(out / "report.json"), "evidence": str(out / "evidence.zip"), "hardware_counters_collected": False}))
        return code
    finally:
        try: zip_evidence(out)
        except Exception: pass

if __name__ == "__main__":
    raise SystemExit(main())
