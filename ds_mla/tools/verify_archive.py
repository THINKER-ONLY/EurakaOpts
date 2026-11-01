"""Offline integrity and consistency checks. Does not execute production kernels."""
from pathlib import Path
import ast
import csv
import hashlib
import json
import math
import re
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = '5e3efcc1da528d72a1909292f505e8fcef560dee0eb7e4f8133b257e8583668c'
errors = []


def check(condition, message):
    if not condition:
        errors.append(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


manifest = json.loads((ROOT / 'SOURCE_MANIFEST.json').read_text(encoding='utf-8'))
for entry in manifest['files']:
    p = ROOT / entry['path']
    check(p.is_file() and digest(p) == entry['sha256'], 'source hash: ' + entry['path'])
    if entry['transformation'] == 'byte-identical':
        check(entry['sha256'] == entry['source_sha256'], 'source identity: ' + entry['path'])
checksums = ROOT / 'CHECKSUMS.json'
if checksums.exists():
    entries = json.loads(checksums.read_text(encoding='utf-8'))
    actual = {p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file() and p.name != 'CHECKSUMS.json' and '__pycache__' not in p.parts}
    check(set(entries) == actual, 'archive checksum file set')
    for name, value in entries.items():
        p = ROOT / name
        check(p.exists() and digest(p) == value, 'archive checksum: ' + name)

for p in ROOT.rglob('*.py'):
    try:
        ast.parse(p.read_text(encoding='utf-8-sig'))
    except Exception as exc:
        errors.append('Python parse: ' + str(p.relative_to(ROOT)) + ': ' + str(exc))
for p in ROOT.rglob('*.json'):
    try:
        json.loads(p.read_text(encoding='utf-8-sig'))
    except Exception:
        errors.append('JSON parse: ' + str(p.relative_to(ROOT)))
for p in [*ROOT.rglob('*.md'), *ROOT.rglob('*.svg')]:
    check(b'\r\r\n' not in p.read_bytes(), 'duplicate CR newline: ' + str(p.relative_to(ROOT)))
check(digest(ROOT / 'code/MLA.py') == EXPECTED, 'best baseline hash')
check(digest(ROOT / 'code/MLA.baseline96p83-20261028.py') == EXPECTED, 'best baseline backup')
tree = ast.parse((ROOT / 'code/MLA.py').read_text(encoding='utf-8'))
for node in tree.body:
    if isinstance(node, ast.Assign):
        try:
            ast.literal_eval(node.value)
        except Exception:
            errors.append('best baseline module literal assignment')
    if isinstance(node, ast.Import):
        check(all(n.name.startswith('tilelang') for n in node.names), 'production imports')
    if isinstance(node, ast.ImportFrom):
        check((node.module or '').startswith('tilelang'), 'production from imports')

rows = list(csv.DictReader((ROOT / 'data/ssh_core.csv').open(encoding='utf-8')))
check(len(rows) == 60, 'core data row count')
for r in rows:
    j = json.loads((ROOT / r['evidence']).read_text(encoding='utf-8'))
    c = next(c for c in j['cases'] if c['case_id'] == int(r['case_id']))
    check(j['status'] == 'passed' and c['status'] == 'passed', 'core case status')
    check(math.isclose(float(r['baseline_us']), c['baseline_ms']*1000, rel_tol=1e-12), 'core baseline data')
    check(math.isclose(float(r['candidate_us']), c['candidate_ms']*1000, rel_tol=1e-12), 'core candidate data')
    check(math.isclose(float(r['speedup']), c['baseline_ms']/c['candidate_ms'], rel_tol=1e-12), 'core speedup')
    check(r['baseline_sha256'] == j['artifacts']['score96_baseline']['sha256'], 'core baseline provenance')
    check(r['candidate_sha256'] == j['artifacts']['score96_candidate']['sha256'], 'core candidate provenance')

for folder, prefix, filename in [
    ('mla-deferredsum-20261030', '3b20aa59e4', 'MLA.candidate-deferredsum-20261030.py'),
    ('mla-pvstream-20261030', '01a2c12bc5', 'MLA.candidate-pvstream-B4-20261030.py'),
    ('mla-9683-20261028', '9699e7d193', 'MLA.candidate-qkchunk128-20261028.py'),
]:
    j = json.loads((ROOT / f'evidence/{folder}/{prefix}-full31.json').read_text(encoding='utf-8'))
    check(j['status'] == 'passed' and len(j['cases']) == 31, 'full31 status: ' + folder)
    check(j['artifacts']['score96_candidate']['sha256'] == digest(ROOT / 'code' / filename), 'full31 candidate hash: ' + folder)
    check(j['artifacts']['score96_baseline']['sha256'] == EXPECTED, 'full31 comparator: ' + folder)
    for c in j['cases']:
        check(c['status'] == 'passed' and {s['seed'] for s in c['seeds']} == {0, 42}, 'full31 seed coverage')
        for s in c['seeds']:
            check(all(s[role]['finite'] and s[role]['accepted'] for role in ['baseline', 'candidate']), 'full31 numeric status')

snippets = json.loads((ROOT / 'slides/snippets/index.json').read_text(encoding='utf-8'))
for s in snippets:
    original = (ROOT / s['source']).read_text(encoding='utf-8').splitlines()
    text = (ROOT / 'slides/snippets' / (s['name'] + '.txt')).read_text(encoding='utf-8').splitlines()
    check(text[2:] == original[s['start_line']-1:s['end_line']], 'verbatim snippet: ' + s['name'])

for p in [ROOT / 'README.md', ROOT / 'code/README.md', ROOT / 'evidence/README.md', *list((ROOT / 'docs').glob('*.md')), ROOT / 'slides/PPT_MATERIALS.md']:
    text = p.read_text(encoding='utf-8')
    for link in re.findall(r'\]\(([^)]+)\)', text):
        if '://' in link or link.startswith('#'):
            continue
        target = unquote(link.split('#')[0])
        check((p.parent / target).exists(), 'relative link: ' + str(p.relative_to(ROOT)) + ' -> ' + link)

check('T.reduce_sum(denom_frag, logsum' in (ROOT / 'slides/snippets/deferred_sum_final.txt').read_text(encoding='utf-8'), 'final reduction excerpt')
check('T.alloc_local([4]' in (ROOT / 'slides/snippets/pv_stream_local.txt').read_text(encoding='utf-8'), 'B4 excerpt')

# No credential values are needed to verify or reproduce the archive.
for p in ROOT.rglob('*'):
    if not p.is_file() or p.suffix.lower() not in {'.py', '.json', '.md', '.txt', '.csv', '.diff'}:
        continue
    text = p.read_text(encoding='utf-8-sig')
    if p.name != 'verify_archive.py':
        check(not re.search(r'114514|root\+vm-[A-Za-z0-9]+@|140\.207\.205\.81|-----BEGIN .*PRIVATE KEY-----|\bsk-[A-Za-z0-9_-]{24,}', text), 'credential/endpoint scan: ' + str(p.relative_to(ROOT)))

if errors:
    raise SystemExit('\n'.join(errors))
print(f'PASS: {len(manifest["files"])} source files, {len(rows)} benchmark rows, 3 full31 reports, verbatim snippets, links, syntax and credential scan.')
