"""Render publication assets from archived evidence; requires matplotlib."""
from pathlib import Path
import csv
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'slides/figures'
DEST.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.spines.top': False,
                     'axes.spines.right': False, 'svg.fonttype': 'none', 'figure.facecolor': 'white'})
BLUE, ORANGE, GREEN = '#2457A7', '#D0642F', '#178475'


def save(fig, name):
    fig.savefig(DEST / (name + '.png'), dpi=200, bbox_inches='tight')
    svg = DEST / (name + '.svg')
    fig.savefig(svg, bbox_inches='tight', metadata={'Date': None})
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n', encoding='utf-8', newline='\n')
    plt.close(fig)


observations = json.loads((ROOT / 'data/oj_observations.json').read_text(encoding='utf-8'))['observations']
milestones = [observations[i] for i in [0, 1, 2, 4]]
fig, ax = plt.subplots(figsize=(10, 4.8))
ax.plot(range(4), [j['score'] for j in milestones], color=BLUE, marker='o', linewidth=3, markersize=9)
for i, j in enumerate(milestones):
    ax.annotate(str(j['score']), (i, j['score']), xytext=(0, 11), textcoords='offset points', ha='center', weight='bold')
ax.set_xticks(range(4), ['Literal-compatible\nsubmission', 'Shape-based\nsplit', 'Hybrid\nmerge', 'Best baseline\nshort target = 64'])
ax.set_ylim(68, 102); ax.set_ylabel('OJ score (user-reported)')
ax.set_title('MLA optimization milestones: 71.5 to 96.83', loc='left', weight='bold', pad=20)
ax.grid(axis='y', alpha=.18)
fig.text(.12, -.03, 'Historical user feedback; not independently fetched from OJ. Scores are not speedups.', fontsize=9, color='#555555')
fig.tight_layout(); save(fig, 'oj_scores')

rows = list(csv.DictReader((ROOT / 'data/ssh_core.csv').open(encoding='utf-8')))
qk = [r for r in rows if r['experiment'] == 'qk_chunk128' and r['case_id'] == '4'][0]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
for ax, title, values, note in [
    (axes[0], 'SSH: same-server paired measurement', [float(qk['baseline_us']), float(qk['candidate_us'])], '1.174x faster on SSH'),
    (axes[1], 'OJ: user-reported case 4', [64, 96], '50% slower on OJ'),
]:
    bars = ax.bar(['Best baseline', 'QK chunk128'], values, color=[BLUE, ORANGE], width=.55)
    ax.bar_label(bars, fmt='%.3f', padding=4)
    ax.set_ylim(0, max(values)*1.24); ax.set_ylabel('Latency (microseconds)')
    ax.set_title(title, fontsize=12, loc='left', pad=18)
    ax.text(.5, .90, note, transform=ax.transAxes, ha='center', weight='bold', color=ORANGE)
    ax.grid(axis='y', alpha=.15); ax.set_axisbelow(True)
fig.suptitle('QK chunk128: local acceleration did not transfer to OJ', weight='bold', x=.06, ha='left')
fig.text(.10, -.015, 'Case 4: batch 1, heads 16, context 16384. Compare within each panel only. OJ total: 474 -> 712 us.', fontsize=9)
fig.tight_layout(rect=(0, 0, 1, .96)); save(fig, 'qk_portability')

fig, ax = plt.subplots(figsize=(10, 4.8))
for label, title, color in [('pv_stream_b4', 'PV stream B4 (OJ pending)', BLUE), ('deferred_sum', 'Deferred sum (OJ pending)', GREEN)]:
    rs = [r for r in rows if r['experiment'] == label and int(r['context']) >= 8192]
    ax.plot([int(r['context'])//1024 for r in rs], [float(r['speedup']) for r in rs], marker='o', color=color, label=title, linewidth=2)
ax.axhline(1, color='#888888', linestyle='--', linewidth=1)
ax.set_xticks([8, 16, 32, 64], ['8K', '16K', '32K', '64K']); ax.set_ylim(.99, 1.04)
ax.set_xlabel('KV context (batch 1, heads 16)'); ax.set_ylabel('SSH baseline / candidate latency')
ax.set_title('Small local gains from two independent candidates', loc='left', weight='bold', pad=18)
ax.legend(frameon=False); ax.grid(alpha=.15)
fig.text(.12, -.03, 'Separate paired benchmark rounds; no OJ result. Unchanged short-context paths omitted.', fontsize=9)
fig.tight_layout(); save(fig, 'candidate_speedups')

elf = json.loads((ROOT / 'evidence/mla-safe-pv-20261028/elf-metadata.json').read_text(encoding='utf-8'))
data = {}
for f in elf['files']:
    if '65536' in f['file']:
        data[f['file'].split('_')[0]] = next(k for k in f['kernels'] if k['.name'] == 'main_split_kernel')
fig, axes = plt.subplots(1, 3, figsize=(11, 4.3))
for ax, field, title in zip(axes, ['.mtreg_count', '.streg_count', '.private_memory_size'], ['mtreg metadata', 'streg metadata', 'Private memory metadata (bytes)']):
    values = [data['baseline'][field], data['candidate'][field]]
    bars = ax.bar(['Before', 'Static PV'], values, color=[BLUE, GREEN], width=.55)
    ax.bar_label(bars, padding=4); ax.set_ylim(0, max(values)*1.25); ax.set_title(title, fontsize=11)
    ax.grid(axis='y', alpha=.15); ax.set_axisbelow(True)
fig.suptitle('Source-static PV: SSH compiler resource observations at context 65536', weight='bold', x=.05, ha='left')
fig.text(.08, -.015, 'ELF resource metadata, not measured dynamic spill, occupancy or throughput. No valid hardware counters.', fontsize=9)
fig.tight_layout(rect=(0, 0, 1, .93)); save(fig, 'static_pv_resources')

fig, ax = plt.subplots(figsize=(14, 4))
ax.set_xlim(0, 14); ax.set_ylim(0, 4); ax.axis('off')
boxes = [(.2, 2, 'Codex', 'Design + fixed SHA'), (3.0, 2, 'HeteMCP', 'Task + persistence'),
         (5.8, 2, 'DSH', 'Source artifact'), (8.6, 2, 'Codex / GPU', 'Review + paired A/B'),
         (11.4, 2, 'User / OJ', 'Accept or reject')]
for x, y, title, body in boxes:
    ax.add_patch(FancyBboxPatch((x, y), 2.25, 1.1, boxstyle='round,pad=.12', facecolor='#EDF3FA', edgecolor=BLUE, linewidth=1.5))
    ax.text(x+1.125, y+.72, title, ha='center', weight='bold', fontsize=13, color=BLUE)
    ax.text(x+1.125, y+.30, body, ha='center', fontsize=10)
for i in range(4):
    x=boxes[i][0]+2.4
    ax.annotate('', xy=(x+.35, 2.55), xytext=(x, 2.55), arrowprops={'arrowstyle': '->', 'color': BLUE, 'lw': 1.8})
ax.text(7, 1.1, 'Single mechanism  |  Artifact SHA  |  Original tolerance  |  Failed experiments retained', ha='center', fontsize=13, color=GREEN)
ax.text(7, .35, 'Task success != compilation success != numerical correctness != SSH speedup != OJ improvement', ha='center', fontsize=11)
ax.set_title('Traceable optimization workflow', loc='left', weight='bold', fontsize=17)
save(fig, 'workflow')
print('Rendered 5 charts in PNG and SVG.')
