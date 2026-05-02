#!/usr/bin/env python3
"""Usage: scripts/plot_resource.py /tmp/handy_resource_<pid>.csv [output.png]"""
import sys, csv
import matplotlib.pyplot as plt

csv_path = sys.argv[1]
out_path = sys.argv[2] if len(sys.argv) > 2 else csv_path.replace('.csv', '.png')

ts, rss, cpu = [], [], []
with open(csv_path) as f:
    r = csv.DictReader(f)
    for row in r:
        ts.append(int(row['ts_unix']))
        rss.append(int(row['rss_kb']) / 1024)  # MB
        cpu.append(float(row['cpu_pct']))

if not ts:
    print("Empty CSV", file=sys.stderr)
    sys.exit(1)

t0 = ts[0]
relt = [t - t0 for t in ts]

fig, ax1 = plt.subplots(figsize=(12, 4))
ax1.plot(relt, rss, 'b-', label='RSS (MB)')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('RSS (MB)', color='b')

ax2 = ax1.twinx()
ax2.plot(relt, cpu, 'r-', label='CPU %', alpha=0.7)
ax2.set_ylabel('CPU %', color='r')

plt.title(f'Resource usage: {csv_path}')
plt.tight_layout()
plt.savefig(out_path, dpi=100)
print(f'Saved: {out_path}')
