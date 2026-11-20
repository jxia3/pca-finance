"""Run from the repository root: python -m scripts.benchmark_pca.

Protocol: use three historical windows per configuration, warm both methods,
alternate execution order, and retain every timed repetition. Include identical
standardization in both timings, but exclude loading, accuracy checks and plots.
Sweep one of stock count, window length and K at a time around (1201, 500, 5).
Check eigenvalues, subspaces and eigenpair residuals against the dense reference.
Timing intervals describe variation across windows, not confidence intervals.
No peak-memory claim is made. BLAS is restricted to one thread before imports.
"""

import os

for name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS',
             'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[name] = '1'

import argparse
import csv
import hashlib
import json
import platform
import subprocess
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.linalg import subspace_angles

from src.data import load_daily_returns
from src.models.pca import correlation_pca, basic_covariance_pca


def dense(x, k):
    z = (x - x.mean(axis=0)) / x.std(axis=0, ddof=1)
    return basic_covariance_pca(z, k)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--output', type=Path, default=Path('report/benchmarks'))
    args = parser.parse_args()
    if args.repeats < 3:
        parser.error('Use at least three repetitions.')
    args.output.mkdir(parents=True, exist_ok=True)
    figures = Path('report/figures')
    figures.mkdir(parents=True, exist_ok=True)
    data = load_daily_returns()
    configs = [(n, 500, 5) for n in (100, 250, 400, 550, 700, 850, 1000, 1201)]
    configs += [(1201, w, 5) for w in (125, 252, 375, 625, 750, 875, 1000)]
    configs += [(1201, 500, k) for k in (1, 2, 3, 8, 10, 15, 20, 30)]
    ends = np.linspace(1000, len(data.dates), 3, dtype=int)
    stock_order = np.random.default_rng(1750).permutation(data.returns.shape[1])
    rows, checks = [], []
    methods = {'dense': dense, 'iterative': correlation_pca}
    for n, window, k in configs:
        for end in ends:
            x = np.ascontiguousarray(data.returns[end-window:end, stock_order[:n]])
            assert np.isfinite(x).all() and (x.std(axis=0) > 0).all()
            outputs = {name: fn(x, k) for name, fn in methods.items()}
            reference = outputs['dense']
            z = (x - x.mean(axis=0)) / x.std(axis=0, ddof=1)
            for repeat in range(args.repeats):
                order = ('dense', 'iterative') if repeat % 2 == 0 else ('iterative', 'dense')
                for name in order:
                    start = perf_counter()
                    result = methods[name](x, k)
                    elapsed = perf_counter() - start
                    rows.append(dict(stocks=n, window=window, k=k,
                                     end=str(data.dates[end-1]), repeat=repeat,
                                     method=name, seconds=elapsed))
                    if name == 'iterative':
                        u, ev = result.components, result.eigenvalues
                        error = np.max(np.abs(ev-reference.eigenvalues) / reference.eigenvalues)
                        angle = np.max(subspace_angles(reference.components, u))
                        residual = np.max(np.linalg.norm(z.T @ (z @ u)/(window-1)-u*ev, axis=0)/ev)
                        checks.append(dict(stocks=n, window=window, k=k, end=str(data.dates[end-1]),
                                           repeat=repeat, eigenvalue_error=float(error),
                                           angle_radians=float(angle), relative_residual=float(residual)))
                        assert error < 1e-8 and angle < 1e-6 and residual < 1e-8
        print(f'Completed N={n}, window={window}, K={k}', flush=True)
    for name, records in [('timings', rows), ('accuracy', checks)]:
        with (args.output / f'{name}.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    summary = []
    for n, w, k in configs:
        values = {}
        for method in methods:
            values[method] = [np.median([r['seconds'] for r in rows
                if (r['stocks'], r['window'], r['k'], r['method'], r['end']) ==
                (n, w, k, method, str(data.dates[end-1]))]) for end in ends]
        summary.append(dict(stocks=n, window=w, k=k,
            dense_seconds=float(np.median(values['dense'])),
            iterative_seconds=float(np.median(values['iterative'])),
            paired_speedup=float(np.median(np.array(values['dense'])/values['iterative'])),
            window_medians=values))
    metadata = dict(platform=platform.platform(), processor=platform.processor(),
        python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
        matplotlib=matplotlib.__version__, repeats=args.repeats, windows=3,
        thread_environment={n:os.environ[n] for n in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS')},
        numpy_config=np.show_config(mode='dicts'),
        data_sha256=hashlib.sha256(Path('materials/daily_returns.csv').read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path('src/models/pca.py').read_bytes()).hexdigest(),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
        seed=1750, eigensolver_tolerance=1e-10,
        window_end_dates=[str(data.dates[end-1]) for end in ends],
        acceptance_limits=dict(eigenvalue_error=1e-8, angle_radians=1e-6, relative_residual=1e-8),
        note='Stock sampling is seeded. The production eigensolver uses its default starting vector. Timings are machine-specific.',
        summary=summary, max_eigenvalue_error=max(c['eigenvalue_error'] for c in checks),
        max_angle_radians=max(c['angle_radians'] for c in checks),
        max_relative_residual=max(c['relative_residual'] for c in checks))
    (args.output/'results.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False})
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.1), layout='constrained')
    for ax, key, label, select in zip(axes, ('stocks','window','k'),
        ('Stocks', 'Window (days)', 'Factors K'),
        (lambda r:r['window']==500 and r['k']==5,
         lambda r:r['stocks']==1201 and r['k']==5,
         lambda r:r['stocks']==1201 and r['window']==500)):
        group = sorted(filter(select, summary), key=lambda r:r[key])
        for method, color in [('dense','#a45336'), ('iterative','#245f83')]:
            y = np.array([r[f'{method}_seconds']*1000 for r in group])
            lo = np.array([min(r['window_medians'][method])*1000 for r in group])
            hi = np.array([max(r['window_medians'][method])*1000 for r in group])
            ax.plot([r[key] for r in group], y, marker='o', markersize=3,
                    lw=1.5, color=color, label=method.capitalize())
            ax.fill_between([r[key] for r in group], lo, hi, color=color, alpha=.12, linewidth=0)
        ax.set(xlabel=label, yscale='log')
        ax.grid(axis='y', alpha=.2)
    axes[0].set_ylabel('Time per fit (ms)')
    for ax, title in zip(axes, ('500 days, K = 5', '1,201 stocks, K = 5', '1,201 stocks, 500 days')):
        ax.set_title(title, fontsize=10, pad=12)
    axes[0].legend(frameon=False)
    for ext in ('pdf','png'):
        fig.savefig(figures/f'pca_benchmark.{ext}', dpi=200)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3), layout='constrained')
    ks = sorted({r['k'] for r in checks if r['stocks']==1201 and r['window']==500})
    for ax, metric, label in zip(axes,
        ('eigenvalue_error','angle_radians','relative_residual'),
        ('Relative eigenvalue error','Largest subspace angle (rad)','Relative eigenpair residual')):
        maxima = [max(r[metric] for r in checks if r['stocks']==1201 and r['window']==500 and r['k']==k) for k in ks]
        ax.plot(ks, maxima, 'o-', color='#245f83', markersize=3, lw=1.5)
        ax.set(xlabel='Factors K', yscale='log')
        ax.set_title(label, fontsize=10, pad=12)
        ax.grid(axis='y', alpha=.2)
    for ext in ('pdf','png'):
        fig.savefig(figures/f'pca_agreement.{ext}', dpi=220)
    plt.close(fig)
    print(json.dumps({k:metadata[k] for k in ('max_eigenvalue_error','max_angle_radians','max_relative_residual')}, indent=2))
    print(json.dumps(next(r for r in summary if (r['stocks'],r['window'],r['k'])==(1201,500,5)), indent=2))


if __name__ == '__main__':
    main()
