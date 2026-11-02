"""Figure generation from run artifacts (serdqn-style training monitor).

    python -m ser_slice.monitor --curve runs/*/train.jsonl --labels A1 A2 A3 A4 --out reports/

Implemented here (M0): F1 learning curves, F2 KPI bars, F3 URLLC delay CDF.
F4-F6 (stochasticity scatter, transfer bars, split heatmaps) land in M4 per
DESIGN.md section 7. Requires matplotlib (optional dependency).
"""
import argparse
import csv
import json


def _plt():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except ImportError:
        raise SystemExit(
            "matplotlib is required for monitor.py (optional dependency): pip install matplotlib"
        )


def load_jsonl(path):
    records = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def fig_learning_curves(paths, labels, out_path, key="return_ewma"):
    """F1: per-arm smoothed training return."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for path, label in zip(paths, labels):
        recs = load_jsonl(path)
        ax.plot([r["episode"] for r in recs], [r[key] for r in recs], label=label)
    ax.set_xlabel("episode")
    ax.set_ylabel(key)
    ax.set_title("SER-Slice learning curves (F1)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


def fig_kpi_bars(summary_paths, labels, out_path, kpis=("thr_e_mbps", "urllc_violation_rate", "mmtc_delay_s")):
    """F2: KPI comparison across runs/baselines."""
    plt = _plt()
    fig, axes = plt.subplots(1, len(kpis), figsize=(4 * len(kpis), 4))
    if len(kpis) == 1:
        axes = [axes]
    for ax, kpi in zip(axes, kpis):
        vals = []
        for path in summary_paths:
            with open(path, encoding="utf-8") as fh:
                s = json.load(fh)
            vals.append(s["kpi_mean"].get(kpi, 0.0))
        ax.bar(range(len(vals)), vals)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30)
        ax.set_title(kpi)
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle("KPIs vs baselines (F2)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


def fig_urllc_delay_cdf(csv_paths, labels, out_path):
    """F3: URLLC SLA CDF over the per-step deadline-violation rate."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for path, label in zip(csv_paths, labels):
        with open(path, encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            vals = sorted(float(row["urllc_violation_rate"]) for row in reader)
        if not vals:
            continue
        ys = [(i + 1) / len(vals) for i in range(len(vals))]
        ax.step(vals, ys, where="post", label=label)
    ax.set_xlabel("URLLC deadline violation rate (per step)")
    ax.set_ylabel("CDF")
    ax.set_title("URLLC SLA (F3)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


def main(argv=None):
    p = argparse.ArgumentParser(description="SER-Slice figures")
    p.add_argument("--curve", nargs="*", help="train.jsonl paths -> F1")
    p.add_argument("--labels", nargs="*", help="labels for F1/F2/F3")
    p.add_argument("--summary", nargs="*", help="summary.json paths -> F2")
    p.add_argument("--evalcsv", nargs="*", help="eval.csv paths -> F3")
    p.add_argument("--out", default="reports")
    args = p.parse_args(argv)
    import os

    os.makedirs(args.out, exist_ok=True)
    labels = args.labels or [f"run{i}" for i in range(max(len(args.curve or []), len(args.summary or []), len(args.evalcsv or [])))]
    if args.curve:
        fig_learning_curves(args.curve, labels, os.path.join(args.out, "f1_learning_curves.png"))
    if args.summary:
        fig_kpi_bars(args.summary, labels, os.path.join(args.out, "f2_kpi_bars.png"))
    if args.evalcsv:
        fig_urllc_delay_cdf(args.evalcsv, labels, os.path.join(args.out, "f3_urllc_cdf.png"))


if __name__ == "__main__":
    main()
