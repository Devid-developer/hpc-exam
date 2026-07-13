#!/usr/bin/env python3
"""Generate presentation-ready plots for the stencil benchmark campaign."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter


CSV_NAMES = {
    "template": "go_template_serial.csv",
    "final": "go_final_serial.csv",
    "openmp": "go_omp_serial.csv",
    "mpi_strong": "go_mpi_strong.csv",
    "mpi_weak": "go_mpi_weak.csv",
    "hybrid": "go_mpi_hybrid.csv",
    "hybrid_weak": "go_mpi_hybrid_weak.csv",
    "cache": "go_cache_serial.csv",
}

# Muted, colour-blind-friendly palette suited to slides and print.
NAVY = "#183B56"
BLUE = "#2F6B9A"
LIGHT_BLUE = "#6FA8C9"
TEAL = "#2A8C82"
ORANGE = "#D47A3A"
CORAL = "#C95D63"
CHARCOAL = "#252B33"
SLATE = "#687684"
LIGHT_SLATE = "#A8B3BD"
GRID = "#DCE3E8"
PANEL = "#F7F9FA"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate all stencil benchmark plots from the go_*.csv files."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="directory searched recursively for CSV files (default: results)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots"),
        help="directory in which figures are written (default: plots)",
    )
    parser.add_argument(
        "--format",
        choices=("png", "pdf", "svg"),
        default="png",
        help="figure format (default: png)",
    )
    parser.add_argument(
        "--dpi", type=int, default=300, help="resolution for raster output (default: 300)"
    )
    return parser.parse_args()


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 15,
            "axes.titleweight": "bold",
            "axes.labelsize": 11.5,
            "axes.labelcolor": CHARCOAL,
            "axes.edgecolor": LIGHT_SLATE,
            "axes.linewidth": 0.8,
            "axes.facecolor": PANEL,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": CHARCOAL,
            "ytick.color": CHARCOAL,
            "legend.fontsize": 9.5,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "grid.alpha": 0.9,
            "lines.linewidth": 2.3,
            "lines.markersize": 6.5,
        }
    )


def job_number(path: Path) -> int:
    numbers = []
    for part in path.parts:
        match = re.match(r"(\d+)_", part)
        if match:
            numbers.append(int(match.group(1)))
    return max(numbers, default=-1)


def discover_csvs(results_dir: Path) -> dict[str, Path]:
    selected: dict[str, Path] = {}
    for key, filename in CSV_NAMES.items():
        candidates = list(results_dir.rglob(filename)) if results_dir.exists() else []
        if not candidates and Path(filename).is_file():
            candidates = [Path(filename)]
        if not candidates:
            print(f"[missing] {filename}")
            continue
        candidates.sort(key=lambda path: (job_number(path), path.stat().st_mtime, str(path)))
        selected[key] = candidates[-1]
        print(f"[use] {key:12s}: {candidates[-1]}")
        if len(candidates) > 1:
            print(f"      ({len(candidates)} candidates found; selected the newest job)")
    return selected


def load_data(paths: dict[str, Path]) -> dict[str, pd.DataFrame]:
    data: dict[str, pd.DataFrame] = {}
    for key, path in paths.items():
        frame = pd.read_csv(path, skipinitialspace=True)
        frame.columns = frame.columns.str.strip()
        if frame.empty:
            print(f"[warning] empty CSV: {path}")
            continue
        data[key] = frame
    return data


def rep01(frame: pd.DataFrame) -> pd.DataFrame:
    """Use the first measured repetition consistently, without aggregation."""
    if "run_name" not in frame:
        return frame.copy()
    first = frame[frame["run_name"].str.endswith("_rep01")]
    return first.copy() if not first.empty else frame.copy()


def integer_formatter(value: float, _position: int) -> str:
    return f"{value:g}"


class Plotter:
    def __init__(self, data: dict[str, pd.DataFrame], output: Path, fmt: str, dpi: int):
        self.data = data
        self.output = output
        self.format = fmt
        self.dpi = dpi
        self.created: list[Path] = []
        output.mkdir(parents=True, exist_ok=True)

    def save(self, fig: plt.Figure, name: str) -> None:
        path = self.output / f"{name}.{self.format}"
        fig.tight_layout()
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        self.created.append(path)

    @staticmethod
    def finish_axis(ax: plt.Axes, *, grid_axis: str = "y") -> None:
        ax.grid(True, axis=grid_axis)
        ax.set_axisbelow(True)
        ax.margins(x=0.04)

    @staticmethod
    def log2_x(ax: plt.Axes, values: pd.Series | np.ndarray) -> None:
        ticks = sorted({int(value) for value in values})
        ax.set_xscale("log", base=2)
        ax.set_xticks(ticks)
        ax.xaxis.set_major_formatter(FuncFormatter(integer_formatter))
        ax.xaxis.set_minor_formatter(NullFormatter())

    @staticmethod
    def log2_y(ax: plt.Axes, values: pd.Series | np.ndarray) -> None:
        maximum = float(np.nanmax(np.asarray(values, dtype=float)))
        upper = 2 ** np.ceil(np.log2(max(maximum, 1.0)))
        ticks = 2.0 ** np.arange(0, int(np.log2(upper)) + 1)
        ax.set_yscale("log", base=2)
        ax.set_yticks(ticks)
        ax.yaxis.set_major_formatter(FuncFormatter(integer_formatter))
        ax.yaxis.set_minor_locator(LogLocator(base=2, subs=()))
        ax.yaxis.set_minor_formatter(NullFormatter())

    @staticmethod
    def value_labels(ax: plt.Axes, x: pd.Series, y: pd.Series, decimals: int = 1) -> None:
        for x_value, y_value in zip(x, y):
            ax.annotate(
                f"{y_value:.{decimals}f}",
                (x_value, y_value),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                color=SLATE,
            )

    @staticmethod
    def bar_labels(ax: plt.Axes, decimals: int = 2) -> None:
        for container in ax.containers:
            if hasattr(container, "datavalues"):
                ax.bar_label(
                    container,
                    fmt=f"%.{decimals}f",
                    padding=4,
                    fontsize=9,
                    color=CHARCOAL,
                )

    def serial_comparison(self) -> None:
        if not {"template", "final", "openmp"}.issubset(self.data):
            return

        values: list[tuple[str, float]] = []
        for key, family in (("template", "Template"), ("final", "Final")):
            frame = rep01(self.data[key])
            frame["optimization"] = frame["run_name"].str.extract(r"_(O[13])_")
            for optimization in ("O1", "O3"):
                row = frame[frame["optimization"] == optimization]
                if not row.empty:
                    values.append((f"{family}\n-{optimization}", float(row.iloc[0]["t_wall"])))

        omp = rep01(self.data["openmp"])
        one_thread = omp[omp["run_name"].str.contains(r"strong_spread_t0*1_", regex=True)]
        if not one_thread.empty:
            values.append(("OpenMP -O3\n1 thread", float(one_thread.iloc[0]["t_wall"])))
        if not values:
            return

        labels, times = zip(*values)
        colors = [SLATE, LIGHT_SLATE, BLUE, NAVY, TEAL][: len(values)]
        fig, ax = plt.subplots(figsize=(9.3, 5.4))
        ax.bar(labels, times, color=colors, width=0.68)
        ax.set_ylabel("Wall-clock time [s]")
        ax.set_title("Serial implementations — large working set", loc="left", pad=14)
        ax.text(
            0.0,
            1.01,
            "25,000 × 25,000 grid · 200 iterations · rep01",
            transform=ax.transAxes,
            color=SLATE,
            fontsize=10,
        )
        self.bar_labels(ax, 1)
        self.finish_axis(ax)
        self.save(fig, "01_serial_large_runtime")

    def cache_comparison(self) -> None:
        if "cache" not in self.data:
            return
        frame = rep01(self.data["cache"])
        frame["label"] = frame.apply(
            lambda row: (
                "OpenMP -O3\n1 thread"
                if row["variant"] == "openmp"
                else f"{str(row['variant']).capitalize()}\n-{row['optimization']}"
            ),
            axis=1,
        )
        order = ["Template\n-O1", "Template\n-O3", "Final\n-O1", "Final\n-O3", "OpenMP -O3\n1 thread"]
        frame = frame.set_index("label").reindex(order).dropna(subset=["t_wall"])
        colors = [SLATE, LIGHT_SLATE, BLUE, NAVY, TEAL][: len(frame)]

        fig, ax = plt.subplots(figsize=(9.3, 5.4))
        ax.bar(frame.index, frame["t_wall"], color=colors, width=0.68)
        ax.set_ylabel("Wall-clock time [s]")
        ax.set_title("Serial implementations — cache-resident workload", loc="left", pad=14)
        ax.text(
            0.0,
            1.01,
            "1,000 × 1,000 grid · 10,000 iterations · rep01",
            transform=ax.transAxes,
            color=SLATE,
            fontsize=10,
        )
        self.bar_labels(ax, 1)
        self.finish_axis(ax)
        self.save(fig, "02_serial_cache_runtime")

    def runtime_breakdowns(self) -> None:
        if "cache" not in self.data:
            return
        frame = rep01(self.data["cache"])
        for variant, label, name, color in (
            ("template", "Template -O3", "03_template_o3_runtime_breakdown", ORANGE),
            ("final", "Final -O3", "04_final_o3_runtime_breakdown", BLUE),
        ):
            row = frame[(frame["variant"] == variant) & (frame["optimization"] == "O3")]
            if row.empty:
                continue
            row = row.iloc[0]
            functions = ["update_plane()", "get_total_energy()", "inject_energy()"]
            times = [
                float(row["t_update_plane"]),
                float(row["t_get_total_energy"]),
                float(row["t_inject_energy"]),
            ]
            wall = float(row["t_wall"])

            fig, ax = plt.subplots(figsize=(9.3, 5.1))
            bars = ax.barh(
                functions,
                times,
                color=[color, LIGHT_BLUE, LIGHT_SLATE],
                height=0.58,
            )
            ax.set_xscale("log")
            ax.axvline(
                wall,
                color=CHARCOAL,
                linestyle=(0, (5, 4)),
                linewidth=1.7,
                label=f"Wall time: {wall:.3f} s",
            )
            for bar, elapsed in zip(bars, times):
                share = 100.0 * elapsed / wall
                ax.annotate(
                    f"{elapsed:.6f} s  ({share:.2f}% of wall time)",
                    (elapsed, bar.get_y() + bar.get_height() / 2),
                    xytext=(7, 0),
                    textcoords="offset points",
                    va="center",
                    fontsize=9,
                    color=CHARCOAL,
                )
            ax.invert_yaxis()
            ax.set_xlabel("Accumulated time [s] — logarithmic scale")
            ax.set_title(f"{label} — runtime breakdown", loc="left", pad=14)
            ax.text(
                0.0,
                1.01,
                "1,000 × 1,000 grid · 10,000 iterations · rep01",
                transform=ax.transAxes,
                color=SLATE,
                fontsize=10,
            )
            ax.legend(loc="lower right")
            self.finish_axis(ax, grid_axis="x")
            self.save(fig, name)

    @staticmethod
    def strong_metrics(frame: pd.DataFrame, resources: str, group: str | None = None) -> pd.DataFrame:
        frame = frame.sort_values(([group] if group else []) + [resources]).copy()
        if group:
            baselines = frame[frame[resources] == 1].set_index(group)["t_wall"]
            frame["speedup"] = frame.apply(
                lambda row: baselines.loc[row[group]] / row["t_wall"], axis=1
            )
        else:
            baseline = float(frame.iloc[0]["t_wall"])
            frame["speedup"] = baseline / frame["t_wall"]
        frame["efficiency"] = 100.0 * frame["speedup"] / frame[resources]
        return frame

    def strong_figure(
        self,
        frame: pd.DataFrame,
        resource: str,
        resource_label: str,
        title: str,
        name: str,
        group: str | None = None,
        styles: dict[str, tuple[str, str]] | None = None,
    ) -> None:
        frame = self.strong_metrics(frame, resource, group)
        fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2))
        ideal_x = np.asarray(sorted(frame[resource].unique()), dtype=float)

        groups = frame.groupby(group, sort=False) if group else [("Measured", frame)]
        for group_name, part in groups:
            part = part.sort_values(resource)
            color, marker = (styles or {}).get(str(group_name), (BLUE, "o"))
            legend = str(group_name).capitalize() if group else "Measured"
            axes[0].plot(part[resource], part["speedup"], color=color, marker=marker, label=legend)
            axes[1].plot(part[resource], part["efficiency"], color=color, marker=marker, label=legend)

        axes[0].plot(
            ideal_x,
            ideal_x,
            color=CHARCOAL,
            linestyle=(0, (4, 3)),
            linewidth=1.6,
            label="Ideal",
            zorder=1,
        )
        axes[1].axhline(
            100.0,
            color=CHARCOAL,
            linestyle=(0, (4, 3)),
            linewidth=1.6,
            label="Ideal",
            zorder=1,
        )

        self.log2_x(axes[0], frame[resource])
        self.log2_x(axes[1], frame[resource])
        self.log2_y(axes[0], np.append(frame["speedup"].to_numpy(), ideal_x))
        axes[0].set_xlabel(resource_label)
        axes[0].set_ylabel("Speedup")
        axes[0].set_title("Speedup", loc="left")
        axes[1].set_xlabel(resource_label)
        axes[1].set_ylabel("Parallel efficiency [%]")
        axes[1].set_title("Efficiency", loc="left")
        axes[1].set_ylim(bottom=0)
        for ax in axes:
            ax.legend(loc="best")
            self.finish_axis(ax, grid_axis="both")
        fig.suptitle(title, x=0.06, ha="left", fontweight="bold", fontsize=16)
        fig.subplots_adjust(top=0.82)
        self.save(fig, name)

    def weak_figure(
        self,
        frame: pd.DataFrame,
        resource: str,
        resource_label: str,
        title: str,
        name: str,
    ) -> None:
        frame = frame.sort_values(resource).copy()
        baseline = float(frame.iloc[0]["t_wall"])
        frame["efficiency"] = 100.0 * baseline / frame["t_wall"]

        fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2))
        axes[0].plot(
            frame[resource], frame["t_wall"], color=BLUE, marker="o", label="Measured"
        )
        axes[0].axhline(
            baseline,
            color=CHARCOAL,
            linestyle=(0, (4, 3)),
            linewidth=1.6,
            label="Ideal",
        )
        axes[1].plot(
            frame[resource], frame["efficiency"], color=TEAL, marker="o", label="Measured"
        )
        axes[1].axhline(
            100.0,
            color=CHARCOAL,
            linestyle=(0, (4, 3)),
            linewidth=1.6,
            label="Ideal",
        )

        self.log2_x(axes[0], frame[resource])
        self.log2_x(axes[1], frame[resource])
        axes[0].set_xlabel(resource_label)
        axes[0].set_ylabel("Wall-clock time [s]")
        axes[0].set_title("Runtime", loc="left")
        axes[1].set_xlabel(resource_label)
        axes[1].set_ylabel("Weak-scaling efficiency [%]")
        axes[1].set_title("Efficiency", loc="left")
        axes[1].set_ylim(bottom=0)
        for ax in axes:
            ax.legend(loc="best")
            self.finish_axis(ax, grid_axis="both")
        fig.suptitle(title, x=0.06, ha="left", fontweight="bold", fontsize=16)
        fig.subplots_adjust(top=0.82)
        self.save(fig, name)

    def openmp(self) -> None:
        if "openmp" not in self.data:
            return
        frame = rep01(self.data["openmp"])
        strong = frame[frame["run_name"].str.startswith("strong_")].copy()
        strong[["binding", "threads"]] = strong["run_name"].str.extract(
            r"strong_(close|spread)_t(\d+)_"
        )
        strong = strong.dropna(subset=["threads"])
        strong["threads"] = strong["threads"].astype(int)
        if not strong.empty:
            self.strong_figure(
                strong,
                "threads",
                "OpenMP threads",
                "OpenMP strong scaling — thread placement",
                "05_openmp_strong_scaling",
                group="binding",
                styles={"spread": (CORAL, "o"), "close": (BLUE, "s")},
            )

        weak = frame[frame["run_name"].str.startswith("weak_")].copy()
        weak["threads"] = pd.to_numeric(weak["run_name"].str.extract(r"_t(\d+)_")[0])
        weak = weak.dropna(subset=["threads"])
        if not weak.empty:
            self.weak_figure(
                weak,
                "threads",
                "OpenMP threads",
                "OpenMP weak scaling — 25 million cells per thread",
                "06_openmp_weak_scaling",
            )

    def mpi(self) -> None:
        if "mpi_strong" in self.data:
            strong = rep01(self.data["mpi_strong"]).sort_values("tasks")
            self.strong_figure(
                strong,
                "tasks",
                "MPI ranks",
                "MPI strong scaling — fixed 25,000 × 25,000 grid",
                "07_mpi_strong_scaling",
            )
        if "mpi_weak" in self.data:
            weak = rep01(self.data["mpi_weak"]).sort_values("tasks")
            self.weak_figure(
                weak,
                "tasks",
                "MPI ranks",
                "MPI weak scaling — 25 million cells per rank",
                "08_mpi_weak_scaling",
            )

    def openmp_mpi_comparison(self) -> None:
        if not {"openmp", "mpi_strong"}.issubset(self.data):
            return
        omp = rep01(self.data["openmp"])
        omp = omp[omp["run_name"].str.startswith("strong_")].copy()
        omp[["binding", "cores"]] = omp["run_name"].str.extract(
            r"strong_(close|spread)_t(\d+)_"
        )
        omp["cores"] = pd.to_numeric(omp["cores"])
        omp = omp.dropna(subset=["cores"])
        omp = self.strong_metrics(omp, "cores", "binding")

        mpi = rep01(self.data["mpi_strong"])
        mpi = mpi[mpi["nodes"] == 1].copy().rename(columns={"tasks": "cores"})
        mpi = self.strong_metrics(mpi, "cores")
        resources = np.asarray(sorted(set(omp["cores"]) | set(mpi["cores"])), dtype=float)

        fig, ax = plt.subplots(figsize=(9.3, 5.5))
        for binding, part in omp.groupby("binding", sort=False):
            style = {"spread": (CORAL, "o"), "close": (LIGHT_BLUE, "s")}[binding]
            ax.plot(
                part["cores"],
                part["speedup"],
                color=style[0],
                marker=style[1],
                label=f"OpenMP {binding}",
            )
        ax.plot(mpi["cores"], mpi["speedup"], color=NAVY, marker="D", label="MPI")
        ax.plot(
            resources,
            resources,
            color=CHARCOAL,
            linestyle=(0, (4, 3)),
            linewidth=1.6,
            label="Ideal",
        )
        self.log2_x(ax, resources)
        self.log2_y(ax, resources)
        ax.set_xlabel("CPU cores on one node")
        ax.set_ylabel("Speedup")
        ax.set_title("OpenMP vs MPI — single-node strong scaling", loc="left", pad=14)
        ax.legend(loc="upper left")
        self.finish_axis(ax, grid_axis="both")
        self.save(fig, "09_openmp_vs_mpi_strong_scaling")

    @staticmethod
    def pair_hybrid_nodes(frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.copy()
        if "ranks_per_node" not in frame:
            frame["ranks_per_node"] = frame["ranks"] // frame["nodes"]
        keys = ["ranks_per_node", "threads_per_rank"]
        one = frame[frame["nodes"] == 1].set_index(keys)["t_wall"].rename("time_1")
        two = frame[frame["nodes"] == 2].set_index(keys)["t_wall"].rename("time_2")
        return pd.concat([one, two], axis=1).dropna().reset_index().sort_values("ranks_per_node")

    def hybrid_figure(self, frame: pd.DataFrame, weak: bool, name: str) -> None:
        paired = self.pair_hybrid_nodes(rep01(frame))
        if paired.empty:
            return
        labels = [
            f"{int(ranks)}×{int(threads)}"
            for ranks, threads in zip(paired["ranks_per_node"], paired["threads_per_rank"])
        ]
        x = np.arange(len(labels))

        fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2))
        axes[0].plot(x, paired["time_1"], color=BLUE, marker="o", label="1 node")
        axes[0].plot(x, paired["time_2"], color=TEAL, marker="s", label="2 nodes")
        if weak:
            ideal_time = paired["time_1"]
            metric = 100.0 * paired["time_1"] / paired["time_2"]
            ideal_metric = 100.0
            metric_label = "Weak-scaling efficiency [%]"
            title = "Hybrid MPI+OpenMP weak scaling"
            ideal_label = "Ideal 2-node runtime = 1-node runtime"
        else:
            ideal_time = paired["time_1"] / 2.0
            metric = paired["time_1"] / paired["time_2"]
            ideal_metric = 2.0
            metric_label = "Speedup from 1 to 2 nodes"
            title = "Hybrid MPI+OpenMP strong scaling"
            ideal_label = "Ideal 2-node runtime"
        axes[0].plot(
            x,
            ideal_time,
            color=CHARCOAL,
            linestyle=(0, (4, 3)),
            linewidth=1.6,
            label=ideal_label,
        )
        axes[1].plot(x, metric, color=CORAL, marker="o", label="Measured")
        axes[1].axhline(
            ideal_metric,
            color=CHARCOAL,
            linestyle=(0, (4, 3)),
            linewidth=1.6,
            label="Ideal",
        )
        axes[0].set_ylabel("Wall-clock time [s]")
        axes[0].set_title("Runtime", loc="left")
        axes[1].set_ylabel(metric_label)
        axes[1].set_title("Scaling quality", loc="left")
        for ax in axes:
            ax.set_xticks(x, labels)
            ax.set_xlabel("MPI ranks per node × OpenMP threads per rank")
            ax.legend(loc="best")
            self.finish_axis(ax)
        self.value_labels(axes[1], pd.Series(x), metric, 2)
        fig.suptitle(title, x=0.06, ha="left", fontweight="bold", fontsize=16)
        fig.subplots_adjust(top=0.82)
        self.save(fig, name)

    def hybrid(self) -> None:
        if "hybrid" in self.data:
            self.hybrid_figure(
                self.data["hybrid"], weak=False, name="10_hybrid_strong_scaling"
            )
        if "hybrid_weak" in self.data:
            self.hybrid_figure(
                self.data["hybrid_weak"], weak=True, name="11_hybrid_weak_scaling"
            )

    def run_all(self) -> None:
        self.serial_comparison()
        self.cache_comparison()
        self.runtime_breakdowns()
        self.openmp()
        self.mpi()
        self.openmp_mpi_comparison()
        self.hybrid()


def main() -> None:
    args = parse_args()
    configure_style()
    paths = discover_csvs(args.results_dir)
    data = load_data(paths)
    if not data:
        raise SystemExit(
            f"No benchmark CSV found below {args.results_dir}. "
            "Copy the result directories locally or pass --results-dir."
        )
    plotter = Plotter(data, args.output_dir, args.format, args.dpi)
    plotter.run_all()
    print(f"\nGenerated {len(plotter.created)} figures in {args.output_dir}:")
    for path in plotter.created:
        print(f"  {path}")


if __name__ == "__main__":
    main()
