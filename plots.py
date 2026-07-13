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

COLORS = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "grey": "#666666",
}


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
        "--dpi", type=int, default=250, help="resolution for raster output (default: 250)"
    )
    return parser.parse_args()


def configure_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 15,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "figure.titlesize": 16,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.alpha": 0.28,
            "lines.linewidth": 2.2,
            "lines.markersize": 7,
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
        candidates.sort(key=lambda p: (job_number(p), p.stat().st_mtime, str(p)))
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
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        self.created.append(path)

    @staticmethod
    def powers_of_two_axis(ax: plt.Axes, values: pd.Series | np.ndarray) -> None:
        ticks = sorted({int(value) for value in values})
        ax.set_xscale("log", base=2)
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(tick) for tick in ticks])

    @staticmethod
    def annotate_bars(ax: plt.Axes, decimals: int = 2) -> None:
        for container in ax.containers:
            if hasattr(container, "datavalues"):
                ax.bar_label(container, fmt=f"%.{decimals}f", padding=3, fontsize=9)

    def serial_large(self) -> None:
        if not {"template", "final", "openmp"}.issubset(self.data):
            return

        rows = []
        for key, prefix in (("template", "Template"), ("final", "Final")):
            frame = self.data[key].copy()
            frame["optimization"] = frame["run_name"].str.extract(r"_(O[13])_")
            for optimization in ("O1", "O3"):
                part = frame[frame["optimization"] == optimization]
                if not part.empty:
                    rows.append(
                        (f"{prefix} {optimization}", part["t_wall"].median(), part["glups"].median())
                    )

        omp = self.data["openmp"]
        omp_one = omp[omp["run_name"].str.contains(r"strong_spread_t0*1_", regex=True)]
        if omp_one.empty:
            omp_one = omp[omp["run_name"].str.contains(r"strong_.*_t0*1_", regex=True)]
        if not omp_one.empty:
            rows.append(("OpenMP O3\n1 thread", omp_one["t_wall"].median(), omp_one["glups"].median()))
        if not rows:
            return

        frame = pd.DataFrame(rows, columns=("label", "t_wall", "glups"))
        colors = [COLORS["grey"], COLORS["sky"], COLORS["orange"], COLORS["red"], COLORS["green"]]
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
        axes[0].bar(frame["label"], frame["t_wall"], color=colors[: len(frame)])
        axes[0].set_ylabel("Tempo totale [s]")
        axes[0].set_title("Tempo di esecuzione")
        axes[1].bar(frame["label"], frame["glups"], color=colors[: len(frame)])
        axes[1].set_ylabel("GLUP/s")
        axes[1].set_title("Throughput")
        for ax in axes:
            ax.tick_params(axis="x", rotation=18)
            self.annotate_bars(ax, 3)
        fig.suptitle("Confronto seriale — griglia 25000×25000")
        self.save(fig, "01_serial_large")

    def serial_cache(self) -> None:
        if "cache" not in self.data:
            return
        frame = self.data["cache"].copy()
        frame["label"] = frame.apply(
            lambda row: (
                "OpenMP O3\n1 thread"
                if row["variant"] == "openmp"
                else f"{str(row['variant']).capitalize()} {row['optimization']}"
            ),
            axis=1,
        )
        order = ["Template O1", "Template O3", "Final O1", "Final O3", "OpenMP O3\n1 thread"]
        grouped = frame.groupby("label", sort=False)
        medians = grouped[["t_wall", "glups"]].median().reindex(order).dropna()
        minima = grouped[["t_wall", "glups"]].min().reindex(medians.index)
        maxima = grouped[["t_wall", "glups"]].max().reindex(medians.index)
        colors = [COLORS["grey"], COLORS["sky"], COLORS["orange"], COLORS["red"], COLORS["green"]]

        fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
        for ax, metric, ylabel, title in (
            (axes[0], "t_wall", "Tempo totale [s]", "Tempo di esecuzione"),
            (axes[1], "glups", "GLUP/s", "Throughput"),
        ):
            values = medians[metric]
            errors = np.vstack((values - minima[metric], maxima[metric] - values))
            ax.bar(values.index, values, yerr=errors, capsize=5, color=colors[: len(values)])
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.tick_params(axis="x", rotation=18)
            self.annotate_bars(ax, 3)
        grid_x = int(frame["grid_x"].iloc[0])
        grid_y = int(frame["grid_y"].iloc[0])
        fig.suptitle(f"Effetto della cache — griglia {grid_x}×{grid_y} (mediana, min–max)")
        self.save(fig, "02_serial_cache")

    def openmp(self) -> None:
        if "openmp" not in self.data:
            return
        frame = self.data["openmp"].copy()
        strong = frame[frame["run_name"].str.startswith("strong_")].copy()
        strong[["binding", "threads"]] = strong["run_name"].str.extract(
            r"strong_(close|spread)_t(\d+)_"
        )
        strong = strong.dropna(subset=["threads"])
        strong["threads"] = strong["threads"].astype(int)
        strong = strong.groupby(["binding", "threads"], as_index=False).median(numeric_only=True)
        if not strong.empty:
            baselines = strong[strong["threads"] == 1].set_index("binding")["t_wall"]
            strong["speedup"] = strong.apply(lambda row: baselines[row["binding"]] / row["t_wall"], axis=1)
            strong["efficiency"] = 100.0 * strong["speedup"] / strong["threads"]
            self.scaling_lines(
                strong,
                group="binding",
                x="threads",
                y="t_wall",
                title="OpenMP strong scaling — tempo",
                xlabel="Thread OpenMP",
                ylabel="Tempo totale [s]",
                name="03_openmp_strong_time",
            )
            self.scaling_lines(
                strong,
                group="binding",
                x="threads",
                y="speedup",
                title="OpenMP strong scaling — speedup",
                xlabel="Thread OpenMP",
                ylabel="Speedup",
                name="04_openmp_strong_speedup",
                ideal="speedup",
            )
            self.scaling_lines(
                strong,
                group="binding",
                x="threads",
                y="efficiency",
                title="OpenMP strong scaling — efficienza",
                xlabel="Thread OpenMP",
                ylabel="Efficienza [%]",
                name="05_openmp_strong_efficiency",
                ideal="efficiency",
            )

        weak = frame[frame["run_name"].str.startswith("weak_")].copy()
        weak["threads"] = pd.to_numeric(weak["run_name"].str.extract(r"_t(\d+)_")[0])
        weak = weak.dropna(subset=["threads"]).groupby("threads", as_index=False).median(numeric_only=True)
        if not weak.empty:
            weak["efficiency"] = 100.0 * weak.iloc[0]["t_wall"] / weak["t_wall"]
            self.single_scaling(
                weak, "threads", "t_wall", "OpenMP weak scaling — tempo", "Thread OpenMP",
                "Tempo totale [s]", "06_openmp_weak_time"
            )
            self.single_scaling(
                weak, "threads", "efficiency", "OpenMP weak scaling — efficienza",
                "Thread OpenMP", "Efficienza [%]", "07_openmp_weak_efficiency", ideal="efficiency"
            )

    def mpi(self) -> None:
        if "mpi_strong" in self.data:
            strong = self.data["mpi_strong"].copy().sort_values("tasks")
            strong = strong.groupby("tasks", as_index=False).median(numeric_only=True)
            baseline = strong.iloc[0]["t_wall"]
            strong["speedup"] = baseline / strong["t_wall"]
            strong["efficiency"] = 100.0 * strong["speedup"] / strong["tasks"]
            strong["comm_percent"] = 100.0 * strong["t_exchange_halos"] / strong["t_wall"]
            self.single_scaling(
                strong, "tasks", "t_wall", "MPI strong scaling — tempo", "Processi MPI",
                "Tempo totale [s]", "08_mpi_strong_time"
            )
            self.single_scaling(
                strong, "tasks", "speedup", "MPI strong scaling — speedup", "Processi MPI",
                "Speedup", "09_mpi_strong_speedup", ideal="speedup"
            )
            self.single_scaling(
                strong, "tasks", "efficiency", "MPI strong scaling — efficienza", "Processi MPI",
                "Efficienza [%]", "10_mpi_strong_efficiency", ideal="efficiency"
            )
            self.single_scaling(
                strong, "tasks", "comm_percent", "MPI strong scaling — comunicazione",
                "Processi MPI", "t_exchange_halos / t_wall [%]", "11_mpi_strong_communication"
            )

        if "mpi_weak" in self.data:
            weak = self.data["mpi_weak"].copy().sort_values("tasks")
            weak = weak.groupby("tasks", as_index=False).median(numeric_only=True)
            weak["efficiency"] = 100.0 * weak.iloc[0]["t_wall"] / weak["t_wall"]
            self.single_scaling(
                weak, "tasks", "t_wall", "MPI weak scaling — tempo", "Processi MPI",
                "Tempo totale [s]", "12_mpi_weak_time"
            )
            self.single_scaling(
                weak, "tasks", "efficiency", "MPI weak scaling — efficienza", "Processi MPI",
                "Efficienza [%]", "13_mpi_weak_efficiency", ideal="efficiency"
            )

    def hybrid(self) -> None:
        if "hybrid" in self.data:
            frame = self.data["hybrid"].copy()
            frame["ranks_per_node"] = frame["ranks"] // frame["nodes"]
            frame = frame.groupby(
                ["nodes", "ranks_per_node", "threads_per_rank"], as_index=False
            ).median(numeric_only=True)
            self.hybrid_time_plot(frame, "14_hybrid_fixed_time", "MPI+OpenMP — problema fisso")
            paired = self.pair_nodes(frame)
            if not paired.empty:
                self.hybrid_efficiency_bar(
                    paired,
                    "speedup",
                    "Speedup 1→2 nodi",
                    "15_hybrid_fixed_speedup",
                    "MPI+OpenMP strong scaling — 1 vs 2 nodi",
                    reference=2.0,
                )

        if "hybrid_weak" in self.data:
            frame = self.data["hybrid_weak"].copy()
            frame = frame.groupby(
                ["nodes", "ranks_per_node", "threads_per_rank"], as_index=False
            ).median(numeric_only=True)
            self.hybrid_time_plot(frame, "16_hybrid_weak_time", "MPI+OpenMP weak scaling")
            paired = self.pair_nodes(frame)
            if not paired.empty:
                paired["efficiency"] = 100.0 * paired["time_1"] / paired["time_2"]
                self.hybrid_efficiency_bar(
                    paired,
                    "efficiency",
                    "Efficienza weak [%]",
                    "17_hybrid_weak_efficiency",
                    "MPI+OpenMP weak scaling — efficienza 1→2 nodi",
                    reference=100.0,
                )

    @staticmethod
    def pair_nodes(frame: pd.DataFrame) -> pd.DataFrame:
        pivot = frame.pivot_table(
            index=["ranks_per_node", "threads_per_rank"], columns="nodes", values="t_wall"
        )
        if 1 not in pivot.columns or 2 not in pivot.columns:
            return pd.DataFrame()
        paired = pivot[[1, 2]].dropna().reset_index().rename(columns={1: "time_1", 2: "time_2"})
        paired["speedup"] = paired["time_1"] / paired["time_2"]
        return paired.sort_values("ranks_per_node")

    def hybrid_time_plot(self, frame: pd.DataFrame, name: str, title: str) -> None:
        fig, ax = plt.subplots(figsize=(9, 5.4))
        for nodes, part in frame.groupby("nodes"):
            part = part.sort_values("ranks_per_node")
            labels = [f"{int(r)}×{int(t)}" for r, t in zip(part["ranks_per_node"], part["threads_per_rank"])]
            ax.plot(labels, part["t_wall"], marker="o", label=f"{int(nodes)} nodo/i")
        ax.set_xlabel("Rank per nodo × thread per rank")
        ax.set_ylabel("Tempo totale [s]")
        ax.set_title(title)
        ax.legend()
        self.save(fig, name)

    def hybrid_efficiency_bar(
        self,
        frame: pd.DataFrame,
        metric: str,
        ylabel: str,
        name: str,
        title: str,
        reference: float,
    ) -> None:
        labels = [
            f"{int(r)}×{int(t)}" for r, t in zip(frame["ranks_per_node"], frame["threads_per_rank"])
        ]
        fig, ax = plt.subplots(figsize=(9, 5.4))
        ax.bar(labels, frame[metric], color=COLORS["purple"])
        ax.axhline(reference, color=COLORS["grey"], linestyle="--", label=f"Ideale: {reference:g}")
        ax.set_xlabel("Rank per nodo × thread per rank")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        self.annotate_bars(ax, 2)
        self.save(fig, name)

    def comparisons(self) -> None:
        if {"openmp", "mpi_strong"}.issubset(self.data):
            omp = self.data["openmp"].copy()
            omp = omp[omp["run_name"].str.startswith("strong_")]
            omp["cores"] = pd.to_numeric(omp["run_name"].str.extract(r"_t(\d+)_")[0])
            omp = omp.dropna(subset=["cores"]).groupby("cores", as_index=False)["glups"].max()
            mpi = self.data["mpi_strong"].copy()
            mpi = mpi[mpi["nodes"] == 1].groupby("tasks", as_index=False)["glups"].median()
            fig, ax = plt.subplots(figsize=(8.8, 5.4))
            ax.plot(omp["cores"], omp["glups"], marker="o", label="OpenMP (binding migliore)")
            ax.plot(mpi["tasks"], mpi["glups"], marker="s", label="MPI")
            self.powers_of_two_axis(ax, pd.concat([omp["cores"], mpi["tasks"]]))
            ax.set_xlabel("Core utilizzati su un nodo")
            ax.set_ylabel("GLUP/s")
            ax.set_title("OpenMP vs MPI — strong scaling su un nodo")
            ax.legend()
            self.save(fig, "18_openmp_vs_mpi_strong")

        required = {"final", "openmp", "mpi_strong", "hybrid"}
        if required.issubset(self.data):
            final = self.data["final"].copy()
            final = final[final["run_name"].str.contains("_O3_")]
            omp = self.data["openmp"]
            omp = omp[omp["run_name"].str.startswith("strong_")]
            mpi = self.data["mpi_strong"]
            mpi = mpi[mpi["nodes"] == 1]
            hybrid = self.data["hybrid"]
            hybrid = hybrid[hybrid["nodes"] == 1]
            if all(not part.empty for part in (final, omp, mpi, hybrid)):
                values = pd.Series(
                    {
                        "Seriale\nFinal O3": final["glups"].median(),
                        "OpenMP\n32 core": omp["glups"].max(),
                        "MPI\n32 core": mpi["glups"].max(),
                        "Ibrido\n32 core": hybrid["glups"].max(),
                    }
                )
                fig, ax = plt.subplots(figsize=(8.8, 5.4))
                ax.bar(values.index, values.values, color=[COLORS["red"], COLORS["green"], COLORS["blue"], COLORS["purple"]])
                ax.set_ylabel("GLUP/s")
                ax.set_title("Migliore configurazione per modello — un nodo")
                self.annotate_bars(ax, 2)
                self.save(fig, "19_best_one_node")

    def scaling_lines(
        self,
        frame: pd.DataFrame,
        group: str,
        x: str,
        y: str,
        title: str,
        xlabel: str,
        ylabel: str,
        name: str,
        ideal: str | None = None,
    ) -> None:
        fig, ax = plt.subplots(figsize=(8.8, 5.4))
        for group_name, part in frame.groupby(group):
            part = part.sort_values(x)
            ax.plot(part[x], part[y], marker="o", label=str(group_name))
        self.powers_of_two_axis(ax, frame[x])
        self.add_ideal(ax, frame[x], ideal)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        self.save(fig, name)

    def single_scaling(
        self,
        frame: pd.DataFrame,
        x: str,
        y: str,
        title: str,
        xlabel: str,
        ylabel: str,
        name: str,
        ideal: str | None = None,
    ) -> None:
        fig, ax = plt.subplots(figsize=(8.8, 5.4))
        ax.plot(frame[x], frame[y], color=COLORS["blue"], marker="o")
        self.powers_of_two_axis(ax, frame[x])
        self.add_ideal(ax, frame[x], ideal)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if ideal:
            ax.legend()
        self.save(fig, name)

    @staticmethod
    def add_ideal(ax: plt.Axes, x: pd.Series, ideal: str | None) -> None:
        if ideal == "speedup":
            values = np.asarray(sorted(set(x)), dtype=float)
            ax.plot(values, values, linestyle="--", color=COLORS["grey"], label="Ideale")
        elif ideal == "efficiency":
            ax.axhline(100.0, linestyle="--", color=COLORS["grey"], label="Ideale")

    def run_all(self) -> None:
        self.serial_large()
        self.serial_cache()
        self.openmp()
        self.mpi()
        self.hybrid()
        self.comparisons()


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
