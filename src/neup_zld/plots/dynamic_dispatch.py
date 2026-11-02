"""
Three-panel dispatch plotting script for ZLD/TES/desalination model outputs.

Creates a three-panel figure:
    (a) Electricity generated, sold, and used, with electricity price on secondary y-axis
    (b) Desalination feed rates for RO, MED, ED, and CRY
    (c) TES inventories

Expected dispatch/hourly CSV columns:
    hour
    hot_tes_kWh
    cold_tes_kWh
    elec_sold_kWe
    elec_gen_kWe
    v_in_RO
    v_in_MED
    v_in_ED
    v_in_CRY

Electricity used is calculated as:
    elec_used_kWe = elec_gen_kWe - elec_sold_kWe

Electricity price can be included in the dispatch CSV or supplied through a separate price CSV.
"""

from pathlib import Path
import argparse
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# Thesis figure style
# =============================================================================

# COLORS = {
#     # Panel 1: electricity, cool family
#     "elec_generated": "#4C78A8",
#     "elec_sold": "#72B7B2",
#     "elec_used": "#9BBCE0",
#     "price": "#D0D0D0",

#     # Panel 2: feed flows, warm family
#     "RO": "#F58518",
#     "MED": "#FFBF79",
#     "ED": "#E45756",
#     "CRY": "#F2A6A6",

#     # Panel 3: TES, blue-violet family
#     "low_temp_tes": "#355F8C",
#     "high_temp_tes": "#8C6BB1",
# }

COLORS = {
    # Panel 1: electricity
    "elec_generated": "#4C78A8",  # blue
    "elec_sold": "#F58518",       # orange
    "elec_used": "#72B7B2",       # teal
    "price": "#D0D0D0",           # light gray

    # Panel 2: process feeds
    "RO": "#4C78A8",              # blue
    "MED": "#F58518",             # orange
    "ED": "#E45756",              # coral red
    "CRY": "#72B7B2",             # teal

    # Panel 3: TES
    "low_temp_tes": "#4C78A8",    # blue
    "high_temp_tes": "#E45756",   # coral red
}

LINESTYLES = {
    "elec_generated": "-",
    "elec_sold": "-",
    "elec_used": "-",
    "price": ":",

    "RO": "-",
    "MED": "-",
    "ED": "-",
    "CRY": "--",

    "low_temp_tes": "-",
    "high_temp_tes": "-",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi": 150,
})


# =============================================================================
# Helper functions
# =============================================================================

def read_dispatch_csv(dispatch_path: Path) -> pd.DataFrame:
    df = pd.read_csv(dispatch_path)
    df.columns = [c.strip() for c in df.columns]

    if "hour" not in df.columns:
        df["hour"] = np.arange(len(df))

    return df


def maybe_add_price(df: pd.DataFrame, price_csv: Path | None = None) -> pd.DataFrame:
    """
    Add electricity price column to the dispatch dataframe.

    Priority:
        1. Use a price-like column already in the dispatch CSV
        2. Use a separate price CSV, matched by hour if possible, otherwise row order
    """

    price_candidates = [
        "electricity_price",
        "price",
        "Price",
        "elec_price",
        "LMP",
        "lmp",
        "Price_elec",
        "price_elec",
    ]

    for col in price_candidates:
        if col in df.columns:
            df["electricity_price"] = df[col]
            return df

    if price_csv is None:
        return df

    price_df = pd.read_csv(price_csv)
    price_df.columns = [c.strip() for c in price_df.columns]

    found_price_col = None
    for col in price_candidates:
        if col in price_df.columns:
            found_price_col = col
            break

    if found_price_col is None:
        warnings.warn(
            f"No price column found in {price_csv}. "
            f"Tried: {price_candidates}"
        )
        return df

    if "hour" in df.columns and "hour" in price_df.columns:
        df = df.merge(
            price_df[["hour", found_price_col]],
            on="hour",
            how="left"
        )
        df["electricity_price"] = df[found_price_col]
    else:
        n = min(len(df), len(price_df))
        df.loc[df.index[:n], "electricity_price"] = price_df[found_price_col].iloc[:n].to_numpy()

    return df


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Create MW/MWh columns."""

    if "elec_gen_kWe" in df.columns:
        df["elec_generated_MW"] = df["elec_gen_kWe"] / 1000.0

    if "elec_sold_kWe" in df.columns:
        df["elec_sold_MW"] = df["elec_sold_kWe"] / 1000.0

    if "elec_used_kWe" in df.columns:
        df["elec_used_MW"] = df["elec_used_kWe"] / 1000.0
    elif {"elec_gen_kWe", "elec_sold_kWe"}.issubset(df.columns):
        df["elec_used_MW"] = (df["elec_gen_kWe"] - df["elec_sold_kWe"]) / 1000.0

    if "hot_tes_kWh" in df.columns:
        df["hot_tes_MWh"] = df["hot_tes_kWh"] / 1000.0

    if "cold_tes_kWh" in df.columns:
        df["cold_tes_MWh"] = df["cold_tes_kWh"] / 1000.0

    return df


def select_window(df: pd.DataFrame, start_hour: int | None, duration: int) -> pd.DataFrame:
    """
    Select a plotting window.

    If start_hour is None, plot the entire dataframe.
    Otherwise, use the hour column if available.
    """

    if start_hour is None:
        return df.copy()

    if "hour" in df.columns:
        out = df[
            (df["hour"] >= start_hour) &
            (df["hour"] < start_hour + duration)
        ].copy()
    else:
        out = df.iloc[start_hour:start_hour + duration].copy()

    if out.empty:
        raise ValueError(
            f"No data found for selected window: start_hour={start_hour}, duration={duration}."
        )

    return out


def make_time_axis(df_window: pd.DataFrame):
    x = np.arange(len(df_window))
    xlabel = "Hour of selected week"
    return x, xlabel


def set_day_ticks(ax, n_hours: int):
    """
    Use day-style ticks for a weekly plot.
    This keeps the x-axis cleaner than labeling every hour.
    """

    if n_hours >= 120:
        day_positions = np.arange(0, min(n_hours, 168) + 1, 24)
        day_labels = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat", ""]
        ax.set_xticks(day_positions)
        ax.set_xticklabels(day_labels[:len(day_positions)])


def clean_axis(ax):
    ax.grid(True, alpha=0.22, linewidth=0.8)
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def add_legend_above(ax, ncol=3, y=1.20):
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, y),
        ncol=ncol,
        frameon=False,
        handlelength=2.6,
        columnspacing=1.2,
        fontsize=13,
    )


def add_panel_labels(axes):
    panel_labels = ["(a)", "(b)", "(c)"]

    for ax, lab in zip(axes, panel_labels):
        ax.text(
            0.015, 0.93, lab,
            transform=ax.transAxes,
            fontsize=17,
            fontweight="bold",
            va="top",
            ha="left",
            bbox=dict(
                facecolor="white",
                edgecolor="none",
                alpha=0.8,
                pad=2.0
            ),
            zorder=10
        )


def align_yaxis_labels(fig, ax_power, ax_feed, ax_tes=None, ax_price=None):
    """Make left-side y-axis titles share the same horizontal position."""
    left_axes = [ax_power, ax_feed]
    if ax_tes is not None:
        left_axes.append(ax_tes)

    # Use a consistent horizontal coordinate for all left y-axis labels.
    for ax in left_axes:
        ax.yaxis.set_label_coords(-0.08, 0.5)

    fig.align_ylabels(left_axes)

    # Nudge the right-side electricity-price label slightly outward so it does not
    # visually crowd the plotting area.
    if ax_price is not None:
        ax_price.yaxis.set_label_coords(1.06, 0.5)


def dispatch_path_from_group_job(root: Path, group: str, job_id: int | str) -> Path:
    """
    Find hourly/dispatch CSV from a group/job ID.

    Supports names like:
        hourly_00.csv
        hourly_0.csv
        hourly_69.csv
        dispatch_69.csv
        dispatch_job_69.csv
    """

    job_id_str = str(job_id)
    job_id_int = int(job_id)
    job_id_2digit = f"{job_id_int:02d}"

    candidates = [
        root / group / f"hourly_{job_id_str}.csv",
        root / group / f"hourly_{job_id_2digit}.csv",
        root / group / f"hourly_job_{job_id_str}.csv",
        root / group / f"dispatch_{job_id_str}.csv",
        root / group / f"dispatch_{job_id_2digit}.csv",
        root / group / f"dispatch_job_{job_id_str}.csv",

        root / f"hourly_{job_id_str}.csv",
        root / f"hourly_{job_id_2digit}.csv",
        root / f"dispatch_{job_id_str}.csv",
        root / f"dispatch_{job_id_2digit}.csv",
    ]

    for p in candidates:
        if p.exists():
            return p

    group_dir = root / group
    patterns = [
        f"*hourly*{job_id_str}*.csv",
        f"*hourly*{job_id_2digit}*.csv",
        f"*dispatch*{job_id_str}*.csv",
        f"*dispatch*{job_id_2digit}*.csv",
    ]

    if group_dir.exists():
        for pattern in patterns:
            matches = sorted(group_dir.glob(pattern))
            if matches:
                return matches[0]

    for pattern in patterns:
        matches = sorted(root.rglob(pattern))
        if matches:
            return matches[0]

    tried = "\n  - ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        f"Could not find hourly/dispatch file for group={group}, job_id={job_id}.\n"
        f"Tried:\n  - {tried}\n"
        f"Also searched recursively for hourly/dispatch files containing "
        f"{job_id_str} or {job_id_2digit}."
    )

def has_thermal_storage(dfw: pd.DataFrame) -> bool:
    """
    Return True if hot or low-temp TES has meaningful nonzero inventory.
    """
    tes_cols = ["cold_tes_MWh", "hot_tes_MWh"]

    for col in tes_cols:
        if col in dfw.columns:
            series = dfw[col].fillna(0)
            if series.abs().max() > 1e-6:
                return True

    return False

# =============================================================================
# Main plotting function
# =============================================================================



def plot_three_panel_dispatch(
    dispatch_path: Path,
    output_dir: Path,
    start_hour: int | None = 3192,
    duration: int = 168,
    price_csv: Path | None = None,
    title: str | None = None,
    save_pdf: bool = True,
    show: bool = False,
):
    dispatch_path = Path(dispatch_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = read_dispatch_csv(dispatch_path)
    df = maybe_add_price(df, price_csv=price_csv)
    df = add_derived_columns(df)
    dfw = select_window(df, start_hour=start_hour, duration=duration)

    x, xlabel = make_time_axis(dfw)

    include_tes_panel = has_thermal_storage(dfw)

    # Always create three panels so side-by-side figures use the same layout,
    # even when no thermal storage is built.
    fig, axes = plt.subplots(
        3, 1,
        figsize=(12, 9.2),
        sharex=True,
        gridspec_kw={
            "height_ratios": [1.25, 1.0, 1.0],
            "hspace": 0.38,
        }
    )

    ax_power = axes[0]
    ax_feed = axes[1]
    ax_tes = axes[2]

    add_panel_labels(axes)

    # -------------------------------------------------------------------------
    # Panel (a): electricity dispatch
    # -------------------------------------------------------------------------

    if "elec_generated_MW" in dfw.columns:
        ax_power.plot(
            x, dfw["elec_generated_MW"],
            color=COLORS["elec_generated"],
            linestyle=LINESTYLES["elec_generated"],
            linewidth=2.6,
            label="Electricity generated"
        )

    if "elec_sold_MW" in dfw.columns:
        ax_power.plot(
            x, dfw["elec_sold_MW"],
            color=COLORS["elec_sold"],
            linestyle=LINESTYLES["elec_sold"],
            linewidth=2.6,
            label="Electricity sold"
        )

    if "elec_used_MW" in dfw.columns:
        ax_power.plot(
            x, dfw["elec_used_MW"],
            color=COLORS["elec_used"],
            linestyle=LINESTYLES["elec_used"],
            linewidth=2.6,
            label="Electricity used"
        )

    ax_power.set_ylabel("Power (MW$_\mathrm{e}$)", fontsize=16, labelpad=18)
    clean_axis(ax_power)

    # Secondary y-axis for electricity price
    ax_price = None

    if "electricity_price" in dfw.columns:
        ax_price = ax_power.twinx()

        ax_price.plot(
            x, dfw["electricity_price"],
            color=COLORS["price"],
            linestyle=LINESTYLES["price"],
            linewidth=2.2,
            label="Electricity price",
            zorder=0
        )

        ax_price.set_ylabel(r"Electricity price (\$/MWh$_\mathrm{e}$)", fontsize=16, labelpad=18)
        ax_price.tick_params(axis="y", colors="#6E6E6E")
        ax_price.spines["right"].set_color("#D0D0D0")
        ax_price.spines["top"].set_visible(False)
        ax_price.grid(False)

        # Keep price visually behind electricity lines
        ax_price.set_zorder(ax_power.get_zorder() - 1)
        ax_power.patch.set_visible(False)

    lines1, labels1 = ax_power.get_legend_handles_labels()

    if ax_price is not None:
        lines2, labels2 = ax_price.get_legend_handles_labels()
    else:
        lines2, labels2 = [], []

    ax_power.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.27),
        ncol=4,
        frameon=False,
        handlelength=2.6,
        columnspacing=1.2,
        fontsize=13,
    )

    # -------------------------------------------------------------------------
    # Panel (b): desalination feed rates
    # -------------------------------------------------------------------------

    feed_map = {
        "RO": "v_in_RO",
        "MED": "v_in_MED",
        "ED": "v_in_ED",
        "CRY": "v_in_CRY",
    }

    for proc, col in feed_map.items():
        if col in dfw.columns:
            series = dfw[col]

            if series.notna().any() and not np.isclose(series.fillna(0).abs().max(), 0.0):
                ax_feed.plot(
                    x, series,
                    color=COLORS[proc],
                    linestyle=LINESTYLES[proc],
                    linewidth=2.6,
                    label=f"{proc}"
                )

    ax_feed.set_ylabel("Feed flow (m$^3$/h)", fontsize=16, labelpad=18)
    clean_axis(ax_feed)

# Prevent matplotlib from zooming in on tiny numerical noise.
# This is especially important for cases where a feed flow is essentially constant.
    feed_cols_present = [
        col for col in ["v_in_RO", "v_in_MED", "v_in_ED", "v_in_CRY"]
        if col in dfw.columns
    ]
    
    if feed_cols_present:
        feed_max = dfw[feed_cols_present].fillna(0).abs().max().max()
    
        if feed_max > 0:
            ax_feed.set_ylim(0, feed_max * 1.08)


    if ax_feed.get_legend_handles_labels()[0]:
        add_legend_above(ax_feed, ncol=4, y=1.24)

    # -------------------------------------------------------------------------
    # Panel (c): TES inventory
    # -------------------------------------------------------------------------


    if "cold_tes_MWh" in dfw.columns:
        series = dfw["cold_tes_MWh"]

        if series.notna().any() and not np.isclose(series.fillna(0).abs().max(), 0.0):
            ax_tes.plot(
                x, series,
                color=COLORS["low_temp_tes"],
                linestyle=LINESTYLES["low_temp_tes"],
                linewidth=2.6,
                label="Low-temp TES"
            )

    if "hot_tes_MWh" in dfw.columns:
        series = dfw["hot_tes_MWh"]

        if series.notna().any() and not np.isclose(series.fillna(0).abs().max(), 0.0):
            ax_tes.plot(
                x, series,
                color=COLORS["high_temp_tes"],
                linestyle=LINESTYLES["high_temp_tes"],
                linewidth=2.6,
                label="High-temp TES"
            )

    ax_tes.set_ylabel("TES inventory (MWh)", fontsize=16, labelpad=18)
    ax_tes.set_xlabel(xlabel, fontsize=16)
    clean_axis(ax_tes)

    if ax_tes.get_legend_handles_labels()[0]:
        add_legend_above(ax_tes, ncol=2, y=1.24)
    else:
        # Preserve the third panel and make the zero-TES result explicit.
        ax_tes.axhline(0.0, color="#B0B0B0", linewidth=1.2)
        ax_tes.set_ylim(-0.05, 1.0)
        ax_tes.text(
            0.5, 0.5, "No thermal storage built",
            transform=ax_tes.transAxes,
            ha="center",
            va="center",
            fontsize=14,
            color="#6E6E6E"
        )

    # -------------------------------------------------------------------------
    # Shared x-axis formatting
    # -------------------------------------------------------------------------

    set_day_ticks(ax_tes, len(dfw))

    align_yaxis_labels(fig, ax_power, ax_feed, ax_tes=ax_tes, ax_price=ax_price)

    if title:
        fig.suptitle(title, fontsize=15, y=0.985)

    fig.subplots_adjust(
        top=0.92,
        bottom=0.08,
        left=0.09,
        right=0.91,
        hspace=0.42
    )
        
    stem = dispatch_path.stem

    if start_hour is not None:
        out_stem = f"{stem}_three_panel_week{start_hour}"
    else:
        out_stem = f"{stem}_three_panel_all"

    png_path = output_dir / f"{out_stem}.png"
    fig.savefig(png_path, dpi=600, bbox_inches="tight")

    pdf_path = None

    if save_pdf:
        pdf_path = output_dir / f"{out_stem}.pdf"
        fig.savefig(pdf_path, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return png_path, pdf_path


# =============================================================================
# Command line interface
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Make three-panel dispatch figures from model hourly/dispatch CSVs."
    )

    parser.add_argument(
        "--dispatch",
        type=str,
        default=None,
        help="Path to a single hourly/dispatch CSV."
    )

    parser.add_argument(
        "--root",
        type=str,
        default=".",
        help="Root directory containing DesA, DesB, etc. folders."
    )

    parser.add_argument(
        "--group",
        type=str,
        default=None,
        help="Design group/folder name, e.g., DesE."
    )

    parser.add_argument(
        "--job-id",
        type=str,
        default=None,
        help="Job ID, e.g., 69. Used with --group to locate hourly/dispatch CSV."
    )

    parser.add_argument(
        "--price-csv",
        type=str,
        default=None,
        help="Optional price CSV if price is not already in dispatch/hourly CSV."
    )

    parser.add_argument(
        "--start-hour",
        type=int,
        default=3192,
        help="Start hour for selected dispatch window. Use -1 to plot all rows."
    )

    parser.add_argument(
        "--duration",
        type=int,
        default=168,
        help="Number of hours to plot."
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="three_panel_dispatch_figures",
        help="Folder where figures will be saved."
    )

    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional figure title. For thesis figures, I recommend leaving this blank."
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Show figure interactively."
    )

    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    price_csv = Path(args.price_csv).expanduser().resolve() if args.price_csv else None

    start_hour = None if args.start_hour == -1 else args.start_hour

    if args.dispatch:
        dispatch_path = Path(args.dispatch).expanduser().resolve()
    else:
        if args.group is None or args.job_id is None:
            raise ValueError(
                "Provide either --dispatch PATH or both --group GROUP and --job-id JOB_ID."
            )

        dispatch_path = dispatch_path_from_group_job(
            root=root,
            group=args.group,
            job_id=args.job_id
        )

    print(f"Using dispatch/hourly CSV: {dispatch_path}")

    png_path, pdf_path = plot_three_panel_dispatch(
        dispatch_path=dispatch_path,
        output_dir=output_dir,
        start_hour=start_hour,
        duration=args.duration,
        price_csv=price_csv,
        title=args.title,
        save_pdf=True,
        show=args.show,
    )

    print(f"Saved PNG: {png_path}")

    if pdf_path is not None:
        print(f"Saved PDF: {pdf_path}")


if __name__ == "__main__":
    main()
