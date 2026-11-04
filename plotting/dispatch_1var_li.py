import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

results_dir = "src/neup_zld/results/sweep_1var_lithium/"

FILES = {
    "Li = 0.25x $/kg": results_dir + "dispatch_lithium_0.25.csv",
    "Li = 0.5x $/kg" : results_dir + "dispatch_lithium_0.5.csv",
    "Li = 1x $/kg"   : results_dir + "dispatch_lithium_1.0.csv",
    "Li = 2.5x $/kg" : results_dir + "dispatch_lithium_2.5.csv",
    "Li = 5x $/kg"   : results_dir + "dispatch_lithium_5.0.csv",
    "Li = 7.5x $/kg" : results_dir + "dispatch_lithium_7.5.csv",
    "Li = 10x $/kg"  : results_dir + "dispatch_lithium_10.0.csv",
    "Li = 20x $/kg"  : results_dir + "dispatch_lithium_20.0.csv",
}

OUTPUT_DIR = Path("dispatch_plots")
OUTPUT_DIR.mkdir(exist_ok=True)

WEEK_HOURS = 168

# ================================
# helper: classify seasons
# ================================
def get_season(hour):

    day = (hour - 1) // 24 + 1

    # approximate seasons
    if day <= 59 or day >= 335:
        return "winter"
    elif 152 <= day <= 243:
        return "summer"
    else:
        return "shoulder"


# ================================
# build representative week
# ================================
def representative_week(df, column, season):

    df["season"] = df["hour"].apply(get_season)

    df = df[df["season"] == season].copy()

    df["hour_of_week"] = (df["hour"] - 1) % WEEK_HOURS + 1

    stats = (
        df.groupby("hour_of_week")[column]
        .agg(["mean", "std"])
        .reset_index()
    )

    return stats


# ================================
# plotting function
# ================================
def plot_rep_week(metric, ylabel, season, outname):

    plt.figure(figsize=(12, 5))

    for label, file in FILES.items():

        df = pd.read_csv(file)

        stats = representative_week(df, metric, season)

        x = stats["hour_of_week"]
        mu = stats["mean"]
        sigma = stats["std"]

        plt.plot(x, mu, linewidth=2, label=label)
        plt.fill_between(x, mu - sigma, mu + sigma, alpha=0.2)

    plt.title(f"{season.capitalize()} Representative Week — {ylabel}")
    plt.xlabel("Hour of Week")
    plt.xticks([0,24,48,72,96,120,144,168])
    plt.ylabel(ylabel)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(OUTPUT_DIR / outname, dpi=300)
    plt.close()


# ================================
# MAIN
# ================================
# plot_rep_week(
#     metric="cold_tes_kg",
#     ylabel="Cold TES Inventory (kg)",
#     season="winter",
#     outname="winter_cold_tes.png",
# )

plot_rep_week(
    metric="elec_sold_kWe",
    ylabel="Electricity Sold (kWe)",
    season="winter",
    outname="winter_electricity.png",
)

# plot_rep_week(
#     metric="cold_tes_kg",
#     ylabel="Cold TES Inventory (kg)",
#     season="summer",
#     outname="summer_cold_tes.png",
# )

plot_rep_week(
    metric="elec_sold_kWe",
    ylabel="Electricity Sold (kWe)",
    season="summer",
    outname="summer_electricity.png",
)

print("Done — representative week plots created.")
