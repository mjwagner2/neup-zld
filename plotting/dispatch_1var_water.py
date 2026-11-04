import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

FILES = {
    "Water = 0.5x $/m3": "dispatch_Water_0.5.csv",
    "Water = 1x $/m3": "dispatch_Water_1.0.csv",
    "Water = 2x $/m3": "dispatch_Water_2.0.csv",
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
    plt.ylabel(ylabel)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(OUTPUT_DIR / outname, dpi=300)
    plt.close()


# ================================
# MAIN
# ================================
plot_rep_week(
    metric="cold_tes_kg",
    ylabel="Cold TES Inventory (kg)",
    season="winter",
    outname="winter_cold_tes_water.png",
)

plot_rep_week(
    metric="elec_sold_kWe",
    ylabel="Electricity Sold (kWe)",
    season="winter",
    outname="winter_electricity_water.png",
)

plot_rep_week(
    metric="cold_tes_kg",
    ylabel="Cold TES Inventory (kg)",
    season="summer",
    outname="summer_cold_tes_water.png",
)

plot_rep_week(
    metric="elec_sold_kWe",
    ylabel="Electricity Sold (kWe)",
    season="summer",
    outname="summer_electricity_water.png",
)

print("Done — representative week plots created.")

