import pyomo.environ as pyo
import pandas as pd
import numpy as np
import os
import csv
from zld_with_tes_v2_ek import build_model


def run_2d_sensitivity():
    # --- 1. SETUP & DATA LOADING ---
    N_design = 24
    N_dispatch = 8760
    results_dir = "sensitivity_output"
    os.makedirs(results_dir, exist_ok=True)
    
    # Load Design Data (Typical Day)
    eff_design = pd.read_csv("avg_day_efficiency.csv")["Average_Efficiency"].to_numpy()[:N_design]
    elec_design = pd.read_csv("average_day_lmp.csv")["LMP"].to_numpy()[:N_design]
    
    # Load Dispatch Data (Full Year)
    eff_dispatch = pd.read_csv("sched_eff.csv")["Efficiency"].to_numpy()[:N_dispatch]
    elec_dispatch = pd.read_csv("DIABLOCN_2_N001_PRICES.csv")["LMP"].to_numpy()[:N_dispatch]

    solver = pyo.SolverFactory("gurobi")
    solver.options['NonConvex'] = 2

    # --- 2. DEFINE SWEEP RANGES ---
    # Example: Lithium Price vs Water Price (Case 10)
    li_mults = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
    water_mults = [0.5, 0.75, 1.0, 1.5, 2, 5]
    
    summary_data = [] # This is your 'summary' list

    # --- 3. THE NESTED LOOP ---
    for l_m in li_mults:
        for w_m in water_mults:
            print(f"\n>>> Solving Case: Li_mult={l_m}, Water_mult={w_m}")
            current_prices = {'lithium': l_m, 'water': w_m}

            # A. DESIGN PHASE
            model_d = build_model(N_design, elec_design, eff_design, price_mults=current_prices)
            solver.solve(model_d, tee=False)

            # Extract Sizing from Design
            caps = {p: pyo.value(model_d.v_dot_cap[p]) for p in model_d.processes}
            hot_max = pyo.value(model_d.m_hot_tes_max)
            cold_max = pyo.value(model_d.m_cold_tes_max)
            y_p = {p: pyo.value(model_d.y_process_active[p]) for p in model_d.processes}
            y_l = {l: pyo.value(model_d.y_link_active[l]) for l in model_d.links}

            # B. DISPATCH PHASE
            model_disp = build_model(N_dispatch, elec_dispatch, eff_dispatch, price_mults=current_prices)
            
            # Fix variables from Design
            for p in model_disp.processes:
                model_disp.v_dot_cap[p].fix(caps[p])
                model_disp.y_process_active[p].fix(round(y_p[p]))
            for l in model_disp.links:
                model_disp.y_link_active[l].fix(round(y_l[l]))
            model_disp.m_hot_tes_max.fix(hot_max)
            model_disp.m_cold_tes_max.fix(cold_max)

            res_disp = solver.solve(model_disp, tee=False)

            # C. DATA COLLECTION
            if res_disp.solver.termination_condition == pyo.TerminationCondition.optimal:
                # Save Hourly Dispatch Data for this specific point
                hourly_filename = f"{results_dir}/dispatch_Li{l_m}_W{w_m}.csv"
                with open(hourly_filename, 'w', newline='') as f:
                    w = csv.writer(f)
                    w.writerow(["hour", "hot_tes_kg", "cold_tes_kg", "elec_sold_kWe", "li_recovered_kg"])
                    for t in model_disp.T:
                        w.writerow([
                            t, 
                            pyo.value(model_disp.m_hot_tes[t]), 
                            pyo.value(model_disp.m_cold_tes[t]),
                            pyo.value(model_disp.elec_sold[t]),
                            pyo.value(model_disp.li_recovered_ed[t])
                        ])

                # Add to Overall Summary
                summary_data.append({
                    'li_mult': l_m,
                    'water_mult': w_m,
                    'annual_profit': pyo.value(model_disp.obj),
                    'total_li_recovered': sum(pyo.value(model_disp.li_recovered_ed[t]) for t in model_disp.T),
                    'total_elec_sold_mwh': sum(pyo.value(model_disp.elec_sold[t]) for t in model_disp.T) / 1000,
                    'hot_tes_size': hot_max
                })

    # --- 4. EXPORT SUMMARY ---
    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv(f"{results_dir}/sweep_summary.csv", index=False)
    print(f"\nSweep complete. Summary saved to {results_dir}/sweep_summary.csv")

if __name__ == "__main__":
    run_2d_sensitivity()