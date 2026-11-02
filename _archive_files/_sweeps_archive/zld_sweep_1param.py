import pyomo.environ as pyo
import pandas as pd
import numpy as np
import os
import csv
from zld_with_tes_v2_ek import build_model

def run_1d_sensitivity():
    # --- 1. SETUP & DATA LOADING ---
    N_design = 24
    N_dispatch = 8760
    results_dir = "sensitivity_1d_output"
    os.makedirs(results_dir, exist_ok=True)
    
    # Load Data
    eff_design = pd.read_csv("avg_day_efficiency.csv")["Average_Efficiency"].to_numpy()[:N_design]
    elec_design = pd.read_csv("average_day_lmp.csv")["LMP"].to_numpy()[:N_design]
    eff_dispatch = pd.read_csv("sched_eff.csv")["Efficiency"].to_numpy()[:N_dispatch]
    elec_dispatch = pd.read_csv("DIABLOCN_2_N001_PRICES.csv")["LMP"].to_numpy()[:N_dispatch]

    solver = pyo.SolverFactory("gurobi")
    solver.options['NonConvex'] = 2

    # --- 2. DEFINE SWEEP RANGE ---
    # Example: Sweeping Lithium Price Multiplier
    water_mults = [0.25, 0.5, 1.0, 1.5, 2.0, 5.0, 10.0, 20.0]
    
    summary_data = []

    # --- 3. THE LOOP ---
    for l_m in water_mults:
        print(f"\n>>> Solving Case: Water_mult={l_m}")
        current_prices = {'tes_cold': l_m}

        # A. DESIGN PHASE
        model_d = build_model(N_design, elec_design, eff_design, price_mults=current_prices)
        solver.solve(model_d, tee=False)

        # Extract Sizing
        caps = {p: pyo.value(model_d.v_dot_cap[p]) for p in model_d.processes}
        hot_max = pyo.value(model_d.m_hot_tes_max)
        cold_max = pyo.value(model_d.m_cold_tes_max)
        y_p = {p: pyo.value(model_d.y_process_active[p]) for p in model_d.processes}
        y_l = {l: pyo.value(model_d.y_link_active[l]) for l in model_d.links}

        # B. DISPATCH PHASE
        model_disp = build_model(N_dispatch, elec_dispatch, eff_dispatch, price_mults=current_prices)
        
        # Fix variables
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
            hourly_filename = f"{results_dir}/dispatch_Tes_cold_{l_m}.csv"
            with open(hourly_filename, 'w', newline='') as f:
                w = csv.writer(f)
                
                # 1. Dynamically build the header
                header = ["hour", "hot_tes_kg", "cold_tes_kg", "elec_sold_kWe", "li_recovered_kg"]
                for p in model_disp.processes:
                    header.append(f"v_in_{p}")
                w.writerow(header)
                
                # 2. Iterate and build the rows
                for t in model_disp.T:
                    # Start the row with base metrics
                    row = [
                        t, 
                        pyo.value(model_disp.m_hot_tes[t]), 
                        pyo.value(model_disp.m_cold_tes[t]),
                        pyo.value(model_disp.elec_sold[t]),
                        pyo.value(model_disp.li_recovered_ed[t])
                    ]
                    
                    # Append flow for each process
                    for p in model_disp.processes:
                        row.append(pyo.value(model_disp.v_dot_in[t, p]))
                    
                    # Write the fully constructed row
                    w.writerow(row)

    # --- 4. EXPORT SUMMARY ---
    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv(f"{results_dir}/sweep_summary_1d.csv", index=False)
    print(f"\nSweep complete. Summary saved to {results_dir}/sweep_summary_1d.csv")

if __name__ == "__main__":
    run_1d_sensitivity()