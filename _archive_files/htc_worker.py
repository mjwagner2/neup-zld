import argparse
import pandas as pd
import pyomo.environ as pyo
import csv
import os
from core.zld_dynamic import build_model

# write csv dispatch results
def write_dispatch_csv(model_disp, hourly_filename):
    os.makedirs(os.path.dirname(hourly_filename), exist_ok=True)
    with open(hourly_filename, "w", newline="") as f:
        w = csv.writer(f)

        header = ["hour",
            "hot_tes_kWh",
            "cold_tes_kWh",
            "elec_sold_kWe",
            "elec_gen_kWe",
            "m_dot_extract_kg_s",
            "li_recovered_kg"]

        for p in model_disp.processes:
            header.append(f"v_in_{p}")
            header.append(f"v_dil_{p}")
        w.writerow(header)

        for t in model_disp.T:
            hot_tes_kWh = (pyo.value(model_disp.m_hot_tes[t])
                * model_disp.Cp_salt.value
                * model_disp.Delta_T_hx1_salt.value / 3600)
            cold_tes_kWh = (pyo.value(model_disp.m_cold_tes[t])
                * model_disp.Delta_H_hx2_water.value / 3600)

            row = [t,
                hot_tes_kWh,
                cold_tes_kWh,
                pyo.value(model_disp.elec_sold[t]),
                pyo.value(model_disp.w_dot_gen[t]),
                pyo.value(model_disp.m_dot_extract[t]),
                pyo.value(model_disp.li_recovered_ed[t])]

            for p in model_disp.processes:
                row.append(pyo.value(model_disp.v_dot_in[t, p]))
                row.append(pyo.value(model_disp.v_dot_dil[t, p]))

            w.writerow(row)
    print(f"Saved: {hourly_filename}")

# writes summary of dispatch file
def build_summary_row(model_disp, hot_max, cold_max, extra_fields=None):
    row = {"annual_profit": pyo.value(model_disp.obj),
        "hot_tes_max_kg": hot_max,
        "cold_tes_max_kg": cold_max,
        "hot_tes_max_kWh": hot_max * model_disp.Cp_salt.value * model_disp.Delta_T_hx1_salt.value / 3600,
        "cold_tes_max_kWh": cold_max * model_disp.Delta_H_hx2_water.value / 3600,
        "total_li_recovered_kg": sum(pyo.value(model_disp.li_recovered_ed[t]) for t in model_disp.T),
        "total_elec_sold_MWh": sum(pyo.value(model_disp.elec_sold[t]) for t in model_disp.T) / 1000}
    if extra_fields:
        row.update(extra_fields)
    return row


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--des_file", type=str, required=True)
    parser.add_argument("--disp_file", type=str, required=True)
    parser.add_argument("--group", type=str, default="Baseline", help="DesignA, DesignB, etc.")
    parser.add_argument("--job_id", type=str, default="0")
    
    # sensitivity parameters
    parser.add_argument("--turbine_mult", type=float, default=1.0)
    parser.add_argument("--li_price", type=float, default=1.0)
    parser.add_argument("--water_price", type=float, default=1.0)
    parser.add_argument("--ctes_price", type=float, default=1.0)
    parser.add_argument("--htes_price", type=float, default=1.0)
    parser.add_argument("--capex_ro", type=float, default=1.0)
    parser.add_argument("--capex_med", type=float, default=1.0)
    parser.add_argument("--capex_ed", type=float, default=1.0)
    parser.add_argument("--energy_ro", type=float, default=1.0)
    parser.add_argument("--energy_med", type=float, default=1.0)
    parser.add_argument("--energy_ed", type=float, default=1.0)
    
    args = parser.parse_args()

    # bundle dictionaries for model
    prices = {'Li': args.li_price, 
              'Water': args.water_price}
    costs  = {'RO': args.capex_ro, 
              'MED': args.capex_med, 
              'ED': args.capex_ed,
              'HTES': args.htes_price, 
              'CTES': args.ctes_price}
    procs  = {'RO': args.energy_ro,
              'MED': args.energy_med,
              'ED': args.energy_ed}

    # define dictionary
    base_dir = os.path.join("results", args.group)
    os.makedirs(base_dir, exist_ok=True)

    # load data
    elec_des  = pd.read_csv(f"data/data_lmp/{args.des_file}")["LMP"].to_numpy()[:24]
    elec_disp = pd.read_csv(f"data/data_lmp/{args.disp_file}")["LMP"].to_numpy()[:8760]
    eff_des   = pd.read_csv(f"data/data_effic/{args.des_file}")["Efficiency"].to_numpy()[:24]
    eff_disp  = pd.read_csv(f"data/data_effic/{args.disp_file}")["Efficiency"].to_numpy()[:8760]

    solver = pyo.SolverFactory("gurobi")
    solver.options["NonConvex"] = 2
    solver.options["TimeLimit"] = 3600 # 1 hour limit

    # run design
    model_d = build_model(24, elec_des, eff_des, 
                          price_mults=prices, 
                          cost_mults=costs, 
                          process_mults=procs,
                          turbine_mults=args.turbine_mult)
    solver.solve(model_d)

    # extract variable data
    caps_val     = {p: pyo.value(model_d.v_dot_cap[p]) for p in model_d.processes}
    hot_max_val  = pyo.value(model_d.m_hot_tes_max)
    cold_max_val = pyo.value(model_d.m_cold_tes_max)
    y_p_val      = {p: pyo.value(model_d.y_process_active[p]) for p in model_d.processes}
    y_l_val      = {l: pyo.value(model_d.y_link_active[l]) for l in model_d.links}

    # run dispatch
    model_disp = build_model(8760, elec_disp, eff_disp, 
                             price_mults=prices, 
                             cost_mults=costs, 
                             process_mults=procs,
                             turbine_mults=args.turbine_mult)

    # fix design variables
    model_disp.m_hot_tes_max.fix(hot_max_val)
    model_disp.m_cold_tes_max.fix(cold_max_val)
    for p in model_disp.processes:
        model_disp.v_dot_cap[p].fix(caps_val[p])
        model_disp.y_process_active[p].fix(y_p_val[p])
    for l in model_disp.links:
        model_disp.y_link_active[l].fix(y_l_val[l])

    solver.solve(model_disp)

    # save results
    # metadata includes all sweep variables for later filtering
    tag = f"{args.des_file.split('.')[0]}_vs_{args.disp_file.split('.')[0]}"
    sweep_metadata = {
        "group": args.group,
        "job_id": args.job_id,
        "scenario": tag,
        "li_price": args.li_price,
        "water_price": args.water_price,
        "ctes_price": args.ctes_price,
        "htes_price": args.htes_price,
        "capex_ro": args.capex_ro,
        "capex_med": args.capex_med,
        "capex_ed": args.capex_ed,
        "energy_ro": args.energy_ro,
        "energy_med": args.energy_med,
        "energy_ed": args.energy_ed,
        "turbine_mult": args.turbine_mult}

    summary = build_summary_row(model_disp, hot_max_val, cold_max_val, 
                                extra_fields=sweep_metadata)

    # save unique files for each job
    pd.DataFrame([summary]).to_csv(f"{base_dir}/summary_{args.job_id}.csv", index=False)
    write_dispatch_csv(model_disp, f"{base_dir}/hourly_{args.job_id}.csv")

if __name__ == "__main__":
    main()
