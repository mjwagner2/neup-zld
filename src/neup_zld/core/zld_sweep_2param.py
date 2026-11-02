import pyomo.environ as pyo
import pandas as pd
import numpy as np
import os
import sys
import csv
import glob
# from zld_model import build_model
from zld_dynamic import build_model
import helpers

wd = sys.path[0]

def run_2d_sensitivity(sweep_params: dict, verbose=False):

    N_design   = 24
    N_dispatch = 8760

    # fixed efficiency data
    eff_design   = pd.read_csv(wd + "/avg_day_efficiency.csv")["Average_Efficiency"].to_numpy()[:N_design]
    eff_dispatch = pd.read_csv(wd + "/year_efficiency.csv")["Efficiency"].to_numpy()[:N_dispatch]

    solver = pyo.SolverFactory("gurobi")
    solver.options["NonConvex"] = 2

    ########## mode selection ##########

    # sweep_mode = "schedule_x_param"
    sweep_mode = "param_x_param"

    ########## parameter sweep ##########
    if len(sweep_params) > 2 or len(sweep_params) < 2  or sweep_params is None:
        raise ValueError("Invalid number of entries for variable 'sweep_param': Must be equal to 2 for a 2-parameter sweep")
    
    [sweep_name_1, sweep_name_2], [sweep_values_1, sweep_values_2] = helpers.unpack_sweep_params(sweep_params)
    
    ########## pricing schedule files ##########

    price_schedule_files = sorted(
        glob.glob(os.path.join("stretched_lmp_cases", "lmp_stretched_*.csv"))
    )

    def get_design_schedule_file(dispatch_file):
        name = os.path.basename(dispatch_file)
        tag = name.replace("lmp_stretched_", "").replace(".csv", "")
        return os.path.join("stretched_lmp_cases", f"avg_day_lmp_{tag}.csv")

    ########## output files ##########

    if sweep_mode == "param_x_param":
        results_dir = os.path.join("sweeps", f"sweep_2var_{sweep_name_1}_{sweep_name_2}")
    else:
        results_dir = os.path.join("sweeps", f"sweep_2var_lmp_stretch_{sweep_name_1}")

    os.makedirs(results_dir, exist_ok=True)

    summary_data = []

    ########## parameter x parameter ##########

    if sweep_mode == "param_x_param":

        elec_design   = pd.read_csv(wd + "/avg_day_lmp.csv")["LMP"].to_numpy()[:N_design]
        elec_dispatch = pd.read_csv(wd + "/year_lmp.csv")["LMP"].to_numpy()[:N_dispatch]

        for val1 in sweep_values_1:
            for val2 in sweep_values_2:

                print(f"\n>>> Solving case: {sweep_name_1}={val1}, {sweep_name_2}={val2}")
                current_prices = {sweep_name_1: val1, sweep_name_2: val2}

                model_d = build_model(N_design, elec_design, eff_design, price_mults=current_prices)
                res_d = solver.solve(model_d, tee=verbose)

                if res_d.solver.termination_condition != pyo.TerminationCondition.optimal:
                    print(f"Design model failed for {sweep_name_1}={val1}, {sweep_name_2}={val2}")
                    continue

                caps     = {p: pyo.value(model_d.v_dot_cap[p]) for p in model_d.processes}
                hot_max  = pyo.value(model_d.m_hot_tes_max)
                cold_max = pyo.value(model_d.m_cold_tes_max)
                y_p      = {p: pyo.value(model_d.y_process_active[p]) for p in model_d.processes}
                y_l      = {l: pyo.value(model_d.y_link_active[l]) for l in model_d.links}

                model_disp = build_model(N_dispatch, elec_dispatch, eff_dispatch, price_mults=current_prices)

                for p in model_disp.processes:
                    model_disp.v_dot_cap[p].fix(caps[p])
                    model_disp.y_process_active[p].fix(round(y_p[p]))

                for l in model_disp.links:
                    model_disp.y_link_active[l].fix(round(y_l[l]))

                model_disp.m_hot_tes_max.fix(hot_max)
                model_disp.m_cold_tes_max.fix(cold_max)

                res_disp = solver.solve(model_disp, tee=verbose)

                if res_disp.solver.termination_condition == pyo.TerminationCondition.optimal:
                    val1_str = str(val1).replace(".", "p")
                    val2_str = str(val2).replace(".", "p")

                    hourly_filename = os.path.join(
                        results_dir,
                        f"dispatch_{sweep_name_1}_{val1_str}_{sweep_name_2}_{val2_str}.csv"
                    )

                    write_dispatch_csv(model_disp, hourly_filename)

                    summary_data.append(build_summary_row(
                        model_disp, hot_max, cold_max,
                        extra_fields={
                            sweep_name_1: val1,
                            sweep_name_2: val2,
                            "schedule_name": "base",
                        }
                    ))
                else:
                    print(f"Dispatch model failed for {sweep_name_1}={val1}, {sweep_name_2}={val2}")

    ########## schedule x parameter ##########

    elif sweep_mode == "schedule_x_param":

        if not price_schedule_files:
            raise FileNotFoundError("No files found matching stretched_lmp_cases/lmp_stretched_*.csv")

        for schedule_file in price_schedule_files:
            design_file = get_design_schedule_file(schedule_file)

            if not os.path.exists(design_file):
                print(f"Skipping {schedule_file} because matching design file was not found: {design_file}")
                continue

            elec_design   = pd.read_csv(design_file)["LMP"].to_numpy()[:N_design]
            elec_dispatch = pd.read_csv(schedule_file)["LMP"].to_numpy()[:N_dispatch]

            schedule_name = os.path.splitext(os.path.basename(schedule_file))[0].replace("lmp_stretched_", "")

            for val1 in sweep_values_1:
                print(f"\n>>> Solving case: schedule={schedule_name}, {sweep_name_1}={val1}")
                current_prices = {sweep_name_1: val1}

                model_d = build_model(N_design, elec_design, eff_design, price_mults=current_prices)
                res_d = solver.solve(model_d, tee=verbose)

                if res_d.solver.termination_condition != pyo.TerminationCondition.optimal:
                    print(f"Design model failed for schedule={schedule_name}, {sweep_name_1}={val1}")
                    continue

                caps     = {p: pyo.value(model_d.v_dot_cap[p]) for p in model_d.processes}
                hot_max  = pyo.value(model_d.m_hot_tes_max)
                cold_max = pyo.value(model_d.m_cold_tes_max)
                y_p      = {p: pyo.value(model_d.y_process_active[p]) for p in model_d.processes}
                y_l      = {l: pyo.value(model_d.y_link_active[l]) for l in model_d.links}

                model_disp = build_model(N_dispatch, elec_dispatch, eff_dispatch, price_mults=current_prices)

                for p in model_disp.processes:
                    model_disp.v_dot_cap[p].fix(caps[p])
                    model_disp.y_process_active[p].fix(round(y_p[p]))

                for l in model_disp.links:
                    model_disp.y_link_active[l].fix(round(y_l[l]))

                model_disp.m_hot_tes_max.fix(hot_max)
                model_disp.m_cold_tes_max.fix(cold_max)

                res_disp = solver.solve(model_disp, tee=verbose)

                if res_disp.solver.termination_condition == pyo.TerminationCondition.optimal:
                    val1_str = str(val1).replace(".", "p")

                    hourly_filename = os.path.join(
                        results_dir,
                        f"dispatch_schedule_{schedule_name}_{sweep_name_1}_{val1_str}.csv"
                    )

                    write_dispatch_csv(model_disp, hourly_filename)

                    summary_data.append(build_summary_row(
                        model_disp, hot_max, cold_max,
                        extra_fields={
                            "schedule_name": schedule_name,
                            sweep_name_1: val1,
                        }
                    ))
                else:
                    print(f"Dispatch model failed for schedule={schedule_name}, {sweep_name_1}={val1}")

    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv(os.path.join(results_dir, "sweep_summary_2d.csv"), index=False)
    print(f"\nSweep complete. Summary saved to {os.path.join(results_dir, 'sweep_summary_2d.csv')}")


def write_dispatch_csv(model_disp, hourly_filename):
    with open(hourly_filename, "w", newline="") as f:
        w = csv.writer(f)

        header = [
            "hour",
            "hot_tes_kWh",
            "cold_tes_kWh",
            "elec_sold_kWe",
            "elec_gen_kWe",
            "m_dot_extract_kg_s",
            "li_recovered_kg",
        ]

        for p in model_disp.processes:
            header.append(f"v_in_{p}")
            header.append(f"v_dil_{p}")
        w.writerow(header)

        for t in model_disp.T:
            hot_tes_kWh = (
                pyo.value(model_disp.m_hot_tes[t])
                * model_disp.Cp_salt.value
                * model_disp.Delta_T_hx1_salt.value / 3600
            )
            cold_tes_kWh = (
                pyo.value(model_disp.m_cold_tes[t])
                * model_disp.Delta_H_hx2_water.value / 3600
            )

            row = [
                t,
                hot_tes_kWh,
                cold_tes_kWh,
                pyo.value(model_disp.elec_sold[t]),
                pyo.value(model_disp.w_dot_gen[t]),
                pyo.value(model_disp.m_dot_extract[t]),
                pyo.value(model_disp.li_recovered_ed[t]),
            ]

            for p in model_disp.processes:
                row.append(pyo.value(model_disp.v_dot_in[t, p]))
                row.append(pyo.value(model_disp.v_dot_dil[t, p]))

            w.writerow(row)

    print(f"Saved: {hourly_filename}")


def build_summary_row(model_disp, hot_max, cold_max, extra_fields=None):
    row = {
        "annual_profit": pyo.value(model_disp.obj),
        "hot_tes_max_kg": hot_max,
        "cold_tes_max_kg": cold_max,
        "hot_tes_max_kWh": hot_max * model_disp.Cp_salt.value * model_disp.Delta_T_hx1_salt.value / 3600,
        "cold_tes_max_kWh": cold_max * model_disp.Delta_H_hx2_water.value / 3600,
        "total_li_recovered_kg": sum(pyo.value(model_disp.li_recovered_ed[t]) for t in model_disp.T),
        "total_elec_sold_MWh": sum(pyo.value(model_disp.elec_sold[t]) for t in model_disp.T) / 1000,
    }
    if extra_fields:
        row.update(extra_fields)
    return row


if __name__ == "__main__":
    run_2d_sensitivity()
