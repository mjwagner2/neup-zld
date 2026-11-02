import pyomo.environ as pyo
import pandas as pd
import os
import csv
import glob
# from zld_model import build_model
from zld_dynamic import build_model

wd = sys.path[0]

def run_2d_sensitivity(sweep_param: list, verbose=False):

    N_design   = 24
    N_dispatch = 8760

    eff_design   = pd.read_csv(wd + "/avg_day_efficiency.csv")["Average_Efficiency"].to_numpy()[:N_design]
    eff_dispatch = pd.read_csv(wd + "/year_efficiency.csv")["Efficiency"].to_numpy()[:N_dispatch]

    solver = pyo.SolverFactory("gurobi")
    solver.options["NonConvex"] = 2

    ##################################################
    # master settings
    ##################################################

    # choose one
    sweep_mode = "price_x_price"
    # sweep_mode = "price_x_process"
    # sweep_mode = "process_x_process"
    # sweep_mode = "schedule_x_process"
    # sweep_mode = "price_x_cost"
    # sweep_mode = "cost_x_cost"

    turbine_scale = 2

    ##################################################
    # sweep definitions
    ##################################################

    # price sweeps
    sweep_name_1   = "lithium"
    sweep_values_1 = [0.25, 0.5, 1.0, 2.5, 5.0, 7.5, 10.0, 20.0]

    sweep_name_2   = "water"
    sweep_values_2 = [0.25, 0.5, 1.0, 1.5, 2.0, 5.0]

    # process sweeps
    # examples:
    # RO_elec, MED_elec, ED_elec, CRY_elec
    # RO_heat, MED_heat, ED_heat, CRY_heat
    # RO_capex, MED_capex, ED_capex, CRY_capex
    process_name_1   = "ED_elec"
    process_values_1 = [0.7, 0.85, 1.0, 1.15, 1.3]

    process_name_2   = "ED_capex"
    process_values_2 = [0.5, 0.75, 1.0, 1.25, 1.5]

    # cost sweeps
    # examples depend on what add_economic_params supports, such as:
    # tes_hot, tes_cold, ramp_power
    cost_name_1   = "tes_hot"
    cost_values_1 = [0.5, 0.75, 1.0, 1.25, 1.5]

    cost_name_2   = "tes_cold"
    cost_values_2 = [0.5, 0.75, 1.0, 1.25, 1.5]

    ##################################################
    # schedule files
    ##################################################

    price_schedule_files = sorted(
        glob.glob(os.path.join("stretched_lmp_cases", "lmp_stretched_*.csv")))

    def get_design_schedule_file(dispatch_file):
        name = os.path.basename(dispatch_file)
        tag = name.replace("lmp_stretched_", "").replace(".csv", "")
        return os.path.join("stretched_lmp_cases", f"avg_day_lmp_{tag}.csv")

    ##################################################
    # output folder
    ##################################################

    turb_tag = str(turbine_scale)

    if sweep_mode == "price_x_price":
        results_dir = os.path.join("sweeps", "3param", f"{sweep_name_1}_{sweep_name_2}_turb")
    elif sweep_mode == "price_x_process":
        results_dir = os.path.join("sweeps", "3param", f"{sweep_name_1}_{process_name_1}_turb")
    elif sweep_mode == "process_x_process":
        results_dir = os.path.join("sweeps", "3param", f"{process_name_1}_{process_name_2}_turb")
    elif sweep_mode == "schedule_x_process":
        results_dir = os.path.join("sweeps", "3param", f"lmpschedule_{process_name_1}_turb")
    elif sweep_mode == "price_x_cost":
        results_dir = os.path.join("sweeps", "3param", f"{sweep_name_1}_{cost_name_1}_turb")
    elif sweep_mode == "cost_x_cost":
        results_dir = os.path.join("sweeps", "3param", f"{cost_name_1}_{cost_name_2}_turb")
    else:
        raise ValueError(f"Unknown sweep_mode: {sweep_mode}")

    os.makedirs(results_dir, exist_ok=True)

    summary_data = []

    ##################################################
    # helpers
    ##################################################

    def safe_tag(x):
        return str(x)

    def solve_case(
        elec_design,
        elec_dispatch,
        current_prices,
        current_costs,
        current_process,
        case_label,
        filename_stub,
        extra_fields):
        
        print(f"\n>>> Solving case: {case_label}")

        model_d = build_model(
            N_design,
            elec_design,
            eff_design,
            price_mults=current_prices,
            cost_mults=current_costs,
            process_mults=current_process,
            turbine_mults=turbine_scale)

        res_d = solver.solve(model_d, tee=verbose)

        if res_d.solver.termination_condition != pyo.TerminationCondition.optimal:
            print(f"Design model failed for {case_label}")
            return None

        caps     = {p: pyo.value(model_d.v_dot_cap[p]) for p in model_d.processes}
        hot_max  = pyo.value(model_d.m_hot_tes_max)
        cold_max = pyo.value(model_d.m_cold_tes_max)
        y_p      = {p: pyo.value(model_d.y_process_active[p]) for p in model_d.processes}
        y_l      = {l: pyo.value(model_d.y_link_active[l]) for l in model_d.links}

        model_disp = build_model(
            N_dispatch,
            elec_dispatch,
            eff_dispatch,
            price_mults=current_prices,
            cost_mults=current_costs,
            process_mults=current_process,
            turbine_mults=turbine_scale
        )

        for p in model_disp.processes:
            model_disp.v_dot_cap[p].fix(caps[p])
            model_disp.y_process_active[p].fix(round(y_p[p]))

        for l in model_disp.links:
            model_disp.y_link_active[l].fix(round(y_l[l]))

        model_disp.m_hot_tes_max.fix(hot_max)
        model_disp.m_cold_tes_max.fix(cold_max)

        res_disp = solver.solve(model_disp, tee=verbose)

        if res_disp.solver.termination_condition != pyo.TerminationCondition.optimal:
            print(f"Dispatch model failed for {case_label}")
            return None

        hourly_filename = os.path.join(results_dir, f"{filename_stub}.csv")
        write_dispatch_csv(model_disp, hourly_filename)

        return build_summary_row(
            model_disp,
            hot_max,
            cold_max,
            extra_fields=extra_fields)

    ##################################################
    # price x price
    ##################################################

    if sweep_mode == "price_x_price":

        elec_design   = pd.read_csv("avg_day_lmp.csv")["LMP"].to_numpy()[:N_design]
        elec_dispatch = pd.read_csv("year_lmp.csv")["LMP"].to_numpy()[:N_dispatch]

        for val1 in sweep_values_1:
            for val2 in sweep_values_2:

                row = solve_case(
                    elec_design=elec_design,
                    elec_dispatch=elec_dispatch,
                    current_prices={sweep_name_1: val1, sweep_name_2: val2},
                    current_costs={},
                    current_process={},
                    case_label=f"{sweep_name_1}={val1}, {sweep_name_2}={val2}, turbine_scale={turbine_scale}",
                    filename_stub=f"dispatch_{sweep_name_1}_{safe_tag(val1)}_{sweep_name_2}_{safe_tag(val2)}",
                    extra_fields={
                        sweep_name_1: val1,
                        sweep_name_2: val2,
                        "schedule_name": "base",
                        "turbine_scale": turbine_scale})

                if row is not None:
                    summary_data.append(row)

    ##################################################
    # price x process
    ##################################################

    elif sweep_mode == "price_x_process":

        elec_design   = pd.read_csv(wd + "/avg_day_lmp.csv")["LMP"].to_numpy()[:N_design]
        elec_dispatch = pd.read_csv(wd + "/year_lmp.csv")["LMP"].to_numpy()[:N_dispatch]

        for val1 in sweep_values_1:
            for pval in process_values_1:

                row = solve_case(
                    elec_design=elec_design,
                    elec_dispatch=elec_dispatch,
                    current_prices={sweep_name_1: val1},
                    current_costs={},
                    current_process={process_name_1: pval},
                    case_label=f"{sweep_name_1}={val1}, {process_name_1}={pval}, turbine_scale={turbine_scale}",
                    filename_stub=f"dispatch_{sweep_name_1}_{safe_tag(val1)}_{process_name_1}_{safe_tag(pval)}",
                    extra_fields={
                        sweep_name_1: val1,
                        process_name_1: pval,
                        "schedule_name": "base",
                        "turbine_scale": turbine_scale})

                if row is not None:
                    summary_data.append(row)

    ##################################################
    # process x process
    ##################################################

    elif sweep_mode == "process_x_process":

        elec_design   = pd.read_csv(wd + "/avg_day_lmp.csv")["LMP"].to_numpy()[:N_design]
        elec_dispatch = pd.read_csv(wd + "/year_lmp.csv")["LMP"].to_numpy()[:N_dispatch]

        for val1 in process_values_1:
            for val2 in process_values_2:

                row = solve_case(
                    elec_design=elec_design,
                    elec_dispatch=elec_dispatch,
                    current_prices={},
                    current_costs={},
                    current_process={process_name_1: val1, process_name_2: val2},
                    case_label=f"{process_name_1}={val1}, {process_name_2}={val2}, turbine_scale={turbine_scale}",
                    filename_stub=f"dispatch_{process_name_1}_{safe_tag(val1)}_{process_name_2}_{safe_tag(val2)}",
                    extra_fields={
                        process_name_1: val1,
                        process_name_2: val2,
                        "schedule_name": "base",
                        "turbine_scale": turbine_scale})

                if row is not None:
                    summary_data.append(row)

    ##################################################
    # schedule x process
    ##################################################

    elif sweep_mode == "schedule_x_process":

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

            for pval in process_values_1:

                row = solve_case(
                    elec_design=elec_design,
                    elec_dispatch=elec_dispatch,
                    current_prices={},
                    current_costs={},
                    current_process={process_name_1: pval},
                    case_label=f"schedule={schedule_name}, {process_name_1}={pval}, turbine_scale={turbine_scale}",
                    filename_stub=f"dispatch_schedule_{schedule_name}_{process_name_1}_{safe_tag(pval)}",
                    extra_fields={
                        "schedule_name": schedule_name,
                        process_name_1: pval,
                        "turbine_scale": turbine_scale})

                if row is not None:
                    summary_data.append(row)

    ##################################################
    # price x cost
    ##################################################

    elif sweep_mode == "price_x_cost":

        elec_design   = pd.read_csv(wd + "/avg_day_lmp.csv")["LMP"].to_numpy()[:N_design]
        elec_dispatch = pd.read_csv(wd + "/year_lmp.csv")["LMP"].to_numpy()[:N_dispatch]

        for val1 in sweep_values_1:
            for cval in cost_values_1:

                row = solve_case(
                    elec_design=elec_design,
                    elec_dispatch=elec_dispatch,
                    current_prices={sweep_name_1: val1},
                    current_costs={cost_name_1: cval},
                    current_process={},
                    case_label=f"{sweep_name_1}={val1}, {cost_name_1}={cval}, turbine_scale={turbine_scale}",
                    filename_stub=f"dispatch_{sweep_name_1}_{safe_tag(val1)}_{cost_name_1}_{safe_tag(cval)}",
                    extra_fields={
                        sweep_name_1: val1,
                        cost_name_1: cval,
                        "schedule_name": "base",
                        "turbine_scale": turbine_scale})

                if row is not None:
                    summary_data.append(row)

    ##################################################
    # cost x cost
    ##################################################

    elif sweep_mode == "cost_x_cost":

        elec_design   = pd.read_csv(wd + "/avg_day_lmp.csv")["LMP"].to_numpy()[:N_design]
        elec_dispatch = pd.read_csv(wd + "/year_lmp.csv")["LMP"].to_numpy()[:N_dispatch]

        for val1 in cost_values_1:
            for val2 in cost_values_2:

                row = solve_case(
                    elec_design=elec_design,
                    elec_dispatch=elec_dispatch,
                    current_prices={},
                    current_costs={cost_name_1: val1, cost_name_2: val2},
                    current_process={},
                    case_label=f"{cost_name_1}={val1}, {cost_name_2}={val2}, turbine_scale={turbine_scale}",
                    filename_stub=f"dispatch_{cost_name_1}_{safe_tag(val1)}_{cost_name_2}_{safe_tag(val2)}",
                    extra_fields={
                        cost_name_1: val1,
                        cost_name_2: val2,
                        "schedule_name": "base",
                        "turbine_scale": turbine_scale})

                if row is not None:
                    summary_data.append(row)

    else:
        raise ValueError(f"Unknown sweep_mode: {sweep_mode}")

    df_summary = pd.DataFrame(summary_data)
    summary_file = os.path.join(results_dir, "sweep_summary_2d.csv")
    df_summary.to_csv(summary_file, index=False)
    print(f"\nSweep complete. Summary saved to {summary_file}")


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
            "li_recovered_kg",]

        for p in model_disp.processes:
            header.append(f"v_in_{p}")
            header.append(f"v_dil_{p}")

        w.writerow(header)

        for t in model_disp.T:
            hot_tes_kWh = (
                pyo.value(model_disp.m_hot_tes[t])
                * model_disp.Cp_salt.value
                * model_disp.Delta_T_hx1_salt.value / 3600)
            
            cold_tes_kWh = (
                pyo.value(model_disp.m_cold_tes[t])
                * model_disp.Delta_H_hx2_water.value / 3600)

            row = [
                t,
                hot_tes_kWh,
                cold_tes_kWh,
                pyo.value(model_disp.elec_sold[t]),
                pyo.value(model_disp.w_dot_gen[t]),
                pyo.value(model_disp.m_dot_extract[t]),
                pyo.value(model_disp.li_recovered_ed[t]),]

            for p in model_disp.processes:
                row.append(pyo.value(model_disp.v_dot_in[t, p]))
                row.append(pyo.value(model_disp.v_dot_dil[t, p]))

            w.writerow(row)

    print(f"Saved: {hourly_filename}")


def build_summary_row(model_disp, hot_max, cold_max, extra_fields=None):
    row = {
        "annual_profit":         pyo.value(model_disp.obj),
        "hot_tes_max_kg":        hot_max,
        "cold_tes_max_kg":       cold_max,
        "hot_tes_max_kWh":       hot_max * model_disp.Cp_salt.value * model_disp.Delta_T_hx1_salt.value / 3600,
        "cold_tes_max_kWh":      cold_max * model_disp.Delta_H_hx2_water.value / 3600,
        "total_li_recovered_kg": sum(pyo.value(model_disp.li_recovered_ed[t]) for t in model_disp.T),
        "total_elec_sold_MWh":   sum(pyo.value(model_disp.elec_sold[t]) for t in model_disp.T) / 1000,}
    if extra_fields:
        row.update(extra_fields)
    return row


if __name__ == "__main__":
    run_2d_sensitivity()
