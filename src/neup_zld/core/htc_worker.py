import argparse
import csv
import os
import shutil
from pathlib import Path

import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition
import subprocess


os.environ['GRB_LICENSE_FILE'] = '/software/groups/gurobi_opt/gurobi.lic'

# 2. Clean out old Python paths just in case
os.environ.pop('PYTHONHOME', None)
os.environ.pop('PYTHONPATH', None)



def save_debug_file(path, outdir):
    if path and os.path.exists(path):
        Path(outdir).mkdir(parents=True, exist_ok=True)
        shutil.copy(path, os.path.join(outdir, os.path.basename(path)))


from zld_dynamic import build_model


def write_dispatch_csv(model_disp, hourly_filename):
    os.makedirs(os.path.dirname(hourly_filename), exist_ok=True)

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
                * model_disp.Delta_T_hx1_salt.value
                / 3600
            )
            cold_tes_kWh = (
                pyo.value(model_disp.m_cold_tes[t])
                * model_disp.Delta_H_hx2_water.value
                / 3600
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

def build_summary_row(model_disp, hot_max, cold_max, caps_val=None, extra_fields=None):
    row = {
        "annual_profit": pyo.value(model_disp.obj),
        "hot_tes_max_kg": hot_max,
        "cold_tes_max_kg": cold_max,
        "hot_tes_max_kWh": (
            hot_max * model_disp.Cp_salt.value * model_disp.Delta_T_hx1_salt.value / 3600
        ),
        "cold_tes_max_kWh": (
            cold_max * model_disp.Delta_H_hx2_water.value / 3600
        ),
        "total_li_recovered_kg": sum(
            pyo.value(model_disp.li_recovered_ed[t]) for t in model_disp.T
        ),
        "total_elec_sold_MWh": sum(
            pyo.value(model_disp.elec_sold[t]) for t in model_disp.T
        ) / 1000,
     
    }
    if caps_val:
        for p, cap in caps_val.items():
            row[f"v_dot_cap_{p}_m3_day"] = cap

    if extra_fields:
        row.update(extra_fields)

    return row


def get_series(path_options, column_name, n_expected=None):
    """
    Try multiple possible file paths and return the requested column as numpy array.
    This helps when the HTC sandbox places inputs under data/... instead of directly
    in the working directory.
    """
    for path in path_options:
        if os.path.exists(path):
            print(f"Reading: {path}")
            series = pd.read_csv(path, sep=r"\s+")[column_name].to_numpy()
            if n_expected is not None:
                return series[:n_expected]
            return series

    raise FileNotFoundError(
        f"Could not find any of these files: {path_options}"
    )


def check_solve(results, label):
    print(f"{label} solver status: {results.solver.status}")
    print(f"{label} termination condition: {results.solver.termination_condition}")

    if results.solver.status != SolverStatus.ok:
        raise RuntimeError(
            f"{label} solve failed: solver status = {results.solver.status}"
        )

    if results.solver.termination_condition not in {
        TerminationCondition.optimal,
        TerminationCondition.locallyOptimal,
        TerminationCondition.feasible,
    }:
        raise RuntimeError(
            f"{label} solve did not return a usable solution: "
            f"{results.solver.termination_condition}"
        )


def main():
    os.makedirs("results", exist_ok=True)
    parser = argparse.ArgumentParser()

    parser.add_argument("--des_file", type=str, required=True)
    parser.add_argument("--disp_file", type=str, required=True)
    parser.add_argument("--group", type=str, default="Baseline")
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
    parser.add_argument("--thermal_med", type=float, default=1.0)
    parser.add_argument("--salt_price", type=float, default=0.0)
    parser.add_argument("--capex_cry", type=float, default=1.0)
    parser.add_argument("--energy_cry", type=float, default=1.0)

    args = parser.parse_args()

    print("Parsed arguments:")
    print(vars(args))

    # Bundle dictionaries for model
    prices = {
        "lithium": args.li_price,
        "water":   args.water_price,
        "salt":    args.salt_price,
    }

    costs = {
        "tes_hot": args.htes_price,
        "tes_cold": args.ctes_price,
    }

    procs = {
        "RO_capex": args.capex_ro,
        "MED_capex": args.capex_med,
        "ED_capex": args.capex_ed,
        "CRY_capex":args.capex_cry,
        "RO_elec": args.energy_ro,
	    "MED_elec": args.energy_med,
	    "ED_elec": args.energy_ed,
        "CRY_elec":args.energy_cry,
        "MED_thermal":args.thermal_med,
    }

    # Output directory
    base_dir = os.path.join("results", args.group)
    os.makedirs(base_dir, exist_ok=True)

     # Load data
    elec_des = get_series(
        [
            os.path.join("src/neup_zld/data_lmp", f"lmp_{args.des_file}.csv"),
            os.path.join("src/neup_zld/data", "data_lmp", f"lmp_{args.des_file}.csv"),
        ],
        "LMP",
        n_expected=24,
    )

    elec_disp = get_series(
        [
            os.path.join("src/neup_zld/data_lmp", f"lmp_{args.disp_file}.csv"),
            os.path.join("src/neup_zld/data", "data_lmp", f"lmp_{args.disp_file}.csv"),
        ],
        "LMP",
        n_expected=8760,
    )

    eff_des = get_series(
        [
            os.path.join("src/neup_zld/data_effic", f"effic_{args.des_file}.csv"),
            os.path.join("src/neup_zld/data", "data_effic", f"effic_{args.des_file}.csv"),
        ],
        "Efficiency",
        n_expected=24,
    )

    eff_disp = get_series(
        [
            os.path.join("src/neup_zld/data_effic", f"effic_{args.disp_file}.csv"),
            os.path.join("src/neup_zld/data", "data_effic", f"effic_{args.disp_file}.csv"),
        ],
        "Efficiency",
        n_expected=8760,
    )


    solver = pyo.SolverFactory("gurobi", solver_io="python")
    solver.options["NonConvex"] = 2
    solver.options["TimeLimit"] = 36000
    solver.options["NodefileStart"] = 0.5
    solver.options["NodefileDir"] = "."
    solver.options["MIPGap"] = 0.01


    # -------------------------
    # Design solve
    # -------------------------
    print("Building design model...")
    model_d = build_model(
        24,
        elec_des,
        eff_des,
        price_mults=prices,
        cost_mults=costs,
        process_mults=procs,
        turbine_mults=args.turbine_mult,
        circular_storage=True
    )

    print("Solving design model...")
    print("li_price multiplier =", args.li_price)
    print("actual Price_li =", pyo.value(model_d.Price_li))
    results_d = solver.solve(model_d, tee=True, keepfiles=True)


    outdir = f"debug_{args.group}_{args.job_id}_design"

    save_debug_file(getattr(solver, "_log_file", None), outdir)

    problem_files = getattr(solver, "_problem_files", [])
    if problem_files is not None:  # Added this check
        for f in problem_files:
            save_debug_file(f, outdir)

    save_debug_file(getattr(solver, "_soln_file", None), outdir)

    # then your check
    check_solve(results_d, "Design")

    print(results_d)

    # Extract design values
    caps_val = {p: pyo.value(model_d.v_dot_cap[p]) for p in model_d.processes}
    hot_max_val = pyo.value(model_d.m_hot_tes_max)
    cold_max_val = pyo.value(model_d.m_cold_tes_max)
    y_p_val = {p: pyo.value(model_d.y_process_active[p]) for p in model_d.processes}
    y_l_val = {l: pyo.value(model_d.y_link_active[l]) for l in model_d.links}

    print("Design capacities:", caps_val)
    print("Design hot TES max:", hot_max_val)
    print("Design cold TES max:", cold_max_val)
    print("Design process actives:", y_p_val)

    # -------------------------
    # Dispatch solve
    # -------------------------
    print("Building dispatch model...")
    model_disp = build_model(
        8760,
        elec_disp,
        eff_disp,
        price_mults=prices,
        cost_mults=costs,
        process_mults=procs,
        turbine_mults=args.turbine_mult,
        circular_storage=False
    )

    # Fix design variables
    model_disp.m_hot_tes_max.fix(hot_max_val)
    model_disp.m_cold_tes_max.fix(cold_max_val)

    for p in model_disp.processes:
        model_disp.v_dot_cap[p].fix(caps_val[p])
        model_disp.y_process_active[p].fix(y_p_val[p])

    for l in model_disp.links:
        model_disp.y_link_active[l].fix(y_l_val[l])

    print("Solving dispatch model...")
    results_disp = solver.solve(model_disp, tee=True, keepfiles=True)

    outdir = f"debug_{args.group}_{args.job_id}_dispatch"

    save_debug_file(getattr(solver, "_log_file", None), outdir)

    problem_files = getattr(solver, "_problem_files", [])
    if problem_files is not None:  # Added this check
        for f in problem_files:
            save_debug_file(f, outdir)

    save_debug_file(getattr(solver, "_soln_file", None), outdir)

    check_solve(results_disp, "Dispatch")

    print(results_disp) 

    # Save results
    tag = f"{args.des_file.split('.')[0]}_vs_{args.disp_file.split('.')[0]}"

    sweep_metadata = {
        "group": args.group,
        "job_id": args.job_id,
        "scenario": tag,
        "li_price": args.li_price,
        "water_price": args.water_price,
        "salt_price":args.salt_price,
        "ctes_price": args.ctes_price,
        "htes_price": args.htes_price,
        "capex_ro": args.capex_ro,
        "capex_med": args.capex_med,
        "capex_ed": args.capex_ed,
        "capex_cry":args.capex_cry,
        "energy_ro": args.energy_ro,
        "energy_med": args.energy_med,
        "energy_ed": args.energy_ed,
        "energy_cry":args.energy_cry,
        "turbine_mult": args.turbine_mult,
        "thermal_med":args.thermal_med,
    }

    summary = build_summary_row(
        model_disp,
        hot_max_val,
        cold_max_val,
        caps_val=caps_val,
        extra_fields=sweep_metadata,
    )

    summary_path = os.path.join(base_dir, f"summary_{args.job_id}.csv")
    hourly_path = os.path.join(base_dir, f"hourly_{args.job_id}.csv")

    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    print(f"Saved: {summary_path}")

    write_dispatch_csv(model_disp, hourly_path)


if __name__ == "__main__":
    main()
