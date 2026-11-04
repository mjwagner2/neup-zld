"""
heatmap_absolute_value_matched_scheme_heat_capacity.py

Run the steady-state Pyomo model over any two selected parameters and save two
figures from the same sweep dataframe:

    1. selected-route heatmap
    2. heat-source stacked-cell heatmap
    3. matched process-capacity heatmaps

The heat-source plot is filtered using the same selected-route label as the
route heatmap. Cells labeled None or RO are left blank. Cells whose selected
route contains MED and/or CRY are drawn as 100% stacked bars showing the share
of total process heat supplied by C1-C7.

This avoids mismatches caused by tiny q_used or v_dot_in artifacts in inactive
thermal processes.

Outputs:
    - steady_state_sweep_results.csv
    - scheme_heatmap_matched_<x>_vs_<y>.png
    - cycle_heat_share_matched_to_scheme_<x>_vs_<y>.png
    - process_capacities_matched_<x>_vs_<y>.png
    - cycle_count_matched_<x>_vs_<y>.png

No PDF files are created.
"""

import argparse
import contextlib
import importlib.util
import inspect
import io
import pathlib
import sys
import traceback

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd


# =============================================================================
# User-adjustable thresholds
# =============================================================================

FLOW_TOL = 10.0
ENERGY_TOL = 1e-6


# =============================================================================
# Nominal absolute values
# =============================================================================

LI_PRICE_BASE = 70.0
WATER_PRICE_BASE = 1.0
SALT_PRICE_BASE = 0.01
ELEC_PRICE_BASE = 0.10

RO_ELEC_REQ_BASE = 1.5
MED_ELEC_REQ_BASE = 1.0
ED_ELEC_REQ_BASE = 5.0

CAPEX_RO_BASE = 700.0
CAPEX_MED_BASE = 2000.0
CAPEX_ED_BASE = 1000.0
CAPEX_CRY_BASE = 3000.0


# =============================================================================
# Colors
# =============================================================================

COLORS = {
    "RO": "#4C78A8",
    "MED": "#F58518",
    "ED": "#E45756",
    "CRY": "#72B7B2",
}

THESIS_BLUE = "#4C78A8"
THESIS_ORANGE = "#F58518"
THESIS_TEAL = "#72B7B2"
THESIS_CORAL = "#E45756"
THESIS_LIGHT = "#F7F7F7"
THESIS_LIGHT_GRAY = "#D0D0D0"
THESIS_DARK = "#222222"

THESIS_BLUE_CMAP = LinearSegmentedColormap.from_list(
    "thesis_blue_cmap",
    ["#F7F7F7", "#D7E5F3", "#9EC1DF", "#4C78A8"]
)

CYCLE_ORDER = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]
PROCESS_ORDER = ["RO", "MED", "ED", "CRY"]

LINK_ORDER = [
    ("SW", "RO"),
    ("SW", "MED"),
    ("SW", "ED"),
    ("SW", "CRY"),
    ("RO", "MED"),
    ("RO", "ED"),
    ("RO", "CRY"),
    ("MED", "ED"),
    ("MED", "CRY"),
    ("ED", "CRY"),
]

CYCLE_COLOR_MAP = {
    # Matches the extraction-curve figure / cycle-temperature order.
    # C1 = 304 C, C2 = 283 C, C3 = 154 C, C4 = 128 C,
    # C5 = 109 C, C6 = 87 C, C7 = 83 C.
    "C1": "#4C78A8",  # 304 deg C, blue
    "C2": "#8CB6D9",  # 283 deg C, light blue
    "C3": "#72B7B2",  # 154 deg C, teal
    "C4": "#E45756",  # 128 deg C, coral
    "C5": "#F58518",  # 109 deg C, orange
    "C6": "#FFBF79",  # 87 deg C, light orange
    "C7": "#BAB0AC",  # 83 deg C, gray
    "No heat": "#D0D0D0",
}

CYCLE_TEMP_LABEL = {
    "C1": "C1 (304 °C)",
    "C2": "C2 (283 °C)",
    "C3": "C3 (154 °C)",
    "C4": "C4 (128 °C)",
    "C5": "C5 (109 °C)",
    "C6": "C6 (87 °C)",
    "C7": "C7 (83 °C)",
}

SCHEME_COLOR_MAP = {
    "None": "#D0D0D0",

    "RO": "#4C78A8",
    "MED": "#F58518",
    "ED": "#E45756",
    "CRY": "#D6F0ED",

    "RO → MED": "#72B7B2",
    "RO → ED": "#8CB6D9",
    "RO → CRY": "#A0CBE8",
    "MED → ED": "#FF9D3D",
    "MED → CRY": "#FECB8B",
    "ED → CRY": "#F28E8C",

    "RO → MED → ED": "#E45756",
    "RO → MED → CRY": "#9CC9C4",
    "RO → ED → CRY": "#D96C6A",
    "MED → ED → CRY": "#F7A35C",
    "RO → MED → ED → CRY": "#FFBF79",

    "Other": "#BAB0AC",
    "Error": "#000000",
}


# =============================================================================
# Sweep parameter definitions
# =============================================================================

PARAM_SPECS = {
    "li_price": {
        "label": "Lithium price ($/kg)",
        "fixed_arg": "li_price",
        "model_kw": "li_price_mult",
        "mode": "mult_from_base",
        "base": LI_PRICE_BASE,
    },
    "water_price": {
        "label": "Water price (\\$/m$^3$)",
        "fixed_arg": "water_price",
        "model_kw": "water_price_mult",
        "mode": "mult_from_base",
        "base": WATER_PRICE_BASE,
    },
    "elec_price": {
        "label": "Electricity price (\\$/kWh$_\\mathrm{e}$)",
        "fixed_arg": "elec_price",
        "model_kw": "elec_price",
        "mode": "direct",
        "base": None,
    },
    "salt_price": {
        "label": "Salt price (\\$/kg)",
        "fixed_arg": "salt_price",
        "model_kw": "salt_price",
        "mode": "direct",
        "base": None,
    },
    "ro_elec_req": {
        "label": "RO electricity requirement (kWh$_\\mathrm{e}$/m$^3$)",
        "fixed_arg": "ro_elec_req",
        "model_kw": "ro_elec_req_mult",
        "mode": "mult_from_base",
        "base": RO_ELEC_REQ_BASE,
    },
    "med_elec_req": {
        "label": "MED electricity requirement (kWh$_\\mathrm{e}$/m$^3$)",
        "fixed_arg": "med_elec_req",
        "model_kw": "med_elec_req_mult",
        "mode": "mult_from_base",
        "base": MED_ELEC_REQ_BASE,
    },
    "ed_elec_req": {
        "label": "ED electricity requirement (kWh$_\\mathrm{e}$/m$^3$)",
        "fixed_arg": "ed_elec_req",
        "model_kw": "ed_elec_req_mult",
        "mode": "mult_from_base",
        "base": ED_ELEC_REQ_BASE,
    },
    "li_price_mult": {
        "label": "Lithium price multiplier",
        "fixed_arg": "li_price_mult",
        "model_kw": "li_price_mult",
        "mode": "mult_direct",
        "base": None,
    },
    "water_price_mult": {
        "label": "Water price multiplier",
        "fixed_arg": "water_price_mult",
        "model_kw": "water_price_mult",
        "mode": "mult_direct",
        "base": None,
    },
    "ro_elec_req_mult": {
        "label": "RO electricity requirement multiplier",
        "fixed_arg": "ro_elec_req_mult",
        "model_kw": "ro_elec_req_mult",
        "mode": "mult_direct",
        "base": None,
    },
    "med_elec_req_mult": {
        "label": "MED electricity requirement multiplier",
        "fixed_arg": "med_elec_req_mult",
        "model_kw": "med_elec_req_mult",
        "mode": "mult_direct",
        "base": None,
    },
    "ed_elec_req_mult": {
        "label": "ED electricity requirement multiplier",
        "fixed_arg": "ed_elec_req_mult",
        "model_kw": "ed_elec_req_mult",
        "mode": "mult_direct",
        "base": None,
    },
    "capex_ro_mult": {
        "label": "RO capex multiplier",
        "fixed_arg": "capex_ro_mult",
        "model_kw": "capex_ro_mult",
        "mode": "mult_direct",
        "base": None,
    },
    "capex_med_mult": {
        "label": "MED capex multiplier",
        "fixed_arg": "capex_med_mult",
        "model_kw": "capex_med_mult",
        "mode": "mult_direct",
        "base": None,
    },
    "capex_ed_mult": {
        "label": "ED capex multiplier",
        "fixed_arg": "capex_ed_mult",
        "model_kw": "capex_ed_mult",
        "mode": "mult_direct",
        "base": None,
    },
    "capex_cry_mult": {
        "label": "CRY capex multiplier",
        "fixed_arg": "capex_cry_mult",
        "model_kw": "capex_cry_mult",
        "mode": "mult_direct",
        "base": None,
    },
    "capex_ro": {
        "label": "RO capex (\\$/(m$^3$/day))",
        "fixed_arg": "capex_ro",
        "model_kw": "capex_ro_mult",
        "mode": "mult_from_base",
        "base": CAPEX_RO_BASE,
    },
    "capex_med": {
        "label": "MED capex (\\$/(m$^3$/day))",
        "fixed_arg": "capex_med",
        "model_kw": "capex_med_mult",
        "mode": "mult_from_base",
        "base": CAPEX_MED_BASE,
    },
    "capex_ed": {
        "label": "ED capex (\\$/(m$^3$/day))",
        "fixed_arg": "capex_ed",
        "model_kw": "capex_ed_mult",
        "mode": "mult_from_base",
        "base": CAPEX_ED_BASE,
    },
    "capex_cry": {
        "label": "CRY capex (\\$/(m$^3$/day))",
        "fixed_arg": "capex_cry",
        "model_kw": "capex_cry_mult",
        "mode": "mult_from_base",
        "base": CAPEX_CRY_BASE,
    },
}


# =============================================================================
# Arguments
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run steady-state ZLD model sweeps and create a heat-share plot "
            "where each heat-share cell is filtered using the same selected-route label "
            "used for the scheme heatmap."
        )
    )

    parser.add_argument("--model-path", required=True)
    parser.add_argument("--outdir", default="heatmap_absolute_value_heat_share_matched_outputs")

    parser.add_argument("--x-param", required=True, choices=sorted(PARAM_SPECS.keys()))
    parser.add_argument("--x-values", required=True, type=float, nargs="+")
    parser.add_argument("--y-param", required=True, choices=sorted(PARAM_SPECS.keys()))
    parser.add_argument("--y-values", required=True, type=float, nargs="+")

    parser.add_argument("--li-price", type=float, default=LI_PRICE_BASE)
    parser.add_argument("--water-price", type=float, default=WATER_PRICE_BASE)
    parser.add_argument("--elec-price", type=float, default=ELEC_PRICE_BASE)
    parser.add_argument("--salt-price", type=float, default=SALT_PRICE_BASE)
    parser.add_argument("--ro-elec-req", type=float, default=RO_ELEC_REQ_BASE)
    parser.add_argument("--med-elec-req", type=float, default=MED_ELEC_REQ_BASE)
    parser.add_argument("--ed-elec-req", type=float, default=ED_ELEC_REQ_BASE)

    parser.add_argument("--li-price-mult", type=float, default=1.0)
    parser.add_argument("--water-price-mult", type=float, default=1.0)
    parser.add_argument("--ro-elec-req-mult", type=float, default=1.0)
    parser.add_argument("--med-elec-req-mult", type=float, default=1.0)
    parser.add_argument("--ed-elec-req-mult", type=float, default=1.0)
    parser.add_argument("--capex-ro-mult", type=float, default=1.0)
    parser.add_argument("--capex-med-mult", type=float, default=1.0)
    parser.add_argument("--capex-ed-mult", type=float, default=1.0)
    parser.add_argument("--capex-cry-mult", type=float, default=1.0)

    parser.add_argument("--capex-ro", type=float, default=CAPEX_RO_BASE)
    parser.add_argument("--capex-med", type=float, default=CAPEX_MED_BASE)
    parser.add_argument("--capex-ed", type=float, default=CAPEX_ED_BASE)
    parser.add_argument("--capex-cry", type=float, default=CAPEX_CRY_BASE)

    parser.add_argument("--flow-tol", type=float, default=FLOW_TOL)
    parser.add_argument("--energy-tol", type=float, default=ENERGY_TOL)
    parser.add_argument("--font-size", type=float, default=14)
    parser.add_argument("--show-title", action="store_true")
    parser.add_argument("--show-times-symbol", action="store_true")
    parser.add_argument("--verbose-model-output", action="store_true")
    parser.add_argument("--debug", action="store_true")

    return parser.parse_args()


# =============================================================================
# Helpers
# =============================================================================

def make_outdir(path):
    path = pathlib.Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def pyomo_value(x, default=np.nan):
    if x is None:
        return default
    try:
        import pyomo.environ as pyo
        val = pyo.value(x, exception=False)
        if val is None:
            return default
        return float(val)
    except Exception:
        pass
    try:
        return float(x)
    except Exception:
        return default


def get_component(model, name):
    if model is None:
        return None
    return getattr(model, name, None)


def get_indexed_value(component, index, default=np.nan):
    if component is None:
        return default
    try:
        return pyomo_value(component[index], default=default)
    except Exception:
        return default


def format_value_label(x, param_name, show_times_symbol=False):
    try:
        x_float = float(x)
    except Exception:
        return str(x)

    if param_name == "water_price":
        return f"{x_float:.2f}"
    if param_name == "li_price":
        return f"{x_float:.0f}"
    if param_name in ["ro_elec_req", "med_elec_req", "ed_elec_req"]:
        return f"{x_float:g}" if not x_float.is_integer() else f"{x_float:.0f}"
    if param_name.endswith("_mult"):
        label = f"{x_float:g}" if not x_float.is_integer() else str(int(x_float))
        return f"{label}×" if show_times_symbol else label
    return f"{x_float:g}" if not x_float.is_integer() else str(int(x_float))


def axis_label(param_name):
    return PARAM_SPECS[param_name]["label"]


def convert_param_to_model_kw(param_name, value):
    spec = PARAM_SPECS[param_name]
    mode = spec["mode"]
    model_kw = spec["model_kw"]

    if mode in ["direct", "mult_direct"]:
        return model_kw, float(value)
    if mode == "mult_from_base":
        base = spec["base"]
        if base is None or abs(base) < 1e-12:
            raise ValueError(f"Cannot convert {param_name}; invalid base {base}.")
        return model_kw, float(value) / float(base)

    raise ValueError(f"Unknown parameter conversion mode: {mode}")


def fixed_value_for_param(args, param_name):
    spec = PARAM_SPECS[param_name]
    return getattr(args, spec["fixed_arg"])


def selected_model_value(args, selected_values, absolute_name, multiplier_name):
    if args.x_param == absolute_name or args.y_param == absolute_name:
        _, model_val = convert_param_to_model_kw(absolute_name, selected_values[absolute_name])
        return model_val
    if args.x_param == multiplier_name or args.y_param == multiplier_name:
        _, model_val = convert_param_to_model_kw(multiplier_name, selected_values[multiplier_name])
        return model_val
    _, model_val = convert_param_to_model_kw(absolute_name, selected_values[absolute_name])
    return model_val


def selected_capex_model_value(args, selected_values, absolute_name, multiplier_name):
    if args.x_param == absolute_name or args.y_param == absolute_name:
        _, model_val = convert_param_to_model_kw(absolute_name, selected_values[absolute_name])
        return model_val
    if args.x_param == multiplier_name or args.y_param == multiplier_name:
        _, model_val = convert_param_to_model_kw(multiplier_name, selected_values[multiplier_name])
        return model_val
    _, model_val = convert_param_to_model_kw(multiplier_name, selected_values[multiplier_name])
    return model_val


def build_model_kwargs(args, x_value, y_value):
    selected_values = {param_name: fixed_value_for_param(args, param_name) for param_name in PARAM_SPECS}
    selected_values[args.x_param] = x_value
    selected_values[args.y_param] = y_value

    model_kwargs = {
        "li_price_mult": selected_model_value(args, selected_values, "li_price", "li_price_mult"),
        "water_price_mult": selected_model_value(args, selected_values, "water_price", "water_price_mult"),
        "ro_elec_req_mult": selected_model_value(args, selected_values, "ro_elec_req", "ro_elec_req_mult"),
        "med_elec_req_mult": selected_model_value(args, selected_values, "med_elec_req", "med_elec_req_mult"),
        "ed_elec_req_mult": selected_model_value(args, selected_values, "ed_elec_req", "ed_elec_req_mult"),
        "capex_ro_mult": selected_capex_model_value(args, selected_values, "capex_ro", "capex_ro_mult"),
        "capex_med_mult": selected_capex_model_value(args, selected_values, "capex_med", "capex_med_mult"),
        "capex_ed_mult": selected_capex_model_value(args, selected_values, "capex_ed", "capex_ed_mult"),
        "capex_cry_mult": selected_capex_model_value(args, selected_values, "capex_cry", "capex_cry_mult"),
        "salt_price": selected_values["salt_price"],
        "elec_price": selected_values["elec_price"],
        "return_model": True,
    }
    return model_kwargs, selected_values


# =============================================================================
# Load and call model
# =============================================================================

def load_steady_model(model_path):
    model_path = pathlib.Path(model_path).resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    module_name = f"steady_model_{abs(hash(str(model_path)))}"
    spec = importlib.util.spec_from_file_location(module_name, model_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)

    if not hasattr(mod, "main"):
        raise RuntimeError(f"Could not find main() in {model_path}.")
    return mod


def accepted_main_kwargs(main_func, candidate_kwargs):
    sig = inspect.signature(main_func)
    params = sig.parameters
    accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    if accepts_kwargs:
        return candidate_kwargs.copy()
    return {key: val for key, val in candidate_kwargs.items() if key in params}


def call_model(mod, candidate_kwargs, verbose_model_output=False):
    kwargs = accepted_main_kwargs(mod.main, candidate_kwargs)
    if verbose_model_output:
        return mod.main(**kwargs)

    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
        result = mod.main(**kwargs)
    return result


def unwrap_model(result):
    if result is None:
        return None
    if hasattr(result, "component_objects") or hasattr(result, "component_map"):
        return result
    if isinstance(result, dict):
        for key in ["model", "m", "instance", "pyomo_model"]:
            if key in result:
                candidate = result[key]
                if hasattr(candidate, "component_objects") or hasattr(candidate, "component_map"):
                    return candidate
        for candidate in result.values():
            if hasattr(candidate, "component_objects") or hasattr(candidate, "component_map"):
                return candidate
    if isinstance(result, (tuple, list)):
        for candidate in result:
            if hasattr(candidate, "component_objects") or hasattr(candidate, "component_map"):
                return candidate
    return None


# =============================================================================
# Extract model results
# =============================================================================

def extract_objective(model):
    if model is None:
        return np.nan
    try:
        import pyomo.environ as pyo
        objs = list(model.component_objects(pyo.Objective, active=True))
        if not objs:
            return np.nan
        return pyomo_value(objs[0], default=np.nan)
    except Exception:
        return np.nan


def extract_common_values(model):
    values = {}

    for c in CYCLE_ORDER:
        values[f"heat_by_{c}"] = np.nan
        values[f"cycle_active_{c}"] = np.nan
        values[f"w_cap_{c}_kWe"] = np.nan
        values[f"m_extract_{c}_kg_s"] = np.nan
        for p in PROCESS_ORDER:
            values[f"q_{c}_to_{p}"] = np.nan

    values["n_cycles_built"] = np.nan
    values["total_w_cap_kWe"] = np.nan
    values["total_w_cap_MWe"] = np.nan

    for p in PROCESS_ORDER:
        values[f"capacity_{p}_m3_per_day"] = np.nan
        values[f"capacity_{p}_m3_per_h"] = np.nan
        values[f"v_in_{p}_m3_per_h"] = np.nan

    for u, p in LINK_ORDER:
        values[f"v_link_{u}_{p}"] = np.nan

    values["total_heat_allocated"] = np.nan
    values["v_in_MED"] = np.nan
    values["v_in_CRY"] = np.nan
    values["q_used_MED"] = np.nan
    values["q_used_CRY"] = np.nan
    values["thermal_process_heat"] = np.nan
    values["thermal_process_flow"] = np.nan
    values["route_from_links"] = "None"
    values["scheme"] = "None"
    values["objective"] = np.nan

    if model is None:
        return values

    q_allocated = get_component(model, "q_allocated")
    q_used = get_component(model, "q_used")
    v_dot_in = get_component(model, "v_dot_in")
    v_dot_link = get_component(model, "v_dot_link")
    v_dot_cap = get_component(model, "v_dot_cap")
    w_dot_cap = get_component(model, "w_dot_cap")
    m_dot_extract = get_component(model, "m_dot_extract")
    y_cycle_active = get_component(model, "y_cycle_active")
    t_day_to_hour = pyomo_value(get_component(model, "T_day_to_hour"), default=24.0)
    if pd.isna(t_day_to_hour) or t_day_to_hour == 0:
        t_day_to_hour = 24.0

    for p in PROCESS_ORDER:
        v_in_p = get_indexed_value(v_dot_in, p, default=np.nan)
        cap_day_p = get_indexed_value(v_dot_cap, p, default=np.nan)

        values[f"v_in_{p}_m3_per_h"] = v_in_p
        values[f"capacity_{p}_m3_per_day"] = cap_day_p

        if not np.isnan(cap_day_p):
            values[f"capacity_{p}_m3_per_h"] = cap_day_p / t_day_to_hour
        else:
            # Fallback if the model does not expose v_dot_cap for some reason.
            values[f"capacity_{p}_m3_per_h"] = v_in_p

    total_w_cap = 0.0
    n_cycles_built = 0

    for c in CYCLE_ORDER:
        y_c = get_indexed_value(y_cycle_active, c, default=np.nan)
        w_c = get_indexed_value(w_dot_cap, c, default=np.nan)
        m_ext_c = get_indexed_value(m_dot_extract, c, default=np.nan)

        values[f"cycle_active_{c}"] = y_c
        values[f"w_cap_{c}_kWe"] = w_c
        values[f"m_extract_{c}_kg_s"] = m_ext_c

        if not np.isnan(y_c) and y_c > 0.5:
            n_cycles_built += 1
        if not np.isnan(w_c):
            total_w_cap += w_c

    values["n_cycles_built"] = n_cycles_built
    values["total_w_cap_kWe"] = total_w_cap
    values["total_w_cap_MWe"] = total_w_cap / 1000.0

    for u, p in LINK_ORDER:
        values[f"v_link_{u}_{p}"] = get_indexed_value(v_dot_link, (u, p), default=np.nan)

    med_flow = get_indexed_value(v_dot_in, "MED", default=np.nan)
    cry_flow = get_indexed_value(v_dot_in, "CRY", default=np.nan)

    values["v_in_MED"] = med_flow
    values["v_in_CRY"] = cry_flow

    thermal_process_flow = 0.0
    if not np.isnan(med_flow):
        thermal_process_flow += med_flow
    if not np.isnan(cry_flow):
        thermal_process_flow += cry_flow
    values["thermal_process_flow"] = thermal_process_flow

    q_med = get_indexed_value(q_used, "MED", default=np.nan)
    q_cry = get_indexed_value(q_used, "CRY", default=np.nan)

    values["q_used_MED"] = q_med
    values["q_used_CRY"] = q_cry

    thermal_process_heat = 0.0
    if not np.isnan(q_med):
        thermal_process_heat += q_med
    if not np.isnan(q_cry):
        thermal_process_heat += q_cry
    values["thermal_process_heat"] = thermal_process_heat

    total_heat_allocated = 0.0

    for c in CYCLE_ORDER:
        heat_by_cycle = 0.0
        for p in PROCESS_ORDER:
            q_val = get_indexed_value(q_allocated, (c, p), default=np.nan)
            values[f"q_{c}_to_{p}"] = q_val
            if not np.isnan(q_val):
                heat_by_cycle += q_val
        values[f"heat_by_{c}"] = heat_by_cycle
        total_heat_allocated += heat_by_cycle

    values["total_heat_allocated"] = total_heat_allocated
    values["objective"] = extract_objective(model)
    return values


def route_from_row_links(row, flow_tol=FLOW_TOL):
    """
    Reconstruct the selected process route using the same link-flow threshold
    idea as the scheme heatmap. This avoids plotting heat-share bars for tiny
    thermal flows that are not part of the selected route.
    """
    downstream = {}

    for u, p in LINK_ORDER:
        v = row.get(f"v_link_{u}_{p}", np.nan)
        if not pd.isna(v) and v > flow_tol:
            downstream[u] = p

    if "SW" not in downstream:
        return []

    route = []
    current = "SW"
    seen = set()

    while current in downstream:
        nxt = downstream[current]

        if nxt in seen:
            break

        if nxt != "SW":
            route.append(nxt)
            seen.add(nxt)

        current = nxt

    return route


def route_label_from_row_links(row, flow_tol=FLOW_TOL):
    route = route_from_row_links(row, flow_tol=flow_tol)
    if not route:
        return "None"
    return " → ".join(route)


def route_has_thermal_process(row, flow_tol=FLOW_TOL):
    route = route_from_row_links(row, flow_tol=flow_tol)
    return ("MED" in route) or ("CRY" in route)


def extract_run_row(result, args, x_value, y_value, selected_values, model_kwargs):
    model = unwrap_model(result)

    row = {
        "x_param": args.x_param,
        "y_param": args.y_param,
        "x_value": x_value,
        "y_value": y_value,
        "flow_tol": args.flow_tol,
        "energy_tol": args.energy_tol,
    }

    for key, val in selected_values.items():
        row[key] = val
    for key, val in model_kwargs.items():
        if key != "return_model":
            row[f"model_{key}"] = val

    if model is None:
        row["model_found"] = False
        row["error"] = "Could not unwrap model"
        row.update(extract_common_values(None))
        return row

    row["model_found"] = True
    row.update(extract_common_values(model))

    # This is the selected-route label used by BOTH the scheme heatmap and the
    # heat-share filter. Keeping both plots tied to this one label prevents the
    # figures from disagreeing because of tiny q_used or v_dot_in artifacts.
    row["scheme"] = route_label_from_row_links(row, flow_tol=args.flow_tol)
    row["route_from_links"] = row["scheme"]
    return row


# =============================================================================
# Run sweep
# =============================================================================

def run_sweep(args):
    outdir = make_outdir(args.outdir)
    model_path = pathlib.Path(args.model_path).resolve()

    mod = load_steady_model(model_path)

    rows = []
    total_runs = len(args.x_values) * len(args.y_values)
    run_counter = 0

    print("\nRunning steady-state sweep")
    print("---------------------------")
    print(f"Model: {model_path}")
    print(f"Output directory: {outdir}")
    print(f"Runs: {total_runs}")
    print(f"X-axis: {args.x_param} = {args.x_values}")
    print(f"Y-axis: {args.y_param} = {args.y_values}")

    for y_value in args.y_values:
        for x_value in args.x_values:
            run_counter += 1
            print(f"[{run_counter:>3}/{total_runs}] {args.x_param}={x_value}, {args.y_param}={y_value}")
            try:
                model_kwargs, selected_values = build_model_kwargs(args, x_value=x_value, y_value=y_value)
                result = call_model(mod, model_kwargs, verbose_model_output=args.verbose_model_output)
                row = extract_run_row(
                    result=result,
                    args=args,
                    x_value=x_value,
                    y_value=y_value,
                    selected_values=selected_values,
                    model_kwargs=model_kwargs,
                )
            except Exception as exc:
                print(f"    ERROR: {exc}")
                if args.debug:
                    traceback.print_exc()
                row = {
                    "x_param": args.x_param,
                    "y_param": args.y_param,
                    "x_value": x_value,
                    "y_value": y_value,
                    "flow_tol": args.flow_tol,
                    "energy_tol": args.energy_tol,
                    "model_found": False,
                    "error": str(exc),
                }
                row.update(extract_common_values(None))
            rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = outdir / "steady_state_sweep_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved sweep results to: {csv_path}")
    return df


# =============================================================================
# Plotting helpers
# =============================================================================

def apply_plot_style(args):
    plt.rcParams.update({
        "font.size": args.font_size,
        "axes.labelsize": args.font_size + 3,
        "xtick.labelsize": args.font_size,
        "ytick.labelsize": args.font_size,
        "legend.title_fontsize": args.font_size + 3,
        "legend.fontsize": args.font_size,
        "text.color": THESIS_DARK,
        "axes.labelcolor": THESIS_DARK,
        "xtick.color": THESIS_DARK,
        "ytick.color": THESIS_DARK,
    })


def setup_axis_ticks(ax, args):
    x_vals = list(args.x_values)
    y_vals = list(args.y_values)

    ax.set_xticks(np.arange(len(x_vals)))
    ax.set_yticks(np.arange(len(y_vals)))

    ax.set_xticklabels([
        format_value_label(x, args.x_param, args.show_times_symbol)
        for x in x_vals
    ])
    ax.set_yticklabels([
        format_value_label(y, args.y_param, args.show_times_symbol)
        for y in y_vals
    ])

    ax.set_xlabel(axis_label(args.x_param))
    ax.set_ylabel(axis_label(args.y_param))


def scheme_has_thermal_process(scheme):
    if not isinstance(scheme, str):
        return False
    parts = [part.strip() for part in scheme.split("→")]
    return ("MED" in parts) or ("CRY" in parts)


def color_for_scheme(scheme):
    return SCHEME_COLOR_MAP.get(scheme, SCHEME_COLOR_MAP["Other"])


def make_scheme_codes(df):
    schemes_in_data = list(pd.unique(df["scheme"]))
    ordered_schemes = []

    for scheme in SCHEME_COLOR_MAP:
        if scheme in schemes_in_data:
            ordered_schemes.append(scheme)

    for scheme in schemes_in_data:
        if scheme not in ordered_schemes:
            ordered_schemes.append(scheme)

    scheme_to_code = {scheme: idx for idx, scheme in enumerate(ordered_schemes)}
    return scheme_to_code, ordered_schemes


def plot_scheme_heatmap(df, args):
    """Save a selected-route heatmap from the exact same sweep dataframe."""
    outdir = make_outdir(args.outdir)
    apply_plot_style(args)

    x_vals = list(args.x_values)
    y_vals = list(args.y_values)
    scheme_to_code, ordered_schemes = make_scheme_codes(df)

    z = np.full((len(y_vals), len(x_vals)), np.nan)
    for i, y_value in enumerate(y_vals):
        for j, x_value in enumerate(x_vals):
            subset = df[np.isclose(df["y_value"], y_value) & np.isclose(df["x_value"], x_value)]
            if subset.empty:
                continue
            scheme = subset.iloc[0]["scheme"]
            z[i, j] = scheme_to_code.get(scheme, scheme_to_code.get("Other", 0))

    colors = [color_for_scheme(scheme) for scheme in ordered_schemes]
    cmap = ListedColormap(colors)
    bounds = np.arange(len(colors) + 1) - 0.5
    norm = BoundaryNorm(bounds, cmap.N)

    fig_width = max(8.5, 0.9 * len(x_vals) + 3.0)
    fig_height = max(6.5, 0.60 * len(y_vals) + 3.0)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)

    ax.imshow(z, cmap=cmap, norm=norm, origin="lower", aspect="auto")
    setup_axis_ticks(ax, args)

    ax.set_xticks(np.arange(-0.5, len(x_vals), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(y_vals), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    if args.show_title:
        ax.set_title("Selected process route")

    legend_handles = [
        mpatches.Patch(
            facecolor=color_for_scheme(scheme),
            edgecolor="#666666",
            linewidth=0.4,
            label=scheme,
        )
        for scheme in ordered_schemes
    ]

    ax.legend(
        handles=legend_handles,
        title="Selected process route",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=3,
        frameon=False,
    )

    base_name = f"scheme_heatmap_matched_{args.x_param}_vs_{args.y_param}"
    png_path = outdir / f"{base_name}.png"
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved matched scheme heatmap to: {png_path}")


def plot_heat_share_stacked_cells(df, args):
    """
    Create a heatmap-style figure where each cell is a 100% stacked
    horizontal bar showing the fraction of total process heat provided by each
    cycle C1-C7.

    Cells with no process heat are filled gray.
    """
    outdir = make_outdir(args.outdir)
    apply_plot_style(args)

    x_vals = list(args.x_values)
    y_vals = list(args.y_values)

    fig_width = max(8.5, 0.9 * len(x_vals) + 3.0)
    fig_height = max(6.5, 0.60 * len(y_vals) + 3.0)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)

    ax.set_xlim(-0.5, len(x_vals) - 0.5)
    ax.set_ylim(-0.5, len(y_vals) - 0.5)
    ax.set_facecolor("white")

    # Draw each cell.
    for i, y_value in enumerate(y_vals):
        for j, x_value in enumerate(x_vals):
            subset = df[
                np.isclose(df["y_value"], y_value)
                & np.isclose(df["x_value"], x_value)
            ]

            # Draw the cell boundary first.
            outer = mpatches.Rectangle(
                (j - 0.5, i - 0.5),
                1.0,
                1.0,
                facecolor="white",
                edgecolor=THESIS_LIGHT_GRAY,
                linewidth=1.0,
            )
            ax.add_patch(outer)

            if subset.empty:
                continue

            row = subset.iloc[0]
            total_heat = row.get("total_heat_allocated", np.nan)
            scheme = row.get("scheme", "None")

            # IMPORTANT: use the same selected-route label as the scheme heatmap.
            # This means cells labeled None or RO are blank, even if q_used has
            # tiny/tie-breaker MED/CRY values. Cells whose selected route contains
            # MED or CRY are plotted.
            if (
                pd.isna(total_heat)
                or total_heat <= args.energy_tol
                or not scheme_has_thermal_process(scheme)
            ):
                continue

            # The cell is still a 100% stacked heat-source bar, but with a small
            # gutter so adjacent cells do not visually merge into continuous stripes.
            cell_pad_x = 0.08
            cell_pad_y = 0.12
            bar_x0 = j - 0.5 + cell_pad_x
            bar_y0 = i - 0.5 + cell_pad_y
            bar_w = 1.0 - 2.0 * cell_pad_x
            bar_h = 1.0 - 2.0 * cell_pad_y
            cursor = bar_x0

            for c in CYCLE_ORDER:
                heat_c = row.get(f"heat_by_{c}", np.nan)
                if pd.isna(heat_c) or heat_c <= args.energy_tol:
                    continue

                frac = heat_c / total_heat
                seg_w = bar_w * frac

                rect = mpatches.Rectangle(
                    (cursor, bar_y0),
                    seg_w,
                    bar_h,
                    facecolor=CYCLE_COLOR_MAP[c],
                    edgecolor="white",
                    linewidth=0.25,
                )
                ax.add_patch(rect)
                cursor += seg_w

            outline = mpatches.Rectangle(
                (bar_x0, bar_y0),
                bar_w,
                bar_h,
                facecolor="none",
                edgecolor=THESIS_LIGHT_GRAY,
                linewidth=1.0,
            )
            ax.add_patch(outline)

    setup_axis_ticks(ax, args)

    ax.set_xticks(np.arange(-0.5, len(x_vals), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(y_vals), 1), minor=True)
    ax.grid(which="minor", color="#D8D8D8", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    if args.show_title:
        ax.set_title("Heat supplied by cycle (share of total process heat)")

    legend_order = CYCLE_ORDER
    legend_handles = []
    for name in legend_order:
        legend_handles.append(
            mpatches.Patch(
                facecolor=CYCLE_COLOR_MAP[name],
                edgecolor="white",
                linewidth=0.4,
                label=CYCLE_TEMP_LABEL.get(name, name),
            )
        )

    ax.legend(
        handles=legend_handles,
        title="Heat source",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=4,
        frameon=False,
    )

    base_name = f"cycle_heat_share_matched_to_scheme_{args.x_param}_vs_{args.y_param}"
    png_path = outdir / f"{base_name}.png"
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved heat-share stacked-cell plot to: {png_path}")



def _matrix_from_df(df, args, column):
    x_vals = list(args.x_values)
    y_vals = list(args.y_values)
    z = np.full((len(y_vals), len(x_vals)), np.nan)

    for i, y_value in enumerate(y_vals):
        for j, x_value in enumerate(x_vals):
            subset = df[
                np.isclose(df["y_value"], y_value)
                & np.isclose(df["x_value"], x_value)
            ]
            if subset.empty:
                continue
            value = subset.iloc[0].get(column, np.nan)
            try:
                z[i, j] = float(value)
            except Exception:
                z[i, j] = np.nan

    return z


def plot_process_capacity_heatmaps(df, args):
    """
    Save one matched 2x2 figure showing RO, MED, ED, and CRY capacities from
    the same solved sweep used for the scheme and heat-share plots.
    """
    outdir = make_outdir(args.outdir)
    apply_plot_style(args)

    x_vals = list(args.x_values)
    y_vals = list(args.y_values)

    matrices = {
        p: _matrix_from_df(df, args, f"capacity_{p}_m3_per_h")
        for p in PROCESS_ORDER
    }

    finite_vals = []
    for z in matrices.values():
        finite = z[np.isfinite(z)]
        if finite.size:
            finite_vals.extend(finite.tolist())

    vmax = max(finite_vals) if finite_vals else 1.0
    if vmax <= 0:
        vmax = 1.0

    fig_width = max(10.0, 0.85 * len(x_vals) + 4.0)
    fig_height = max(8.0, 0.72 * len(y_vals) + 4.0)
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(fig_width, fig_height),
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )

    axes = axes.ravel()
    last_im = None

    for ax, p in zip(axes, PROCESS_ORDER):
        z = matrices[p]
        z_plot = np.nan_to_num(z, nan=0.0)

        last_im = ax.imshow(
            z_plot,
            origin="lower",
            aspect="auto",
            cmap=THESIS_BLUE_CMAP,
            vmin=0,
            vmax=vmax,
        )

        ax.set_title(f"{p} capacity")
        setup_axis_ticks(ax, args)

        ax.set_xticks(np.arange(-0.5, len(x_vals), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(y_vals), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.2)
        ax.tick_params(which="minor", bottom=False, left=False)

    # Only outer labels, so the 2x2 plot is less crowded.
    axes[0].set_xlabel("")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("")
    axes[3].set_ylabel("")

    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes.tolist(), shrink=0.85)
        cbar.set_label("Capacity (m$^3$/h)")

    base_name = f"process_capacities_matched_{args.x_param}_vs_{args.y_param}"
    png_path = outdir / f"{base_name}.png"
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved matched process-capacity heatmaps to: {png_path}")


def plot_cycle_count_heatmap(df, args):
    """
    Save a simple heatmap showing how many power-cycle configurations are built.
    This helps interpret heat-source cells where only a few cycle configurations
    are available.
    """
    outdir = make_outdir(args.outdir)
    apply_plot_style(args)

    x_vals = list(args.x_values)
    y_vals = list(args.y_values)
    z = _matrix_from_df(df, args, "n_cycles_built")
    z_plot = np.nan_to_num(z, nan=0.0)

    fig_width = max(8.5, 0.9 * len(x_vals) + 3.0)
    fig_height = max(6.5, 0.60 * len(y_vals) + 3.0)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)

    im = ax.imshow(
        z_plot,
        origin="lower",
        aspect="auto",
        cmap=THESIS_BLUE_CMAP,
        vmin=0,
        vmax=max(7, np.nanmax(z_plot) if np.isfinite(z_plot).any() else 7),
    )

    setup_axis_ticks(ax, args)

    ax.set_xticks(np.arange(-0.5, len(x_vals), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(y_vals), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i in range(len(y_vals)):
        for j in range(len(x_vals)):
            val = z_plot[i, j]
            if val > 0:
                ax.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=max(args.font_size - 2, 8))

    if args.show_title:
        ax.set_title("Number of power-cycle configurations built")

    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Cycles built")

    base_name = f"cycle_count_matched_{args.x_param}_vs_{args.y_param}"
    png_path = outdir / f"{base_name}.png"
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved matched cycle-count heatmap to: {png_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    if args.x_param == args.y_param:
        raise ValueError("x-param and y-param must be different.")

    df = run_sweep(args)

    # Save all figures from the same dataframe. Use these together.
    plot_scheme_heatmap(df, args)
    plot_heat_share_stacked_cells(df, args)
    plot_process_capacity_heatmaps(df, args)
    plot_cycle_count_heatmap(df, args)

    print("\nDone.")


if __name__ == "__main__":
    main()
