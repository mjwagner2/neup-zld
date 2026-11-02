import argparse
import json
import sys

from core.zld_sweep_1param import run_1d_sensitivity
from core.zld_sweep_2param import run_2d_sensitivity
# from neup_zld.core.zld_sweep_3param import run_3d_sensitivity  # I don't think this code works right now, only has a function named "run_2d_sensitivity"

wd = sys.path[0]

def _validate_params(params, label="params"):
    if not isinstance(params, dict):
        raise argparse.ArgumentTypeError(f"{label} must be a JSON object (dict)")
    for key, values in params.items():
        if not isinstance(values, list):
            raise argparse.ArgumentTypeError(
                f"value for {key!r} in {label} must be a list, got {type(values).__name__}"
            )
    return params


def parse_params(text):
    """Parse an inline JSON string into a validated params dict."""
    try:
        params = json.loads(text)
    except json.JSONDecodeError as e:
        raise argparse.ArgumentTypeError(f"params must be valid JSON: {e}")
    return _validate_params(params)


def load_case(case_name, path=wd + "/cases.json"):
    """Load one named case from a JSON file and validate it."""
    try:
        with open(path) as f:
            cases = json.load(f)
    except FileNotFoundError:
        raise argparse.ArgumentTypeError(f"cases file not found: {path}")
    except json.JSONDecodeError as e:
        raise argparse.ArgumentTypeError(f"{path} is not valid JSON: {e}")

    if case_name not in cases:
        raise argparse.ArgumentTypeError(
            f"case {case_name!r} not found in {path}. "
            f"Available cases: {', '.join(cases)}"
        )
    return _validate_params(cases[case_name], label=repr(case_name))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Run the optimization.")
    p.add_argument(
        "--case",
        metavar="NAME",
        help="name of a case to load from the cases file, e.g. case_1",
    )
    p.add_argument(
        "--params",
        type=parse_params,
        metavar="JSON",
        help='inline JSON dict of parameter name -> list of values. Used alone, '
             'it defines the sweep; combined with --case, it adds to the case '
             '(and overrides any parameter with the same name)',
    )
    p.add_argument(
        "--cases-file",
        help="JSON file containing named cases (default: %(default)s)",
    )
    p.add_argument("-v", "--verbose", action="store_true", help="Stream solver log")

    args = p.parse_args(argv)

    if args.case is None and args.params is None:
        p.error("provide --case, --params, or both")

    # If a case was chosen, resolve it to a params dict so downstream code
    # always sees args.params as a dict regardless of which option was used.
    merged = {}
    if args.case is not None:
        if args.cases_file is not None:
            try:
                merged.update(load_case(args.case, args.cases_file))
            except argparse.ArgumentTypeError as e:
                p.error(str(e))
        else:
            try:
                merged.update(load_case(args.case))
            except argparse.ArgumentTypeError as e:
                p.error(str(e))
    if args.params is not None:
        merged.update(args.params)

    args.params = merged
    return args

def run_sweep(sweep_params: dict, verbose: bool = False):

    if len(sweep_params) == 1:
        run_1d_sensitivity(sweep_params, verbose)
    elif len(sweep_params) == 2:
        run_2d_sensitivity(sweep_params, verbose)
    # elif num_params == 3: 
    #     run_3d_sensitivity()


if __name__ == "__main__":
    args = parse_args()
    run_sweep(sweep_params=args.params, verbose=args.verbose)
