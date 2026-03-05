import argparse
import os
import json

import carb

from syn_utils.models import State, States

def readSolveStateJson(args: argparse.Namespace) -> dict:
    """Resolve and load solve_state.json.

    Resolution order: args.solve_state → <usdc_dir>/.. → <usdc_dir> → <usdc_dir>/../..
    """
    if args.solve_state and os.path.isfile(args.solve_state):
        path = os.path.abspath(args.solve_state)
        carb.log_info(f"Using CLI-provided solve_state: {path}")
    else:
        usdc_dir = os.path.dirname(os.path.abspath(args.usdc_path))
        path = None
        for rel in ["..", ".", "../.."]:
            candidate = os.path.normpath(os.path.join(usdc_dir, rel, "solve_state.json"))
            if os.path.isfile(candidate):
                path = candidate
                break
    if path is None:
        carb.log_error("solve_state.json not found in any search path")
        raise FileNotFoundError("solve_state.json not found")
    carb.log_info(f"Loaded solve_state.json from {path}")
    with open(path) as f:
        objs = json.load(f)
    objs = objs.get("objs", {})

    states : States = {}

    for obj_key, obj_value in objs.items():
        state = State.from_dict(obj_key, obj_value)
        if not state.active:
            continue
        states.setdefault(state.state_type, {})[state.id] = state
    return states
