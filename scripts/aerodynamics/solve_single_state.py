from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from common import (
    add_common_arguments,
    build_body,
    build_system_model,
    output_dir,
    parsed_common,
    print_trim_summary,
    save_figure,
    write_json,
)

from awetrim.aerodynamics.vsm_quasi_steady import solve_vsm_quasi_steady_trim


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Solve one VSM aerodynamic trim state."
    )
    add_common_arguments(parser)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()
    values = parsed_common(args)
    out_dir = output_dir(args, "single_state")

    body, _ = build_body(args)  # unpack tuple, use body only
    result, _ = solve_vsm_quasi_steady_trim(
        body_aero=body,
        center_of_gravity=values["center_of_gravity"],
        reference_point=values["reference_point"],
        system_model=build_system_model(args),
        x_guess=values["x_guess"],
        bounds_lower=values["bounds_lower"],
        bounds_upper=values["bounds_upper"],
        include_gravity=args.include_gravity,
        moment_tolerance=args.moment_tolerance,
        return_timing_breakdown=True,
        max_nfev=args.max_nfev,
    )
    print_trim_summary(result)

    json_path = (
        Path(args.output_json) if args.output_json else out_dir / "trim_result.json"
    )
    write_json(json_path, result)

    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["cmx", "cmy", "cmz", "cfx", "cfy"]
    values_plot = np.r_[
        np.asarray(result["cm"], dtype=float), result["cfx"], result["cfy"]
    ]
    ax.bar(
        labels,
        values_plot,
        color=["#4C78A8", "#4C78A8", "#4C78A8", "#F58518", "#F58518"],
    )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_ylabel("Residual coefficient [-]")
    ax.set_title("VSM aerodynamic trim residuals")
    fig.tight_layout()
    save_figure(fig, out_dir / "trim_residuals.pdf")
    if args.no_show:
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
