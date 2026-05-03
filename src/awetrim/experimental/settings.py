"""AWETrim-aware replacement for awes_ekf.setup.settings.

Provides the same interface as awes_ekf.setup.settings but resolves config
files from this repo's data/ layout (data/<kite>/ekf_config/*.yaml and
data/<kite>/*_config.yaml) instead of the hardcoded data/config/ path.

Re-exports SimulationConfig, TuningParameters, and validate_config from the
installed awes_ekf package so callers only need to change the import line.

Output convention: results/<kite_name>/ekf/<model>_<YYYY>-<MM>-<DD>.h5
matching the repo-wide results/<kite_name>/<analysis_type>/ pattern.
"""

from __future__ import annotations

import h5py
import pandas as pd
from pathlib import Path

import yaml

from awes_ekf.setup.settings import (  # re-export unchanged classes
    SimulationConfig,
    TuningParameters,
    validate_config,
)

_EKF_CONFIG_KEYS = {"simulation_parameters", "tuning_parameters", "kite", "kcu", "tether"}
_KITE_NAME_KEY = "_awetrim_kite_name"

# Root of the AWETrim project (src/awetrim/experimental/ → three parents up).
_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _find_ekf_configs(project_dir: Path) -> list[Path]:
    """Return YAML files under data/ that look like EKF configs."""
    candidates = []
    for yaml_path in sorted((project_dir / "data").rglob("*.yaml")):
        try:
            with yaml_path.open("r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh)
            if isinstance(data, dict) and _EKF_CONFIG_KEYS.issubset(data.keys()):
                candidates.append(yaml_path)
        except Exception:
            pass
    return candidates


def load_config(project_dir: Path | None = None) -> dict:
    """Interactively select and load an EKF configuration file.

    Searches data/ in the project root for YAML files containing the required
    EKF keys (simulation_parameters, tuning_parameters, kite, kcu, tether).

    Injects _awetrim_kite_name into the returned dict so save_ekf_results can
    place output under results/<kite_name>/ekf/ without extra arguments.
    """
    root = Path(project_dir) if project_dir is not None else _PROJECT_ROOT
    configs = _find_ekf_configs(root)

    if not configs:
        raise FileNotFoundError(
            f"No EKF config files found under {root / 'data'}. "
            "Expected YAML files with keys: " + ", ".join(sorted(_EKF_CONFIG_KEYS))
        )

    print("Available configuration files:")
    for idx, path in enumerate(configs, start=1):
        print(f"  {idx}: {path.relative_to(root)}")

    selection = int(input("Select a configuration file by number: ")) - 1
    if not (0 <= selection < len(configs)):
        raise ValueError("Invalid selection.")

    selected = configs[selection]
    with selected.open("r", encoding="utf-8") as fh:
        config_data = yaml.safe_load(fh)

    if not validate_config(config_data):
        raise ValueError(f"Configuration file is missing required data: {selected}")

    # Derive kite directory name from path: data/<kite_name>/...
    kite_name = selected.relative_to(root / "data").parts[0]
    config_data[_KITE_NAME_KEY] = kite_name

    print(f"Configuration loaded from: {selected.relative_to(root)}")
    return config_data


def save_ekf_results(
    ekf_output_df: pd.DataFrame,
    flight_data: pd.DataFrame,
    kite_model: str,
    year: str,
    month: str,
    day: str,
    config_data: dict,
    addition: str = "",
    project_dir: Path | None = None,
) -> Path:
    """Save EKF results to results/<kite_name>/ekf/<model>_<date>.h5.

    Matches the repo convention results/<kite_name>/<analysis>/ and uses an
    absolute path from the project root rather than the CWD.

    Returns the path of the written file.
    """
    root = Path(project_dir) if project_dir is not None else _PROJECT_ROOT
    kite_name = config_data.get(_KITE_NAME_KEY, kite_model)

    out_dir = root / "results" / kite_name / "ekf"
    out_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{kite_model}_{year}-{month}-{day}{addition}.h5"
    h5_path = out_dir / filename

    def _encode_strings(df: pd.DataFrame) -> pd.DataFrame:
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].astype("S")
        return df

    def _sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
        df.columns = (
            df.columns.str.replace(" ", "_")
            .str.replace("(", "")
            .str.replace(")", "")
            .str.replace("/", "_")
        )
        return df

    ekf_output_df = _encode_strings(ekf_output_df.copy())
    flight_data = _sanitize_columns(_encode_strings(flight_data.copy()))

    def _save_dict(group: h5py.Group, d: dict) -> None:
        for key, value in d.items():
            if key.startswith("_"):
                continue  # skip internal AWETrim keys
            if isinstance(value, dict):
                _save_dict(group.create_group(key), value)
            else:
                group.attrs[key] = value.encode("utf-8") if isinstance(value, str) else value

    with h5py.File(h5_path, "w") as hf:
        ekf_group = hf.create_group("ekf_output")
        ekf_group.attrs["description"] = (
            "Extended Kalman Filter output, including system parameters derived "
            "from postprocessing the EKF state vector with experimental data."
        )
        for col in ekf_output_df.columns:
            ekf_group.create_dataset(col, data=ekf_output_df[col].values)

        flight_group = hf.create_group("flight_data")
        flight_group.attrs["description"] = (
            "Experimental data collected during the flight test. "
            "Offsets are applied to orientation data and tether length."
        )
        for col in flight_data.columns:
            flight_group.create_dataset(col, data=flight_data[col].values)

        config_group = hf.create_group("config_data")
        config_group.attrs["description"] = (
            "Configuration data used for the simulation and postprocessing."
        )
        _save_dict(config_group, config_data)

    print(f"EKF results saved to: {h5_path.relative_to(root)}")
    return h5_path
