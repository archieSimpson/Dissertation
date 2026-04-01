from __future__ import annotations

import json
import dataclasses
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from sheep_sim.simulation import SimulationResult

def export_run(result: "SimulationResult", output_dir: Path, save_png: bool = True) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result.metrics.group_dataframe().to_csv(output_dir / "metrics.csv", index=False)
    result.metrics.position_dataframe().to_csv(output_dir / "positions.csv", index=False)
    result.metrics.field_dataframe().to_csv(output_dir / "field_health.csv", index=False)

    cfg_dict = _config_to_dict(result.cfg)
    with open(output_dir / "config.json", "w") as fh:
        json.dump(cfg_dict, fh, indent=2)

    if save_png:
        from sheep_sim.rendering.static import save_final_frame
        save_final_frame(
            result.environment,
            result.food,
            result.flock,
            output_dir / "final_frame.png",
            f"Sheep simulation — {result.cfg.scenario} (seed={result.cfg.seed})",
        )

    return output_dir

def load_run(output_dir: Path) -> dict[str, pd.DataFrame]:
    output_dir = Path(output_dir)
    return {
        "metrics":      pd.read_csv(output_dir / "metrics.csv"),
        "positions":    pd.read_csv(output_dir / "positions.csv"),
        "field_health": pd.read_csv(output_dir / "field_health.csv"),
    }

def load_config(output_dir: Path) -> dict:
    with open(Path(output_dir) / "config.json") as fh:
        return json.load(fh)

def _config_to_dict(cfg) -> dict:
    if dataclasses.is_dataclass(cfg) and not isinstance(cfg, type):
        return {
            f.name: _config_to_dict(getattr(cfg, f.name))
            for f in dataclasses.fields(cfg)
        }
    return cfg
