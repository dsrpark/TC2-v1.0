"""
Load tracker configuration from release config.json.

Detection constants and environmental parameters are hardcoded (fixed for this release).
Operational limits, model-specific overrides (var_*, dim_*, file_pattern),
classifier environment, sfc_vars, and oceanid are read from config.json.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


_config_path = None  # set by CLI before import-time access


def set_config_path(path: str):
    """Override config path. Must be called before any config access."""
    global _config_path, _cfg, _limits, MAX_CENTERS, MAX_TRACKS, MAX_TRACKLEN, OCEANID
    _config_path = path
    _cfg = load_release_config()
    _limits = _cfg.get("tracker", {}).get("limits", {
        "max_centers": 500, "max_tracks": 200000, "max_tracklen": 1000
    })
    MAX_CENTERS = _limits["max_centers"]
    MAX_TRACKS = _limits["max_tracks"]
    MAX_TRACKLEN = _limits["max_tracklen"]
    OCEANID = _cfg.get("tracker", {}).get("oceanid", 0)


def load_release_config():
    """Load config.json from set path or default release/config.json."""
    path = _config_path or str(Path(__file__).parent.parent.parent / "config.json")
    with open(path) as f:
        return json.load(f)


_cfg = load_release_config()
_limits = _cfg["tracker"]["limits"]

# Detection constants (hardcoded)
CRI_VOR = 5.0e-5
CRI_LAT = 45.0
CRI_GEN_LAT = 40.0
MINDIST_KM = 350.0
MINDHR = 24
MAXD_DEG = 2.5
PSL_SEARCH_KM = 500.0
WSPD_SEARCH_KM = 350.0
M_SEARCH_DEG = 7.0
DT_HOURS = 6
OCEANID = _cfg.get("tracker", {}).get("oceanid", 0)

# Operational limits (from config, may vary by dataset)
MAX_CENTERS = _limits["max_centers"]
MAX_TRACKS = _limits["max_tracks"]
MAX_TRACKLEN = _limits["max_tracklen"]

# Environmental parameter constants (hardcoded)
ENV_REGION_DEG = 12.0
ENV_WSPD_RADIUS_KM = 500.0
ENV_VOR_INNER_KM = 350.0
ENV_VOR_OUTER_KM = 800.0
ENV_VOR_OUTER_START_KM = 400.0
ENV_PSL_RADIUS_KM = 200.0
ENV_WARMCORE_RADIUS_KM = 278.0

STANDARD_UNITS = {
    "wind": "m s-1",
    "pressure": "Pa",
    "temperature": "K",
    "vorticity": "s-1",
}

DEFAULT_SFC_VARS = ["psl", "uas", "vas", "ts", "pr"]


def get_active_models() -> list:
    """Get active_models list from config."""
    return _cfg.get("active_models", [])


@dataclass
class ModelConfig:
    name: str
    ncdir: str
    maskfile: str
    styr: int
    edyr: int
    caltype: int = 1
    var_ua: Optional[str] = None
    var_va: Optional[str] = None
    var_ta: Optional[str] = None
    var_psl: Optional[str] = None
    var_mask: Optional[str] = None
    dim_lat: Optional[str] = None
    dim_lon: Optional[str] = None
    dim_lev: Optional[str] = None
    file_pattern: str = "{var}.{year}.nc"
    file_layout: str = "yearly"  # "yearly" | "split_monthly" | "cmip6"
    sfc_vars: List[str] = field(default_factory=lambda: list(DEFAULT_SFC_VARS))


def build_model_config(model_name: str, scenario: str, styr: int, edyr: int,
                       caltype: int = 1) -> ModelConfig:
    """Build ModelConfig for a model+scenario from release config.

    Config can specify ncdir/maskfile per model, or use basedir for legacy layout.
    Supports CMIP6, HighResMIP, reanalysis — any layout as long as config provides paths.
    """
    mcfg = _cfg["models"].get(model_name, {})
    name = f"{model_name}_{scenario}"

    # ncdir: config에 직접 지정 가능, scenario별로도 가능
    if "ncdir" in mcfg:
        ncdir = mcfg["ncdir"]
        if "{scenario}" in ncdir or "{experiment}" in ncdir:
            exp_map = _cfg.get("experiment_map", {"hist": "historical"})
            experiment = exp_map.get(scenario, scenario)
            ncdir = ncdir.format(scenario=scenario, experiment=experiment,
                                 model=model_name, member=mcfg.get("member", ""))
    else:
        basedir = _cfg.get("basedir", "")
        ncdir = f"{basedir}/{model_name}/fortracking/{scenario}"

    # maskfile: config에 직접 지정 가능
    if "maskfile" in mcfg:
        maskfile = mcfg["maskfile"]
    else:
        basedir = _cfg.get("basedir", "")
        maskfile = f"{basedir}/{model_name}/fortracking/landmask.nc"

    # sfc_vars: model → top-level → default
    if "sfc_vars" in mcfg:
        sfc_vars = mcfg["sfc_vars"]
    elif "sfc_vars" in _cfg:
        sfc_vars = _cfg["sfc_vars"]
    else:
        sfc_vars = list(DEFAULT_SFC_VARS)

    return ModelConfig(
        name=name,
        ncdir=ncdir,
        maskfile=maskfile,
        styr=styr,
        edyr=edyr,
        caltype=caltype,
        var_ua=mcfg.get("var_ua"),
        var_va=mcfg.get("var_va"),
        var_ta=mcfg.get("var_ta"),
        var_psl=mcfg.get("var_psl"),
        var_mask=mcfg.get("var_mask"),
        dim_lat=mcfg.get("dim_lat"),
        dim_lon=mcfg.get("dim_lon"),
        dim_lev=mcfg.get("dim_lev"),
        file_pattern=mcfg.get("file_pattern", _cfg.get("file_pattern", "{var}.{year}.nc")),
        file_layout=mcfg.get("file_layout", _cfg.get("file_layout", "yearly")),
        sfc_vars=sfc_vars,
    )
